"""Self-Organizing Neural Network (Self-ONN) layers for 1D audio processing.

Based on "Blind Restoration of Real-World Audio by 1D Operational GANs"
(Kiranyaz et al., 2022). Implements learnable operator selection (sin, cos,
tanh, exp, linear) via softmax-weighted combination.
"""

from __future__ import annotations

import logging
import math
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger(__name__)


class OptimizedSelfONN(nn.Module):
    """Fully-connected Self-ONN layer with operator pruning and gradient checkpointing.

    Each output neuron learns a weighted combination of q nonlinear operators
    (linear, sin, cos, tanh, exp) applied to the input via softmax selection.

    Args:
        input_size: Number of input features.
        output_size: Number of output features.
        q: Number of operators per neuron (default 5).
        bias: Whether to include a bias term.
        prune_threshold: Operators below this weight are pruned at inference.
        use_checkpointing: Use gradient checkpointing to save memory.
        gradient_clip_value: Max gradient norm for clip_gradients().
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        q: int = 5,
        bias: bool = True,
        prune_threshold: float = 0.01,
        use_checkpointing: bool = False,
        gradient_clip_value: float = 1.0,
    ) -> None:
        super().__init__()
        if q < 1:
            raise ValueError(f"q must be >= 1, got {q}")
        self.input_size = input_size
        self.output_size = output_size
        self.q = q
        self.prune_threshold = prune_threshold
        self.use_checkpointing = use_checkpointing
        self.gradient_clip_value = gradient_clip_value

        # Weights: [output_size, input_size, q]
        self.weights = nn.Parameter(torch.empty(output_size, input_size, q))

        # Operator probabilities: [output_size, q]
        self.operator_probs = nn.Parameter(torch.empty(output_size, q))

        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size))
        else:
            self.register_parameter("bias", None)

        # Pruning mask (buffer, not a parameter)
        self.register_buffer("operator_mask", torch.ones(output_size, q, dtype=torch.bool))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Conservative initialization for gradient stability."""
        std = math.sqrt(1.0 / self.input_size) * 0.5
        nn.init.normal_(self.weights, 0, std)

        with torch.no_grad():
            self.operator_probs.fill_(0.1)
            self.operator_probs[:, 0] = 0.5  # Slight preference for linear

    def _vectorized_operators(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Apply all q operators in parallel.

        Args:
            x: Input tensor [batch, input_size].
            w: Weight tensor [output_size, input_size, q].

        Returns:
            Operator outputs [batch, output_size, q].
        """
        x = torch.clamp(x, -5.0, 5.0)

        wx = torch.einsum("bi,oiq->boq", x, w)
        wx = torch.clamp(wx, -10.0, 10.0)

        batch_size = x.size(0)
        outputs = torch.empty(batch_size, self.output_size, self.q, device=x.device, dtype=x.dtype)

        outputs[:, :, 0] = wx[:, :, 0]  # Linear
        outputs[:, :, 1] = torch.sin(wx[:, :, 1])
        outputs[:, :, 2] = torch.cos(wx[:, :, 2])
        outputs[:, :, 3] = torch.tanh(wx[:, :, 3])

        if self.q > 4:
            outputs[:, :, 4] = torch.exp(torch.clamp(wx[:, :, 4], -5, 5))

        return outputs

    def _compute_operator_weights(self) -> torch.Tensor:
        """Softmax over operator probs, with pruning applied at inference."""
        operator_weights = F.softmax(self.operator_probs, dim=1)

        if not self.training:
            operator_weights = operator_weights * self.operator_mask.float()  # type: ignore[operator]
            operator_weights = operator_weights / (operator_weights.sum(dim=1, keepdim=True) + 1e-8)

        return operator_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing."""
        if self.use_checkpointing and self.training:
            return checkpoint(self._forward_impl, x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Core forward: operators -> weighted sum -> bias -> clamp."""
        operator_outputs = self._vectorized_operators(x, self.weights)
        operator_weights = self._compute_operator_weights()
        output = torch.einsum("boq,oq->bo", operator_outputs, operator_weights)

        if self.bias is not None:
            output.add_(self.bias)

        return torch.clamp(output, -10.0, 10.0)

    def clip_gradients(self) -> None:
        """Clip gradients to prevent explosion."""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_value)

    def prune_operators(self, threshold: float | None = None) -> None:
        """Prune low-importance operators for inference speedup."""
        if threshold is None:
            threshold = self.prune_threshold

        with torch.no_grad():
            operator_weights = F.softmax(self.operator_probs, dim=1)
            new_mask = operator_weights > threshold

            # Ensure at least one operator per neuron is active
            for i in range(self.output_size):
                if not new_mask[i].any():
                    best_op = torch.argmax(operator_weights[i])
                    new_mask[i, best_op] = True

            self.operator_mask.copy_(new_mask)  # type: ignore[operator]
            pruned = int((~self.operator_mask).sum().item())  # type: ignore[operator]
            total = self.output_size * self.q
            logger.info("Pruned %d/%d operators (threshold=%.3f)", pruned, total, threshold)

    def get_operator_usage(self) -> torch.Tensor:
        """Return current operator usage probabilities [output_size, q]."""
        return F.softmax(self.operator_probs, dim=1)


class OptimizedConv1DSelfONN(nn.Module):
    """1D Convolutional Self-ONN layer for audio processing.

    Applies q separate convolutions (one per operator), then combines them
    via learnable softmax weights. Supports operator caching at inference.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        q: Number of operators.
        groups: Convolution groups.
        gradient_clip_value: Max gradient norm for clip_gradients().
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        q: int = 5,
        groups: int = 1,
        gradient_clip_value: float = 1.0,
    ) -> None:
        super().__init__()
        if q < 1:
            raise ValueError(f"q must be >= 1, got {q}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.q = q
        self.groups = groups
        self.gradient_clip_value = gradient_clip_value

        # Weight layout: [out_channels, in_channels // groups, kernel_size, q]
        self.conv_weights = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, q)
        )

        # Operator probabilities: [out_channels, q]
        self.operator_probs = nn.Parameter(torch.empty(out_channels, q))

        # Inference cache (lock protects concurrent read/write)
        self._cache_lock = threading.Lock()
        self.register_buffer("_operator_weights_cache", None)
        self.register_buffer("_cache_valid", torch.tensor(False))

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Conservative He initialization for stable training."""
        fan_in = self.in_channels * self.kernel_size
        std = math.sqrt(1.0 / fan_in) * 0.3
        nn.init.normal_(self.conv_weights, 0, std)

        with torch.no_grad():
            self.operator_probs.fill_(0.1)
            self.operator_probs[:, 0] = 0.3  # Prefer linear

    def _get_operator_weights(self) -> torch.Tensor:
        """Cached softmax over operator probs (cache valid only at inference)."""
        if self.training:
            return F.softmax(self.operator_probs, dim=1)

        # Fast path: cache already populated (lock-free, safe under CPython GIL)
        if self._cache_valid:
            return self._operator_weights_cache  # type: ignore[has-type]

        with self._cache_lock:
            if self._cache_valid:
                return self._operator_weights_cache  # type: ignore[has-type]
            weights = F.softmax(self.operator_probs, dim=1)
            self._operator_weights_cache = weights.detach()
            self._cache_valid.fill_(True)  # type: ignore[operator]
            return weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: per-operator conv -> weighted sum -> clamp."""
        x = torch.clamp(x, -3.0, 3.0)

        batch_size, in_channels, length = x.size()
        out_length = (length + 2 * self.padding - self.kernel_size) // self.stride + 1

        outputs = torch.zeros(
            batch_size, self.out_channels, out_length, self.q, device=x.device, dtype=x.dtype
        )

        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding))

        for q_idx in range(self.q):
            conv_out = F.conv1d(
                x, self.conv_weights[:, :, :, q_idx], stride=self.stride, groups=self.groups
            )
            conv_out = torch.clamp(conv_out, -8.0, 8.0)

            if q_idx == 0:
                outputs[:, :, :, q_idx] = conv_out  # Linear
            elif q_idx == 1:
                outputs[:, :, :, q_idx] = torch.sin(conv_out)
            elif q_idx == 2:
                outputs[:, :, :, q_idx] = torch.cos(conv_out)
            elif q_idx == 3:
                outputs[:, :, :, q_idx] = torch.tanh(conv_out)
            elif q_idx == 4:
                outputs[:, :, :, q_idx] = torch.exp(torch.clamp(conv_out, -5, 5))

        operator_weights = self._get_operator_weights()
        final_output = torch.einsum("bcto,co->bct", outputs, operator_weights)

        return torch.clamp(final_output, -5.0, 5.0)

    def clip_gradients(self) -> None:
        """Clip gradients for this layer."""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_value)

    def invalidate_cache(self) -> None:
        """Invalidate operator weights cache."""
        if hasattr(self, "_cache_valid"):
            self._cache_valid.fill_(False)  # type: ignore[operator]

    def train(self, mode: bool = True) -> OptimizedConv1DSelfONN:
        """Override to invalidate cache when entering training mode."""
        super().train(mode)
        if mode:
            self.invalidate_cache()
        return self
