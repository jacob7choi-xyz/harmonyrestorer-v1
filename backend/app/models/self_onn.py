"""Self-Organizing Neural Network (Self-ONN) 1D convolutional layer.

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

logger = logging.getLogger(__name__)


class OptimizedConv1DSelfONN(nn.Module):
    """1D Convolutional Self-ONN layer for audio processing.

    Applies a batched convolution across q operators, then combines them
    via learnable softmax weights. Supports operator caching at inference.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        q: Number of operators (max 5: linear, sin, cos, tanh, exp).
        groups: Convolution groups.
    """

    _NUM_OPERATORS = 5  # linear, sin, cos, tanh, exp

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        q: int = 5,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if q < 1 or q > self._NUM_OPERATORS:
            raise ValueError(f"q must be in [1, {self._NUM_OPERATORS}], got {q}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.q = q
        self.groups = groups

        # Weight layout: [out_channels, in_channels // groups, kernel_size, q]
        self.conv_weights = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, q)
        )

        # Operator probabilities: [out_channels, q]
        self.operator_probs = nn.Parameter(torch.empty(out_channels, q))

        # Inference cache (lock protects concurrent read/write)
        self._cache_lock = threading.Lock()
        self._operator_weights_cache: torch.Tensor | None = None
        self._cache_valid: bool = False

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
        if self._cache_valid and self._operator_weights_cache is not None:
            return self._operator_weights_cache

        with self._cache_lock:
            if self._cache_valid and self._operator_weights_cache is not None:
                return self._operator_weights_cache
            weights = F.softmax(self.operator_probs, dim=1)
            self._operator_weights_cache = weights.detach()
            self._cache_valid = True
            return weights

    _OPERATOR_FNS = [
        lambda t: t,  # linear
        torch.sin,
        torch.cos,
        torch.tanh,
        lambda t: torch.exp(torch.clamp(t, -5, 5)),
    ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: batched conv -> per-operator nonlinearity -> weighted sum."""
        x = torch.clamp(x, -3.0, 3.0)

        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding))

        # Batched convolution: reshape [out, in, k, q] -> [out*q, in, k]
        w = self.conv_weights.permute(3, 0, 1, 2).reshape(
            self.out_channels * self.q, self.in_channels // self.groups, self.kernel_size
        )
        conv_out = F.conv1d(x, w, stride=self.stride, groups=self.groups)
        conv_out = torch.clamp(conv_out, -8.0, 8.0)

        # Reshape to [batch, q, out_channels, out_length]
        batch_size = x.size(0)
        out_length = conv_out.size(2)
        conv_out = conv_out.view(batch_size, self.q, self.out_channels, out_length)

        # Apply operator nonlinearities per slice
        outputs = torch.zeros_like(conv_out)
        for q_idx in range(self.q):
            outputs[:, q_idx] = self._OPERATOR_FNS[q_idx](conv_out[:, q_idx])  # type: ignore[operator]

        # Weighted sum: [batch, q, out_channels, length] -> [batch, out_channels, length]
        operator_weights = self._get_operator_weights()  # [out_channels, q]
        final_output = torch.einsum("bqct,cq->bct", outputs, operator_weights)

        return torch.clamp(final_output, -5.0, 5.0)

    def invalidate_cache(self) -> None:
        """Invalidate operator weights cache."""
        self._cache_valid = False
        self._operator_weights_cache = None

    def train(self, mode: bool = True) -> OptimizedConv1DSelfONN:
        """Override to invalidate cache when entering training mode."""
        super().train(mode)
        if mode:
            self.invalidate_cache()
        return self
