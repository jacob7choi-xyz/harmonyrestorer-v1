"""1D Operational GAN for audio denoising.

Based on "Blind Restoration of Real-World Audio by 1D Operational GANs"
(Kiranyaz et al., 2022). Generator uses a 10-layer U-Net with Self-ONN
encoder/decoder and skip connections. Discriminator uses Self-ONN only in
the first layer for feature extraction, with standard convolutions for speed.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .self_onn import OptimizedConv1DSelfONN as Conv1DSelfONN

logger = logging.getLogger(__name__)


class OpGANGenerator(nn.Module):
    """U-Net generator with Self-ONN encoder/decoder layers.

    Architecture:
        Encoder: 5 Self-ONN layers with stride-2 downsampling
        Bottleneck: Standard conv + GroupNorm (stability anchor)
        Decoder: 4 Self-ONN layers with interpolation upsampling + skip connections
        Final: Standard conv + Tanh (bounded output)

    Args:
        input_length: Expected input length in samples (default 32000 = 2s @ 16kHz).
        q: Operator count per Self-ONN neuron.
        gradient_clip_value: Max gradient norm for clip_gradients().
    """

    def __init__(
        self,
        input_length: int = 32000,
        q: int = 3,
        gradient_clip_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.q = q
        self.gradient_clip_value = gradient_clip_value

        # Encoder (downsampling)
        self.enc1 = Conv1DSelfONN(1, 16, kernel_size=5, stride=2, padding=2, q=q)
        self.enc2 = Conv1DSelfONN(16, 32, kernel_size=5, stride=2, padding=2, q=q)
        self.enc3 = Conv1DSelfONN(32, 64, kernel_size=5, stride=2, padding=2, q=q)
        self.enc4 = Conv1DSelfONN(64, 128, kernel_size=5, stride=2, padding=2, q=q)
        self.enc5 = Conv1DSelfONN(128, 128, kernel_size=5, stride=2, padding=2, q=q)

        # Bottleneck: standard conv for stability
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Decoder (upsampling with skip connections)
        self.dec1 = Conv1DSelfONN(128, 128, kernel_size=5, stride=1, padding=2, q=q)
        self.dec2 = Conv1DSelfONN(256, 64, kernel_size=5, stride=1, padding=2, q=q)
        self.dec3 = Conv1DSelfONN(128, 32, kernel_size=5, stride=1, padding=2, q=q)
        self.dec4 = Conv1DSelfONN(64, 16, kernel_size=5, stride=1, padding=2, q=q)

        # Final: standard conv for stability
        self.final_conv = nn.Sequential(
            nn.Conv1d(32, 1, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
        )

        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(0.2)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier init with small gain for standard conv/norm layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.

        Args:
            x: Corrupted audio [batch_size, 1, 32000].

        Returns:
            Restored audio [batch_size, 1, 32000].
        """
        x = torch.clamp(x, -1.0, 1.0)

        # Encoder: 32000 -> 16000 -> 8000 -> 4000 -> 2000 -> 1000
        enc1_out = self.leaky_relu(self.enc1(x))
        enc2_out = self.leaky_relu(self.enc2(enc1_out))
        enc3_out = self.leaky_relu(self.enc3(enc2_out))
        enc4_out = self.leaky_relu(self.enc4(enc3_out))
        enc5_out = self.leaky_relu(self.enc5(enc4_out))

        bottleneck_out = self.bottleneck(enc5_out)

        # Decoder: upsample + skip connections
        dec1_out = F.interpolate(
            self.leaky_relu(self.dec1(bottleneck_out)),
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )
        dec1_out = self.dropout(dec1_out)

        skip4 = torch.cat([dec1_out, enc4_out], dim=1)
        dec2_out = F.interpolate(
            self.leaky_relu(self.dec2(skip4)),
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )
        dec2_out = self.dropout(dec2_out)

        skip3 = torch.cat([dec2_out, enc3_out], dim=1)
        dec3_out = F.interpolate(
            self.leaky_relu(self.dec3(skip3)),
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )
        dec3_out = self.dropout(dec3_out)

        skip2 = torch.cat([dec3_out, enc2_out], dim=1)
        dec4_out = F.interpolate(
            self.leaky_relu(self.dec4(skip2)),
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )

        skip1 = torch.cat([dec4_out, enc1_out], dim=1)
        final_input = F.interpolate(
            skip1,
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )

        restored = self.final_conv(final_input)
        return restored  # final_conv ends with Tanh, already in [-1, 1]

    def clip_gradients(self) -> None:
        """Clip gradients for all parameters and Self-ONN sublayers."""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_value)
        for module in self.modules():
            if hasattr(module, "clip_gradients") and module is not self:
                module.clip_gradients()

    def get_operator_usage_summary(self) -> dict[str, torch.Tensor]:
        """Return mean operator usage per encoder/decoder layer."""
        usage: dict[str, torch.Tensor] = {}

        for i, layer in enumerate([self.enc1, self.enc2, self.enc3, self.enc4, self.enc5], 1):
            if hasattr(layer, "operator_probs"):
                usage[f"enc{i}"] = F.softmax(layer.operator_probs, dim=1).mean(dim=0)

        for i, layer in enumerate([self.dec1, self.dec2, self.dec3, self.dec4], 1):
            if hasattr(layer, "operator_probs"):
                usage[f"dec{i}"] = F.softmax(layer.operator_probs, dim=1).mean(dim=0)

        return usage


class OpGANDiscriminator(nn.Module):
    """PatchGAN-style discriminator with Self-ONN first layer.

    Only the first layer uses Self-ONN (nonlinear feature extraction).
    Remaining layers use standard convolutions for computational efficiency.

    Args:
        input_length: Expected input length in samples.
        q: Operator count for the Self-ONN layer.
        gradient_clip_value: Max gradient norm for clip_gradients().
    """

    def __init__(
        self,
        input_length: int = 32000,
        q: int = 2,
        gradient_clip_value: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_length = input_length
        self.q = q
        self.gradient_clip_value = gradient_clip_value

        # Self-ONN for first-layer feature extraction
        self.conv1 = Conv1DSelfONN(1, 32, kernel_size=4, stride=2, padding=1, q=q)

        # Standard convolutions for speed
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(16, 64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GroupNorm(64, 256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=4, stride=1, padding=1),
            nn.GroupNorm(64, 256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv6 = nn.Conv1d(256, 1, kernel_size=4, stride=2, padding=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Xavier init with small gain for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: progressive downsampling to patch-level predictions.

        Args:
            x: Audio [batch_size, 1, 32000].

        Returns:
            Patch validity scores [batch_size, 1, output_length].
        """
        x = torch.clamp(x, -1.0, 1.0)

        x = self.conv1(x)  # Self-ONN: [batch, 32, 16000]
        x = self.conv2(x)  # [batch, 64, 8000]
        x = self.conv3(x)  # [batch, 128, 4000]
        x = self.conv4(x)  # [batch, 256, 2000]
        x = self.conv5(x)  # [batch, 256, 2000]
        x = self.conv6(x)  # [batch, 1, 1000]

        return x

    def clip_gradients(self) -> None:
        """Clip gradients for stable training."""
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip_value)


class OpGANLoss(nn.Module):
    """Composite loss combining adversarial, temporal (L1), and spectral (STFT) terms.

    Uses label smoothing (0.9/0.1) for GAN stability.

    Args:
        lambda_temporal: Weight for L1 temporal loss.
        lambda_spectral: Weight for STFT magnitude loss.
        n_fft: FFT size for spectral loss.
        hop_length: Hop size for spectral loss.
    """

    def __init__(
        self,
        lambda_temporal: float = 10,
        lambda_spectral: float = 5,
        n_fft: int = 128,
        hop_length: int = 64,
    ) -> None:
        super().__init__()
        self.lambda_temporal = lambda_temporal
        self.lambda_spectral = lambda_spectral
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def temporal_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 loss in the time domain."""
        return self.l1_loss(generated, target)

    def spectral_loss(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 loss on STFT magnitudes."""
        window = torch.hann_window(self.n_fft, device=generated.device)

        gen_stft = torch.stft(
            generated.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            normalized=True,
        )
        target_stft = torch.stft(
            target.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
            normalized=True,
        )

        gen_mag = torch.abs(gen_stft) + 1e-8
        target_mag = torch.abs(target_stft) + 1e-8

        return self.l1_loss(gen_mag, target_mag)

    def generator_loss(
        self,
        discriminator_output: torch.Tensor,
        generated: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute generator loss: adversarial + temporal + spectral.

        Args:
            discriminator_output: Discriminator scores for generated audio.
            generated: Generator output.
            target: Ground truth clean audio.

        Returns:
            Tuple of (total_loss, loss_breakdown_dict).
        """
        real_labels = torch.ones_like(discriminator_output) * 0.9
        adversarial_loss = self.mse_loss(discriminator_output, real_labels)

        temporal = self.temporal_loss(generated, target)
        spectral = self.spectral_loss(generated, target)

        total_loss = (
            adversarial_loss + self.lambda_temporal * temporal + self.lambda_spectral * spectral
        )

        return total_loss, {
            "adversarial": adversarial_loss.item(),
            "temporal": temporal.item(),
            "spectral": spectral.item(),
            "total": total_loss.item(),
        }

    def discriminator_loss(
        self,
        real_output: torch.Tensor,
        fake_output: torch.Tensor,
    ) -> torch.Tensor:
        """Compute discriminator loss with label smoothing.

        Args:
            real_output: Discriminator scores for real audio.
            fake_output: Discriminator scores for generated audio.

        Returns:
            Total discriminator loss.
        """
        real_labels = torch.ones_like(real_output) * 0.9
        fake_labels = torch.zeros_like(fake_output) + 0.1

        real_loss = self.mse_loss(real_output, real_labels)
        fake_loss = self.mse_loss(fake_output, fake_labels)

        return 0.5 * (real_loss + fake_loss)


class GradientHealthMonitor:
    """Monitor gradient norms and apply clipping during training.

    Tracks gradient history for post-training analysis and warns on
    pathological gradients (vanishing or exploding).

    Args:
        max_grad_norm: Clip threshold.
        log_frequency: Log every N steps.
    """

    def __init__(self, max_grad_norm: float = 1.0, log_frequency: int = 10) -> None:
        self.max_grad_norm = max_grad_norm
        self.log_frequency = log_frequency
        self.step_count = 0
        self.gradient_history: list[dict[str, float | int]] = []

    def check_and_clip_gradients(self, model: nn.Module, model_name: str = "") -> float:
        """Check gradient health, clip if needed, and log periodically.

        Args:
            model: The model to check.
            model_name: Label for logging.

        Returns:
            Pre-clip gradient norm.
        """
        self.step_count += 1

        total_norm = 0.0
        param_count = 0
        max_grad = 0.0

        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm**2
                max_grad = max(max_grad, param_norm)
                param_count += 1

        total_norm = total_norm**0.5

        if self.step_count % self.log_frequency == 0:
            logger.info(
                "Step %d %s: grad_norm=%.2e, max_grad=%.2e",
                self.step_count,
                model_name,
                total_norm,
                max_grad,
            )

        if total_norm > self.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
            if self.step_count % self.log_frequency == 0:
                logger.info(
                    "  Gradients clipped from %.2e to %.1f",
                    total_norm,
                    self.max_grad_norm,
                )

        if total_norm > 1000:
            logger.warning("Very large gradients detected: %.2e", total_norm)
        elif total_norm < 1e-8 and param_count > 0:
            logger.warning("Very small gradients detected: %.2e", total_norm)

        self.gradient_history.append(
            {
                "step": self.step_count,
                "total_norm": total_norm,
                "max_grad": max_grad,
                "param_count": param_count,
            }
        )

        return total_norm

    def get_gradient_stats(self) -> dict[str, float]:
        """Return summary statistics over gradient history."""
        if not self.gradient_history:
            return {}

        norms = [h["total_norm"] for h in self.gradient_history]
        return {
            "mean_norm": sum(norms) / len(norms),
            "max_norm": max(norms),
            "min_norm": min(norms),
            "recent_norm": norms[-1],
        }
