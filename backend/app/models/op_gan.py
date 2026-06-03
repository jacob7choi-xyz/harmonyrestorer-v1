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
        q: Operator count per Self-ONN neuron.
    """

    def __init__(
        self,
        q: int = 3,
    ) -> None:
        super().__init__()
        self.q = q

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

        # Decoder: upsample + skip connections (trim to match encoder sizes)
        dec1_out = F.interpolate(
            self.leaky_relu(self.dec1(bottleneck_out)),
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )
        dec1_out = dec1_out[:, :, : enc4_out.size(2)]
        dec1_out = self.dropout(dec1_out)

        skip4 = torch.cat([dec1_out, enc4_out], dim=1)
        dec2_out = F.interpolate(
            self.leaky_relu(self.dec2(skip4)),
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )
        dec2_out = dec2_out[:, :, : enc3_out.size(2)]
        dec2_out = self.dropout(dec2_out)

        skip3 = torch.cat([dec2_out, enc3_out], dim=1)
        dec3_out = F.interpolate(
            self.leaky_relu(self.dec3(skip3)),
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )
        dec3_out = dec3_out[:, :, : enc2_out.size(2)]
        dec3_out = self.dropout(dec3_out)

        skip2 = torch.cat([dec3_out, enc2_out], dim=1)
        dec4_out = F.interpolate(
            self.leaky_relu(self.dec4(skip2)),
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )
        dec4_out = dec4_out[:, :, : enc1_out.size(2)]

        skip1 = torch.cat([dec4_out, enc1_out], dim=1)
        final_input = F.interpolate(
            skip1,
            scale_factor=2,
            mode="linear",
            align_corners=False,
        )

        restored = self.final_conv(final_input)
        return restored  # final_conv ends with Tanh, already in [-1, 1]


class OpGANDiscriminator(nn.Module):
    """PatchGAN-style discriminator with Self-ONN first layer.

    Only the first layer uses Self-ONN (nonlinear feature extraction).
    Remaining layers use standard convolutions for computational efficiency.

    Args:
        q: Operator count for the Self-ONN layer.
    """

    def __init__(
        self,
        q: int = 2,
    ) -> None:
        super().__init__()
        self.q = q

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
        x = self.conv5(x)  # [batch, 256, 1999] (k=4, s=1, p=1)
        x = self.conv6(x)  # [batch, 1, 999] (k=4, s=2, p=1)

        return x
