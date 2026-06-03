"""Tests for OpGAN generator and discriminator."""

import torch
import torch.nn as nn
from app.models.op_gan import (
    OpGANDiscriminator,
    OpGANGenerator,
)


class TestOpGANGenerator:
    """Tests for U-Net generator."""

    def test_output_shape(self) -> None:
        """Output shape matches input shape."""
        gen = OpGANGenerator(q=2)
        x = torch.randn(1, 1, 320)
        out = gen(x)
        assert out.shape == (1, 1, 320)

    def test_output_shape_batch(self) -> None:
        """Batch dimension is preserved."""
        gen = OpGANGenerator(q=2)
        x = torch.randn(2, 1, 320)
        out = gen(x)
        assert out.shape == (2, 1, 320)

    def test_output_bounded_by_tanh(self) -> None:
        """Output is in [-1, 1] due to final Tanh."""
        gen = OpGANGenerator(q=2)
        x = torch.randn(1, 1, 320) * 5
        out = gen(x)
        assert out.min() >= -1.0
        assert out.max() <= 1.0

    def test_skip_connection_non_power_of_32(self) -> None:
        """Forward pass works for input lengths not divisible by 32."""
        gen = OpGANGenerator(q=2)
        x = torch.randn(1, 1, 352)  # 32 * 11
        out = gen(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 1
        assert out.shape[2] > 0

    def test_dropout_differs_in_train(self) -> None:
        """Train mode produces different outputs due to dropout."""
        gen = OpGANGenerator(q=2)
        gen.train()
        x = torch.randn(1, 1, 320)
        torch.manual_seed(0)
        out1 = gen(x)
        torch.manual_seed(1)
        out2 = gen(x)
        assert not torch.equal(out1, out2)

    def test_eval_deterministic(self) -> None:
        """Eval mode produces identical outputs."""
        gen = OpGANGenerator(q=2)
        gen.eval()
        x = torch.randn(1, 1, 320)
        out1 = gen(x)
        out2 = gen(x)
        assert torch.equal(out1, out2)

    def test_weight_initialization(self) -> None:
        """Standard conv layers have zero bias, GroupNorm weight=1."""
        gen = OpGANGenerator(q=2)
        for m in gen.modules():
            if isinstance(m, nn.GroupNorm):
                assert torch.allclose(m.weight, torch.ones_like(m.weight))
                assert torch.allclose(m.bias, torch.zeros_like(m.bias))

    def test_q_stored(self) -> None:
        """q parameter is accessible."""
        gen = OpGANGenerator(q=3)
        assert gen.q == 3

    def test_gradient_flows(self) -> None:
        """Backward pass produces gradients on all parameters."""
        gen = OpGANGenerator(q=2)
        x = torch.randn(1, 1, 320)
        out = gen(x)
        out.sum().backward()
        params_with_grad = sum(1 for p in gen.parameters() if p.grad is not None)
        assert params_with_grad > 0


class TestOpGANDiscriminator:
    """Tests for PatchGAN discriminator."""

    def test_output_shape(self) -> None:
        """Discriminator produces patch-level scores."""
        disc = OpGANDiscriminator(q=2)
        x = torch.randn(1, 1, 320)
        out = disc(x)
        assert out.shape[0] == 1
        assert out.shape[1] == 1
        assert out.shape[2] > 0

    def test_output_shape_batch(self) -> None:
        """Batch dimension is preserved."""
        disc = OpGANDiscriminator(q=2)
        x = torch.randn(2, 1, 320)
        out = disc(x)
        assert out.shape[0] == 2

    def test_weight_initialization(self) -> None:
        """GroupNorm layers initialized with weight=1, bias=0."""
        disc = OpGANDiscriminator(q=2)
        for m in disc.modules():
            if isinstance(m, nn.GroupNorm):
                assert torch.allclose(m.weight, torch.ones_like(m.weight))
                assert torch.allclose(m.bias, torch.zeros_like(m.bias))

    def test_gradient_flows(self) -> None:
        """Backward pass produces gradients."""
        disc = OpGANDiscriminator(q=2)
        x = torch.randn(1, 1, 320)
        out = disc(x)
        out.sum().backward()
        params_with_grad = sum(1 for p in disc.parameters() if p.grad is not None)
        assert params_with_grad > 0
