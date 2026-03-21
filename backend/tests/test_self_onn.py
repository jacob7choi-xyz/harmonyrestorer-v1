"""Tests for OptimizedConv1DSelfONN layer."""

import pytest
import torch
from app.models.self_onn import OptimizedConv1DSelfONN


class TestInit:
    """Tests for layer construction and validation."""

    def test_default_construction(self) -> None:
        """Layer stores all attributes correctly."""
        layer = OptimizedConv1DSelfONN(4, 8, kernel_size=5, stride=1, padding=2, q=3)
        assert layer.in_channels == 4
        assert layer.out_channels == 8
        assert layer.kernel_size == 5
        assert layer.q == 3

    def test_q_too_low_raises(self) -> None:
        """q=0 is invalid."""
        with pytest.raises(ValueError, match="q must be in"):
            OptimizedConv1DSelfONN(1, 1, kernel_size=3, q=0)

    def test_q_too_high_raises(self) -> None:
        """q=6 exceeds the number of defined operators."""
        with pytest.raises(ValueError, match="q must be in"):
            OptimizedConv1DSelfONN(1, 1, kernel_size=3, q=6)

    def test_q_boundary_values(self) -> None:
        """q=1 and q=5 are both valid."""
        OptimizedConv1DSelfONN(1, 1, kernel_size=3, q=1)
        OptimizedConv1DSelfONN(1, 1, kernel_size=3, q=5)

    def test_weight_shapes(self) -> None:
        """conv_weights and operator_probs have correct shapes."""
        layer = OptimizedConv1DSelfONN(4, 8, kernel_size=5, q=3)
        assert layer.conv_weights.shape == (8, 4, 5, 3)
        assert layer.operator_probs.shape == (8, 3)

    def test_initialization_prefers_linear(self) -> None:
        """operator_probs[:, 0] initialized higher than other slots."""
        layer = OptimizedConv1DSelfONN(4, 8, kernel_size=5, q=3)
        assert (layer.operator_probs[:, 0] == 0.3).all()
        assert (layer.operator_probs[:, 1] == 0.1).all()
        assert (layer.operator_probs[:, 2] == 0.1).all()


class TestForward:
    """Tests for forward pass shapes and behavior."""

    def test_output_shape_basic(self) -> None:
        """Standard forward pass produces correct output shape."""
        layer = OptimizedConv1DSelfONN(4, 8, kernel_size=5, stride=1, padding=2, q=3)
        x = torch.randn(2, 4, 64)
        out = layer(x)
        assert out.shape == (2, 8, 64)

    def test_output_shape_with_stride(self) -> None:
        """Stride-2 halves the output length."""
        layer = OptimizedConv1DSelfONN(1, 16, kernel_size=5, stride=2, padding=2, q=3)
        x = torch.randn(1, 1, 64)
        out = layer(x)
        assert out.shape == (1, 16, 32)

    def test_output_shape_q_equals_1(self) -> None:
        """q=1 uses only the linear operator."""
        layer = OptimizedConv1DSelfONN(1, 8, kernel_size=5, stride=1, padding=2, q=1)
        x = torch.randn(1, 1, 32)
        out = layer(x)
        assert out.shape == (1, 8, 32)

    def test_output_shape_q_equals_5(self) -> None:
        """q=5 exercises all 5 operator functions."""
        layer = OptimizedConv1DSelfONN(1, 4, kernel_size=3, stride=1, padding=1, q=5)
        x = torch.randn(1, 1, 16)
        out = layer(x)
        assert out.shape == (1, 4, 16)

    def test_output_clamped(self) -> None:
        """Output values stay within [-5, 5]."""
        layer = OptimizedConv1DSelfONN(1, 8, kernel_size=5, stride=1, padding=2, q=3)
        x = torch.randn(2, 1, 64) * 100
        out = layer(x)
        assert out.min() >= -5.0
        assert out.max() <= 5.0

    def test_no_padding_case(self) -> None:
        """padding=0 skips the F.pad call and reduces output length."""
        layer = OptimizedConv1DSelfONN(1, 4, kernel_size=5, stride=1, padding=0, q=2)
        x = torch.randn(1, 1, 32)
        out = layer(x)
        assert out.shape == (1, 4, 28)  # 32 - 5 + 1

    def test_gradient_flows(self) -> None:
        """Gradients propagate through all operator paths."""
        layer = OptimizedConv1DSelfONN(1, 4, kernel_size=3, padding=1, q=3)
        x = torch.randn(1, 1, 16)
        out = layer(x)
        out.sum().backward()
        assert layer.conv_weights.grad is not None
        assert layer.operator_probs.grad is not None


class TestCache:
    """Tests for operator weight caching in eval mode."""

    def test_train_mode_no_cache(self) -> None:
        """Training mode does not populate cache."""
        layer = OptimizedConv1DSelfONN(1, 4, kernel_size=3, padding=1, q=2)
        layer.train()
        layer(torch.randn(1, 1, 16))
        assert not layer._cache_valid

    def test_eval_mode_caches(self) -> None:
        """Eval mode populates cache after first forward pass."""
        layer = OptimizedConv1DSelfONN(1, 4, kernel_size=3, padding=1, q=2)
        layer.eval()
        layer(torch.randn(1, 1, 16))
        assert layer._cache_valid
        assert layer._operator_weights_cache is not None

    def test_invalidate_cache(self) -> None:
        """invalidate_cache clears both flag and tensor."""
        layer = OptimizedConv1DSelfONN(1, 4, kernel_size=3, padding=1, q=2)
        layer.eval()
        layer(torch.randn(1, 1, 16))
        layer.invalidate_cache()
        assert not layer._cache_valid
        assert layer._operator_weights_cache is None

    def test_train_mode_invalidates_cache(self) -> None:
        """Switching from eval to train invalidates cache."""
        layer = OptimizedConv1DSelfONN(1, 4, kernel_size=3, padding=1, q=2)
        layer.eval()
        layer(torch.randn(1, 1, 16))
        assert layer._cache_valid
        layer.train()
        assert not layer._cache_valid

    def test_eval_deterministic(self) -> None:
        """Two forward passes in eval mode produce identical output."""
        layer = OptimizedConv1DSelfONN(1, 4, kernel_size=3, padding=1, q=3)
        layer.eval()
        x = torch.randn(1, 1, 16)
        out1 = layer(x)
        out2 = layer(x)
        assert torch.equal(out1, out2)
