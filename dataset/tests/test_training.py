"""Tests for OpGANLoss and GradientHealthMonitor (training-only components)."""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from dataset.train import GradientHealthMonitor, OpGANLoss


class TestOpGANLoss:
    """Tests for composite loss function."""

    @pytest.fixture()
    def loss_fn(self) -> OpGANLoss:
        """Small loss function for testing."""
        return OpGANLoss(n_fft=64, hop_length=32)

    def test_temporal_loss_identical_is_zero(self, loss_fn: OpGANLoss) -> None:
        """L1 of identical signals is 0."""
        x = torch.randn(1, 1, 320)
        assert loss_fn.temporal_loss(x, x).item() == 0.0

    def test_temporal_loss_different_is_positive(self, loss_fn: OpGANLoss) -> None:
        """L1 of different signals is positive."""
        a = torch.randn(1, 1, 320)
        b = torch.randn(1, 1, 320)
        assert loss_fn.temporal_loss(a, b).item() > 0.0

    def test_spectral_loss_identical_near_zero(self, loss_fn: OpGANLoss) -> None:
        """STFT magnitude L1 of identical signals is near 0."""
        x = torch.randn(1, 1, 320)
        loss = loss_fn.spectral_loss(x, x)
        assert loss.item() < 1e-6

    def test_spectral_loss_different_is_positive(self, loss_fn: OpGANLoss) -> None:
        """STFT magnitude L1 of different signals is positive."""
        a = torch.randn(1, 1, 320)
        b = torch.randn(1, 1, 320)
        assert loss_fn.spectral_loss(a, b).item() > 0.0

    def test_generator_loss_returns_tuple(self, loss_fn: OpGANLoss) -> None:
        """Generator loss returns (tensor, dict) with expected keys."""
        disc_out = torch.randn(1, 1, 10)
        generated = torch.randn(1, 1, 320)
        target = torch.randn(1, 1, 320)
        total, breakdown = loss_fn.generator_loss(disc_out, generated, target)
        assert isinstance(total, torch.Tensor)
        assert set(breakdown.keys()) == {"adversarial", "temporal", "spectral", "total"}

    def test_generator_loss_total_formula(self, loss_fn: OpGANLoss) -> None:
        """Total = adversarial + lambda_temporal*temporal + lambda_spectral*spectral."""
        disc_out = torch.randn(1, 1, 10)
        generated = torch.randn(1, 1, 320)
        target = torch.randn(1, 1, 320)
        total, bd = loss_fn.generator_loss(disc_out, generated, target)
        expected = (
            bd["adversarial"]
            + loss_fn.lambda_temporal * bd["temporal"]
            + loss_fn.lambda_spectral * bd["spectral"]
        )
        assert abs(total.item() - expected) < 1e-4

    def test_discriminator_loss_label_smoothing(self, loss_fn: OpGANLoss) -> None:
        """Perfect predictions still produce nonzero loss due to smoothing."""
        real_out = torch.ones(1, 1, 10)
        fake_out = torch.zeros(1, 1, 10)
        loss = loss_fn.discriminator_loss(real_out, fake_out)
        assert loss.item() > 0.0

    def test_custom_lambda_values(self) -> None:
        """Custom lambda weights affect the total loss."""
        loss_a = OpGANLoss(lambda_temporal=100, lambda_spectral=0, n_fft=64, hop_length=32)
        loss_b = OpGANLoss(lambda_temporal=0, lambda_spectral=100, n_fft=64, hop_length=32)
        disc_out = torch.randn(1, 1, 10)
        generated = torch.randn(1, 1, 320)
        target = torch.randn(1, 1, 320)
        total_a, _ = loss_a.generator_loss(disc_out, generated, target)
        total_b, _ = loss_b.generator_loss(disc_out, generated, target)
        assert total_a.item() != total_b.item()

    def test_hann_window_is_buffer(self, loss_fn: OpGANLoss) -> None:
        """Hann window is a registered buffer."""
        buffers = dict(loss_fn.named_buffers())
        assert "_hann_window" in buffers
        assert buffers["_hann_window"].shape == (64,)


class TestGradientHealthMonitor:
    """Tests for gradient monitoring and clipping."""

    @pytest.fixture()
    def model_with_grad(self) -> nn.Linear:
        """A small model with computed gradients."""
        model = nn.Linear(4, 4, bias=False)
        x = torch.randn(1, 4)
        loss = model(x).sum()
        loss.backward()
        return model

    def test_returns_norm(self, model_with_grad: nn.Linear) -> None:
        """Returns a positive gradient norm."""
        monitor = GradientHealthMonitor(max_grad_norm=10.0)
        norm = monitor.check_and_clip_gradients(model_with_grad, "test")
        assert norm > 0.0

    def test_clips_gradients(self) -> None:
        """Gradients are clipped when norm exceeds threshold."""
        model = nn.Linear(4, 4, bias=False)
        x = torch.randn(1, 4) * 100
        (model(x).sum() * 1000).backward()
        monitor = GradientHealthMonitor(max_grad_norm=0.01)
        pre_clip = monitor.check_and_clip_gradients(model, "test")
        post_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        assert post_norm.item() <= 0.01 + 1e-6
        assert pre_clip > 0.01

    def test_step_count_increments(self, model_with_grad: nn.Linear) -> None:
        """Step count increments on each call."""
        monitor = GradientHealthMonitor()
        monitor.check_and_clip_gradients(model_with_grad)
        x = torch.randn(1, 4)
        model_with_grad(x).sum().backward()
        monitor.check_and_clip_gradients(model_with_grad)
        assert monitor.step_count == 2

    def test_history_recorded(self, model_with_grad: nn.Linear) -> None:
        """Gradient history entries contain step and total_norm."""
        monitor = GradientHealthMonitor()
        monitor.check_and_clip_gradients(model_with_grad, "test")
        assert len(monitor.gradient_history) == 1
        entry = monitor.gradient_history[0]
        assert "step" in entry
        assert "total_norm" in entry

    def test_history_bounded(self, model_with_grad: nn.Linear) -> None:
        """History respects max_history via deque."""
        monitor = GradientHealthMonitor(max_history=3)
        for _ in range(5):
            x = torch.randn(1, 4)
            model_with_grad(x).sum().backward()
            monitor.check_and_clip_gradients(model_with_grad)
        assert len(monitor.gradient_history) == 3

    def test_get_stats_empty(self) -> None:
        """Empty history returns empty dict."""
        monitor = GradientHealthMonitor()
        assert monitor.get_gradient_stats() == {}

    def test_get_stats_computed(self, model_with_grad: nn.Linear) -> None:
        """Stats contain expected keys after recording history."""
        monitor = GradientHealthMonitor()
        for _ in range(3):
            x = torch.randn(1, 4)
            model_with_grad(x).sum().backward()
            monitor.check_and_clip_gradients(model_with_grad)
        stats = monitor.get_gradient_stats()
        assert set(stats.keys()) == {"mean_norm", "max_norm", "min_norm", "recent_norm"}

    def test_large_gradient_warning(
        self, model_with_grad: nn.Linear, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Very large gradients trigger a warning."""
        for p in model_with_grad.parameters():
            if p.grad is not None:
                p.grad.fill_(1000.0)
        monitor = GradientHealthMonitor(max_grad_norm=10000.0)
        with caplog.at_level(logging.WARNING, logger="dataset.train"):
            monitor.check_and_clip_gradients(model_with_grad, "test")
        assert any("Very large gradients" in r.message for r in caplog.records)

    def test_small_gradient_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Near-zero gradients trigger a warning."""
        model = nn.Linear(4, 4, bias=False)
        x = torch.zeros(1, 4)
        model(x).sum().backward()
        monitor = GradientHealthMonitor()
        with caplog.at_level(logging.WARNING, logger="dataset.train"):
            monitor.check_and_clip_gradients(model, "test")
        assert any("Very small gradients" in r.message for r in caplog.records)

    def test_periodic_logging(
        self, model_with_grad: nn.Linear, caplog: pytest.LogCaptureFixture
    ) -> None:
        """INFO logs appear at log_frequency intervals."""
        monitor = GradientHealthMonitor(log_frequency=2, max_grad_norm=100.0)
        with caplog.at_level(logging.INFO, logger="dataset.train"):
            for _ in range(4):
                x = torch.randn(1, 4)
                model_with_grad(x).sum().backward()
                monitor.check_and_clip_gradients(model_with_grad, "test")
        info_messages = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_messages) >= 2
