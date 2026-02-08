"""Shared test fixtures."""

from unittest.mock import MagicMock

import pytest
from app.config import settings
from app.main import app
from app.middleware import RateLimitMiddleware
from app.services.jobs import job_manager
from fastapi.testclient import TestClient


def _clear_rate_limiter(application) -> None:  # type: ignore[no-untyped-def]
    """Walk the middleware stack and clear rate limiter state."""
    stack = application.middleware_stack
    while stack is not None:
        if isinstance(stack, RateLimitMiddleware):
            stack._requests.clear()
            return
        stack = getattr(stack, "app", None)


@pytest.fixture()
def client():
    """TestClient with clean job state for each test."""
    job_manager._jobs.clear()
    with TestClient(app) as c:
        _clear_rate_limiter(app)
        yield c
    job_manager._jobs.clear()


@pytest.fixture()
def mock_denoiser():
    """Mock DenoiserService that writes a dummy output file."""
    denoiser = MagicMock()

    def fake_denoise(input_path):
        # Output goes in processed_dir so rename() stays on same filesystem
        output = settings.processed_dir / f"{input_path.stem}_(No Noise).wav"
        output.write_bytes(b"RIFF" + b"\x00" * 100)
        return output

    denoiser.denoise.side_effect = fake_denoise
    job_manager._denoiser = denoiser
    yield denoiser
    job_manager._denoiser = None
