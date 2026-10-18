"""Shared test fixtures."""

from unittest.mock import MagicMock, patch

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


def _fake_decode(source, destination, **kwargs):  # type: ignore[no-untyped-def]
    """Stand in for decode_to_wav: write a 1 s 16 kHz WAV and report its duration."""
    import numpy as np
    import soundfile as sf

    sf.write(str(destination), np.zeros(16_000, dtype=np.float32), 16_000, subtype="PCM_16")
    return 1.0


@pytest.fixture()
def mock_decode():
    """Replace the upload decode with a stand-in that writes a 1 s WAV.

    Required in tests that upload the minimal byte fixtures (SMALL_WAV, SAMPLE_BYTES),
    which carry the right magic bytes but are not audio, so the real ffmpeg decode
    would reject them. Apply it whenever an upload is expected to succeed and decoding
    is not the behavior under test; tests of decoding upload real files instead.
    """
    with patch("app.routes.denoise.decode_to_wav", side_effect=_fake_decode) as fake:
        yield fake


@pytest.fixture()
def mock_denoiser(mock_decode):
    """Mock DenoiserService for upload-success tests.

    Depends on mock_decode, because the SMALL_WAV and SAMPLE_BYTES fixtures are
    magic-byte sequences, not real audio, and the real decode would reject them.
    """
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
