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


@pytest.fixture()
def mock_audio_duration():
    """Mock audio duration reads to return 1.0 second for all formats.

    Required in tests that use minimal byte fixtures (SMALL_WAV, SAMPLE_BYTES)
    which are not valid audio files. Without this fixture, soundfile and librosa
    raise parse errors that now fail closed with 400 under the M1 security fix.
    Apply this fixture whenever an upload is expected to succeed and duration
    validation is not the behavior under test.
    """
    mock_info = MagicMock()
    mock_info.duration = 1.0
    with (
        patch("app.routes.denoise.sf.info", return_value=mock_info),
        patch("librosa.get_duration", return_value=1.0),
    ):
        yield


@pytest.fixture()
def mock_denoiser(mock_audio_duration):
    """Mock DenoiserService for upload-success tests.

    Depends on mock_audio_duration, which patches sf.info and librosa.get_duration
    to return 1.0 second. This is required because the SMALL_WAV and SAMPLE_BYTES
    fixtures are minimal magic-byte sequences, not real audio, and the fail-closed
    duration check in _check_audio_duration would otherwise reject them with 400.

    Tests that exercise duration-validation behavior should explicitly override
    the duration mock with their own controlled values. Test-level monkeypatches
    or patches take precedence over this fixture because they are applied after
    fixture setup.
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
