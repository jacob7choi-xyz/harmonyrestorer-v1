"""Shared test fixtures."""

from unittest.mock import MagicMock, patch

import pytest
from app.config import settings
from app.main import app
from app.services.artifacts import ArtifactBudget
from app.services.file_store import LocalFileStore
from app.services.job_store import build_job_store
from app.services.jobs import job_manager
from app.services.rate_limit import build_rate_limiter
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_artifact_dirs(tmp_path_factory, monkeypatch) -> None:
    """Point uploads and results at folders made for this test.

    Without this, tests write into backend/uploads and backend/processed and
    nothing removes the files. The artifact budget is replaced too, because the
    shared one captured the real folders when it was built at import, and the
    cached denoiser is cleared, because a built one keeps the folder it was
    given. The app's file store is rebuilt on the new folder, because the
    composition root rooted it at the real one when the app was imported.
    """
    root = tmp_path_factory.mktemp("artifacts")
    uploads = root / "uploads"
    processed = root / "processed"
    uploads.mkdir()
    processed.mkdir()
    monkeypatch.setattr(settings, "upload_dir", uploads)
    monkeypatch.setattr(settings, "processed_dir", processed)
    budget = ArtifactBudget(
        limit_bytes=settings.max_artifact_bytes, directories=[uploads, processed]
    )
    monkeypatch.setattr("app.routes.denoise.artifact_budget", budget)
    monkeypatch.setattr("app.services.transcode.artifact_budget", budget)
    monkeypatch.setattr(job_manager, "_denoiser", None)
    monkeypatch.setattr(app.state, "file_store", LocalFileStore(processed))


def _clear_rate_limiter(application: FastAPI) -> None:
    """Give the app a fresh rate limiter, so earlier requests do not count."""
    application.state.rate_limiter = build_rate_limiter(settings.rate_limiter)


def _clear_job_store(application: FastAPI) -> None:
    """Give the app an empty job store, built with the current settings' caps."""
    application.state.job_store = build_job_store(settings.job_store)


@pytest.fixture()
def client():
    """TestClient with clean job state for each test."""
    _clear_job_store(app)
    with TestClient(app) as c:
        _clear_rate_limiter(app)
        yield c
    _clear_job_store(app)
    _clear_rate_limiter(app)


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
