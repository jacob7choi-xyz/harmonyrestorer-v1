"""Shared test fixtures."""

import pytest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.services.jobs import job_manager


@pytest.fixture()
def client():
    """TestClient with clean job state for each test."""
    job_manager._jobs.clear()
    with TestClient(app) as c:
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
