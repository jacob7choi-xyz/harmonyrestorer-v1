"""Tests for what the application logs and loads at startup."""

import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
from app.main import app
from fastapi.testclient import TestClient


def test_startup_logs_the_ffmpeg_version(caplog: pytest.LogCaptureFixture) -> None:
    """The ffmpeg build is logged at startup, since the image does not pin it and
    decode results differ slightly between builds."""
    with caplog.at_level(logging.INFO, logger="app.main"), TestClient(app):
        pass
    assert any("ffmpeg version" in r.getMessage() for r in caplog.records)


def test_startup_logs_an_error_when_ffmpeg_is_missing(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ffmpeg every upload fails with 500, so its absence is logged at ERROR."""
    monkeypatch.setattr("app.main.ffmpeg_version", lambda timeout_s: None)
    with caplog.at_level(logging.INFO, logger="app.main"), TestClient(app):
        pass
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("ffmpeg" in r.getMessage() for r in errors)


def test_importing_the_app_does_not_load_the_uvr_package(tmp_path: Path) -> None:
    """With the default OpGAN engine, importing the app leaves audio-separator unloaded.

    It is needed only for UVR jobs. On two Lambda cold starts measured on 2026-11-08
    (PYTHONPROFILEIMPORTTIME), its import took about 1.2 and 1.7 s beyond torch. A fresh
    process is used because other tests may already have imported it into this one.
    """
    backend_dir = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "UPLOAD_DIR": str(tmp_path / "uploads"),
        "PROCESSED_DIR": str(tmp_path / "processed"),
        # Set explicitly: config.py loads backend/.env, which could select UVR
        "DENOISER_ENGINE": "opgan",
    }
    probe = "import sys, app.main; print('audio_separator' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=backend_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"
