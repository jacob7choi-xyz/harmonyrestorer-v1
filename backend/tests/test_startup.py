"""Tests for what the application logs at startup."""

import logging

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
