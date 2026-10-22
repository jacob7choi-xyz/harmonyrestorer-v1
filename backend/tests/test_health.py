"""Tests for health and root endpoints."""

import pytest
from app.main import app
from fastapi.testclient import TestClient


def test_root_returns_platform_info(client):
    r = client.get("/")
    assert r.status_code == 200

    data = r.json()
    assert data["platform"] == "HarmonyRestorer v1"
    # Engine-neutral: DENOISER_ENGINE can select UVR, so naming one model would be wrong
    assert data["description"] == "AI-powered audio denoising"
    assert data["status"] == "ready"
    assert ".wav" in data["supported_formats"]


def test_health_reports_real_checks(client):
    r = client.get("/health")
    assert r.status_code == 200

    data = r.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["checks"]["disk_space"]["ok"] is True
    assert data["checks"]["directories"]["ok"] is True
    assert isinstance(data["checks"]["disk_space"]["free_mb"], int)


def test_health_includes_job_counts(client):
    r = client.get("/health")
    data = r.json()
    assert "jobs" in data
    assert data["jobs"] == {}  # no jobs yet


def test_health_reports_ffmpeg_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The startup ffmpeg check is reported, without exposing the version line."""
    version_line = "ffmpeg version test-build"
    monkeypatch.setattr("app.main.ffmpeg_version", lambda timeout_s: version_line)
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["checks"]["ffmpeg_at_startup"] == {"ok": True}
    assert version_line not in r.text


def test_health_is_degraded_when_ffmpeg_failed_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ffmpeg, an upload that reaches the decode fails with 500, so /health
    should not report healthy."""
    monkeypatch.setattr("app.main.ffmpeg_version", lambda timeout_s: None)
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 503
    data = r.json()
    assert data["status"] == "degraded"
    assert data["checks"]["ffmpeg_at_startup"] == {"ok": False}
    assert data["checks"]["disk_space"]["ok"] is True
    assert data["checks"]["directories"]["ok"] is True
