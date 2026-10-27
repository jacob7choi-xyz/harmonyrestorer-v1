"""Tests for health and root endpoints."""

import hashlib
from pathlib import Path

import pytest
from app.config import DenoiserEngine, settings
from app.main import app
from fastapi.testclient import TestClient

_CHECKPOINT_STARTUP_ERROR = "OpGAN checkpoint failed its startup check"


@pytest.fixture
def matching_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run the OpGAN engine against a temp checkpoint whose digest the startup check expects.

    CI and the image's test stage have no final.pt, so no test here may depend on it.
    """
    checkpoint = tmp_path / "final.pt"
    checkpoint.write_bytes(b"test checkpoint")
    monkeypatch.setattr(settings, "denoiser_engine", DenoiserEngine.OPGAN)
    monkeypatch.setattr(settings, "opgan_checkpoint", checkpoint)
    monkeypatch.setattr(
        "app.main.OPGAN_CHECKPOINT_SHA256", hashlib.sha256(b"test checkpoint").hexdigest()
    )
    return checkpoint


def test_root_returns_platform_info(client):
    r = client.get("/")
    assert r.status_code == 200

    data = r.json()
    assert data["platform"] == "HarmonyRestorer v1"
    # Engine-neutral: DENOISER_ENGINE can select UVR, so naming one model would be wrong
    assert data["description"] == "AI-powered audio denoising"
    assert data["status"] == "ready"
    assert ".wav" in data["supported_formats"]


def test_health_reports_real_checks(matching_checkpoint: Path, client: TestClient) -> None:
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


def test_health_reports_ffmpeg_at_startup(
    matching_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The startup ffmpeg check is reported, without exposing the version line."""
    version_line = "ffmpeg version test-build"
    monkeypatch.setattr("app.main.ffmpeg_version", lambda timeout_s: version_line)
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["checks"]["ffmpeg_at_startup"] == {"ok": True}
    assert version_line not in r.text


def test_health_is_degraded_when_ffmpeg_failed_at_startup(
    matching_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
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
    assert data["checks"]["opgan_checkpoint_at_startup"] == {"ok": True}
    assert data["checks"]["disk_space"]["ok"] is True
    assert data["checks"]["directories"]["ok"] is True


@pytest.mark.parametrize(
    "contents",
    [b"some other checkpoint", None],
    ids=["digest-mismatch", "missing-file"],
)
def test_health_is_degraded_when_the_opgan_checkpoint_failed_at_startup(
    contents: bytes | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With a wrong or missing checkpoint every OpGAN job fails, so /health should not
    report healthy."""
    checkpoint = tmp_path / "final.pt"
    if contents is not None:
        checkpoint.write_bytes(contents)
    monkeypatch.setattr(settings, "denoiser_engine", DenoiserEngine.OPGAN)
    monkeypatch.setattr(settings, "opgan_checkpoint", checkpoint)
    monkeypatch.setattr("app.main.ffmpeg_version", lambda timeout_s: "ffmpeg version test")
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 503
    data = r.json()
    assert data["status"] == "degraded"
    assert data["checks"]["opgan_checkpoint_at_startup"] == {"ok": False}
    assert data["checks"]["ffmpeg_at_startup"] == {"ok": True}
    assert data["checks"]["disk_space"]["ok"] is True
    assert data["checks"]["directories"]["ok"] is True
    assert _CHECKPOINT_STARTUP_ERROR in caplog.text


def test_health_reports_a_matching_opgan_checkpoint(
    matching_checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint with the expected digest is reported ok without degrading /health."""
    monkeypatch.setattr("app.main.ffmpeg_version", lambda timeout_s: "ffmpeg version test")
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["checks"]["opgan_checkpoint_at_startup"] == {"ok": True}


def test_health_omits_the_opgan_checkpoint_for_the_uvr_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UVR does not use the OpGAN checkpoint, so a missing one does not degrade /health."""
    monkeypatch.setattr(settings, "denoiser_engine", DenoiserEngine.UVR)
    monkeypatch.setattr(settings, "opgan_checkpoint", tmp_path / "missing.pt")
    monkeypatch.setattr("app.main.ffmpeg_version", lambda timeout_s: "ffmpeg version test")
    with TestClient(app) as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert "opgan_checkpoint_at_startup" not in r.json()["checks"]


def test_health_fails_closed_when_no_opgan_checkpoint_result_was_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a recorded startup result the checkpoint is reported not ok."""
    monkeypatch.setattr(settings, "denoiser_engine", DenoiserEngine.OPGAN)
    monkeypatch.delattr(app.state, "opgan_checkpoint_at_startup_ok", raising=False)
    # Without a with block TestClient does not run the lifespan, so nothing is recorded
    r = TestClient(app).get("/health")
    assert r.status_code == 503
    assert r.json()["checks"]["opgan_checkpoint_at_startup"] == {"ok": False}
