"""Tests for the denoise API endpoints."""

from unittest.mock import patch

import pytest
from app.config import settings

SMALL_WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 32

# Valid magic bytes per format for content-type validation tests
SAMPLE_BYTES: dict[str, bytes] = {
    ".wav": SMALL_WAV,
    ".mp3": b"ID3" + b"\x00" * 40,
    ".flac": b"fLaC" + b"\x00" * 40,
    ".ogg": b"OggS" + b"\x00" * 40,
    ".m4a": b"\x00" * 4 + b"ftyp" + b"\x00" * 36,
    ".aac": b"\xff\xf1" + b"\x00" * 40,
}


# --- Upload ---


def test_upload_returns_job_id(client, mock_denoiser):
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_upload_rejects_unsupported_format(client):
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert "Unsupported format" in r.json()["detail"]


def test_upload_rejects_oversized_file(client):
    with patch.object(settings, "max_upload_bytes", 10):
        r = client.post(
            "/api/v1/denoise",
            files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
        )
    assert r.status_code == 413


def test_upload_rejects_missing_file(client):
    r = client.post("/api/v1/denoise")
    assert r.status_code == 422  # FastAPI validation error


@pytest.mark.parametrize("ext", [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"])
def test_upload_accepts_all_supported_formats(client, mock_denoiser, ext):
    """Every format listed in settings.supported_formats should be accepted."""
    content = SAMPLE_BYTES[ext]
    r = client.post(
        "/api/v1/denoise",
        files={"file": (f"audio{ext}", content, "audio/octet-stream")},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "queued"


# --- Status ---


def test_status_after_processing(client, mock_denoiser):
    """Background task runs synchronously in TestClient, so job completes immediately."""
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )
    job_id = r.json()["job_id"]

    r = client.get(f"/api/v1/status/{job_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "completed"
    assert r.json()["progress"] == 100


def test_status_invalid_uuid(client):
    r = client.get("/api/v1/status/not-a-uuid")
    assert r.status_code == 400
    assert r.json()["detail"] == "Invalid job ID"


def test_status_nonexistent_job(client):
    r = client.get("/api/v1/status/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404


# --- Download ---


def test_download_completed_job(client, mock_denoiser):
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )
    job_id = r.json()["job_id"]

    r = client.get(f"/api/v1/download/{job_id}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "audio/wav"


def test_download_not_completed(client):
    from datetime import datetime

    from app.schemas import JobStatus, JobStatusEnum
    from app.services.jobs import job_manager

    job_id = "00000000-0000-0000-0000-000000000001"
    job_manager._jobs[job_id] = JobStatus(
        job_id=job_id,
        status=JobStatusEnum.PROCESSING,
        progress=50,
        message="Working...",
        created_at=datetime.now(),
    )

    r = client.get(f"/api/v1/download/{job_id}")
    assert r.status_code == 400


def test_download_invalid_uuid(client):
    r = client.get("/api/v1/download/not-a-valid-uuid")
    assert r.status_code == 400


def test_download_after_job_cleanup(client, mock_denoiser):
    """Download should return 404 after the job has been cleaned up."""
    from app.services.jobs import job_manager

    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )
    job_id = r.json()["job_id"]

    # Simulate cleanup removing the job
    with job_manager._lock:
        del job_manager._jobs[job_id]

    r = client.get(f"/api/v1/download/{job_id}")
    assert r.status_code == 404
