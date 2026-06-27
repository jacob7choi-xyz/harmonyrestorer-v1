"""Tests for the denoise API endpoints."""

from unittest.mock import Mock, patch

import pytest
import soundfile as sf
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


def test_upload_leaves_no_temp_files(client, mock_denoiser):
    """Atomic upload write: no .tmp files survive after a successful upload.

    The upload is written via temp file + atomic rename. The background task
    (run synchronously by TestClient) then deletes the renamed file. This test
    proves no partial .tmp files are left behind at any stage.
    """
    before_tmp = set(settings.upload_dir.glob("*.tmp"))

    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )
    assert r.status_code == 200

    new_tmp = set(settings.upload_dir.glob("*.tmp")) - before_tmp
    assert not new_tmp, f"Orphaned temp files after upload: {[f.name for f in new_tmp]}"


def test_upload_write_failure_cleans_up_temp_file(client, monkeypatch, mock_audio_duration):
    """If the atomic rename fails, no traces remain in the upload dir and 500 is returned."""
    from pathlib import Path

    before = set(settings.upload_dir.iterdir())

    original_replace = Path.replace

    def fail_replace(self, target):  # type: ignore[no-untyped-def]
        if self.parent == settings.upload_dir and str(self).endswith(".tmp"):
            raise OSError("simulated atomic rename failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_replace)

    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )

    after = set(settings.upload_dir.iterdir())

    assert r.status_code == 500
    assert "Failed to save file" in r.json()["detail"]
    assert after == before, f"Upload dir changed after write failure: {after - before}"


def test_upload_rejects_unsupported_format(client):
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert "Unsupported format" in r.json()["detail"]


def test_upload_rejects_oversized_file(client) -> None:
    """Oversized uploads return 413 and do not create a job or temp file."""
    from app.services.jobs import job_manager

    original_job_count = len(job_manager._jobs)
    with patch.object(settings, "max_upload_bytes", 10):
        r = client.post(
            "/api/v1/denoise",
            files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
        )
    assert r.status_code == 413
    assert len(job_manager._jobs) == original_job_count


def test_upload_accepts_file_at_size_limit(client, mock_denoiser, monkeypatch) -> None:
    """A file exactly at max_upload_bytes is accepted."""
    from app.services.jobs import job_manager

    original_job_count = len(job_manager._jobs)
    monkeypatch.setattr(settings, "max_upload_bytes", len(SMALL_WAV))
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )
    assert r.status_code == 200
    assert len(job_manager._jobs) == original_job_count + 1


def test_upload_rejects_file_one_byte_over_limit(client, monkeypatch) -> None:
    """A file one byte over max_upload_bytes is rejected with 413 and no job is created."""
    from app.services.jobs import job_manager

    original_job_count = len(job_manager._jobs)
    monkeypatch.setattr(settings, "max_upload_bytes", len(SMALL_WAV) - 1)
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )
    assert r.status_code == 413
    assert len(job_manager._jobs) == original_job_count


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


# --- Duration validation ---


class TestDurationValidation:
    """Tests for audio duration gate in the upload endpoint."""

    def test_wav_exceeding_max_duration_returns_400(
        self, client, mock_denoiser, monkeypatch
    ) -> None:
        """WAV longer than the configured limit is rejected with 400."""
        mock_info = Mock()
        mock_info.duration = settings.max_audio_duration_seconds + 1
        monkeypatch.setattr("app.routes.denoise.sf.info", lambda _: mock_info)

        r = client.post(
            "/api/v1/denoise",
            files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
        )
        assert r.status_code == 400
        assert "too long" in r.json()["detail"].lower()

    def test_wav_within_max_duration_succeeds(self, client, mock_denoiser, monkeypatch) -> None:
        """WAV within the duration limit is accepted."""
        mock_info = Mock()
        mock_info.duration = settings.max_audio_duration_seconds - 1
        monkeypatch.setattr("app.routes.denoise.sf.info", lambda _: mock_info)

        r = client.post(
            "/api/v1/denoise",
            files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
        )
        assert r.status_code == 200

    def test_soundfile_error_returns_400_and_creates_no_job(self, client, monkeypatch) -> None:
        """A known soundfile parse error during duration reading returns 400 and creates no job.

        Covers the fail-closed invariant: duration validation failure must not allow
        the upload to proceed to inference.
        """
        from app.services.jobs import job_manager

        original_job_count = len(job_manager._jobs)
        monkeypatch.setattr(
            "app.routes.denoise.sf.info",
            Mock(side_effect=sf.SoundFileError("corrupt header")),
        )

        r = client.post(
            "/api/v1/denoise",
            files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
        )
        assert r.status_code == 400
        assert "duration" in r.json()["detail"].lower()
        assert len(job_manager._jobs) == original_job_count

    def test_unexpected_duration_error_returns_500_and_creates_no_job(
        self, client, monkeypatch
    ) -> None:
        """An unexpected exception during duration reading returns 500 and creates no job."""
        from app.services.jobs import job_manager

        original_job_count = len(job_manager._jobs)
        monkeypatch.setattr(
            "app.routes.denoise.sf.info",
            Mock(side_effect=RuntimeError("unexpected internal error")),
        )

        r = client.post(
            "/api/v1/denoise",
            files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
        )
        assert r.status_code == 500
        assert len(job_manager._jobs) == original_job_count

    @pytest.mark.parametrize("ext", [".mp3", ".m4a", ".aac"])
    def test_librosa_format_exceeding_max_duration_returns_400(
        self, client, mock_denoiser, monkeypatch, ext: str
    ) -> None:
        """MP3/M4A/AAC files longer than the configured limit are rejected with 400."""
        monkeypatch.setattr(
            "librosa.get_duration",
            Mock(return_value=float(settings.max_audio_duration_seconds + 1)),
        )
        r = client.post(
            "/api/v1/denoise",
            files={"file": (f"test{ext}", SAMPLE_BYTES[ext], "audio/octet-stream")},
        )
        assert r.status_code == 400
        assert "too long" in r.json()["detail"].lower()


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
