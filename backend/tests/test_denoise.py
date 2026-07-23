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


def test_upload_returns_job_id(client, mock_denoiser) -> None:
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )
    assert r.status_code == 200
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "completed"


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
def test_upload_accepts_all_supported_formats(client, mock_denoiser, ext) -> None:
    """Every format listed in settings.supported_formats should be accepted."""
    content = SAMPLE_BYTES[ext]
    r = client.post(
        "/api/v1/denoise",
        files={"file": (f"audio{ext}", content, "audio/octet-stream")},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "completed"


def test_upload_returns_failed_status_when_processing_fails(
    client, mock_audio_duration, monkeypatch
) -> None:
    """A processing failure is a 200 with status failed, and the input is cleaned up.

    The transitional contract keeps failure reporting on the job API rather
    than converting inference errors into HTTP 5xx.
    """
    from app.services.jobs import job_manager

    failing = Mock()
    failing.denoise.side_effect = RuntimeError("model exploded")
    monkeypatch.setattr(job_manager, "_denoiser", failing)

    before_files = set(settings.upload_dir.iterdir())
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )

    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    # The input file must not linger after a failed job
    assert set(settings.upload_dir.iterdir()) == before_files


def test_download_immediately_after_upload(client, mock_denoiser) -> None:
    """Processing completes within the upload request, so download works with no polling."""
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )
    assert r.json()["status"] == "completed"

    d = client.get(f"/api/v1/download/{r.json()['job_id']}")
    assert d.status_code == 200


def test_download_filename_uses_original_name(client, mock_denoiser) -> None:
    """The download is named after the uploaded file with a _restored suffix."""
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("My Lonely Heart.wav", SMALL_WAV, "audio/wav")},
    )
    d = client.get(f"/api/v1/download/{r.json()['job_id']}")

    assert d.status_code == 200
    # Starlette emits an RFC 5987 encoded filename; decode before comparing
    from urllib.parse import unquote

    assert "My Lonely Heart_restored.wav" in unquote(d.headers["content-disposition"])


def test_download_filename_sanitizes_hostile_names(client, mock_denoiser) -> None:
    """Path tricks and unsafe characters never reach the download header."""
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("../../etc/pass#wd$!.wav", SMALL_WAV, "audio/wav")},
    )
    d = client.get(f"/api/v1/download/{r.json()['job_id']}")

    assert d.status_code == 200
    disposition = d.headers["content-disposition"]
    assert "passwd_restored.wav" in disposition
    assert "/" not in disposition.split("filename")[1]
    assert ".." not in disposition


def test_download_filename_falls_back_when_nothing_safe_remains(client, mock_denoiser) -> None:
    """A name with no safe characters gets the generic audio stem."""
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("....wav", SMALL_WAV, "audio/wav")},
    )
    d = client.get(f"/api/v1/download/{r.json()['job_id']}")

    assert d.status_code == 200
    assert "audio_restored.wav" in d.headers["content-disposition"]


def test_lifecycle_start_failure_rolls_back_the_job_transaction(
    client, mock_audio_duration, monkeypatch
) -> None:
    """A create_task failure at the ownership-transfer boundary leaves nothing.

    The client never received the job_id, so this is a failed transaction,
    not a failed job: no record, no input file, and the admission slot is
    restored for the next upload.
    """
    import asyncio as asyncio_module

    from app.routes import denoise as denoise_module
    from app.services.jobs import job_manager

    real_create_task = asyncio_module.create_task

    def selective_boom(coro, **kwargs):  # type: ignore[no-untyped-def]
        if getattr(coro, "__name__", "") == "_run_inference":
            coro.close()
            raise RuntimeError("loop unavailable")
        return real_create_task(coro, **kwargs)

    monkeypatch.setattr(denoise_module.asyncio, "create_task", selective_boom)

    files_before = set(settings.upload_dir.iterdir())
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )

    assert r.status_code == 500
    assert job_manager._jobs == {}
    assert set(settings.upload_dir.iterdir()) == files_before
    # Admission capacity restored
    assert denoise_module.admission.try_acquire() is True
    denoise_module.admission.release()


def test_download_rejects_unknown_format(client, mock_denoiser) -> None:
    """Formats outside the allowlist are rejected before any file access."""
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )
    d = client.get(f"/api/v1/download/{r.json()['job_id']}?format=exe")

    assert d.status_code == 400
    assert "Unsupported format" in d.json()["detail"]


def test_download_converts_to_requested_format(client, mock_denoiser, monkeypatch) -> None:
    """A non-wav format is served with matching name and media type."""
    from app.routes import denoise as denoise_module

    def fake_transcode(wav_path, fmt):  # type: ignore[no-untyped-def]
        converted = wav_path.with_suffix(f".{fmt}")
        converted.write_bytes(b"encoded")
        return converted

    monkeypatch.setattr(denoise_module, "transcode_output", fake_transcode)

    r = client.post(
        "/api/v1/denoise",
        files={"file": ("song.wav", SMALL_WAV, "audio/wav")},
    )
    d = client.get(f"/api/v1/download/{r.json()['job_id']}?format=mp3")

    assert d.status_code == 200
    assert d.headers["content-type"].startswith("audio/mpeg")
    assert "song_restored.mp3" in d.headers["content-disposition"]


def test_download_conversion_failure_returns_500_and_releases_guard(
    client, mock_denoiser, monkeypatch
) -> None:
    """A failed conversion reports 500 and does not leave the job download-locked."""
    from app.routes import denoise as denoise_module
    from app.services.jobs import job_manager
    from app.services.transcode import TranscodeError

    def failing_transcode(wav_path, fmt):  # type: ignore[no-untyped-def]
        if fmt == "wav":
            return wav_path
        raise TranscodeError("encoder unavailable")

    monkeypatch.setattr(denoise_module, "transcode_output", failing_transcode)

    r = client.post(
        "/api/v1/denoise",
        files={"file": ("song.wav", SMALL_WAV, "audio/wav")},
    )
    job_id = r.json()["job_id"]
    d = client.get(f"/api/v1/download/{job_id}?format=ogg")

    assert d.status_code == 500
    assert d.json()["detail"] == "Format conversion failed"
    # Guard released: a wav download still succeeds afterwards
    assert job_id not in job_manager._downloading
    assert client.get(f"/api/v1/download/{job_id}").status_code == 200


def test_status_response_does_not_leak_download_stem(client, mock_denoiser) -> None:
    """download_stem is internal-only, like client_ip."""
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("secret project name.wav", SMALL_WAV, "audio/wav")},
    )
    s = client.get(f"/api/v1/status/{r.json()['job_id']}")

    assert s.status_code == 200
    assert "download_stem" not in s.json()


def test_upload_rejected_when_inference_busy(client, mock_denoiser) -> None:
    """Busy admission is an immediate 503 with no job record and no new files."""
    from app.routes import denoise as denoise_module
    from app.services.jobs import job_manager

    assert denoise_module.admission.try_acquire()
    try:
        jobs_before = dict(job_manager._jobs)
        files_before = set(settings.upload_dir.iterdir())

        r = client.post(
            "/api/v1/denoise",
            files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
        )

        assert r.status_code == 503
        assert r.headers.get("retry-after") == "30"
        assert job_manager._jobs == jobs_before
        assert set(settings.upload_dir.iterdir()) == files_before
    finally:
        denoise_module.admission.release()


def test_upload_processes_exactly_once(client, mock_denoiser, monkeypatch) -> None:
    """The synchronous path runs processing once, with no scheduled background task."""
    from app.services.jobs import job_manager

    calls: list[str] = []
    original_process = job_manager.process

    def counting_process(job_id, input_path) -> None:
        calls.append(job_id)
        original_process(job_id, input_path)

    monkeypatch.setattr(job_manager, "process", counting_process)

    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )

    assert r.status_code == 200
    assert calls == [r.json()["job_id"]]


# --- Status ---


def test_status_after_processing(client, mock_denoiser) -> None:
    """Processing completes within the upload request, so status is terminal immediately."""
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


def test_per_ip_cap_returns_429_and_removes_input_file(
    client, mock_audio_duration, monkeypatch
) -> None:
    """Per-IP cap rejection returns 429, creates no new job, and leaves no input file behind.

    Uses max_jobs_per_ip=1 (a valid production config) with one existing COMPLETED job
    from the same IP so the cap is already at its limit. Covers the route cleanup path
    that fires after the temp file has been renamed to input_path: if that cleanup were
    missing, rejected uploads would accumulate on disk indefinitely.
    """
    from datetime import UTC, datetime

    from app.schemas import JobStatus, JobStatusEnum
    from app.services.jobs import job_manager

    monkeypatch.setattr(settings, "max_jobs_per_ip", 1)
    # "testclient" is the peer host Starlette's TestClient reports via request.client.host
    job_manager._jobs["existing"] = JobStatus(
        job_id="existing",
        client_ip="testclient",
        status=JobStatusEnum.COMPLETED,
        progress=100,
        message="done",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    upload_files_before = set(settings.upload_dir.iterdir())

    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )

    assert r.status_code == 429
    assert "Too many active jobs" in r.json()["detail"]
    assert "existing" in job_manager._jobs
    assert len(job_manager._jobs) == 1
    assert set(settings.upload_dir.iterdir()) == upload_files_before


def test_global_cap_returns_503_and_removes_input_file(
    client, mock_audio_duration, monkeypatch
) -> None:
    """Global job cap rejection returns 503, creates no new job, and leaves no input file behind.

    Uses max_total_jobs=1 with an existing job from a different IP so the global cap
    fires before the per-IP check. Verifies the same route cleanup path as the 429 case.
    """
    from datetime import UTC, datetime

    from app.schemas import JobStatus, JobStatusEnum
    from app.services.jobs import job_manager

    monkeypatch.setattr(settings, "max_total_jobs", 1)
    monkeypatch.setattr(settings, "max_jobs_per_ip", 1)
    job_manager._jobs["existing"] = JobStatus(
        job_id="existing",
        client_ip="other-client",
        status=JobStatusEnum.COMPLETED,
        progress=100,
        message="done",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    upload_files_before = set(settings.upload_dir.iterdir())

    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )

    assert r.status_code == 503
    assert "Server is busy" in r.json()["detail"]
    assert "existing" in job_manager._jobs
    assert len(job_manager._jobs) == 1
    assert set(settings.upload_dir.iterdir()) == upload_files_before


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
