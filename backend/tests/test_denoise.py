"""Tests for the denoise API endpoints."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
import soundfile as sf
from app.config import DenoiserEngine, settings
from app.services.audio_decode import decoded_size_cap
from app.services.engines import input_sample_rate, peak_output_bytes

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


def test_upload_write_failure_cleans_up_temp_file(client, monkeypatch, mock_decode):
    """If writing the upload fails, no traces remain in the upload dir and 500 is returned."""
    before = set(settings.upload_dir.iterdir())

    def fail_fsync(fd: int) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr("app.routes.denoise.os.fsync", fail_fsync)

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
    client, mock_decode, monkeypatch
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


def test_upload_rejected_when_artifact_budget_exhausted(client, mock_decode, monkeypatch) -> None:
    """A full byte budget rejects the upload with 503 and leaves nothing:
    no job, no files, and the inference slot is restored."""
    from app.routes import denoise as denoise_module
    from app.services.artifacts import ArtifactBudget
    from app.services.jobs import job_manager

    tiny = ArtifactBudget(limit_bytes=1, directories=[])
    monkeypatch.setattr(denoise_module, "artifact_budget", tiny)

    files_before = set(settings.upload_dir.iterdir())
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
    )

    assert r.status_code == 503
    assert "storage" in r.json()["detail"].lower()
    assert r.headers.get("retry-after") == "60"
    assert job_manager._jobs == {}
    assert set(settings.upload_dir.iterdir()) == files_before
    assert denoise_module.admission.try_acquire() is True
    denoise_module.admission.release()


@pytest.mark.parametrize("engine", list(DenoiserEngine))
def test_upload_reserves_the_configured_engine_peak(
    client, mock_denoiser, monkeypatch, engine: DenoiserEngine
) -> None:
    """The upload reserves its input bytes plus the configured engine's peak
    output bytes at the maximum accepted duration."""
    from app.routes import denoise as denoise_module
    from app.services.artifacts import ArtifactBudget

    budget = ArtifactBudget(limit_bytes=10**12, directories=[])
    reserve = budget.try_reserve
    requested: list[int] = []

    def recording_reserve(amount: int) -> bool:
        requested.append(amount)
        return reserve(amount)

    monkeypatch.setattr(budget, "try_reserve", recording_reserve)
    monkeypatch.setattr(denoise_module, "artifact_budget", budget)
    monkeypatch.setattr(settings, "denoiser_engine", engine)

    r = client.post("/api/v1/denoise", files={"file": ("test.wav", SMALL_WAV, "audio/wav")})

    assert r.status_code == 200
    max_s = settings.max_audio_duration_seconds
    expected = decoded_size_cap(input_sample_rate(engine), max_s) + peak_output_bytes(engine, max_s)
    assert requested == [len(SMALL_WAV) + expected]


def test_lifecycle_start_failure_rolls_back_the_job_transaction(
    client, mock_decode, monkeypatch
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


def test_download_conversion_busy_returns_503_with_retry_after(
    client, mock_denoiser, monkeypatch
) -> None:
    """A busy conversion slot maps to 503 with Retry-After, guard released."""
    from app.routes import denoise as denoise_module
    from app.services.jobs import job_manager
    from app.services.transcode import TranscodeBusyError

    def busy_transcode(wav_path, fmt):  # type: ignore[no-untyped-def]
        if fmt == "wav":
            return wav_path
        raise TranscodeBusyError("Another conversion is in progress")

    monkeypatch.setattr(denoise_module, "transcode_output", busy_transcode)

    r = client.post(
        "/api/v1/denoise",
        files={"file": ("song.wav", SMALL_WAV, "audio/wav")},
    )
    job_id = r.json()["job_id"]
    d = client.get(f"/api/v1/download/{job_id}?format=mp3")

    assert d.status_code == 503
    assert d.headers.get("retry-after") == "15"
    assert job_id not in job_manager._downloading
    assert client.get(f"/api/v1/download/{job_id}").status_code == 200


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


def test_per_ip_cap_returns_429_and_removes_input_file(client, mock_decode, monkeypatch) -> None:
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


def test_global_cap_returns_503_and_removes_input_file(client, mock_decode, monkeypatch) -> None:
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


def _ffmpeg(*args: str) -> None:
    """Build a real test input with ffmpeg, which CI installs."""
    import subprocess

    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


def _tone_wav(path, seconds: float) -> None:
    import numpy as np

    t = np.arange(round(seconds * 16_000)) / 16_000
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 16_000)


@pytest.fixture()
def stub_denoiser():
    """A denoiser stand-in that records its input and writes a small WAV, so the upload
    path runs with real decoding but without a model."""
    from unittest.mock import MagicMock

    from app.services.jobs import job_manager

    denoiser = MagicMock()
    denoiser.inputs = []

    def fake_denoise(input_path):
        info = sf.info(str(input_path))
        denoiser.inputs.append((input_path.suffix, info.samplerate, info.channels, info.duration))
        output = settings.processed_dir / f"{input_path.stem}_out.wav"
        _tone_wav(output, 0.1)
        return output

    denoiser.denoise.side_effect = fake_denoise
    job_manager._denoiser = denoiser
    yield denoiser
    job_manager._denoiser = None


class TestUploadDecoding:
    """Uploads are decoded once and measured by their decoded length."""

    def test_vbr_mp3_longer_than_the_cap_is_rejected_by_its_decoded_length(
        self, client, stub_denoiser, monkeypatch, tmp_path
    ) -> None:
        """A VBR MP3 without a Xing header reads as shorter than it is through
        soundfile. Measured by its decoded length, a 30 s file does not pass a 25 s
        cap, and the rejected upload leaves no job and no file."""
        from app.services.jobs import job_manager

        wav = tmp_path / "tone.wav"
        _tone_wav(wav, 30.0)
        mp3 = tmp_path / "vbr.mp3"
        _ffmpeg("-i", str(wav), "-c:a", "libmp3lame", "-q:a", "2", "-write_xing", "0", str(mp3))
        assert sf.info(str(mp3)).duration < 25.0  # what the metadata-based check read
        monkeypatch.setattr(settings, "max_audio_duration_seconds", 25)
        uploads_before = set(settings.upload_dir.iterdir())

        r = client.post(
            "/api/v1/denoise", files={"file": ("vbr.mp3", mp3.read_bytes(), "audio/mpeg")}
        )

        assert r.status_code == 400
        assert "too long" in r.json()["detail"].lower()
        assert job_manager._jobs == {}
        assert set(settings.upload_dir.iterdir()) == uploads_before
        stub_denoiser.denoise.assert_not_called()

    @pytest.mark.parametrize(
        ("extension", "codec_args"),
        [
            (".wav", ["-c:a", "pcm_s16le"]),
            (".flac", ["-c:a", "flac"]),
            (".ogg", None),  # written with soundfile; not every ffmpeg build has libvorbis
            (".mp3", ["-c:a", "libmp3lame"]),
            (".m4a", ["-c:a", "aac", "-f", "mp4"]),
            (".aac", ["-c:a", "aac", "-f", "adts"]),
        ],
    )
    def test_each_format_reaches_the_engine_as_a_wav_at_its_rate(
        self, client, stub_denoiser, tmp_path, extension: str, codec_args: list[str] | None
    ) -> None:
        """Every accepted format is decoded, and the engine reads a WAV at its own rate.
        M4A and AAC failed with the default engine before, because OpGAN read the
        upload with soundfile."""
        wav = tmp_path / "tone.wav"
        _tone_wav(wav, 1.0)
        src = tmp_path / f"in{extension}"
        if codec_args is None:
            audio, _ = sf.read(str(wav), dtype="float32")
            sf.write(
                str(src), np.stack([audio, audio], axis=1), 16_000, format="OGG", subtype="VORBIS"
            )
        else:
            _ffmpeg("-i", str(wav), "-ar", "44100", "-ac", "2", *codec_args, str(src))

        r = client.post("/api/v1/denoise", files={"file": (src.name, src.read_bytes(), "audio/x")})

        assert r.status_code == 200, r.text
        assert r.json()["status"] == "completed"
        suffix, rate, channels, duration = stub_denoiser.inputs[0]
        assert suffix == ".wav"
        assert rate == input_sample_rate(settings.denoiser_engine)
        assert channels == 2
        assert duration == pytest.approx(1.0, abs=0.15)

    def _assert_nothing_left(self, uploads_before: set) -> None:
        """No job, no upload file, and the admission slot and reservation released."""
        from app.routes import denoise as denoise_module
        from app.services.jobs import job_manager

        assert job_manager._jobs == {}
        assert set(settings.upload_dir.iterdir()) == uploads_before
        assert denoise_module.artifact_budget._reserved == 0
        assert denoise_module.admission.try_acquire() is True
        denoise_module.admission.release()

    def test_wav_over_the_cap_is_rejected_and_leaves_nothing(
        self, client, stub_denoiser, monkeypatch, tmp_path
    ) -> None:
        """A real WAV longer than the cap is rejected with 400."""
        wav = tmp_path / "long.wav"
        _tone_wav(wav, 3.0)
        monkeypatch.setattr(settings, "max_audio_duration_seconds", 2)
        before = set(settings.upload_dir.iterdir())

        r = client.post(
            "/api/v1/denoise", files={"file": ("long.wav", wav.read_bytes(), "audio/wav")}
        )

        assert r.status_code == 400
        assert "too long" in r.json()["detail"].lower()
        self._assert_nothing_left(before)
        stub_denoiser.denoise.assert_not_called()

    def test_undecodable_upload_is_rejected_and_leaves_nothing(self, client, stub_denoiser) -> None:
        """Bytes with the right magic but no audio are rejected with 400."""
        before = set(settings.upload_dir.iterdir())

        r = client.post("/api/v1/denoise", files={"file": ("test.wav", SMALL_WAV, "audio/wav")})

        assert r.status_code == 400
        assert "decode" in r.json()["detail"].lower()
        self._assert_nothing_left(before)
        stub_denoiser.denoise.assert_not_called()

    def test_unavailable_decoder_is_a_server_error_and_leaves_nothing(
        self, client, stub_denoiser, monkeypatch, caplog
    ) -> None:
        """A missing ffmpeg or a timeout is the server's failure: 500, logged at ERROR,
        not a 400 telling the user to re-export the file."""
        import logging

        from app.services.audio_decode import DecoderUnavailableError

        def unavailable(*args, **kwargs):
            raise DecoderUnavailableError("ffmpeg is not installed")

        monkeypatch.setattr("app.routes.denoise.decode_to_wav", unavailable)
        before = set(settings.upload_dir.iterdir())

        with caplog.at_level(logging.ERROR, logger="app.routes.denoise"):
            r = client.post("/api/v1/denoise", files={"file": ("test.wav", SMALL_WAV, "audio/wav")})

        assert r.status_code == 500
        assert any(rec.levelno == logging.ERROR for rec in caplog.records)
        self._assert_nothing_left(before)

    def test_unexpected_decode_error_is_a_server_error_and_leaves_nothing(
        self, client, stub_denoiser, monkeypatch
    ) -> None:
        """An unexpected exception while decoding returns 500 and creates no job."""

        def broken(*args, **kwargs):
            raise RuntimeError("unexpected")

        monkeypatch.setattr("app.routes.denoise.decode_to_wav", broken)
        before = set(settings.upload_dir.iterdir())

        r = client.post("/api/v1/denoise", files={"file": ("test.wav", SMALL_WAV, "audio/wav")})

        assert r.status_code == 500
        self._assert_nothing_left(before)


async def test_cancelled_request_waits_for_the_decode_and_removes_its_output(
    monkeypatch, tmp_path
) -> None:
    """If the request is cancelled while ffmpeg runs, _decode_upload does not return
    until the decode stops, and removes what it wrote, so the caller does not release
    the admission slot or the reservation while the decode still uses them."""
    import asyncio
    import threading

    from app.routes import denoise as denoise_module

    release = threading.Event()
    started = threading.Event()

    def slow_decode(source, destination, **kwargs):
        started.set()
        release.wait(timeout=10)
        _tone_wav(destination, 0.1)
        return 0.1

    monkeypatch.setattr(denoise_module, "decode_to_wav", slow_decode)
    destination = tmp_path / "out.wav"
    task = asyncio.ensure_future(
        denoise_module._decode_upload(tmp_path / "in.wav", destination, "wav", 16_000)
    )
    while not started.is_set():
        await asyncio.sleep(0.01)
    task.cancel()
    await asyncio.sleep(0.05)
    assert not task.done()  # still waiting for ffmpeg

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not destination.exists()
