"""Tests for JobManager: the denoiser it builds and runs, the download guard, and cleanup."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from app.config import OPGAN_CHECKPOINT_SHA256, DenoiserEngine, settings
from app.services.file_store import LocalFileStore
from app.services.job_store import InMemoryJobStore
from app.services.jobs import JobManager, _DownloadGuard

from tests.job_store_contract import FakeWallClock

_TTL = 60

# Canonical UUIDs, since the file store refuses any other job ID
OLD = "0a1b2c3d-0000-4000-8000-000000000001"
ACTIVE = "0a1b2c3d-0000-4000-8000-000000000002"
NEW = "0a1b2c3d-0000-4000-8000-000000000003"
FRESH = "0a1b2c3d-0000-4000-8000-000000000004"
OLD_FAIL = "0a1b2c3d-0000-4000-8000-000000000005"
STUCK = "0a1b2c3d-0000-4000-8000-000000000006"
FINE = "0a1b2c3d-0000-4000-8000-000000000007"
JOB = "0a1b2c3d-0000-4000-8000-000000000008"
ORPHAN = "0a1b2c3d-0000-4000-8000-000000000009"


def _files() -> LocalFileStore:
    """A file store on the processed folder the test is using now."""
    return LocalFileStore(settings.processed_dir)


async def _finished(store: InMemoryJobStore, job_id: str, *, failed: bool = False) -> None:
    """Create a job and take it to COMPLETED, or to FAILED if *failed*."""
    await store.create(job_id, "", "audio")
    await store.start(job_id)
    if failed:
        await store.fail(job_id)
    else:
        await store.complete(job_id, 1.0, f"/api/v1/download/{job_id}")


async def _store_with_jobs(clock: FakeWallClock) -> InMemoryJobStore:
    """A store with an expired COMPLETED job OLD, a PROCESSING job ACTIVE
    started at the same time, and a COMPLETED job NEW still within the TTL."""
    store = InMemoryJobStore(10, 10, _TTL, clock=clock)
    await _finished(store, OLD)
    await store.create(ACTIVE, "", "audio")
    await store.start(ACTIVE)
    clock.advance(_TTL + 1)
    await _finished(store, NEW)
    return store


# --- engine selection ---


def test_get_denoiser_builds_opgan_service_for_opgan_engine(tmp_path, monkeypatch) -> None:
    """DENOISER_ENGINE=opgan builds the OpGAN service with the configured paths."""
    from app.services.opgan_denoiser import OpGANDenoiserService

    checkpoint = tmp_path / "final.pt"
    monkeypatch.setattr(settings, "denoiser_engine", DenoiserEngine.OPGAN)
    monkeypatch.setattr(settings, "processed_dir", tmp_path / "processed")
    monkeypatch.setattr(settings, "opgan_checkpoint", checkpoint)

    denoiser = JobManager()._get_denoiser()

    assert isinstance(denoiser, OpGANDenoiserService)
    assert denoiser.checkpoint_path == checkpoint.resolve()
    assert denoiser.expected_sha256 == OPGAN_CHECKPOINT_SHA256
    assert denoiser.output_dir == (tmp_path / "processed").resolve()


def test_get_denoiser_builds_uvr_service_for_uvr_engine(tmp_path, monkeypatch) -> None:
    """DENOISER_ENGINE=uvr builds the UVR service with the configured model name."""
    from app.services.denoiser import DenoiserService

    monkeypatch.setattr(settings, "denoiser_engine", DenoiserEngine.UVR)
    monkeypatch.setattr(settings, "processed_dir", tmp_path / "processed")
    monkeypatch.setattr(settings, "uvr_model_name", "UVR-DeNoise.pth")

    denoiser = JobManager()._get_denoiser()

    assert isinstance(denoiser, DenoiserService)
    assert denoiser.model_name == "UVR-DeNoise.pth"
    assert denoiser.output_dir == (tmp_path / "processed").resolve()


def test_get_denoiser_returns_the_same_instance_on_later_calls(tmp_path, monkeypatch) -> None:
    """A later call returns the same denoiser instance."""
    monkeypatch.setattr(settings, "denoiser_engine", DenoiserEngine.OPGAN)
    monkeypatch.setattr(settings, "processed_dir", tmp_path / "processed")

    mgr = JobManager()

    assert mgr._get_denoiser() is mgr._get_denoiser()


# --- worker ---


def test_run_denoiser_moves_output_into_place(tmp_path, monkeypatch) -> None:
    """On success the output lands at the job's path; the input and intermediate are gone."""
    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setattr(settings, "processed_dir", processed)
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"RIFF" + b"\x00" * 16)
    intermediate = processed / "job_opgan.wav"

    def fake_denoise(path):  # type: ignore[no-untyped-def]
        intermediate.write_bytes(b"RIFF" + b"\x00" * 16)
        return intermediate

    mgr = JobManager()
    mgr._denoiser = MagicMock()
    mgr._denoiser.denoise.side_effect = fake_denoise

    processing_time = mgr.run_denoiser(JOB, input_file, LocalFileStore(processed))

    assert processing_time >= 0
    assert (processed / f"{JOB}_denoised.wav").exists()
    assert not intermediate.exists()
    assert not input_file.exists()


def test_run_denoiser_failure_removes_intermediate_output(tmp_path, monkeypatch) -> None:
    """A failed final rename raises and leaves no orphan file.

    The intermediate output is not matched by the cleanup glob, so it must
    never survive run_denoiser().
    """
    processed = tmp_path / "processed"
    processed.mkdir()
    monkeypatch.setattr(settings, "processed_dir", processed)
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"RIFF" + b"\x00" * 16)
    intermediate = processed / "orphan-test_opgan.wav"

    def fake_denoise(path):  # type: ignore[no-untyped-def]
        intermediate.write_bytes(b"RIFF" + b"\x00" * 16)
        return intermediate

    mgr = JobManager()
    mgr._denoiser = MagicMock()
    mgr._denoiser.denoise.side_effect = fake_denoise
    # Force the final rename to fail: the destination is a directory
    (processed / f"{ORPHAN}_denoised.wav").mkdir()

    with pytest.raises(OSError):
        mgr.run_denoiser(ORPHAN, input_file, LocalFileStore(processed))

    assert not intermediate.exists()
    assert not input_file.exists()


# --- cleanup ---


async def test_cleanup_removes_old_completed_jobs() -> None:
    """A COMPLETED job past the TTL is removed from the store."""
    store = await _store_with_jobs(FakeWallClock())

    removed = await JobManager().cleanup_expired(store, _files())

    assert removed == 1
    assert OLD not in await store.expired_ids()


async def test_cleanup_keeps_recent_jobs() -> None:
    """A COMPLETED job within the TTL stays."""
    store = await _store_with_jobs(FakeWallClock())

    await JobManager().cleanup_expired(store, _files())

    assert await store.get(NEW) is not None


async def test_cleanup_never_touches_in_progress() -> None:
    """A PROCESSING job stays, however long ago it started."""
    store = await _store_with_jobs(FakeWallClock())

    await JobManager().cleanup_expired(store, _files())

    assert await store.get(ACTIVE) is not None


async def test_cleanup_returns_zero_when_nothing_expired() -> None:
    """With only a QUEUED job, nothing is removed."""
    store = InMemoryJobStore(10, 10, _TTL, clock=FakeWallClock())
    await store.create(FRESH, "", "audio")

    assert await JobManager().cleanup_expired(store, _files()) == 0


async def test_cleanup_removes_old_failed_jobs() -> None:
    """Expired FAILED jobs are removed; their (absent) output file deletion is a no-op."""
    clock = FakeWallClock()
    store = InMemoryJobStore(10, 10, _TTL, clock=clock)
    await _finished(store, OLD_FAIL, failed=True)
    clock.advance(_TTL + 1)

    removed = await JobManager().cleanup_expired(store, _files())

    assert removed == 1
    assert await store.expired_ids() == []


async def test_cleanup_deletes_all_cached_format_variants(tmp_path, monkeypatch) -> None:
    """Expiry removes the WAV and every cached download conversion beside it."""
    monkeypatch.setattr(settings, "processed_dir", tmp_path)
    store = await _store_with_jobs(FakeWallClock())
    files = [tmp_path / f"{OLD}_denoised.{ext}" for ext in ("wav", "mp3", "flac")]
    for f in files:
        f.write_bytes(b"x")

    removed = await JobManager().cleanup_expired(store, _files())

    assert removed == 1
    assert not any(f.exists() for f in files)


async def test_cleanup_keeps_the_record_when_its_files_cannot_be_deleted(
    tmp_path, monkeypatch
) -> None:
    """A job whose file cannot be deleted keeps its record, so the next pass retries it.

    Other expired jobs are still cleaned up in the same pass.
    """
    monkeypatch.setattr(settings, "processed_dir", tmp_path)
    clock = FakeWallClock()
    store = InMemoryJobStore(10, 10, _TTL, clock=clock)
    await _finished(store, STUCK)
    await _finished(store, FINE)
    clock.advance(_TTL + 1)
    stuck_file = tmp_path / f"{STUCK}_denoised.wav"
    fine_file = tmp_path / f"{FINE}_denoised.wav"
    for f in (stuck_file, fine_file):
        f.write_bytes(b"x")
    real_unlink = Path.unlink

    def failing_unlink(self: Path, missing_ok: bool = False) -> None:
        if self == stuck_file:
            raise PermissionError("cannot delete")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    mgr = JobManager()

    removed = await mgr.cleanup_expired(store, _files())

    assert removed == 1
    assert await store.expired_ids() == [STUCK]
    assert stuck_file.exists()
    assert not fine_file.exists()

    monkeypatch.setattr(Path, "unlink", real_unlink)
    assert await mgr.cleanup_expired(store, _files()) == 1
    assert await store.expired_ids() == []
    assert not stuck_file.exists()


# --- download guard ---


async def test_download_guard_refcount_protects_concurrent_downloads() -> None:
    """Two holds, one release: cleanup must not claim; last release: it may.

    Pins the B3 regression where a single marker let the first finished
    download strip protection from a second still-streaming one.
    """
    store = await _store_with_jobs(FakeWallClock())
    mgr = JobManager()
    mgr.mark_downloading(OLD)
    mgr.mark_downloading(OLD)

    mgr.unmark_downloading(OLD)
    assert await mgr.cleanup_expired(store, _files()) == 0
    assert OLD in await store.expired_ids()

    mgr.unmark_downloading(OLD)
    assert await mgr.cleanup_expired(store, _files()) == 1
    assert OLD not in await store.expired_ids()


async def test_cleanup_skips_downloading_jobs() -> None:
    """Jobs marked as downloading should not be cleaned up."""
    store = await _store_with_jobs(FakeWallClock())
    mgr = JobManager()

    mgr.mark_downloading(OLD)
    assert await mgr.cleanup_expired(store, _files()) == 0
    assert OLD in await store.expired_ids()

    mgr.unmark_downloading(OLD)
    assert await mgr.cleanup_expired(store, _files()) == 1


def test_unmark_downloading_is_idempotent() -> None:
    """Releasing a hold that was never taken does not raise."""
    mgr = JobManager()

    mgr.unmark_downloading("nonexistent")


async def test_cleanup_evicts_stale_downloading_job() -> None:
    """An expired job whose download hold is older than download_ttl_seconds is cleaned up."""
    store = await _store_with_jobs(FakeWallClock())
    mgr = JobManager()
    mgr._downloading[OLD] = _DownloadGuard(
        count=1,
        started_at=datetime.now(UTC) - timedelta(seconds=settings.download_ttl_seconds + 1),
    )

    removed = await mgr.cleanup_expired(store, _files())

    assert removed == 1
    assert OLD not in await store.expired_ids()
    assert OLD not in mgr._downloading
