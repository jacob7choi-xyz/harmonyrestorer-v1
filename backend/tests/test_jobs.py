"""Tests for JobManager business logic."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

import pytest
from app.config import settings
from app.schemas import JobStatus, JobStatusEnum
from app.services.jobs import IPJobCapError, JobCapError, JobManager


def _make_manager_with_jobs() -> JobManager:
    """Helper: manager with one old, one new, and one in-progress job."""
    mgr = JobManager()

    mgr._jobs["old"] = JobStatus(
        job_id="old",
        status=JobStatusEnum.COMPLETED,
        progress=100,
        message="done",
        created_at=datetime.now(UTC) - timedelta(hours=2),
        completed_at=datetime.now(UTC) - timedelta(hours=2),
    )
    mgr._jobs["new"] = JobStatus(
        job_id="new",
        status=JobStatusEnum.COMPLETED,
        progress=100,
        message="done",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    mgr._jobs["active"] = JobStatus(
        job_id="active",
        status=JobStatusEnum.PROCESSING,
        progress=50,
        message="working",
        created_at=datetime.now(UTC) - timedelta(hours=2),
    )
    return mgr


# --- create / get ---


def test_create_job():
    mgr = JobManager()
    job = mgr.create_job("abc", client_ip="127.0.0.1")
    assert job.status == "queued"
    assert job.job_id == "abc"


def test_get_existing_job():
    mgr = JobManager()
    mgr.create_job("abc")
    assert mgr.get_job("abc") is not None


def test_get_missing_job():
    mgr = JobManager()
    assert mgr.get_job("nope") is None


# --- cleanup ---


def test_cleanup_removes_old_completed_jobs():
    mgr = _make_manager_with_jobs()
    removed = mgr.cleanup_expired()

    assert removed == 1
    assert mgr.get_job("old") is None


def test_cleanup_keeps_recent_jobs():
    mgr = _make_manager_with_jobs()
    mgr.cleanup_expired()

    assert mgr.get_job("new") is not None


def test_cleanup_never_touches_in_progress():
    mgr = _make_manager_with_jobs()
    mgr.cleanup_expired()

    assert mgr.get_job("active") is not None


def test_cleanup_returns_zero_when_nothing_expired():
    mgr = JobManager()
    mgr.create_job("fresh")
    assert mgr.cleanup_expired() == 0


def test_cleanup_deletes_all_cached_format_variants() -> None:
    """Expiry removes the WAV and every cached download conversion beside it."""
    from app.config import settings

    mgr = _make_manager_with_jobs()
    wav = settings.processed_dir / "old_denoised.wav"
    mp3 = settings.processed_dir / "old_denoised.mp3"
    flac = settings.processed_dir / "old_denoised.flac"
    for f in (wav, mp3, flac):
        f.write_bytes(b"x")

    try:
        removed = mgr.cleanup_expired()
        assert removed == 1
        assert not wav.exists()
        assert not mp3.exists()
        assert not flac.exists()
    finally:
        for f in (wav, mp3, flac):
            f.unlink(missing_ok=True)


# --- job_counts ---


def test_job_counts():
    mgr = _make_manager_with_jobs()
    counts = mgr.job_counts
    assert counts == {JobStatusEnum.COMPLETED: 2, JobStatusEnum.PROCESSING: 1}


def test_job_counts_empty():
    mgr = JobManager()
    assert mgr.job_counts == {}


# --- process() error paths ---


def test_process_sets_failed_on_denoiser_error(tmp_path):
    """When the denoiser raises, job status should be FAILED."""
    mgr = JobManager()
    mock_denoiser = MagicMock()
    mock_denoiser.denoise.side_effect = RuntimeError("model crashed")
    mgr._denoiser = mock_denoiser

    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"fake audio")

    mgr.create_job("fail-job")
    mgr.process("fail-job", input_file)

    job = mgr.get_job("fail-job")
    assert job is not None
    assert job.status == JobStatusEnum.FAILED
    assert job.progress == -1


def test_process_handles_vanished_job(tmp_path):
    """If a job is removed before processing starts, process() exits cleanly."""
    mgr = JobManager()
    input_file = tmp_path / "input.wav"
    input_file.write_bytes(b"fake audio")

    mgr.create_job("vanish-job")
    # Simulate cleanup removing the job before process runs
    del mgr._jobs["vanish-job"]

    # Should not raise
    mgr.process("vanish-job", input_file)


# --- download guard ---


def test_cleanup_skips_downloading_jobs():
    """Jobs marked as downloading should not be cleaned up."""
    mgr = JobManager()
    mgr._jobs["dl-job"] = JobStatus(
        job_id="dl-job",
        status=JobStatusEnum.COMPLETED,
        progress=100,
        message="done",
        created_at=datetime.now(UTC) - timedelta(hours=2),
        completed_at=datetime.now(UTC) - timedelta(hours=2),
    )

    mgr.mark_downloading("dl-job")
    removed = mgr.cleanup_expired()

    assert removed == 0
    assert mgr.get_job("dl-job") is not None

    mgr.unmark_downloading("dl-job")
    removed = mgr.cleanup_expired()
    assert removed == 1


def test_mark_downloading_returns_false_for_missing_job():
    mgr = JobManager()
    assert mgr.mark_downloading("nonexistent") is False


def test_unmark_downloading_is_idempotent():
    mgr = JobManager()
    # Should not raise even if job was never marked
    mgr.unmark_downloading("nonexistent")


# --- job caps ---


def test_global_cap_rejects_when_at_limit(monkeypatch) -> None:
    """create_job raises JobCapError when resource-occupying jobs reach max_total_jobs."""
    mgr = JobManager()
    monkeypatch.setattr(settings, "max_total_jobs", 2)
    monkeypatch.setattr(settings, "max_jobs_per_ip", 2)
    mgr._jobs["j1"] = JobStatus(
        job_id="j1",
        status=JobStatusEnum.COMPLETED,
        progress=100,
        message="done",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    mgr._jobs["j2"] = JobStatus(
        job_id="j2",
        status=JobStatusEnum.QUEUED,
        progress=0,
        message="queued",
        created_at=datetime.now(UTC),
    )
    with pytest.raises(JobCapError):
        mgr.create_job("j3", client_ip="1.2.3.4")


def test_per_ip_cap_rejects_same_ip(monkeypatch) -> None:
    """create_job raises IPJobCapError when a client has too many resource-occupying jobs."""
    mgr = JobManager()
    monkeypatch.setattr(settings, "max_jobs_per_ip", 2)
    mgr._jobs["j1"] = JobStatus(
        job_id="j1",
        client_ip="1.2.3.4",
        status=JobStatusEnum.QUEUED,
        progress=0,
        message="queued",
        created_at=datetime.now(UTC),
    )
    mgr._jobs["j2"] = JobStatus(
        job_id="j2",
        client_ip="1.2.3.4",
        status=JobStatusEnum.PROCESSING,
        progress=50,
        message="processing",
        created_at=datetime.now(UTC),
    )
    with pytest.raises(IPJobCapError):
        mgr.create_job("j3", client_ip="1.2.3.4")


def test_different_ip_not_blocked_by_per_ip_cap(monkeypatch) -> None:
    """A per-IP cap for one client does not affect a different client."""
    mgr = JobManager()
    monkeypatch.setattr(settings, "max_jobs_per_ip", 1)
    mgr._jobs["j1"] = JobStatus(
        job_id="j1",
        client_ip="1.2.3.4",
        status=JobStatusEnum.QUEUED,
        progress=0,
        message="queued",
        created_at=datetime.now(UTC),
    )
    job = mgr.create_job("j2", client_ip="5.6.7.8")
    assert job.status == JobStatusEnum.QUEUED


def test_failed_does_not_count_toward_global_cap(monkeypatch) -> None:
    """FAILED jobs are excluded from the resource-occupying count."""
    mgr = JobManager()
    monkeypatch.setattr(settings, "max_total_jobs", 1)
    monkeypatch.setattr(settings, "max_jobs_per_ip", 1)
    mgr._jobs["fail"] = JobStatus(
        job_id="fail",
        status=JobStatusEnum.FAILED,
        progress=-1,
        message="failed",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    job = mgr.create_job("new", client_ip="1.2.3.4")
    assert job.status == JobStatusEnum.QUEUED


def test_completed_counts_toward_global_cap(monkeypatch) -> None:
    """COMPLETED jobs count toward the global cap until TTL cleanup removes them."""
    mgr = JobManager()
    monkeypatch.setattr(settings, "max_total_jobs", 1)
    monkeypatch.setattr(settings, "max_jobs_per_ip", 1)
    mgr._jobs["done"] = JobStatus(
        job_id="done",
        status=JobStatusEnum.COMPLETED,
        progress=100,
        message="done",
        created_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    with pytest.raises(JobCapError):
        mgr.create_job("new", client_ip="1.2.3.4")


def test_client_ip_stored_on_job() -> None:
    """client_ip is persisted on the job record for per-IP cap enforcement."""
    mgr = JobManager()
    job = mgr.create_job("abc", client_ip="192.168.1.1")
    assert job.client_ip == "192.168.1.1"


# --- stale download eviction ---


def test_cleanup_evicts_stale_downloading_job() -> None:
    """A completed job with a download marker older than download_ttl_seconds is cleaned up."""
    mgr = JobManager()
    mgr._jobs["stale-dl"] = JobStatus(
        job_id="stale-dl",
        status=JobStatusEnum.COMPLETED,
        progress=100,
        message="done",
        created_at=datetime.now(UTC) - timedelta(hours=2),
        completed_at=datetime.now(UTC) - timedelta(hours=2),
    )
    mgr._downloading["stale-dl"] = datetime.now(UTC) - timedelta(
        seconds=settings.download_ttl_seconds + 1
    )
    removed = mgr.cleanup_expired()
    assert removed == 1
    assert mgr.get_job("stale-dl") is None
    assert "stale-dl" not in mgr._downloading


def test_cleanup_removes_old_failed_jobs() -> None:
    """Expired FAILED jobs are removed; their (absent) output file deletion is a no-op."""
    mgr = JobManager()
    mgr._jobs["old-fail"] = JobStatus(
        job_id="old-fail",
        status=JobStatusEnum.FAILED,
        progress=-1,
        message="failed",
        created_at=datetime.now(UTC) - timedelta(hours=2),
        completed_at=datetime.now(UTC) - timedelta(hours=2),
    )
    removed = mgr.cleanup_expired()
    assert removed == 1
    assert mgr.get_job("old-fail") is None
