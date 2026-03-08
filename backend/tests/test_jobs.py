"""Tests for JobManager business logic."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from app.schemas import JobStatus, JobStatusEnum
from app.services.jobs import JobManager


def _make_manager_with_jobs() -> JobManager:
    """Helper: manager with one old, one new, and one in-progress job."""
    mgr = JobManager()

    mgr._jobs["old"] = JobStatus(
        job_id="old",
        status=JobStatusEnum.COMPLETED,
        progress=100,
        message="done",
        created_at=datetime.now() - timedelta(hours=2),
        completed_at=datetime.now() - timedelta(hours=2),
    )
    mgr._jobs["new"] = JobStatus(
        job_id="new",
        status=JobStatusEnum.COMPLETED,
        progress=100,
        message="done",
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    mgr._jobs["active"] = JobStatus(
        job_id="active",
        status=JobStatusEnum.PROCESSING,
        progress=50,
        message="working",
        created_at=datetime.now() - timedelta(hours=2),
    )
    return mgr


# --- create / get ---


def test_create_job():
    mgr = JobManager()
    job = mgr.create_job("abc")
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
        created_at=datetime.now() - timedelta(hours=2),
        completed_at=datetime.now() - timedelta(hours=2),
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
