"""Tests for JobManager business logic."""

from datetime import datetime, timedelta

from app.schemas import JobStatus
from app.services.jobs import JobManager


def _make_manager_with_jobs() -> JobManager:
    """Helper: manager with one old, one new, and one in-progress job."""
    mgr = JobManager()

    mgr._jobs["old"] = JobStatus(
        job_id="old",
        status="completed",
        progress=100,
        message="done",
        created_at=datetime.now() - timedelta(hours=2),
        completed_at=datetime.now() - timedelta(hours=2),
    )
    mgr._jobs["new"] = JobStatus(
        job_id="new",
        status="completed",
        progress=100,
        message="done",
        created_at=datetime.now(),
        completed_at=datetime.now(),
    )
    mgr._jobs["active"] = JobStatus(
        job_id="active",
        status="processing",
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
    assert counts == {"completed": 2, "processing": 1}


def test_job_counts_empty():
    mgr = JobManager()
    assert mgr.job_counts == {}
