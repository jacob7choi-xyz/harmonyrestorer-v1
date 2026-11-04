"""Behaviour every JobStore adapter must show (ADR 010).

An adapter's test module subclasses JobStoreContract and provides the
make_store fixture. The class name does not start with Test, so pytest
collects these tests only through a subclass.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest
from app.schemas import JobStatusEnum
from app.services.job_store import IPJobCapError, JobCapError, JobStore


class FakeWallClock:
    """A UTC clock that moves only when the test advances it."""

    def __init__(self, start: datetime = datetime(2026, 1, 1, tzinfo=UTC)) -> None:
        """Start the clock at *start*."""
        self.now = start

    def __call__(self) -> datetime:
        """Return the current fake time."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by *seconds*."""
        self.now += timedelta(seconds=seconds)


MakeStore = Callable[[int, int, float, FakeWallClock], JobStore]


async def _processing(store: JobStore, job_id: str, client_ip: str = "") -> None:
    """Create a job and move it to PROCESSING."""
    await store.create(job_id, client_ip, "audio")
    assert await store.start(job_id)


class JobStoreContract:
    """Tests run against every JobStore adapter."""

    @pytest.fixture()
    def make_store(self) -> MakeStore:
        """Return a factory taking (max_total_jobs, max_jobs_per_ip, job_ttl_seconds, clock)."""
        raise NotImplementedError("Adapter test modules provide make_store")

    # --- create and get ---

    async def test_create_adds_a_queued_job(self, make_store: MakeStore) -> None:
        """A new job is QUEUED, with its client, stem and creation time."""
        clock = FakeWallClock()
        store = make_store(10, 3, 3600, clock)

        await store.create("job", "client-a", "song")

        job = await store.get("job")
        assert job is not None
        assert job.status == JobStatusEnum.QUEUED
        assert job.progress == 0
        assert job.client_ip == "client-a"
        assert job.download_stem == "song"
        assert job.created_at == clock.now
        assert job.completed_at is None

    async def test_get_returns_none_for_an_unknown_job(self, make_store: MakeStore) -> None:
        """Reading a job that was never created returns None."""
        store = make_store(10, 3, 3600, FakeWallClock())

        assert await store.get("missing") is None

    async def test_changing_a_returned_job_does_not_change_the_stored_one(
        self, make_store: MakeStore
    ) -> None:
        """Jobs returned by create and get are copies."""
        store = make_store(10, 3, 3600, FakeWallClock())
        created = await store.create("job", "", "audio")
        created.status = JobStatusEnum.FAILED
        read = await store.get("job")
        assert read is not None
        read.status = JobStatusEnum.COMPLETED

        stored = await store.get("job")

        assert stored is not None
        assert stored.status == JobStatusEnum.QUEUED

    # --- caps ---

    async def test_global_cap_counts_queued_processing_and_completed(
        self, make_store: MakeStore
    ) -> None:
        """QUEUED, PROCESSING and COMPLETED jobs all hold a place under the global cap."""
        store = make_store(3, 3, 3600, FakeWallClock())
        await store.create("queued", "", "audio")
        await _processing(store, "processing")
        await _processing(store, "completed")
        assert await store.complete("completed", 1.0, "/download/completed")

        with pytest.raises(JobCapError):
            await store.create("one-more", "", "audio")
        assert await store.get("one-more") is None

    async def test_failed_jobs_do_not_count_toward_the_caps(self, make_store: MakeStore) -> None:
        """A FAILED job frees its place under both caps."""
        store = make_store(1, 1, 3600, FakeWallClock())
        await _processing(store, "failed", "client-a")
        assert await store.fail("failed")

        await store.create("next", "client-a", "audio")

        assert await store.get("next") is not None

    async def test_per_ip_cap_rejects_only_that_client(self, make_store: MakeStore) -> None:
        """A client at its cap is rejected while another client is not."""
        store = make_store(10, 2, 3600, FakeWallClock())
        await store.create("a1", "client-a", "audio")
        await store.create("a2", "client-a", "audio")

        with pytest.raises(IPJobCapError):
            await store.create("a3", "client-a", "audio")
        await store.create("b1", "client-b", "audio")

        assert await store.get("a3") is None
        assert await store.get("b1") is not None

    async def test_empty_client_ip_counts_only_toward_the_global_cap(
        self, make_store: MakeStore
    ) -> None:
        """Jobs created without a client identity are not grouped under one per-IP cap."""
        store = make_store(10, 1, 3600, FakeWallClock())
        await store.create("first", "", "audio")

        await store.create("second", "", "audio")

        assert await store.get("second") is not None

    async def test_concurrent_creates_never_exceed_the_global_cap(
        self, make_store: MakeStore
    ) -> None:
        """Creates issued together still respect the cap."""
        store = make_store(3, 3, 3600, FakeWallClock())

        results = await asyncio.gather(
            *(store.create(f"job-{i}", "", "audio") for i in range(10)),
            return_exceptions=True,
        )

        assert sum(1 for r in results if not isinstance(r, BaseException)) == 3
        assert all(isinstance(r, JobCapError) for r in results if isinstance(r, BaseException))
        assert sum((await store.counts()).values()) == 3

    # --- status changes ---

    async def test_a_job_moves_from_queued_to_processing_to_completed(
        self, make_store: MakeStore
    ) -> None:
        """start then complete record progress, timing, the download URL and completion time."""
        clock = FakeWallClock()
        store = make_store(10, 3, 3600, clock)
        await store.create("job", "", "audio")

        assert await store.start("job")
        processing = await store.get("job")
        assert processing is not None
        assert processing.status == JobStatusEnum.PROCESSING
        assert processing.progress == 10

        clock.advance(5)
        assert await store.complete("job", 4.5, "/download/job")
        job = await store.get("job")
        assert job is not None
        assert job.status == JobStatusEnum.COMPLETED
        assert job.progress == 100
        assert job.processing_time == 4.5
        assert job.download_url == "/download/job"
        assert job.completed_at == clock.now

    async def test_fail_moves_a_processing_job_to_failed(self, make_store: MakeStore) -> None:
        """fail records progress -1 and the completion time, with no download URL."""
        clock = FakeWallClock()
        store = make_store(10, 3, 3600, clock)
        await _processing(store, "job")

        assert await store.fail("job")

        job = await store.get("job")
        assert job is not None
        assert job.status == JobStatusEnum.FAILED
        assert job.progress == -1
        assert job.download_url is None
        assert job.completed_at == clock.now

    async def test_status_changes_are_refused_from_the_wrong_state(
        self, make_store: MakeStore
    ) -> None:
        """Each change is refused unless the job is in the state it expects."""
        store = make_store(10, 3, 3600, FakeWallClock())
        await store.create("queued", "", "audio")
        await _processing(store, "completed")
        assert await store.complete("completed", 1.0, "/download/completed")
        await _processing(store, "failed")
        assert await store.fail("failed")

        assert not await store.complete("queued", 1.0, "/download/queued")
        assert not await store.fail("queued")
        assert not await store.start("completed")
        assert not await store.fail("completed")
        assert not await store.start("failed")
        assert not await store.complete("failed", 1.0, "/download/failed")

        statuses = {job_id: await store.get(job_id) for job_id in ("queued", "completed", "failed")}
        assert {k: v.status for k, v in statuses.items() if v is not None} == {
            "queued": JobStatusEnum.QUEUED,
            "completed": JobStatusEnum.COMPLETED,
            "failed": JobStatusEnum.FAILED,
        }

    async def test_a_job_starts_only_once(self, make_store: MakeStore) -> None:
        """A second start of the same job is refused."""
        store = make_store(10, 3, 3600, FakeWallClock())
        await store.create("job", "", "audio")

        assert await store.start("job")
        assert not await store.start("job")

    async def test_status_changes_are_refused_for_an_unknown_job(
        self, make_store: MakeStore
    ) -> None:
        """No status change creates a job."""
        store = make_store(10, 3, 3600, FakeWallClock())

        assert not await store.start("missing")
        assert not await store.complete("missing", 1.0, "/download/missing")
        assert not await store.fail("missing")
        assert await store.get("missing") is None

    # --- delete, counts, expiry ---

    async def test_delete_removes_the_job_and_frees_its_place(self, make_store: MakeStore) -> None:
        """A deleted job is gone and no longer counts toward the caps."""
        store = make_store(1, 1, 3600, FakeWallClock())
        await store.create("job", "client-a", "audio")

        await store.delete("job")
        await store.delete("never-created")

        assert await store.get("job") is None
        await store.create("next", "client-a", "audio")

    async def test_counts_reports_jobs_by_status(self, make_store: MakeStore) -> None:
        """counts has one entry per status that has jobs."""
        store = make_store(10, 10, 3600, FakeWallClock())
        await store.create("q1", "", "audio")
        await store.create("q2", "", "audio")
        await _processing(store, "p")

        assert await store.counts() == {JobStatusEnum.QUEUED: 2, JobStatusEnum.PROCESSING: 1}

    async def test_counts_is_empty_with_no_jobs(self, make_store: MakeStore) -> None:
        """A new store reports no counts."""
        store = make_store(10, 3, 3600, FakeWallClock())

        assert await store.counts() == {}

    async def test_expired_ids_lists_finished_jobs_past_the_ttl(
        self, make_store: MakeStore
    ) -> None:
        """Only COMPLETED and FAILED jobs finished more than the TTL ago are listed."""
        clock = FakeWallClock()
        store = make_store(10, 10, 60, clock)
        await _processing(store, "old-completed")
        assert await store.complete("old-completed", 1.0, "/download/old-completed")
        await _processing(store, "old-failed")
        assert await store.fail("old-failed")
        await store.create("old-queued", "", "audio")
        await _processing(store, "old-processing")
        clock.advance(30)
        await _processing(store, "new-completed")
        assert await store.complete("new-completed", 1.0, "/download/new-completed")

        clock.advance(31)

        assert sorted(await store.expired_ids()) == ["old-completed", "old-failed"]

    async def test_a_job_exactly_one_ttl_old_has_not_expired(self, make_store: MakeStore) -> None:
        """Expiry needs strictly more than the TTL to have passed."""
        clock = FakeWallClock()
        store = make_store(10, 10, 60, clock)
        await _processing(store, "job")
        assert await store.complete("job", 1.0, "/download/job")

        clock.advance(60)
        assert await store.expired_ids() == []
        clock.advance(1)
        assert await store.expired_ids() == ["job"]
