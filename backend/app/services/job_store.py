"""The contract for storing job records, and how the configured store is built (ADR 010)."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, assert_never

from app.config import JobStoreBackend, settings
from app.schemas import JobStatus, JobStatusEnum


class JobCapError(Exception):
    """Raised when the global resource-occupying job cap is reached."""


class IPJobCapError(Exception):
    """Raised when the per-IP resource-occupying job cap is reached."""


# Jobs in these states hold files or inference capacity, so they count toward the caps
RESOURCE_STATUSES: frozenset[JobStatusEnum] = frozenset(
    {JobStatusEnum.QUEUED, JobStatusEnum.PROCESSING, JobStatusEnum.COMPLETED}
)


class JobStore(Protocol):
    """Keeps job records and enforces the job caps (ADR 010).

    Jobs returned by create and get are copies, so changing one never changes
    the stored job. Status changes name the state they expect and do nothing
    if the job is in another state.
    """

    async def create(self, job_id: str, client_ip: str, download_stem: str) -> JobStatus:
        """Add a QUEUED job, unless that would exceed a job cap.

        Checking the caps and adding the job happen as one step, so concurrent
        creates cannot together exceed a cap.

        Args:
            job_id: UUID string for the new job.
            client_ip: The client's identity; an empty string skips the per-IP cap.
            download_stem: Sanitized stem for the eventual download filename.

        Returns:
            The new job.

        Raises:
            JobCapError: If the global cap is reached.
            IPJobCapError: If the per-IP cap is reached for client_ip.
        """
        ...

    async def get(self, job_id: str) -> JobStatus | None:
        """Return the job, or None if there is no such job."""
        ...

    async def start(self, job_id: str) -> bool:
        """Move a job from QUEUED to PROCESSING. Returns False if it was not QUEUED."""
        ...

    async def complete(self, job_id: str, processing_time: float, download_url: str) -> bool:
        """Move a job from PROCESSING to COMPLETED. Returns False if it was not PROCESSING."""
        ...

    async def fail(self, job_id: str) -> bool:
        """Move a job from PROCESSING to FAILED. Returns False if it was not PROCESSING."""
        ...

    async def delete(self, job_id: str) -> None:
        """Remove the job if it exists."""
        ...

    async def counts(self) -> dict[str, int]:
        """Return the number of jobs in each status that has any."""
        ...

    async def expired_ids(self) -> list[str]:
        """Return the IDs of COMPLETED or FAILED jobs finished more than the TTL ago."""
        ...


def _utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


class InMemoryJobStore:
    """Job records kept in a dict inside this process.

    The records belong to one process, so they are complete only while one
    process serves every request for a job.
    """

    def __init__(
        self,
        max_total_jobs: int,
        max_jobs_per_ip: int,
        job_ttl_seconds: float,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        """Initialize the store.

        Args:
            max_total_jobs: Cap on resource-occupying jobs across all clients.
            max_jobs_per_ip: Cap on resource-occupying jobs per client.
            job_ttl_seconds: How long a finished job is kept.
            clock: Source of the current UTC time; tests pass a fake one.
        """
        self._max_total_jobs = max_total_jobs
        self._max_jobs_per_ip = max_jobs_per_ip
        self._job_ttl_seconds = job_ttl_seconds
        self._clock = clock
        self._jobs: dict[str, JobStatus] = {}

    # No method below awaits, so on one event loop no other call can run
    # between a method's reads and its writes. That is what makes each one atomic.

    async def create(self, job_id: str, client_ip: str, download_stem: str) -> JobStatus:
        """Add a QUEUED job, unless that would exceed a job cap."""
        global_count = self._count_resource_occupying_jobs()
        if global_count >= self._max_total_jobs:
            raise JobCapError(f"Global job cap reached ({global_count}/{self._max_total_jobs})")
        if client_ip:
            ip_count = self._count_resource_occupying_jobs(client_ip)
            if ip_count >= self._max_jobs_per_ip:
                raise IPJobCapError(
                    f"Per-IP job cap reached for {client_ip} ({ip_count}/{self._max_jobs_per_ip})"
                )
        job = JobStatus(
            job_id=job_id,
            client_ip=client_ip,
            download_stem=download_stem,
            status=JobStatusEnum.QUEUED,
            progress=0,
            message="Audio uploaded, queued for processing",
            created_at=self._clock(),
        )
        self._jobs[job_id] = job
        return job.model_copy()

    async def get(self, job_id: str) -> JobStatus | None:
        """Return a copy of the job, or None if there is no such job."""
        job = self._jobs.get(job_id)
        return None if job is None else job.model_copy()

    async def start(self, job_id: str) -> bool:
        """Move a job from QUEUED to PROCESSING."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatusEnum.QUEUED:
            return False
        job.status = JobStatusEnum.PROCESSING
        job.progress = 10
        job.message = "Denoising audio..."
        return True

    async def complete(self, job_id: str, processing_time: float, download_url: str) -> bool:
        """Move a job from PROCESSING to COMPLETED."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatusEnum.PROCESSING:
            return False
        job.status = JobStatusEnum.COMPLETED
        job.progress = 100
        job.message = "Denoising complete"
        job.completed_at = self._clock()
        job.download_url = download_url
        job.processing_time = processing_time
        return True

    async def fail(self, job_id: str) -> bool:
        """Move a job from PROCESSING to FAILED."""
        job = self._jobs.get(job_id)
        if job is None or job.status != JobStatusEnum.PROCESSING:
            return False
        job.status = JobStatusEnum.FAILED
        job.progress = -1
        job.message = "Processing failed"
        job.completed_at = self._clock()
        return True

    async def delete(self, job_id: str) -> None:
        """Remove the job if it exists."""
        self._jobs.pop(job_id, None)

    async def counts(self) -> dict[str, int]:
        """Return the number of jobs in each status that has any."""
        counts: dict[str, int] = {}
        for job in self._jobs.values():
            counts[job.status] = counts.get(job.status, 0) + 1
        return counts

    async def expired_ids(self) -> list[str]:
        """Return the IDs of COMPLETED or FAILED jobs finished more than the TTL ago."""
        now = self._clock()
        # Only complete() and fail() set completed_at, so it also selects the finished jobs
        return [
            job_id
            for job_id, job in self._jobs.items()
            if job.completed_at is not None
            and (now - job.completed_at).total_seconds() > self._job_ttl_seconds
        ]

    def _count_resource_occupying_jobs(self, client_ip: str | None = None) -> int:
        """Count jobs in a resource-occupying status, for one client or for all."""
        return sum(
            1
            for job in self._jobs.values()
            if job.status in RESOURCE_STATUSES and (client_ip is None or job.client_ip == client_ip)
        )


def build_job_store(backend: JobStoreBackend) -> JobStore:
    """Build the job store for *backend* from the current settings.

    Args:
        backend: The configured job store backend.

    Returns:
        A job store using settings.max_total_jobs, settings.max_jobs_per_ip
        and settings.job_ttl_seconds.
    """
    # match with assert_never lets mypy report a backend that has no branch here
    match backend:
        case JobStoreBackend.MEMORY:
            return InMemoryJobStore(
                max_total_jobs=settings.max_total_jobs,
                max_jobs_per_ip=settings.max_jobs_per_ip,
                job_ttl_seconds=settings.job_ttl_seconds,
            )
        case _:
            assert_never(backend)
