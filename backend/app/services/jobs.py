"""Job management service."""

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.schemas import JobStatus, JobStatusEnum
from app.services.denoiser import DenoiserService
from app.services.opgan_denoiser import OpGANDenoiserService

logger = logging.getLogger(__name__)


class JobCapError(Exception):
    """Raised when the global resource-occupying job cap is reached."""


class IPJobCapError(Exception):
    """Raised when the per-IP resource-occupying job cap is reached."""


_RESOURCE_STATUSES: frozenset[JobStatusEnum] = frozenset(
    {JobStatusEnum.QUEUED, JobStatusEnum.PROCESSING, JobStatusEnum.COMPLETED}
)


class JobManager:
    """Manages denoising jobs and their lifecycle."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._denoiser: DenoiserService | OpGANDenoiserService | None = None
        self._lock = threading.Lock()
        self._downloading: dict[str, datetime] = {}

    def _get_denoiser(self) -> DenoiserService | OpGANDenoiserService:
        """Lazy-load the denoiser singleton based on configured engine."""
        with self._lock:
            if self._denoiser is None:
                if settings.denoiser_engine == "opgan":
                    self._denoiser = OpGANDenoiserService(
                        output_dir=settings.processed_dir,
                        checkpoint_path=settings.opgan_checkpoint,
                    )
                    logger.info("OpGAN Denoiser initialized")
                else:
                    self._denoiser = DenoiserService(
                        output_dir=settings.processed_dir,
                        model_name=settings.uvr_model_name,
                    )
                    logger.info("UVR Denoiser initialized")
        return self._denoiser

    @property
    def job_counts(self) -> dict[str, int]:
        """Return counts by status for health reporting."""
        with self._lock:
            counts: dict[str, int] = {}
            for job in self._jobs.values():
                counts[job.status] = counts.get(job.status, 0) + 1
            return counts

    def _count_resource_occupying_jobs(self, client_ip: str | None = None) -> int:
        """Count resource-occupying jobs. Must be called with self._lock held.

        Args:
            client_ip: If provided, count only jobs from this client IP.
                       If None, count all resource-occupying jobs globally.

        Returns:
            Number of jobs in QUEUED, PROCESSING, or COMPLETED status.
        """
        return sum(
            1
            for job in self._jobs.values()
            if job.status in _RESOURCE_STATUSES
            and (client_ip is None or job.client_ip == client_ip)
        )

    def create_job(
        self, job_id: str, client_ip: str = "", download_stem: str = "audio"
    ) -> JobStatus:
        """Register a new processing job, enforcing global and per-IP caps.

        Args:
            job_id: UUID string for the new job.
            client_ip: Verified client IP address from the upload request.
            download_stem: Sanitized stem for the eventual download filename.

        Returns:
            The newly created JobStatus.

        Raises:
            JobCapError: If the global resource-occupying job cap is reached.
            IPJobCapError: If the per-IP resource-occupying job cap is reached.
        """
        with self._lock:
            global_count = self._count_resource_occupying_jobs()
            if global_count >= settings.max_total_jobs:
                raise JobCapError(
                    f"Global job cap reached ({global_count}/{settings.max_total_jobs})"
                )
            if client_ip:
                ip_count = self._count_resource_occupying_jobs(client_ip)
                if ip_count >= settings.max_jobs_per_ip:
                    raise IPJobCapError(
                        f"Per-IP job cap reached for {client_ip}"
                        f" ({ip_count}/{settings.max_jobs_per_ip})"
                    )
            job = JobStatus(
                job_id=job_id,
                client_ip=client_ip,
                download_stem=download_stem,
                status=JobStatusEnum.QUEUED,
                progress=0,
                message="Audio uploaded, queued for processing",
                created_at=datetime.now(UTC),
            )
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> JobStatus | None:
        """Look up a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def mark_downloading(self, job_id: str) -> bool:
        """Mark a job as actively being downloaded. Returns False if job gone."""
        with self._lock:
            if job_id not in self._jobs:
                return False
            self._downloading[job_id] = datetime.now(UTC)
            return True

    def unmark_downloading(self, job_id: str) -> None:
        """Remove download guard from a job."""
        with self._lock:
            self._downloading.pop(job_id, None)

    def process(self, job_id: str, input_path: Path) -> None:
        """Run denoising. Called as a background task."""
        start_time = datetime.now(UTC)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                logger.error("Job %s vanished before processing started", job_id)
                return

        try:
            with self._lock:
                if job_id not in self._jobs:
                    logger.error("Job %s removed during processing", job_id)
                    return
                job.status = JobStatusEnum.PROCESSING
                job.progress = 10
                job.message = "Denoising audio..."

            denoiser = self._get_denoiser()
            output_path = denoiser.denoise(input_path)

            final_path = settings.processed_dir / f"{job_id}_denoised.wav"
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            with self._lock:
                if job_id not in self._jobs:
                    logger.error("Job %s removed during processing", job_id)
                    Path(output_path).unlink(missing_ok=True)
                    return
                Path(output_path).replace(final_path)
                job.status = JobStatusEnum.COMPLETED
                job.progress = 100
                job.message = "Denoising complete"
                job.completed_at = datetime.now(UTC)
                job.download_url = f"/api/v1/download/{job_id}"
                job.processing_time = processing_time

            logger.info("Completed job %s in %.1fs", job_id, processing_time)

        except Exception as e:
            logger.error("Job %s failed: %s", job_id, e, exc_info=True)
            with self._lock:
                if job_id in self._jobs:
                    job.status = JobStatusEnum.FAILED
                    job.progress = -1
                    job.message = "Processing failed"
                    job.completed_at = datetime.now(UTC)
        finally:
            input_path.unlink(missing_ok=True)

    def cleanup_expired(self) -> int:
        """Remove finished jobs older than TTL and delete their output files.

        First evicts stale download markers left by disconnected clients, then
        removes jobs in COMPLETED or FAILED status that have exceeded the TTL.
        Skips jobs actively being downloaded to prevent races with concurrent
        downloads and background tasks.
        """
        now = datetime.now(UTC)
        with self._lock:
            # Evict stale download markers left by disconnected clients
            stale_markers = [
                job_id
                for job_id, started_at in self._downloading.items()
                if (now - started_at).total_seconds() > settings.download_ttl_seconds
            ]
            for job_id in stale_markers:
                del self._downloading[job_id]
                logger.warning("Evicted stale download marker for job %s", job_id)

            expired_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job_id not in self._downloading
                and job.status in (JobStatusEnum.COMPLETED, JobStatusEnum.FAILED)
                and job.completed_at
                and (now - job.completed_at).total_seconds() > settings.job_ttl_seconds
            ]
            for job_id in expired_ids:
                del self._jobs[job_id]

        # Delete files outside the lock to avoid blocking other operations
        for job_id in expired_ids:
            output_file = settings.processed_dir / f"{job_id}_denoised.wav"
            output_file.unlink(missing_ok=True)

        if expired_ids:
            logger.info("Cleaned up %d expired job(s)", len(expired_ids))

        return len(expired_ids)


job_manager = JobManager()
