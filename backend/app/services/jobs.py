"""Job management service."""

import logging
import threading
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.schemas import JobStatus
from app.services.denoiser import DenoiserService

logger = logging.getLogger(__name__)


class JobManager:
    """Manages denoising jobs and their lifecycle."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._denoiser: DenoiserService | None = None
        self._lock = threading.Lock()

    def _get_denoiser(self) -> DenoiserService:
        """Lazy-load the denoiser singleton."""
        if self._denoiser is None:
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

    def create_job(self, job_id: str) -> JobStatus:
        """Register a new processing job."""
        job = JobStatus(
            job_id=job_id,
            status="queued",
            progress=0,
            message="Audio uploaded, queued for processing",
            created_at=datetime.now(),
        )
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> JobStatus | None:
        """Look up a job by ID."""
        with self._lock:
            return self._jobs.get(job_id)

    def process(self, job_id: str, input_path: Path) -> None:
        """Run denoising. Called as a background task."""
        start_time = datetime.now()
        with self._lock:
            job = self._jobs[job_id]

        try:
            with self._lock:
                job.status = "processing"
                job.progress = 10
                job.message = "Denoising audio..."

            denoiser = self._get_denoiser()
            output_path = denoiser.denoise(input_path)

            final_path = settings.processed_dir / f"{job_id}_denoised.wav"
            output_path.rename(final_path)

            processing_time = (datetime.now() - start_time).total_seconds()
            with self._lock:
                job.status = "completed"
                job.progress = 100
                job.message = "Denoising complete"
                job.completed_at = datetime.now()
                job.download_url = f"/api/v1/download/{job_id}"
                job.processing_time = processing_time

            logger.info("Completed job %s in %.1fs", job_id, processing_time)

        except Exception as e:
            logger.error("Job %s failed: %s", job_id, e, exc_info=True)
            with self._lock:
                job.status = "failed"
                job.progress = -1
                job.message = "Processing failed"
                job.completed_at = datetime.now()
        finally:
            input_path.unlink(missing_ok=True)

    def cleanup_expired(self) -> int:
        """Remove finished jobs older than TTL and delete their output files."""
        now = datetime.now()
        with self._lock:
            expired_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status in ("completed", "failed")
                and job.completed_at
                and (now - job.completed_at).total_seconds() > settings.job_ttl_seconds
            ]
            # Remove from dict first so no new downloads can start
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
