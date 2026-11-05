"""Runs the denoiser for jobs and protects their files while they are downloaded."""

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.services.engines import Denoiser, build_denoiser
from app.services.job_store import JobStore

logger = logging.getLogger(__name__)


@dataclass
class _DownloadGuard:
    """Reference-counted guard keeping a job's files alive during downloads.

    A single marker is not enough: with concurrent downloads of different
    formats, the first to finish would drop the guard while the second is
    still streaming, letting cleanup unlink files under it.
    """

    count: int
    started_at: datetime


class JobManager:
    """Runs the denoiser for jobs and guards their files during downloads.

    Job records are kept by the configured JobStore (app.state.job_store),
    not here.
    """

    def __init__(self) -> None:
        self._denoiser: Denoiser | None = None
        # Taken by worker threads building the denoiser
        self._denoiser_lock = threading.Lock()
        self._downloading: dict[str, _DownloadGuard] = {}
        # Starlette runs the synchronous unmark_downloading background task in a
        # worker thread, so the guards are changed from both threads and the loop
        self._guard_lock = threading.Lock()

    def _get_denoiser(self) -> Denoiser:
        """Lazy-load the denoiser singleton based on configured engine."""
        with self._denoiser_lock:
            if self._denoiser is None:
                self._denoiser = build_denoiser(settings.denoiser_engine)
        return self._denoiser

    def mark_downloading(self, job_id: str) -> None:
        """Take a download hold on a job.

        Holds are reference counted so concurrent downloads of different
        formats each keep the files alive independently. The caller must have
        read the job from the store, and found it COMPLETED, without awaiting
        anything since: cleanup_expired then cannot select the job between
        that read and this hold.
        """
        with self._guard_lock:
            guard = self._downloading.get(job_id)
            if guard is None:
                self._downloading[job_id] = _DownloadGuard(count=1, started_at=datetime.now(UTC))
            else:
                guard.count += 1
                guard.started_at = datetime.now(UTC)

    def unmark_downloading(self, job_id: str) -> None:
        """Release one download hold; the guard drops only at zero holds."""
        with self._guard_lock:
            guard = self._downloading.get(job_id)
            if guard is None:
                return
            guard.count -= 1
            if guard.count <= 0:
                del self._downloading[job_id]

    def run_denoiser(self, job_id: str, input_path: Path) -> float:
        """Denoise *input_path* into the job's output file. Runs in a worker thread.

        Reads and changes no job record; the caller records the outcome. The
        input file and any intermediate output are removed whether or not
        this succeeds.

        Args:
            job_id: The job whose output file is written.
            input_path: The decoded WAV to denoise.

        Returns:
            Seconds from the start of this call until the output was produced.

        Raises:
            Whatever the denoiser or the final rename raises.
        """
        start_time = datetime.now(UTC)
        intermediate_output: Path | None = None
        try:
            intermediate_output = Path(self._get_denoiser().denoise(input_path))
            processing_time = (datetime.now(UTC) - start_time).total_seconds()
            intermediate_output.replace(settings.processed_dir / f"{job_id}_denoised.wav")
            return processing_time
        finally:
            input_path.unlink(missing_ok=True)
            # Invariant: the intermediate output never survives this call.
            # After a successful rename this path no longer exists; after
            # any failure it must not linger (cleanup only globs _denoised.*)
            if intermediate_output is not None:
                intermediate_output.unlink(missing_ok=True)

    async def cleanup_expired(self, store: JobStore) -> int:
        """Remove finished jobs older than the TTL and delete their output files.

        First evicts stale download markers left by disconnected clients, then
        removes jobs that store.expired_ids() lists, except jobs being
        downloaded. A job whose files cannot be deleted keeps its record and
        is retried on the next pass.

        Args:
            store: The job store holding the records.

        Returns:
            The number of jobs removed.
        """
        self._evict_stale_download_markers()
        # The download guard holds only while no await between reading the
        # guards and deleting the files and records lets a download start: true for the
        # in-memory store, whose methods never await. A networked JobStore must
        # not go live before FileStore replaces the guard (ADR 010).
        candidates = await store.expired_ids()
        with self._guard_lock:
            expired_ids = [job_id for job_id in candidates if job_id not in self._downloading]
        removed = 0
        for job_id in expired_ids:
            # Files first: if one cannot be deleted, the record stays, so the
            # next pass finds the job again and retries
            try:
                self._delete_result_files(job_id)
            except OSError as e:
                logger.warning("Could not delete files of expired job %s: %s", job_id, e)
                continue
            await store.delete(job_id)
            removed += 1

        if removed:
            logger.info("Cleaned up %d expired job(s)", removed)

        return removed

    @staticmethod
    def _delete_result_files(job_id: str) -> None:
        """Delete a job's result WAV and any cached download-format conversions."""
        for output_file in settings.processed_dir.glob(f"{job_id}_denoised.*"):
            output_file.unlink(missing_ok=True)

    def _evict_stale_download_markers(self) -> None:
        """Drop download holds older than DOWNLOAD_TTL_SECONDS, left by disconnected clients."""
        now = datetime.now(UTC)
        with self._guard_lock:
            stale_markers = [
                job_id
                for job_id, guard in self._downloading.items()
                if (now - guard.started_at).total_seconds() > settings.download_ttl_seconds
            ]
            for job_id in stale_markers:
                del self._downloading[job_id]
                logger.warning("Evicted stale download marker for job %s", job_id)


job_manager = JobManager()
