"""The contract for storing job result files, and how the configured store is built (ADR 010)."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Protocol, assert_never

from app.config import FileStoreBackend, settings


class FileStore(Protocol):
    """Keeps each job's result WAV and its cached download conversions (ADR 010).

    The methods are synchronous and may block. The work that produces and
    reads these files (the denoiser, ffmpeg) takes local paths and runs in
    worker threads, which can call the store directly. The local adapter's
    methods are local filesystem calls (a rename, a directory glob, unlinks);
    for an adapter that does network I/O, a caller on the event loop would
    need to move the call off the loop.
    Job IDs must be canonical UUID strings, since they become part of storage
    names; any other value raises ValueError.
    """

    def put_result(self, job_id: str, source: Path) -> None:
        """Move the finished WAV at *source* in as the job's result.

        *source* no longer exists afterwards.
        """
        ...

    def result_path(self, job_id: str) -> Path:
        """Return the local path of the job's result WAV, which may not exist.

        Meaningful only for adapters that stream results from local disk. An
        adapter that issues download grants is expected to replace this
        method (ADR 010).
        """
        ...

    def delete_results(self, job_id: str) -> None:
        """Remove the job's result and any cached conversions. Safe to repeat.

        Raises:
            OSError: If a file exists but cannot be removed.
        """
        ...


def _checked_job_id(job_id: str) -> str:
    """Return *job_id* if it is a canonical UUID string, else raise ValueError."""
    # uuid.UUID also accepts braces, a urn:uuid: prefix and upper case; only the
    # form the server generates is allowed into a storage name
    if str(uuid.UUID(job_id)) != job_id:
        raise ValueError(f"Job ID is not a canonical UUID: {job_id!r}")
    return job_id


class LocalFileStore:
    """Result files kept in one directory on this instance's disk.

    The result is named {job_id}_denoised.wav, and cached conversions sit
    beside it as {job_id}_denoised.{format}. The artifact byte budget scans
    this directory, so the root must be one the budget covers.
    """

    def __init__(self, root: Path) -> None:
        """Initialize the store.

        Args:
            root: Directory that holds the result files.
        """
        self._root = root

    def put_result(self, job_id: str, source: Path) -> None:
        """Move *source* in as the job's result.

        A rename, so *source* must be on the same filesystem as the root; the
        denoisers write their output into the root, so it is.
        """
        source.replace(self.result_path(job_id))

    def result_path(self, job_id: str) -> Path:
        """Return the path of the job's result WAV, which may not exist."""
        return self._root / f"{_checked_job_id(job_id)}_denoised.wav"

    def delete_results(self, job_id: str) -> None:
        """Remove the job's result and any cached conversions. Safe to repeat."""
        for path in self._root.glob(f"{_checked_job_id(job_id)}_denoised.*"):
            path.unlink(missing_ok=True)


def build_file_store(backend: FileStoreBackend) -> FileStore:
    """Build the file store for *backend* from the current settings.

    Args:
        backend: The configured file store backend.

    Returns:
        A file store rooted at settings.processed_dir.
    """
    # match with assert_never lets mypy report a backend that has no branch here
    match backend:
        case FileStoreBackend.LOCAL:
            return LocalFileStore(settings.processed_dir)
        case _:
            assert_never(backend)
