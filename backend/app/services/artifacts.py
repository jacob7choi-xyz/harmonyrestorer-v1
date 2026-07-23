"""Atomic byte budget for artifacts on the memory-backed filesystem.

Cloud Run's writable filesystem consumes instance memory, so artifact
occupancy is a memory-safety control, not a disk-space concern. Free-space
checks cannot express that; this module enforces an explicit budget:

    persisted bytes + in-flight reserved bytes + requested bytes
    <= MAX_ARTIFACT_BYTES

Reservations close the race where the two allowed producers (one inference,
one transcode) each scan the directory, both observe the same headroom, and
jointly overspend it. The scan happens inside the lock, so reconciliation
and reservation are atomic together; releases run in the owner that
performs the work, after materialization, which briefly double-counts file
plus reservation. That bias is deliberate: over-counting can only reject,
never oversubscribe.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class ArtifactBudget:
    """Lock-protected byte budget over a set of managed directories."""

    def __init__(self, limit_bytes: int, directories: list[Path]) -> None:
        """Initialize the budget.

        Args:
            limit_bytes: Maximum combined persisted plus reserved bytes.
            directories: Managed artifact directories to scan; only the
                application's own namespace counts against the budget.
        """
        self._limit = limit_bytes
        self._directories = directories
        self._lock = threading.Lock()
        self._reserved = 0

    def _scan_persisted(self) -> int:
        """Sum managed artifact sizes. Callers must hold the lock."""
        total = 0
        for directory in self._directories:
            for entry in directory.iterdir():
                if entry.is_file():
                    total += entry.stat().st_size
        return total

    def try_reserve(self, requested: int) -> bool:
        """Atomically reserve bytes for an operation about to start.

        Args:
            requested: Worst-case additional bytes the operation may occupy
                while running (derived from its file lifecycle, not from
                final sizes alone).

        Returns:
            True if the reservation fits; False if it must be rejected.
        """
        with self._lock:
            persisted = self._scan_persisted()
            projected = persisted + self._reserved + requested
            if projected > self._limit:
                logger.warning(
                    "Artifact budget rejected reservation of %d bytes"
                    " (persisted=%d reserved=%d limit=%d)",
                    requested,
                    persisted,
                    self._reserved,
                    self._limit,
                )
                return False
            self._reserved += requested
            return True

    def release(self, reserved: int) -> None:
        """Return a reservation once its operation has finished.

        Args:
            reserved: The exact amount previously reserved.
        """
        with self._lock:
            self._reserved -= reserved
            if self._reserved < 0:
                # Over-release indicates a lifecycle bug; recover loudly
                logger.error("Artifact reservation underflow (%d); resetting", self._reserved)
                self._reserved = 0


artifact_budget = ArtifactBudget(
    limit_bytes=settings.max_artifact_bytes,
    directories=[settings.upload_dir, settings.processed_dir],
)
