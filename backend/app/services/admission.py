"""Fail-fast admission control for expensive inference work."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


class InferenceAdmission:
    """Bound the number of concurrently executing inference operations.

    Backed by an asyncio.Queue of capacity tokens, which provides documented
    non-blocking semantics (get_nowait and put_nowait). The bound is
    process-local: the deployment must run exactly one application worker per
    instance (see the Dockerfile CMD) for it to hold instance-wide. Combined
    with the Cloud Run instance cap, the invariant is:

        max instances x workers per instance x slots per worker
        = maximum simultaneous inferences

    A double release raises QueueFull, so an over-release bug fails loudly
    instead of silently inflating capacity.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize with the given number of inference slots.

        Args:
            capacity: Maximum concurrently executing inferences (minimum 1,
                enforced by config validation).
        """
        self._queue: asyncio.Queue[None] = asyncio.Queue(maxsize=capacity)
        for _ in range(capacity):
            self._queue.put_nowait(None)

    def try_acquire(self) -> bool:
        """Take an inference slot without waiting.

        Returns:
            True if a slot was acquired, False if all slots are busy.
        """
        try:
            self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return False
        return True

    def release(self) -> None:
        """Return an inference slot.

        Raises:
            asyncio.QueueFull: If released more times than acquired, which
                indicates a lifecycle bug that must not be masked.
        """
        self._queue.put_nowait(None)
