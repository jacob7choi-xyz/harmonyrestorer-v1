"""The contract for counting uploads per client, and how the configured counter is built."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol, assert_never

from app.config import RateLimiterBackend, settings


class RateLimiter(Protocol):
    """Decides whether one more upload from a client is allowed (ADR 010)."""

    async def allow(self, client_id: str) -> bool:
        """Count one request from *client_id* if it is within the limit.

        Args:
            client_id: The client's identity, from client_identity().

        Returns:
            True if the request is allowed and has been counted. False if the
            client is at its limit, in which case the request is not counted.
        """
        ...


class InMemoryRateLimiter:
    """Sliding-window rate limit counted inside this process.

    Keeps, per client, the times of its allowed requests. A request is allowed
    while fewer than max_requests of them are newer than one window ago. The
    count belongs to one process, so it is the whole limit only while one
    process serves uploads.
    """

    def __init__(
        self,
        max_requests: int,
        window_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize the limiter.

        Args:
            max_requests: Requests allowed per client within one window.
            window_seconds: Length of the sliding window.
            clock: Source of the current time in seconds; tests pass a fake one.
        """
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._clock = clock
        self._requests: dict[str, list[float]] = {}

    async def allow(self, client_id: str) -> bool:
        """Count one request from *client_id* if it is within the limit.

        Args:
            client_id: The client's identity, from client_identity().

        Returns:
            True if the request is allowed and has been counted, otherwise False.
        """
        # Nothing below awaits, so on one event loop no other request can run
        # between reading this client's count and recording the new request
        now = self._clock()
        cutoff = now - self._window_seconds
        active = [t for t in self._requests.get(client_id, []) if t > cutoff]
        if len(active) >= self._max_requests:
            self._requests[client_id] = active
            return False
        active.append(now)
        self._requests[client_id] = active
        return True


def build_rate_limiter(backend: RateLimiterBackend) -> RateLimiter:
    """Build the rate limiter for *backend* from the current settings.

    Args:
        backend: The configured rate limiter backend.

    Returns:
        A rate limiter using settings.rate_limit_max_requests per
        settings.rate_limit_window_seconds.
    """
    # match with assert_never lets mypy report a backend that has no branch here
    match backend:
        case RateLimiterBackend.MEMORY:
            return InMemoryRateLimiter(
                max_requests=settings.rate_limit_max_requests,
                window_seconds=settings.rate_limit_window_seconds,
            )
        case _:
            assert_never(backend)
