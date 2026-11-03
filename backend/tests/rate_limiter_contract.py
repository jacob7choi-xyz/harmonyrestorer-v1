"""Behaviour every RateLimiter adapter must show (ADR 010).

An adapter's test module subclasses RateLimiterContract and provides the
make_limiter fixture. The class name does not start with Test, so pytest
collects these tests only through a subclass. The exact shape of the window
is not part of the contract: a shared or edge limiter may count differently,
so those details are tested per adapter.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import pytest
from app.services.rate_limit import RateLimiter


class FakeClock:
    """A clock that moves only when the test advances it."""

    def __init__(self, start: float = 1_000.0) -> None:
        """Start the clock at *start* seconds."""
        self.now = start

    def __call__(self) -> float:
        """Return the current fake time in seconds."""
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by *seconds*."""
        self.now += seconds


MakeLimiter = Callable[[int, float, FakeClock], RateLimiter]


class RateLimiterContract:
    """Tests run against every RateLimiter adapter."""

    @pytest.fixture()
    def make_limiter(self) -> MakeLimiter:
        """Return a factory taking (max_requests, window_seconds, clock)."""
        raise NotImplementedError("Adapter test modules provide make_limiter")

    async def test_allows_up_to_the_limit_then_rejects(self, make_limiter: MakeLimiter) -> None:
        """Within one window, max_requests are allowed and the next is not."""
        limiter = make_limiter(3, 60, FakeClock())

        results = [await limiter.allow("client-a") for _ in range(4)]

        assert results == [True, True, True, False]

    async def test_rejected_request_is_not_counted(self, make_limiter: MakeLimiter) -> None:
        """A rejected request does not extend the client's wait."""
        clock = FakeClock()
        limiter = make_limiter(1, 60, clock)
        assert await limiter.allow("client-a")

        clock.advance(30)
        assert not await limiter.allow("client-a")

        # 61 s after the allowed request, but only 31 s after the rejected one
        clock.advance(31)
        assert await limiter.allow("client-a")

    async def test_requests_older_than_the_window_stop_counting(
        self, make_limiter: MakeLimiter
    ) -> None:
        """Once more than one window has passed, the client is allowed again."""
        clock = FakeClock()
        limiter = make_limiter(2, 60, clock)
        assert await limiter.allow("client-a")
        assert await limiter.allow("client-a")
        assert not await limiter.allow("client-a")

        clock.advance(61)

        assert await limiter.allow("client-a")

    async def test_clients_are_counted_separately(self, make_limiter: MakeLimiter) -> None:
        """One client at its limit does not limit another."""
        limiter = make_limiter(1, 60, FakeClock())
        assert await limiter.allow("client-a")
        assert not await limiter.allow("client-a")

        assert await limiter.allow("client-b")

    async def test_concurrent_requests_never_exceed_the_limit(
        self, make_limiter: MakeLimiter
    ) -> None:
        """Requests made at the same time are allowed at most max_requests times.

        The in-memory adapter passes this without any locking, because it does
        not await between reading and writing. The test is here for adapters
        that read and write a shared store.
        """
        limiter = make_limiter(5, 60, FakeClock())

        results = await asyncio.gather(*(limiter.allow("client-a") for _ in range(20)))

        assert results.count(True) == 5
