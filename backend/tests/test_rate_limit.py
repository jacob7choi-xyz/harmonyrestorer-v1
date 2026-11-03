"""Tests for the in-memory RateLimiter adapter, its builder, and the composition root."""

import pytest
from app.composition import install_adapters
from app.config import RateLimiterBackend
from app.services.rate_limit import InMemoryRateLimiter, build_rate_limiter
from fastapi import FastAPI

from tests.rate_limiter_contract import FakeClock, MakeLimiter, RateLimiterContract


class TestInMemoryRateLimiter(RateLimiterContract):
    """The shared contract, plus the window shape this adapter keeps from before ADR 010."""

    @pytest.fixture()
    def make_limiter(self) -> MakeLimiter:
        """Build in-memory limiters for the contract tests."""
        return lambda max_requests, window_seconds, clock: InMemoryRateLimiter(
            max_requests, window_seconds, clock=clock
        )

    async def test_window_slides_with_each_request(self) -> None:
        """Each request stops counting one window after it was made, not at a fixed boundary."""
        clock = FakeClock()
        limiter = InMemoryRateLimiter(2, 60, clock=clock)
        assert await limiter.allow("client-a")
        clock.advance(30)
        assert await limiter.allow("client-a")

        # The first request has expired, the second still counts
        clock.advance(31)
        assert await limiter.allow("client-a")
        assert not await limiter.allow("client-a")

    async def test_request_exactly_one_window_old_no_longer_counts(self) -> None:
        """A request made exactly window_seconds ago is outside the window."""
        clock = FakeClock()
        limiter = InMemoryRateLimiter(1, 60, clock=clock)
        assert await limiter.allow("client-a")

        clock.advance(60)

        assert await limiter.allow("client-a")

    async def test_clients_idle_for_a_window_are_dropped(self) -> None:
        """A client that stops sending requests is forgotten once a window has passed.

        Without this, every address that ever uploaded would stay in memory.
        """
        clock = FakeClock()
        limiter = InMemoryRateLimiter(10, 60, clock=clock)
        for i in range(100):
            assert await limiter.allow(f"client-{i}")

        clock.advance(61)
        assert await limiter.allow("client-new")

        assert list(limiter._requests) == ["client-new"]


def test_build_rate_limiter_returns_the_in_memory_adapter_for_memory() -> None:
    """The memory backend builds an InMemoryRateLimiter."""
    assert isinstance(build_rate_limiter(RateLimiterBackend.MEMORY), InMemoryRateLimiter)


def test_install_adapters_attaches_a_rate_limiter() -> None:
    """The composition root puts the configured rate limiter on app.state."""
    app = FastAPI()

    install_adapters(app)

    assert isinstance(app.state.rate_limiter, InMemoryRateLimiter)
