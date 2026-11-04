"""Tests for the in-memory JobStore adapter and its builder."""

import pytest
from app.config import JobStoreBackend
from app.services.job_store import InMemoryJobStore, build_job_store

from tests.job_store_contract import JobStoreContract, MakeStore


class TestInMemoryJobStore(JobStoreContract):
    """The shared contract, run against the in-memory adapter."""

    @pytest.fixture()
    def make_store(self) -> MakeStore:
        """Build in-memory stores for the contract tests."""
        return lambda max_total_jobs, max_jobs_per_ip, job_ttl_seconds, clock: InMemoryJobStore(
            max_total_jobs, max_jobs_per_ip, job_ttl_seconds, clock=clock
        )


def test_build_job_store_returns_the_in_memory_adapter_for_memory() -> None:
    """The memory backend builds an InMemoryJobStore."""
    assert isinstance(build_job_store(JobStoreBackend.MEMORY), InMemoryJobStore)
