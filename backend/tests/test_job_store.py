"""Tests for the in-memory JobStore adapter, its builder, and the composition root."""

import pytest
from app.composition import install_adapters
from app.config import JobStoreBackend
from app.services.job_store import InMemoryJobStore, build_job_store
from fastapi import FastAPI

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


def test_install_adapters_attaches_a_job_store() -> None:
    """The composition root puts the configured job store on app.state."""
    app = FastAPI()

    install_adapters(app)

    assert isinstance(app.state.job_store, InMemoryJobStore)
