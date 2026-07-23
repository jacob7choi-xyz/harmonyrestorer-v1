"""Tests for the atomic artifact byte budget."""

from pathlib import Path

import pytest
from app.services.artifacts import ArtifactBudget


@pytest.fixture()
def managed_dir(tmp_path) -> Path:
    d = tmp_path / "managed"
    d.mkdir()
    return d


class TestArtifactBudget:
    def test_reserves_within_limit(self, managed_dir) -> None:
        budget = ArtifactBudget(limit_bytes=100, directories=[managed_dir])
        assert budget.try_reserve(60) is True
        assert budget.try_reserve(40) is True
        assert budget.try_reserve(1) is False

    def test_concurrent_reservations_cannot_oversubscribe(self, managed_dir) -> None:
        """Two producers observing the same headroom cannot both take it."""
        budget = ArtifactBudget(limit_bytes=100, directories=[managed_dir])
        assert budget.try_reserve(80) is True
        # The second producer sees reserved bytes, not just persisted ones
        assert budget.try_reserve(80) is False

    def test_release_restores_capacity(self, managed_dir) -> None:
        budget = ArtifactBudget(limit_bytes=100, directories=[managed_dir])
        assert budget.try_reserve(100) is True
        assert budget.try_reserve(1) is False
        budget.release(100)
        assert budget.try_reserve(100) is True

    def test_persisted_files_count_against_the_budget(self, managed_dir) -> None:
        (managed_dir / "job_denoised.wav").write_bytes(b"x" * 70)
        budget = ArtifactBudget(limit_bytes=100, directories=[managed_dir])
        assert budget.try_reserve(40) is False
        assert budget.try_reserve(30) is True

    def test_reservation_and_persisted_double_count_is_conservative(self, managed_dir) -> None:
        """Materialized file plus unreleased reservation may only reject, never
        oversubscribe."""
        budget = ArtifactBudget(limit_bytes=100, directories=[managed_dir])
        assert budget.try_reserve(50) is True
        # Operation materializes its file before releasing the reservation
        (managed_dir / "out.wav").write_bytes(b"x" * 50)
        assert budget.try_reserve(10) is False
        budget.release(50)
        assert budget.try_reserve(50) is True

    def test_over_release_recovers_to_zero(self, managed_dir) -> None:
        budget = ArtifactBudget(limit_bytes=100, directories=[managed_dir])
        budget.release(9999)
        # Reserved floor is zero, not negative: full capacity, no more
        assert budget.try_reserve(100) is True
        assert budget.try_reserve(1) is False
