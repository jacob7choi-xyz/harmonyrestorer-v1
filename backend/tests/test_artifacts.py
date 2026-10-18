"""Tests for the atomic artifact byte budget."""

import os
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

    def test_files_in_subdirectories_count_against_the_budget(self, managed_dir) -> None:
        """A file below a managed directory counts, as a top-level file does."""
        nested = managed_dir / ".staging"
        nested.mkdir()
        (nested / "stem.wav").write_bytes(b"x" * 70)
        budget = ArtifactBudget(limit_bytes=100, directories=[managed_dir])
        assert budget.try_reserve(40) is False
        assert budget.try_reserve(30) is True

    def test_symlinks_are_not_followed(self, managed_dir, tmp_path) -> None:
        """A symlink inside a managed directory does not count its target's
        bytes, and a symlinked directory is not descended into."""
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "big.bin").write_bytes(b"x" * 1000)
        (managed_dir / "file_link").symlink_to(outside / "big.bin")
        (managed_dir / "dir_link").symlink_to(outside, target_is_directory=True)
        budget = ArtifactBudget(limit_bytes=100, directories=[managed_dir])
        assert budget.try_reserve(100) is True

    def test_missing_managed_directory_raises(self, tmp_path) -> None:
        """A managed directory that no longer exists fails the reservation instead
        of counting as empty, which would undercount the budget."""
        budget = ArtifactBudget(limit_bytes=100, directories=[tmp_path / "gone"])
        with pytest.raises(FileNotFoundError):
            budget.try_reserve(1)

    def test_file_removed_during_scan_is_skipped(self, managed_dir, monkeypatch) -> None:
        """A file listed by the scan but removed before its size is read, as
        cleanup and transcode do outside the budget lock, is skipped instead of
        failing the reservation."""
        from app.services import artifacts

        (managed_dir / "kept.wav").write_bytes(b"x" * 70)
        vanished = managed_dir / "vanished.wav"
        vanished.write_bytes(b"x" * 10)
        real_lstat = artifacts.os.lstat

        def lstat_after_removal(path):
            if os.fspath(path) == os.fspath(vanished):
                raise FileNotFoundError(path)
            return real_lstat(path)

        monkeypatch.setattr(artifacts.os, "lstat", lstat_after_removal)
        budget = ArtifactBudget(limit_bytes=100, directories=[managed_dir])
        assert budget.try_reserve(30) is True
        assert budget.try_reserve(1) is False
