"""Tests for writing a file so that it is either complete or absent."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmark import atomic_write
from benchmark.atomic_write import write_atomically, write_directory_atomically


class TestWriteAtomically:
    """The target holds the new bytes after success and the old bytes after a failure."""

    def test_a_new_file_holds_the_data_and_no_temporary_file_remains(self, tmp_path: Path) -> None:
        target = tmp_path / "record.json"
        write_atomically(target, b'{"a": 1}')
        assert target.read_bytes() == b'{"a": 1}'
        assert sorted(path.name for path in tmp_path.iterdir()) == ["record.json"]

    def test_an_existing_file_is_replaced(self, tmp_path: Path) -> None:
        target = tmp_path / "record.json"
        target.write_bytes(b"old")
        write_atomically(target, memoryview(b"new"))
        assert target.read_bytes() == b"new"

    def test_a_failed_fsync_leaves_the_old_file_and_no_temporary_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure before the replace must not leave a partial target or a stray file."""
        target = tmp_path / "record.json"
        target.write_bytes(b"old")

        def failing_fsync(fd: int) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(atomic_write.os, "fsync", failing_fsync)
        with pytest.raises(OSError, match="No space left"):
            write_atomically(target, b"new")
        assert target.read_bytes() == b"old"
        assert sorted(path.name for path in tmp_path.iterdir()) == ["record.json"]
        assert os.path.exists(target)


def test_a_directory_is_published_with_all_its_files(tmp_path: Path) -> None:
    """Every file is written, and no staging directory remains beside the published one."""
    target = tmp_path / "out"
    write_directory_atomically(target, {"a.txt": b"one", "b.txt": b"two"})
    assert (target / "a.txt").read_bytes() == b"one"
    assert (target / "b.txt").read_bytes() == b"two"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_a_failed_file_leaves_no_directory_and_no_staging(tmp_path: Path) -> None:
    """A file that cannot be written leaves neither the target nor the staging directory."""
    target = tmp_path / "out"
    with pytest.raises(OSError):
        write_directory_atomically(target, {"a.txt": b"one", "missing/b.txt": b"two"})
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_an_existing_directory_is_not_replaced(tmp_path: Path) -> None:
    """A move onto a non-empty directory fails, and the existing contents are left as they were."""
    target = tmp_path / "out"
    target.mkdir()
    (target / "keep.txt").write_bytes(b"kept")
    with pytest.raises(OSError):
        write_directory_atomically(target, {"a.txt": b"one"})
    assert (target / "keep.txt").read_bytes() == b"kept"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_the_target_name_is_absent_until_every_file_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No partial directory is visible under the target name while the files are being written."""
    target = tmp_path / "out"
    seen: list[bool] = []
    real_write = atomic_write.write_atomically

    def observe(path: Path, data: bytes) -> None:
        seen.append(target.exists())
        real_write(path, data)

    monkeypatch.setattr(atomic_write, "write_atomically", observe)
    write_directory_atomically(target, {"a.txt": b"one", "b.txt": b"two"})
    assert seen == [False, False]
