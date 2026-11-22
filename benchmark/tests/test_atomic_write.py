"""Tests for writing a file so that it is either complete or absent."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmark import atomic_write
from benchmark.atomic_write import write_atomically


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
