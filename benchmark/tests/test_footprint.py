"""Tests for reading a process's peak footprint from the process or from its parent.

The exit codes hold on every platform; the footprint is read on macOS only, so the tests
that need a number are skipped elsewhere.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from benchmark.footprint import _wait_without_reaping, peak_footprint_bytes, wait_with_footprint

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="footprint is macOS only")

_ALLOCATE_200_MB = "import numpy as np\na = np.ones(200_000_000 // 8)\n"


def spawn(code: str) -> subprocess.Popen[bytes]:
    """Start a child interpreter running *code*, without waiting for it."""
    return subprocess.Popen(
        [sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )


def wait(child: subprocess.Popen[bytes]) -> tuple[int, int | None]:
    """`wait_with_footprint` on *child*, recording the status so Popen does not wait again."""
    returncode, footprint = wait_with_footprint(child.pid)
    child.returncode = returncode
    return returncode, footprint


class TestWaitWithFootprint:
    """The parent learns how the child ended, and on macOS how much memory it used."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("raise SystemExit(3)", 3),
            ("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)", -9),
        ],
        ids=["exit_status", "sigkill"],
    )
    def test_the_exit_code_is_returned_in_subprocess_form(self, code: str, expected: int) -> None:
        returncode, _ = wait(spawn(code))
        assert returncode == expected

    def test_the_child_is_reaped(self) -> None:
        child = spawn("pass")
        wait(child)
        with pytest.raises(ChildProcessError):
            os.waitpid(child.pid, os.WNOHANG)

    @darwin_only
    def test_a_killed_child_still_has_its_footprint(self) -> None:
        """The case a worker cannot report itself: it is read before the child is reaped."""
        child = spawn(_ALLOCATE_200_MB + "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)")
        returncode, footprint = wait(child)
        assert returncode == -9
        assert footprint is not None
        assert footprint >= 200_000_000

    @pytest.mark.skipif(sys.platform == "darwin", reason="footprint is read on macOS")
    def test_no_footprint_is_claimed_off_macos(self) -> None:
        assert wait(spawn("pass")) == (0, None)


@darwin_only
class TestPeakFootprintBytes:
    """The read itself, and what it gives when the process cannot be read."""

    def test_this_process_has_a_footprint(self) -> None:
        footprint = peak_footprint_bytes()
        assert footprint is not None
        assert footprint > 0

    def test_a_reaped_child_gives_none_rather_than_a_number(self) -> None:
        child = spawn("pass")
        child.wait()
        assert peak_footprint_bytes(child.pid) is None

    def test_waiting_on_a_process_that_is_not_a_child_raises(self) -> None:
        with pytest.raises(ChildProcessError):
            _wait_without_reaping(os.getpid())
