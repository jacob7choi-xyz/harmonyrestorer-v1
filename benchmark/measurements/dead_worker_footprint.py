"""Whether a parent can read a worker's peak footprint after the worker dies, however it dies.

Each case is a fresh child interpreter that allocates a known amount and then exits, is
killed by SIGKILL, segfaults, or raises. The parent reads the child's footprint with
`benchmark.footprint.wait_with_footprint` before reaping it. The first case also reads its
own footprint just before exiting, for comparison. On the reference platform, before the
canonicalisation worker was written, every case returned a footprint close to its
allocation on two runs. Run from the repository root:
`uv run --frozen python -m benchmark.measurements.dead_worker_footprint`. The cases need
about 1 GB of memory.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

from benchmark.footprint import wait_with_footprint

_ALLOCATE = "import numpy as np\na = np.ones({} // 8)\n"
CASES = {
    "exit_0_500_mb": _ALLOCATE.format(500_000_000)
    + "from benchmark.footprint import peak_footprint_bytes\n"
    + "print('self-reported', peak_footprint_bytes())",
    "sigkill_500_mb": _ALLOCATE.format(500_000_000)
    + "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)",
    "segfault_800_mb": _ALLOCATE.format(800_000_000) + "import ctypes\nctypes.string_at(0)",
    "uncaught_300_mb": _ALLOCATE.format(300_000_000) + "raise RuntimeError('uncaught')",
}


@dataclass(frozen=True)
class Ended:
    """How a child ended, the footprint its parent read, and what it printed."""

    returncode: int
    footprint: int | None
    stdout: str


def run_case(code: str) -> Ended:
    """Run *code* in a fresh interpreter, read its footprint after it ends, then reap it."""
    child = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert child.stdout is not None
    # The pipe closes when the child ends, so reading it all cannot block the wait below.
    stdout = child.stdout.read()
    child.stdout.close()
    returncode, footprint = wait_with_footprint(child.pid)
    # Reaped above; recording it stops Popen from waiting for the pid again.
    child.returncode = returncode
    return Ended(returncode, footprint, stdout.strip())


def main() -> None:
    """Print each case's return code, the footprint the parent read, and the child's output."""
    for name, code in CASES.items():
        ended = run_case(code)
        print(f"{name}: returncode {ended.returncode}, footprint {ended.footprint}")
        if ended.stdout:
            print(f"  stdout: {ended.stdout}")


if __name__ == "__main__":
    main()
