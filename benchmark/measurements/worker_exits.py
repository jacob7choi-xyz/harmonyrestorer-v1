"""How a worker's failures reach its parent, the observations behind v8's exit mapping.

Each case runs as a fresh child interpreter, and its return code and last lines of output
are printed. On the reference platform, before v8 was frozen: SIGKILL returned -9, a
segmentation fault -11, and an uncaught MemoryError 1, and setrlimit refused RLIMIT_AS at
1, 6, 8, and 64 GiB. Run from the repository root:
`uv run --frozen python -m benchmark.measurements.worker_exits`.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

CASES = {
    "sigkill": "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
    "memoryerror_caught": (
        "import numpy as np\n"
        "try:\n    np.empty(10**14)\n"
        "except MemoryError:\n    print('caught MemoryError'); raise SystemExit(3)"
    ),
    "memoryerror_uncaught": "import numpy as np; np.empty(10**14)",
    "segfault": "import ctypes; ctypes.string_at(0)",
    "rlimit_as": (
        "import resource\n"
        "for gib in (1, 6, 8, 64):\n"
        "    try:\n"
        "        resource.setrlimit(resource.RLIMIT_AS, (gib << 30, resource.RLIM_INFINITY))\n"
        "        print(f'{gib} GiB accepted'); break\n"
        "    except (ValueError, OSError) as e:\n"
        "        print(f'{gib} GiB refused: {e}')"
    ),
}


@dataclass(frozen=True)
class Exit:
    """A child's return code and everything it wrote to each stream."""

    returncode: int
    stdout: str
    stderr: str


def run_case(code: str) -> Exit:
    """Run *code* in a fresh interpreter and return how it ended, with all of its output.

    Every line is kept: reading only the last line once made a refused setrlimit look like
    an accepted one.
    """
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    return Exit(out.returncode, out.stdout, out.stderr)


def main() -> None:
    """Print each case's return code and every line of its output."""
    for name, code in CASES.items():
        ended = run_case(code)
        print(f"{name}: returncode {ended.returncode}")
        for stream, text in (("stdout", ended.stdout), ("stderr", ended.stderr)):
            for line in text.splitlines():
                print(f"  {stream}: {line}")


if __name__ == "__main__":
    main()
