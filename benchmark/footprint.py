"""A process's lifetime peak memory footprint on macOS, read by the process or by its parent.

Protocol v8 records each worker's peak footprint per source, as `lifetime_max_phys_footprint`
from `proc_pid_rusage`. That counts memory macOS has compressed, where the peak resident set
does not; on the capacity fixture the resident set varied between 3.63 and 4.94 GB over five
runs while the footprint varied by about 1 MB.

**Read from the parent, so a killed worker still has one.** A worker that is killed cannot
report anything. `wait_with_footprint` waits for the child with `waitid(WEXITED | WNOWAIT)`,
which leaves it unreaped so its pid still names it, reads the footprint, and only then reaps
it. On the reference platform that read succeeded for a child that exited 0, one killed
by SIGKILL (-9), one that segfaulted (-11), and one that raised an uncaught exception (1),
on four runs each: two from a script written before this module and two through
`benchmark/measurements/dead_worker_footprint.py`. For the child that also read its own
footprint just before exiting, the parent's reading was 16,384 or 98,304 bytes higher, at
about 500 MB. On v8's capacity child (`benchmark/measurements/capacity.py`), over three runs,
the parent's reading was 1,589,248 to 1,605,632 bytes higher than the child's own. v8's
recorded footprints were read by the child itself, so a footprint read by the parent is not
directly comparable with them.

The footprint is read on macOS only; elsewhere these functions return None for it. Where
Python provides `os.waitid`, as CPython does on Linux, it is used to wait without reaping.
The CPython 3.11.17 interpreter this was checked with on macOS has no `os.waitid`, so
where it is missing libc's `waitid` is called through ctypes.

`wait_with_footprint` calls `before_reap` while the exited child is still unreaped, so its
pid cannot yet be given to another process. The runner stops its timeout timer there, so a
timer that fires late cannot signal a process that reused the pid.
"""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from collections.abc import Callable

__all__ = ["peak_footprint_bytes", "wait_with_footprint"]

_RUSAGE_INFO_V4 = 4
_LIBPROC = "/usr/lib/libproc.dylib"
# siginfo_t is filled by waitid and not read here. sizeof(siginfo_t) printed 104 on the
# reference platform; the buffer is larger so a different layout cannot write past it.
_SIGINFO_BUFFER_BYTES = 256
_RUSAGE_V4_COUNTERS = (
    "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
    "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
    "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
    "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
    "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
    "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
    "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
    "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
    "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
    "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy",
    "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time",
)  # fmt: skip


class _RusageInfoV4(ctypes.Structure):
    """`struct rusage_info_v4` from macOS `<sys/resource.h>`, in field order."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64) for name in _RUSAGE_V4_COUNTERS
    ]


def peak_footprint_bytes(pid: int | None = None) -> int | None:
    """A process's lifetime peak memory footprint on macOS, or None elsewhere.

    Agreed with `/usr/bin/time -l`'s "peak memory footprint" to within about 16 KB at 1 GB
    and 3 GB on the reference platform.

    Args:
        pid: The process to read, this one when None. An exited child can be read only
            while it is unreaped.

    Returns:
        The footprint in bytes, or None off macOS or when the read fails.
    """
    if sys.platform != "darwin":
        return None
    info = _RusageInfoV4()
    target = os.getpid() if pid is None else pid
    if ctypes.CDLL(_LIBPROC).proc_pid_rusage(target, _RUSAGE_INFO_V4, ctypes.byref(info)):
        return None
    return int(info.ri_lifetime_max_phys_footprint)


def _wait_without_reaping(pid: int) -> None:
    """Block until the child exits, leaving it unreaped so its pid can still be read.

    Raises:
        OSError: If waitid fails for any reason but an interrupted call, which is retried.
    """
    if hasattr(os, "waitid"):
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    siginfo = ctypes.create_string_buffer(_SIGINFO_BUFFER_BYTES)
    options = os.WEXITED | os.WNOWAIT
    while libc.waitid(ctypes.c_int(os.P_PID), ctypes.c_uint(pid), siginfo, ctypes.c_int(options)):
        code = ctypes.get_errno()
        if code != errno.EINTR:
            raise OSError(code, os.strerror(code))


def wait_with_footprint(
    pid: int, *, before_reap: Callable[[], None] | None = None
) -> tuple[int, int | None]:
    """Wait for a child to exit, read its lifetime peak footprint, then reap it.

    The caller must not also wait for the child: this function reaps it.

    Args:
        pid: A child of this process.
        before_reap: Called after the child has exited and before it is reaped, while its
            pid still names it.

    Returns:
        The exit code in `subprocess` form (negative for a signal) and the footprint in
        bytes, or None for the footprint off macOS or when the read fails.
    """
    footprint = None
    _wait_without_reaping(pid)
    if sys.platform == "darwin":
        footprint = peak_footprint_bytes(pid)
    if before_reap is not None:
        before_reap()
    _, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status), footprint
