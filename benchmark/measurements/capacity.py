"""Peak resident memory of a worker canonicalising the capacity fixture, as v8 recorded it.

The parent writes the capacity fixture that `runner.capacity_fixture` describes, then runs a
fresh child process that performs acquisition, inspection, decode, and canonicalisation
exactly as a run would, and reports its own peak memory footprint and peak resident set
size. The footprint, `ri_lifetime_max_phys_footprint` from `proc_pid_rusage`, counts memory
macOS has compressed; the resident set (`ru_maxrss`, bytes on macOS) does not, and on one
fixture it varied by hundreds of megabytes between runs while the footprint did not. The
footprint is read on macOS only. Run from the repository root:
`uv run --frozen python -m benchmark.measurements.capacity`. The fixture is written to a
temporary directory and needs about 200 MB of disk and several GB of memory.
"""

from __future__ import annotations

import ctypes
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from benchmark.canonicalisation import CanonicalWaveform, canonicalise
from benchmark.protocol import CapacityFixture, load_protocol
from benchmark.source_acquisition import ArtifactBuffered, acquire_local_artifact
from benchmark.source_decode import decode_source
from benchmark.source_inspection import InspectedSource, inspect_source

_BLOCK = 1 << 20
_RESULT = "RESULT "
_RUSAGE_INFO_V4 = 4
_LIBPROC = "/usr/lib/libproc.dylib"
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


def peak_footprint_bytes() -> int | None:
    """This process's lifetime peak memory footprint on macOS, or None elsewhere.

    Agreed with `/usr/bin/time -l`'s "peak memory footprint" to within about 16 KB at 1 GB
    and 3 GB on the reference platform.
    """
    if sys.platform != "darwin":
        return None
    info = _RusageInfoV4()
    if ctypes.CDLL(_LIBPROC).proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)):
        return None
    return int(info.ri_lifetime_max_phys_footprint)


def write_tone_flac(path: Path, fixture: CapacityFixture) -> None:
    """Write the fixture's two tones, 0.3 at 440 Hz plus 0.2 at 1234.5 Hz, block by block."""
    rate, channels, frames = fixture.sample_rate, fixture.channels, fixture.frames
    with sf.SoundFile(
        path,
        "w",
        samplerate=rate,
        channels=channels,
        format=fixture.container,
        subtype=fixture.subtype,
    ) as out:
        for start in range(0, frames, _BLOCK):
            t = (np.arange(start, min(start + _BLOCK, frames)) / rate)[:, None]
            x = 0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 1234.5 * t)
            out.write(np.repeat(x, channels, axis=1))


def run_pipeline(path: Path) -> dict[str, Any]:
    """Acquire, inspect, decode, and canonicalise one source; report what each stage returned."""
    protocol = load_protocol()
    config = protocol.canonicalisation
    source_id = path.stem
    start = time.perf_counter()
    result: dict[str, Any] = {}
    artifact = acquire_local_artifact(path, logical_source_id=source_id, config=config.decode)
    result["acquired"] = type(artifact).__name__
    if isinstance(artifact, ArtifactBuffered):
        inspected = inspect_source(
            artifact,
            logical_source_id=source_id,
            protocol_sha256=protocol.source_sha256,
            config=config,
        )
        result["inspected"] = type(inspected).__name__
        if isinstance(inspected, InspectedSource):
            decoded = decode_source(
                artifact,
                inspected,
                logical_source_id=source_id,
                protocol_sha256=protocol.source_sha256,
                config=config,
            )
            result["decoded"] = type(decoded).__name__
            if isinstance(decoded, np.ndarray):
                canonical = canonicalise(
                    decoded, inspected, logical_source_id=source_id, config=config
                )
                result["canonical"] = type(canonical).__name__
                if isinstance(canonical, CanonicalWaveform):
                    result["canonical_bytes"] = canonical.samples.nbytes
    result["seconds"] = round(time.perf_counter() - start, 1)
    return result


def parse_result(stdout: str) -> dict[str, Any] | None:
    """The child's result line, or None when it printed none."""
    line = next((line for line in stdout.splitlines() if line.startswith(_RESULT)), None)
    return None if line is None else dict(json.loads(line[len(_RESULT) :]))


def child(path: str) -> None:
    """Run the pipeline in this process and print the result with this process's peak RSS."""
    result = run_pipeline(Path(path))
    result["peak_rss"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["peak_footprint"] = peak_footprint_bytes()
    print(_RESULT + json.dumps(result))


def measure_once(path: Path) -> dict[str, Any] | None:
    """One fresh worker on *path*: its result with its peak RSS, or None if it gave none."""
    command = [sys.executable, "-m", "benchmark.measurements.capacity", "child", str(path)]
    out = subprocess.run(command, capture_output=True, text=True, check=False)
    result = parse_result(out.stdout)
    if result is None:
        # The last 600 characters of stderr, the end of any traceback.
        print(f"worker exited {out.returncode} without a result: {out.stderr[-600:]}")
    return result


def main() -> None:
    """Write the capacity fixture, measure a fresh worker on it repeatedly, print the peaks.

    The peak varies between runs on the same input on the reference platform, so the
    measurement is repeated as many times as v8 records and every peak is printed.
    """
    protocol = load_protocol()
    fixture = protocol.runner.capacity_fixture
    budget = protocol.canonicalisation.decode.max_projected_live_bytes
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "capacity_fixture.flac"
        write_tone_flac(path, fixture)
        print(f"capacity fixture FLAC: {path.stat().st_size} bytes")
        footprints = []
        for _ in range(fixture.measured_runs):
            result = measure_once(path)
            if result is None:
                return
            footprints.append(result["peak_footprint"])
            print(json.dumps(result))
    if None in footprints:
        print("the peak footprint is read on macOS only")
        return
    largest = max(footprints)
    print(f"peak footprints {footprints}; smallest {min(footprints)}, largest {largest}")
    print(f"largest is {largest / budget:.3f} of max_projected_live_bytes")
    print(
        f"v8 recorded {fixture.measured_min_peak_footprint_bytes} to "
        f"{fixture.measured_peak_footprint_bytes} over {fixture.measured_runs} runs "
        f"on {fixture.measured_on}"
    )


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "child":
        child(sys.argv[2])
    else:
        main()
