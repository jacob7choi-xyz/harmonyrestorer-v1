"""The engine boundary of the systems stage: one system's call on one degraded frame.

Protocol v13 (`system_execution`, `scoring`) sorts every failure into one of two kinds, and the
boundary keeps them apart:

- An exception raised inside one system's `denoise` call is that system's failure for that
  frame. `call_system` wraps only that call, and it catches every `Exception` subclass,
  `MemoryError` included, because the engines raise many unrelated types. `KeyboardInterrupt`
  and `SystemExit` are not `Exception` subclasses and propagate.
- An output the stage can read but that is not 16 kHz mono is also that system's failure. The
  protocol is silent on this case, and the systems stage follows the decision made for it.
- Anything outside the call is a harness failure and raises `HarnessFailure`, which aborts the
  run: writing the input frame, writing the progress marker, and reading the output file. A
  harness failure is never charged to a system.

Before each call the boundary writes a progress marker, atomically, naming the system and the
recording and frame it is called on. A native crash ends the process before any result is
written, so the marker is how the frame that crashed is found afterwards.
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import soundfile as sf

from benchmark.atomic_write import write_atomically

__all__ = [
    "FrameIdentity",
    "HarnessFailure",
    "System",
    "SystemCallResult",
    "SystemFailed",
    "SystemOutput",
    "call_system",
    "read_output",
    "write_frame_input",
    "write_progress_marker",
]

logger = logging.getLogger(__name__)


class HarnessFailure(RuntimeError):
    """A failure outside a system's call. It aborts the run and is charged to no system."""


class System(Protocol):
    """A denoising system the stage calls: a name, and one `denoise` call per input file."""

    name: str

    def denoise(self, input_path: Path) -> Path:
        """Denoise the WAV at `input_path` and return the path of the output WAV."""
        ...


@dataclass(frozen=True)
class FrameIdentity:
    """The recording and frame a call is made on, as the progress marker names them.

    `canonical_sha256` is the recording's canonical digest, the protocol's recording identifier.
    """

    canonical_sha256: str
    frame_index: int


@dataclass(frozen=True)
class SystemFailed:
    """One system's failure on one frame: an exception in its call, or a wrong output format.

    `exception_type` is the class name of the exception raised inside the call, and None when
    the output was read and rejected for its format.
    """

    reason: str
    exception_type: str | None


@dataclass(frozen=True)
class SystemOutput:
    """A system's output for one frame: mono, at the conditioning rate, as float64 samples."""

    samples: npt.NDArray[np.float64]


SystemCallResult = SystemFailed | SystemOutput


def write_frame_input(samples: npt.NDArray[np.float32], path: Path, sample_rate: int) -> None:
    """Write one degraded frame as a mono float WAV, atomically.

    Args:
        samples: The degraded frame, one-dimensional.
        path: Where the WAV is written. Its directory must exist.
        sample_rate: The rate the samples are at, in Hz.

    Raises:
        HarnessFailure: If the frame cannot be encoded or written. No file is left at `path`.
    """
    buffer = io.BytesIO()
    try:
        sf.write(buffer, samples, sample_rate, format="WAV", subtype="FLOAT")
        write_atomically(path, buffer.getvalue())
    except (OSError, sf.SoundFileError) as error:
        logger.error("Cannot write frame input %s: %s", path, error)
        raise HarnessFailure(f"frame input {path} could not be written: {error}") from error


def write_progress_marker(path: Path, system: str, identity: FrameIdentity) -> None:
    """Write the marker that names the system, recording, and frame about to be called.

    Args:
        path: The marker file, replaced atomically on every call.
        system: The system's name.
        identity: The recording and frame.

    Raises:
        HarnessFailure: If the marker cannot be written. The call must not then be made.
    """
    payload = {
        "system": system,
        "canonical_sha256": identity.canonical_sha256,
        "frame_index": identity.frame_index,
    }
    try:
        write_atomically(path, json.dumps(payload, sort_keys=True).encode("utf-8"))
    except OSError as error:
        logger.error("Cannot write progress marker %s: %s", path, error)
        raise HarnessFailure(f"progress marker {path} could not be written: {error}") from error


def call_system(
    system: System,
    input_path: Path,
    *,
    identity: FrameIdentity,
    marker_path: Path,
    sample_rate: int,
) -> SystemCallResult:
    """Call one system on one written frame and return its output, or its failure.

    The marker is written before the call. Exceptions inside the call are the system's failure;
    the output is then read, and a read that fails is a harness failure.

    Args:
        system: The system to call.
        input_path: The degraded frame's WAV, already written.
        identity: The recording and frame, for the progress marker.
        marker_path: Where the progress marker is written.
        sample_rate: The rate the output must have, in Hz (mono is also required).

    Returns:
        The output as float64 samples, or a `SystemFailed` for this system and frame.

    Raises:
        HarnessFailure: If the marker cannot be written or the output cannot be read.
    """
    write_progress_marker(marker_path, system.name, identity)
    returned = _invoke(system, input_path)
    if isinstance(returned, SystemFailed):
        return returned
    return read_output(returned, sample_rate)


def _invoke(system: System, input_path: Path) -> Path | SystemFailed:
    """Run the system's `denoise` call, and turn any exception it raises into a failure."""
    try:
        return system.denoise(input_path)
    except Exception as error:
        # Protocol v13: an exception inside the call is this system's failure for the frame.
        logger.warning(
            "System %s failed on %s: %s: %s",
            system.name,
            input_path.name,
            type(error).__name__,
            error,
        )
        return SystemFailed(
            reason=f"{type(error).__name__}: {error}", exception_type=type(error).__name__
        )


def read_output(path: Path, sample_rate: int) -> SystemCallResult:
    """Read a system's output file and check that it is mono at `sample_rate`.

    Args:
        path: The output file the system returned.
        sample_rate: The required rate, in Hz.

    Returns:
        The samples as float64 when the format matches, otherwise a `SystemFailed`.

    Raises:
        HarnessFailure: If the file cannot be opened or read as audio. The protocol places
            reading the output outside the call, so this aborts the run.
    """
    info = _output_info(path)
    if info.samplerate != sample_rate or info.channels != 1:
        reason = (
            f"output is {info.samplerate} Hz with {info.channels} channels, "
            f"expected {sample_rate} Hz mono"
        )
        logger.warning("System output rejected for %s: %s", path.name, reason)
        return SystemFailed(reason=reason, exception_type=None)
    return SystemOutput(_read_samples(path))


def _output_info(path: Path) -> sf.SoundFileInfo:
    """The header of an output file, or a harness failure if it cannot be read."""
    try:
        return sf.info(str(path))
    except sf.SoundFileError as error:
        logger.error("Cannot read output %s: %s", path, error)
        raise HarnessFailure(f"output {path} cannot be read: {error}") from error


def _read_samples(path: Path) -> npt.NDArray[np.float64]:
    """The samples of an output file as float64, or a harness failure if they cannot be read."""
    try:
        samples, _ = sf.read(str(path), dtype="float64", always_2d=False)
    except sf.SoundFileError as error:
        logger.error("Cannot read output samples %s: %s", path, error)
        raise HarnessFailure(f"output {path} samples cannot be read: {error}") from error
    return np.asarray(samples, dtype=np.float64)
