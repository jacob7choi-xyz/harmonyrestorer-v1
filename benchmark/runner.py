"""The parent side of canonicalisation: preconditions, one worker per source, exit mapping.

Protocol v8's runner duties, for the canonicalisation stage:

- **Preconditions first.** `verify_run_preconditions` runs before any worker starts, and a
  failure aborts the run with nothing opened.
- **One child process per source, one at a time.** Each worker runs
  `benchmark.canonical_worker` in its own directory under the run's work directory, named
  by position because logical source identifiers contain `/`. The directory must not exist
  yet, so a record left by an earlier run can never be read as this one's.
- **How a worker ended decides what is recorded.** A source's record is accepted only from
  a worker that exited 0 and left a record that rebuilds through the record constructors,
  names this source, and carries this run's protocol digest; a canonical waveform is
  accepted only if its file hashes to the digest the record states and holds the stated
  number of samples. Otherwise:

  - the worker's caught-`MemoryError` status, or death by SIGKILL, aborts with
    `runner.capacity_abort` (`ENVIRONMENT_CAPACITY_FAILURE`);
  - any other end aborts with `runner.other_worker_death_abort`
    (`INTERNAL_CANONICALISATION_ERROR`) naming the source: another signal, a non-zero
    status, or a record that is missing, malformed, or disagrees with its file.

  An abort the worker recorded itself, such as `LOCAL_ARTIFACT_UNSTABLE`, stands as
  recorded. The first abort stops the run.
- **Each worker's peak footprint is recorded per source,** read by this process after the
  worker ends and before it is reaped (`benchmark.footprint`), so a killed worker has one
  too. It is read on macOS only and is None elsewhere. v8's recorded footprints were read by
  the measured process itself; on the capacity fixture a reading taken by the parent was
  about 1.6 MB higher (`benchmark.footprint`), so the two are not directly comparable.

The worker's output goes to `worker.log` in its directory rather than to a pipe, so a long
traceback cannot fill a pipe nobody is reading and stall the worker.

> **Note:** There is no timeout. The protocol permits one and defines no abort code for
> it, and stopping a slow worker with SIGKILL would be recorded as a capacity failure.

Nothing calls `run_canonicalisation` yet: candidate selection does not exist, so the caller
supplies the candidates. No engine is loaded here, so the requirement that the model files
are checked before the UVR separator is built is not reached in this stage.
"""

from __future__ import annotations

import json
import logging
import signal
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.canonical_worker import (
    CANONICAL_FILE,
    CANONICAL_KIND,
    EXIT_MEMORY_ERROR,
    RECORD_FILE,
)
from benchmark.footprint import wait_with_footprint
from benchmark.outcome_wire import OutcomeWireError, outcome_from_wire
from benchmark.protocol import Protocol
from benchmark.run_preconditions import file_sha256, verify_run_preconditions
from benchmark.source_outcome import AbortReason, RunAbortOutcome, SourceIneligibleOutcome

__all__ = [
    "WORKER_LOG",
    "Candidate",
    "CanonicalFile",
    "CanonicalisationRun",
    "SourceRun",
    "run_canonicalisation",
    "run_source",
    "worker_command",
]

logger = logging.getLogger(__name__)

WORKER_LOG = "worker.log"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FLOAT64_BYTES = 8
_RECORD_KEYS = frozenset({"protocol_sha256", "result"})
_CANONICAL_KEYS = frozenset({"kind", "sha256", "samples"})


@dataclass(frozen=True)
class Candidate:
    """A staged source the caller asks the run to canonicalise."""

    logical_source_id: str
    path: Path
    expected_sha256: str | None = None


@dataclass(frozen=True)
class CanonicalFile:
    """A canonical waveform on disk whose bytes hashed to `sha256` when the worker ended."""

    path: Path
    sha256: str
    samples: int


SourceOutcome = CanonicalFile | SourceIneligibleOutcome | RunAbortOutcome


@dataclass(frozen=True)
class SourceRun:
    """What one source became, how its worker ended, and the worker's peak footprint."""

    logical_source_id: str
    outcome: SourceOutcome
    returncode: int
    peak_footprint_bytes: int | None


@dataclass(frozen=True)
class CanonicalisationRun:
    """Every source run in order, and the abort that stopped the run, if one did."""

    abort: RunAbortOutcome | None
    sources: tuple[SourceRun, ...]


WorkerCommand = Callable[[Candidate, Path], list[str]]


def worker_command(candidate: Candidate, out_dir: Path) -> list[str]:
    """The command that runs `benchmark.canonical_worker` on one candidate.

    `-m` without `-I`, from the repository root, because the `benchmark` package is not
    installed and `-I` would remove the working directory from the import path.
    """
    command = [
        sys.executable,
        "-m",
        "benchmark.canonical_worker",
        str(candidate.path),
        candidate.logical_source_id,
        str(out_dir),
    ]
    if candidate.expected_sha256 is not None:
        command += ["--expected-sha256", candidate.expected_sha256]
    return command


def _run_worker(command: list[str], out_dir: Path) -> tuple[int, int | None]:
    """Start the worker, wait for it, and return its exit code and peak footprint."""
    with (out_dir / WORKER_LOG).open("wb") as log:
        child = subprocess.Popen(
            command, cwd=_REPO_ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT
        )
    returncode, footprint = wait_with_footprint(child.pid)
    # Reaped above; recording it stops Popen from waiting for the pid again.
    child.returncode = returncode
    return returncode, footprint


def _abort(reason: str, candidate: Candidate, detail: str, out_dir: Path) -> RunAbortOutcome:
    """An abort naming the source, with the worker's log location in the run log only."""
    logger.error("Source %s: %s; worker output in %s", candidate.logical_source_id, detail, out_dir)
    return RunAbortOutcome(
        reason=AbortReason(reason), detail=detail, logical_source_id=candidate.logical_source_id
    )


def _canonical_file(result: dict[str, Any], out_dir: Path) -> CanonicalFile | str:
    """The canonical file the record describes, or why it does not match the record."""
    if set(result) != _CANONICAL_KEYS:
        return f"canonical record has keys {sorted(result)}"
    sha256, samples = result["sha256"], result["samples"]
    if not isinstance(sha256, str) or isinstance(samples, bool) or not isinstance(samples, int):
        return "canonical record has a malformed digest or sample count"
    path = out_dir / CANONICAL_FILE
    if file_sha256(path) != sha256:
        return "canonical file is missing or does not hash to the recorded digest"
    if path.stat().st_size != samples * _FLOAT64_BYTES:
        return f"canonical file does not hold the recorded {samples} samples"
    return CanonicalFile(path=path, sha256=sha256, samples=samples)


def _reported(candidate: Candidate, out_dir: Path, protocol: Protocol) -> SourceOutcome | str:
    """The outcome the worker's record states, or why the record cannot be accepted."""
    try:
        record = json.loads((out_dir / RECORD_FILE).read_bytes())
    except FileNotFoundError:
        return "worker exited 0 without a record"
    except ValueError:
        return "worker record is not valid JSON"
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        return "worker record does not have the expected keys"
    if record["protocol_sha256"] != protocol.source_sha256:
        return "worker record names a different protocol digest"
    result = record["result"]
    if isinstance(result, dict) and result.get("kind") == CANONICAL_KIND:
        return _canonical_file(result, out_dir)
    try:
        outcome = outcome_from_wire(result, protocol.canonicalisation.decode)
    except OutcomeWireError as error:
        logger.error("Source %s record refused: %s", candidate.logical_source_id, error)
        return "worker record does not rebuild"
    if outcome.logical_source_id != candidate.logical_source_id:
        return "worker record does not name this source"
    return outcome


def _outcome(
    candidate: Candidate, out_dir: Path, returncode: int, protocol: Protocol
) -> SourceOutcome:
    """Map how the worker ended, and what it left, to what the source became."""
    runner = protocol.runner
    if returncode in (EXIT_MEMORY_ERROR, -signal.SIGKILL):
        detail = f"worker ended with {returncode}, a capacity exit"
        return _abort(runner.capacity_abort, candidate, detail, out_dir)
    if returncode != 0:
        detail = f"worker ended with {returncode} without a verdict"
        return _abort(runner.other_worker_death_abort, candidate, detail, out_dir)
    reported = _reported(candidate, out_dir, protocol)
    if isinstance(reported, str):
        return _abort(runner.other_worker_death_abort, candidate, reported, out_dir)
    return reported


def run_source(
    candidate: Candidate,
    out_dir: Path,
    protocol: Protocol,
    *,
    command: WorkerCommand = worker_command,
) -> SourceRun:
    """Run one candidate in a worker of its own and record what it became.

    Args:
        candidate: The source to canonicalise.
        out_dir: A directory that does not exist yet, for the worker's files.
        protocol: The loaded protocol.
        command: Builds the worker's command line; injected in tests.

    Returns:
        The source's outcome, the worker's exit code, and its peak footprint.

    Raises:
        FileExistsError: If `out_dir` already exists.
    """
    out_dir.mkdir()
    returncode, footprint = _run_worker(command(candidate, out_dir), out_dir)
    outcome = _outcome(candidate, out_dir, returncode, protocol)
    logger.info(
        "Source %s: %s, exit %d, peak footprint %s bytes",
        candidate.logical_source_id,
        type(outcome).__name__,
        returncode,
        footprint,
    )
    return SourceRun(candidate.logical_source_id, outcome, returncode, footprint)


def run_canonicalisation(
    protocol: Protocol,
    candidates: Iterable[Candidate],
    *,
    work_dir: Path,
    runtime_platform: str,
    opgan_checkpoint: Path,
    uvr_model_dir: Path,
    command: WorkerCommand = worker_command,
) -> CanonicalisationRun:
    """Check the run preconditions, then canonicalise each candidate in turn.

    Args:
        protocol: The loaded protocol.
        candidates: The sources, in the order they are run.
        work_dir: An existing directory; each source gets a new directory inside it.
        runtime_platform: This runtime's platform, normally `platform_token()`.
        opgan_checkpoint: The checkpoint the OpGAN engine loads.
        uvr_model_dir: The directory audio-separator loads the UVR files from.
        command: Builds each worker's command line; injected in tests.

    Returns:
        The sources run, in order, and the abort that stopped the run, if any.
    """
    abort = verify_run_preconditions(
        protocol,
        runtime_platform=runtime_platform,
        opgan_checkpoint=opgan_checkpoint,
        uvr_model_dir=uvr_model_dir,
    )
    if abort is not None:
        return CanonicalisationRun(abort, ())
    runs: list[SourceRun] = []
    for index, candidate in enumerate(candidates):
        run = run_source(candidate, work_dir / f"source-{index:06d}", protocol, command=command)
        runs.append(run)
        if isinstance(run.outcome, RunAbortOutcome):
            return CanonicalisationRun(run.outcome, tuple(runs))
    return CanonicalisationRun(None, tuple(runs))
