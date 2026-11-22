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

A worker that runs past `timeout_s` is stopped and recorded as `WORKER_TIMEOUT` (protocol
v11), not as a capacity failure, although it is stopped with SIGKILL: a timer sends the
signal and records that it did, and a worker that then ends with -9 is a timeout. The timer
is stopped in `wait_with_footprint`'s `before_reap` hook, while the exited worker is still
unreaped, so it cannot signal a process that later reuses the pid. If the wait ends in an
exception, the worker is killed and reaped before the exception propagates, unless the reap
had already begun or the wait reported that the pid is not this process's child.

> **Note:** A worker killed by SIGKILL for another reason, such as an out-of-memory kill,
> is recorded as `WORKER_TIMEOUT` if the timer also fires before it is stopped, between the
> worker's exit and the `before_reap` hook. Both codes abort the run and neither charges the
> source, so only the abort's label depends on it.

Nothing calls `run_canonicalisation` yet: candidate selection does not exist, so the caller
supplies the candidates. No engine is loaded here, so the requirement that the model files
are checked before the UVR separator is built is not reached in this stage.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark._source_validation import SourceOutcomeError, checked_digest
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
_CANONICAL_KEYS = frozenset(
    {"kind", "sha256", "samples", "logical_source_id", "raw_sha256", "decoded_pcm_sha256"}
)
# The parent reads at most this much of a record the worker wrote, so a damaged or
# oversized file cannot make it allocate without bound.
_MAX_RECORD_BYTES = 1 << 20


@dataclass(frozen=True)
class Candidate:
    """A staged source the caller asks the run to canonicalise."""

    logical_source_id: str
    path: Path
    expected_sha256: str | None = None


@dataclass(frozen=True)
class CanonicalFile:
    """A canonical waveform on disk whose bytes hashed to `sha256` when the worker ended.

    `raw_sha256` and `decoded_pcm_sha256` identify the source the worker canonicalised, as
    the raw-digest and decoded-PCM rules compare them.
    """

    path: Path
    sha256: str
    samples: int
    raw_sha256: str
    decoded_pcm_sha256: str


SourceOutcome = CanonicalFile | SourceIneligibleOutcome | RunAbortOutcome


@dataclass(frozen=True)
class SourceRun:
    """What one source became, how its worker ended, its peak footprint, and its wall time.

    `wall_seconds` runs from just before the worker starts to just after it is reaped, on a
    monotonic clock, so it includes process start-up and the footprint read.
    """

    logical_source_id: str
    outcome: SourceOutcome
    returncode: int
    peak_footprint_bytes: int | None
    wall_seconds: float


@dataclass(frozen=True)
class CanonicalisationRun:
    """Every source run in order, and the abort that stopped the run, if one did."""

    abort: RunAbortOutcome | None
    sources: tuple[SourceRun, ...]


WorkerCommand = Callable[[Candidate, Path], list[str]]


def worker_command(candidate: Candidate, out_dir: Path) -> list[str]:
    """The command that runs `benchmark.canonical_worker` on one candidate.

    `-m` without `-I`, from the repository root, because the `benchmark` package is not
    installed and `-I` would remove the working directory from the import path. The option
    comes first and `--` ends the options, so a path or identifier that begins with `-` is
    read as a positional argument rather than refused by the parser.
    """
    command = [sys.executable, "-m", "benchmark.canonical_worker"]
    if candidate.expected_sha256 is not None:
        command += ["--expected-sha256", candidate.expected_sha256]
    return [*command, "--", str(candidate.path), candidate.logical_source_id, str(out_dir)]


class _Watchdog:
    """Sends SIGKILL to an unreaped worker once its time is up, unless stopped first."""

    def __init__(self, pid: int, timeout_s: float) -> None:
        """Arm the timer for *pid*."""
        self._pid = pid
        self._lock = threading.Lock()
        self._stopped = False
        self.fired = False
        self._timer = threading.Timer(timeout_s, self._fire)
        self._timer.daemon = True
        self._timer.start()

    def _fire(self) -> None:
        """Kill the worker, if the caller has not stopped the timer yet."""
        with self._lock:
            if self._stopped:
                return
            os.kill(self._pid, signal.SIGKILL)
            self.fired = True

    def stop(self) -> None:
        """Prevent any later signal; once this returns, the timer never signals."""
        with self._lock:
            self._stopped = True
        self._timer.cancel()


def _end_unwaited(child: subprocess.Popen[bytes], error: BaseException) -> None:
    """Kill and reap a worker the wait did not reap, unless the pid is not this process's child.

    A ChildProcessError from the wait means the pid no longer names a child this process can
    reap, so it may belong to another process and is not signalled.
    """
    if isinstance(error, ChildProcessError):
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(child.pid, signal.SIGKILL)
    with contextlib.suppress(ChildProcessError):
        os.waitpid(child.pid, 0)
    child.returncode = -signal.SIGKILL


def _run_worker(
    command: list[str], out_dir: Path, timeout_s: float
) -> tuple[int, int | None, bool]:
    """Start the worker, wait for it, and return its exit code, footprint, and timeout flag."""
    child: subprocess.Popen[bytes] | None = None
    watchdog: _Watchdog | None = None
    reaping = False

    def stop_before_reap() -> None:
        nonlocal reaping
        if watchdog is not None:
            watchdog.stop()
        reaping = True

    try:
        # Once Popen returns, an exception reaches the handler with the worker known, so a
        # failed timer or an interrupt cannot leave it running. An interrupt inside Popen,
        # after the child has started, is not covered: child is not yet bound.
        with (out_dir / WORKER_LOG).open("wb") as log:
            child = subprocess.Popen(
                command,
                cwd=_REPO_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        watchdog = _Watchdog(child.pid, timeout_s)
        returncode, footprint = wait_with_footprint(child.pid, before_reap=stop_before_reap)
    except BaseException as error:
        # broad: cleanup only, so an interrupt cannot leave the worker running; re-raised.
        pid = None if child is None else child.pid
        logger.error("Worker %s: run ended by %s before a result", pid, type(error).__name__)
        if watchdog is not None:
            watchdog.stop()
        if child is not None and not reaping:
            _end_unwaited(child, error)
        raise
    # Reaped above; recording it stops Popen from waiting for the pid again.
    child.returncode = returncode
    return returncode, footprint, watchdog.fired and returncode == -signal.SIGKILL


def _abort(reason: str, candidate: Candidate, detail: str, out_dir: Path) -> RunAbortOutcome:
    """An abort naming the source, with the worker's log location in the run log only."""
    logger.error("Source %s: %s; worker output in %s", candidate.logical_source_id, detail, out_dir)
    return RunAbortOutcome(
        reason=AbortReason(reason), detail=detail, logical_source_id=candidate.logical_source_id
    )


def _record_digest(value: object, field: str) -> str | None:
    """*value* if it is a SHA-256 digest in canonical form, otherwise None."""
    if not isinstance(value, str):
        return None
    try:
        return checked_digest(value, field)
    except SourceOutcomeError:
        return None


def _canonical_file(
    result: dict[str, Any], candidate: Candidate, out_dir: Path
) -> CanonicalFile | str:
    """The canonical file the record describes, or why it does not match the record."""
    if set(result) != _CANONICAL_KEYS:
        return f"canonical record has keys {sorted(result)}"
    sha256, samples = result["sha256"], result["samples"]
    if not isinstance(sha256, str) or isinstance(samples, bool) or not isinstance(samples, int):
        return "canonical record has a malformed digest or sample count"
    if result["logical_source_id"] != candidate.logical_source_id:
        return "canonical record does not name this source"
    raw = _record_digest(result["raw_sha256"], "raw_sha256")
    decoded = _record_digest(result["decoded_pcm_sha256"], "decoded_pcm_sha256")
    if raw is None or decoded is None:
        return "canonical record has a malformed raw or decoded-PCM digest"
    if candidate.expected_sha256 is not None and raw != candidate.expected_sha256:
        return "canonical record names bytes other than the expected digest"
    path = out_dir / CANONICAL_FILE
    if file_sha256(path) != sha256:
        return "canonical file is missing or does not hash to the recorded digest"
    if path.stat().st_size != samples * _FLOAT64_BYTES:
        return f"canonical file does not hold the recorded {samples} samples"
    return CanonicalFile(
        path=path, sha256=sha256, samples=samples, raw_sha256=raw, decoded_pcm_sha256=decoded
    )


def _reported(candidate: Candidate, out_dir: Path, protocol: Protocol) -> SourceOutcome | str:
    """The outcome the worker's record states, or why the record cannot be accepted."""
    try:
        with (out_dir / RECORD_FILE).open("rb") as handle:
            data = handle.read(_MAX_RECORD_BYTES + 1)
    except FileNotFoundError:
        return "worker exited 0 without a record"
    if len(data) > _MAX_RECORD_BYTES:
        return f"worker record is larger than {_MAX_RECORD_BYTES} bytes"
    try:
        record = json.loads(data)
    except ValueError:
        return "worker record is not valid JSON"
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        return "worker record does not have the expected keys"
    if record["protocol_sha256"] != protocol.source_sha256:
        return "worker record names a different protocol digest"
    result = record["result"]
    if isinstance(result, dict) and result.get("kind") == CANONICAL_KIND:
        return _canonical_file(result, candidate, out_dir)
    try:
        outcome = outcome_from_wire(result, protocol.canonicalisation.decode)
    except OutcomeWireError as error:
        logger.error("Source %s record refused: %s", candidate.logical_source_id, error)
        return "worker record does not rebuild"
    if outcome.logical_source_id != candidate.logical_source_id:
        return "worker record does not name this source"
    if isinstance(outcome, SourceIneligibleOutcome) and _names_other_bytes(outcome, candidate):
        return "worker record names bytes other than the expected digest"
    return outcome


def _names_other_bytes(outcome: SourceIneligibleOutcome, candidate: Candidate) -> bool:
    """Whether a verdict carries a digest that is not the one the candidate was staged with.

    Abort records are not checked: a SOURCE_HASH_MISMATCH record differs from the expected
    digest by definition, and every abort ends the run whatever bytes it names. A verdict
    that carries no digest is accepted: the worker records none only for a source above
    `max_identity_stream_bytes`, which protocol v11 decides by size rather than by digest.
    Like the canonical record's raw digest, this binds only when an expected digest is given.
    """
    digest = outcome.identity.sha256
    expected = candidate.expected_sha256
    return expected is not None and digest is not None and digest != expected


def _outcome(
    candidate: Candidate, out_dir: Path, returncode: int, protocol: Protocol, *, timed_out: bool
) -> SourceOutcome:
    """Map how the worker ended, and what it left, to what the source became."""
    runner = protocol.runner
    if timed_out:
        detail = f"worker stopped by the timeout, ended with {returncode}"
        return _abort(runner.timeout_abort, candidate, detail, out_dir)
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


def _check_inputs(candidate: Candidate, out_dir: Path, timeout_s: float) -> None:
    """Refuse what the worker would read differently from this process, before it starts.

    The worker runs with the repository root as its working directory, so a relative path
    would name one file here and another there. A malformed expected digest would reach the
    worker's comparison, which raises on a non-ASCII value instead of recording a mismatch.
    """
    for name, path in (("candidate path", candidate.path), ("output directory", out_dir)):
        if not path.is_absolute():
            raise ValueError(f"{name} must be absolute, got {path}")
    for name, text in (
        ("candidate path", str(candidate.path)),
        ("output directory", str(out_dir)),
        ("logical source id", candidate.logical_source_id),
    ):
        if "\x00" in text:
            raise ValueError(f"{name} contains a NUL character, which no argument can carry")
    expected = candidate.expected_sha256
    if expected is not None and _record_digest(expected, "expected_sha256") is None:
        raise ValueError(f"expected digest is not 64 lowercase hex characters: {expected!r}")
    _check_timeout(timeout_s)


def _check_timeout(timeout_s: float) -> None:
    """Refuse a timeout that is not a positive, finite number of seconds."""
    if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool):
        raise ValueError(f"timeout must be a number of seconds, got {timeout_s!r}")
    if not (math.isfinite(timeout_s) and timeout_s > 0):
        raise ValueError(f"timeout must be positive and finite, got {timeout_s!r}")


def run_source(
    candidate: Candidate,
    out_dir: Path,
    protocol: Protocol,
    *,
    timeout_s: float,
    command: WorkerCommand = worker_command,
) -> SourceRun:
    """Run one candidate in a worker of its own and record what it became.

    Args:
        candidate: The source to canonicalise.
        out_dir: A directory that does not exist yet, for the worker's files.
        protocol: The loaded protocol.
        timeout_s: Seconds the worker may run before it is stopped, a harness setting.
        command: Builds the worker's command line; injected in tests.

    Returns:
        The source's outcome, the worker's exit code, and its peak footprint.

    Raises:
        FileExistsError: If `out_dir` already exists.
        ValueError: If a path is relative, the expected digest is malformed, or the timeout
            is not a positive finite number, checked before anything is created or started.
    """
    _check_inputs(candidate, out_dir, timeout_s)
    out_dir.mkdir()
    started = time.monotonic()
    returncode, footprint, timed_out = _run_worker(command(candidate, out_dir), out_dir, timeout_s)
    wall_seconds = time.monotonic() - started
    outcome = _outcome(candidate, out_dir, returncode, protocol, timed_out=timed_out)
    logger.info(
        "Source %s: %s, exit %d, peak footprint %s bytes, %.2f s",
        candidate.logical_source_id,
        type(outcome).__name__,
        returncode,
        footprint,
        wall_seconds,
    )
    return SourceRun(candidate.logical_source_id, outcome, returncode, footprint, wall_seconds)


def run_canonicalisation(
    protocol: Protocol,
    candidates: Iterable[Candidate],
    *,
    work_dir: Path,
    runtime_platform: str,
    opgan_checkpoint: Path,
    uvr_model_dir: Path,
    timeout_s: float,
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
        timeout_s: Seconds each worker may run before it is stopped.
        command: Builds each worker's command line; injected in tests.

    Returns:
        The sources run, in order, and the abort that stopped the run, if any.

    Raises:
        ValueError: If `work_dir` is relative or `timeout_s` is not a positive finite number,
            checked before the preconditions run.
    """
    if not work_dir.is_absolute():
        raise ValueError(f"work directory must be absolute, got {work_dir}")
    _check_timeout(timeout_s)
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
        out_dir = work_dir / f"source-{index:06d}"
        run = run_source(candidate, out_dir, protocol, timeout_s=timeout_s, command=command)
        runs.append(run)
        if isinstance(run.outcome, RunAbortOutcome):
            return CanonicalisationRun(run.outcome, tuple(runs))
    return CanonicalisationRun(None, tuple(runs))
