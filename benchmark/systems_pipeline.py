"""The systems stage's pipeline: the walk manifest, and each test recording's frames scored.

The walk writes `walk_manifest.json` (the walk-to-systems contract) and `audit_trail.jsonl`
beside it. This module reads the manifest with exact keys and types, refusing anything else, and
refuses it unless the audit trail beside it exists and hashes to the digest the manifest names.
Then, for each test admission of each complete stratum, it:

1. loads the canonical file, which re-checks its bytes against the recorded digest;
2. cuts the frames and takes the eligible ones from `frame_verdicts`, the function the walk's
   eligibility check uses;
3. takes the recording peak over the complete canonical waveform, including the trailing
   remainder that framing drops, and degrades each eligible frame at training level with its
   frozen seed (`frame_seed` of the protocol's degradation seed, the recording digest, and the
   frame's true index);
4. writes the degraded frame as a 16 kHz mono float WAV and calls each system on it through
   `call_system`;
5. scores each valid output against the clean reference, and scores the degraded input the
   same way as the v13 input baseline. The baseline is a diagnostic, not a system.

Before the frame loop, each system gets one unscored call on a fixed tone one admitted frame long
(`warm_up_all`), preceded by a progress marker that names the system and the warm-up phase.
That call is what loads the model, so a load fault aborts the run with the exception the call
raised, and the marker written just before it names the system; a wrong output format aborts it
as a harness failure that names the system. No exception is caught there. Only calls on admitted
frames are inside the engine boundary, where an exception is a system failure. The warm-up output is never scored or written
to a result file. After each system's warm-up, the caller's `after_warm_up` check runs; the
run uses it to compare the UVR model files again, since the warm-up is what loads them.

A system failure stays in the population and is recorded against that system. A harness
failure raises `HarnessFailure` and aborts the run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt

from benchmark.atomic_write import write_atomically
from benchmark.canonicalisation import CanonicalWaveform
from benchmark.eligibility import frame_floors
from benchmark.eligibility_outcomes import FrameVerdict, frame_verdicts
from benchmark.frame_scoring import (
    FrameOutcome,
    FrameScore,
    SystemFailure,
    degrade_with_reference,
    score_output,
)
from benchmark.framing import FramedWaveform, frame
from benchmark.protocol import Protocol
from benchmark.runner import CanonicalFile, CanonicalFileError
from benchmark.scoring import SYSTEMS, SystemFrameResult, coverage
from benchmark.systems_stage import (
    FrameIdentity,
    HarnessFailure,
    System,
    SystemCallResult,
    SystemFailed,
    call_system,
    read_output,
    write_frame_input,
)

__all__ = [
    "AUDIT_NAME",
    "BASELINE",
    "MANIFEST_NAME",
    "Admission",
    "AdmissionResult",
    "FailureKind",
    "FrameRecord",
    "Partition",
    "StageContext",
    "StratumEntry",
    "StratumStatus",
    "WalkManifest",
    "build_summary",
    "load_walk_manifest",
    "parse_walk_manifest",
    "population_to_json",
    "read_walk_manifest_bytes",
    "record_to_json",
    "score_admission",
    "score_manifest",
    "warm_up_all",
    "warm_up_system",
    "write_warm_up_input",
]

logger = logging.getLogger(__name__)

BASELINE = "degraded_input"
MANIFEST_NAME = "walk_manifest.json"
# The walk writes this trail beside the manifest; its digest is the manifest's audit_trail_sha256.
AUDIT_NAME = "audit_trail.jsonl"
WARM_UP_NAME = "warm_up.wav"
WARM_UP_PHASE = "warm_up"
_WARM_UP_TONES = ((440.0, 0.3), (1234.5, 0.2))
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TOP_KEYS = frozenset(
    {
        "protocol_sha256",
        "universe_sha256",
        "archive_sha256",
        "strata",
        "audit_trail_sha256",
        "admissions",
    }
)
_UNIVERSE_KEYS = frozenset({"solo_piano", "chamber"})
_STRATUM_KEYS = frozenset({"status", "admitted"})
_ADMISSION_KEYS = frozenset(
    {
        "stratum",
        "partition",
        "position",
        "identifier",
        "cluster",
        "canonical_path",
        "canonical_sha256",
        "samples",
        "raw_sha256",
        "decoded_pcm_sha256",
    }
)


class StratumStatus(StrEnum):
    """Whether a stratum's walk filled its partitions."""

    COMPLETE = "complete"
    SHORT = "short"


class Partition(StrEnum):
    """The partition an admitted recording was assigned to."""

    TEST = "test"
    DEMO = "demo"
    TEST_RESERVE = "test_reserve"


class FailureKind(StrEnum):
    """Where a system failure came from: its call, its output format, or conditioning."""

    EXCEPTION = "exception"
    OUTPUT_FORMAT = "output_format"
    CONDITIONING = "conditioning"


@dataclass(frozen=True)
class StratumEntry:
    """A stratum's walk status and the number of recordings it admitted."""

    status: StratumStatus
    admitted: int


@dataclass(frozen=True)
class Admission:
    """One admitted recording, as the walk manifest lists it."""

    stratum: str
    partition: Partition
    position: int
    identifier: str
    cluster: str
    canonical_path: Path
    canonical_sha256: str
    samples: int
    raw_sha256: str
    decoded_pcm_sha256: str


@dataclass(frozen=True)
class WalkManifest:
    """The walk manifest, validated: its digests, its strata, its admissions, and its trail."""

    protocol_sha256: str
    universe_sha256: Mapping[str, str]
    archive_sha256: str | None
    audit_trail_sha256: str
    strata: Mapping[str, StratumEntry]
    admissions: tuple[Admission, ...]


@dataclass(frozen=True)
class FrameRecord:
    """One system's outcome, or the baseline's, on one eligible frame of one recording."""

    admission: Admission
    frame_index: int
    system: str
    outcome: FrameOutcome
    failure_kind: FailureKind | None
    exception_type: str | None


@dataclass(frozen=True)
class AdmissionResult:
    """A scored recording: every candidate frame's verdict, the eligible indices, and the records.

    `verdicts` covers every complete frame, eligible or not, and `dropped_samples` is the trailing
    remainder that framing drops. Both are kept so the population is recorded, not inferred.
    """

    admission: Admission
    verdicts: tuple[FrameVerdict, ...]
    dropped_samples: int
    eligible: tuple[int, ...]
    records: tuple[FrameRecord, ...]


@dataclass(frozen=True)
class StageContext:
    """What scoring a recording needs: the protocol, the systems, and the run's paths."""

    protocol: Protocol
    systems: Sequence[System]
    input_dir: Path
    marker_path: Path


def _refuse(message: str) -> NoReturn:
    """Log why the walk manifest is refused, then raise the harness failure."""
    logger.error("Walk manifest refused: %s", message)
    raise HarnessFailure(f"walk manifest refused: {message}")


def _mapping(data: object, expected: frozenset[str], where: str) -> Mapping[str, Any]:
    """The object at `where`, which must be a dict with exactly the expected keys."""
    if not isinstance(data, dict) or set(data) != expected:
        _refuse(f"{where} must hold exactly the keys {sorted(expected)}")
    return data


def _int(value: object, where: str) -> int:
    """An integer that is not a bool."""
    if type(value) is not int:
        _refuse(f"{where} must be an integer")
    return value


def _sha256(value: object, where: str) -> str:
    """A lowercase hex SHA-256 digest."""
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        _refuse(f"{where} must be 64 lowercase hex digits")
    return value


def _text(value: object, where: str) -> str:
    """A string."""
    if not isinstance(value, str):
        _refuse(f"{where} must be a string")
    return value


def _stratum_entry(data: object, name: str) -> StratumEntry:
    """One stratum's status and admitted count, validated."""
    where = f"strata.{name}"
    fields = _mapping(data, _STRATUM_KEYS, where)
    status = _text(fields["status"], f"{where}.status")
    if status not in {member.value for member in StratumStatus}:
        _refuse(f"{where}.status is {status!r}")
    admitted = _int(fields["admitted"], f"{where}.admitted")
    if admitted < 0:
        _refuse(f"{where}.admitted is negative")
    return StratumEntry(StratumStatus(status), admitted)


def _admission(
    data: object, index: int, strata: Mapping[str, StratumEntry], walk_dir: Path
) -> Admission:
    """One admission, validated against the strata it must belong to."""
    where = f"admissions[{index}]"
    fields = _mapping(data, _ADMISSION_KEYS, where)
    stratum = _text(fields["stratum"], f"{where}.stratum")
    if stratum not in strata:
        _refuse(f"{where}.stratum {stratum!r} is not a stratum in the manifest")
    partition = _text(fields["partition"], f"{where}.partition")
    if partition not in {member.value for member in Partition}:
        _refuse(f"{where}.partition is {partition!r}")
    canonical_path = _text(fields["canonical_path"], f"{where}.canonical_path")
    if not Path(canonical_path).is_absolute():
        _refuse(f"{where}.canonical_path must be absolute")
    if not is_within_walk(Path(canonical_path), walk_dir):
        _refuse(f"{where}.canonical_path resolves outside the walk directory")
    samples = _int(fields["samples"], f"{where}.samples")
    if samples <= 0:
        _refuse(f"{where}.samples must be positive")
    return Admission(
        stratum=stratum,
        partition=Partition(partition),
        position=_int(fields["position"], f"{where}.position"),
        identifier=_text(fields["identifier"], f"{where}.identifier"),
        cluster=_text(fields["cluster"], f"{where}.cluster"),
        canonical_path=Path(canonical_path),
        canonical_sha256=_sha256(fields["canonical_sha256"], f"{where}.canonical_sha256"),
        samples=samples,
        raw_sha256=_sha256(fields["raw_sha256"], f"{where}.raw_sha256"),
        decoded_pcm_sha256=_sha256(fields["decoded_pcm_sha256"], f"{where}.decoded_pcm_sha256"),
    )


def _check_strata(strata: Mapping[str, StratumEntry], admissions: Sequence[Admission]) -> None:
    """Each stratum's admitted count equals its admissions, which are numbered 1 to that count."""
    for name, entry in strata.items():
        positions = [a.position for a in admissions if a.stratum == name]
        if len(positions) != entry.admitted:
            _refuse(f"strata.{name}.admitted is {entry.admitted} but {len(positions)} are listed")
        if positions != list(range(1, entry.admitted + 1)):
            _refuse(f"admissions of {name} are not numbered 1 to {entry.admitted} in order")


def _check_unique_recordings(admissions: Sequence[Admission]) -> None:
    """No two admissions share a canonical digest, since the digest names their frames."""
    digests = [a.canonical_sha256 for a in admissions]
    if len(set(digests)) != len(digests):
        _refuse("two admissions share a canonical_sha256, so their frames cannot be told apart")


def _refuse_constant(token: str) -> NoReturn:
    """Refuse NaN, Infinity and -Infinity, which a walk manifest does not allow."""
    _refuse(f"non-finite JSON value {token} is not allowed")


def is_within_walk(canonical_path: Path, walk_dir: Path) -> bool:
    """Whether a canonical file resolves to a location inside the walk directory.

    Both paths are resolved first, so a symbolic link or a `..` segment that leads out of the
    walk directory counts as outside it.
    """
    return canonical_path.resolve().is_relative_to(walk_dir.resolve())


def _decode(raw: bytes) -> Any:
    """The manifest's JSON value, or a harness failure if it is not UTF-8 JSON."""
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_refuse_constant)
    except ValueError as error:
        logger.error("Walk manifest is not UTF-8 JSON: %s", error)
        raise HarnessFailure(f"walk manifest is not UTF-8 JSON: {error}") from error


def _check_audit_trail(walk_dir: Path, expected: str) -> None:
    """Refuse unless the audit trail beside the manifest exists and hashes to `expected`.

    Raises:
        HarnessFailure: If the trail is missing, cannot be read, or hashes to another digest.
    """
    audit_path = walk_dir / AUDIT_NAME
    if not audit_path.is_file():
        _refuse(f"{AUDIT_NAME} is missing from {walk_dir}, so the manifest cannot be checked")
    try:
        actual = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    except OSError as error:
        logger.error("Cannot read audit trail %s: %s", audit_path, error)
        raise HarnessFailure(f"audit trail {audit_path} cannot be read: {error}") from error
    if actual != expected:
        _refuse(f"{AUDIT_NAME} hashes to {actual}, but the manifest names {expected}")


def read_walk_manifest_bytes(path: Path) -> bytes:
    """The manifest file's bytes, read once. Callers hash and parse these same bytes.

    Raises:
        HarnessFailure: If the file cannot be read, for example because it is a directory.
    """
    try:
        return path.read_bytes()
    except OSError as error:
        logger.error("Cannot read walk manifest %s: %s", path, error)
        raise HarnessFailure(f"walk manifest {path} cannot be read: {error}") from error


def parse_walk_manifest(raw: bytes, walk_dir: Path, protocol_sha256: str) -> WalkManifest:
    """Validate the bytes of a walk manifest, refusing anything the contract does not allow.

    The audit trail is read from `walk_dir`, so the manifest and its trail must sit together,
    as the walk writes them.

    Args:
        raw: The manifest's bytes, as `read_walk_manifest_bytes` returned them.
        walk_dir: The directory the manifest was read from, which holds the audit trail.
        protocol_sha256: The loaded protocol's source digest, which the manifest must name.

    Returns:
        The validated manifest.

    Raises:
        HarnessFailure: If the bytes are not UTF-8 JSON, have a shape other than the contract's,
            name another protocol, or have an audit trail that is missing or edited.
    """
    data = _mapping(_decode(raw), _TOP_KEYS, "manifest")
    if _sha256(data["protocol_sha256"], "protocol_sha256") != protocol_sha256:
        _refuse("protocol_sha256 names another protocol than the one loaded")
    universe = _mapping(data["universe_sha256"], _UNIVERSE_KEYS, "universe_sha256")
    universe_digests = {k: _sha256(v, f"universe_sha256.{k}") for k, v in universe.items()}
    archive = data["archive_sha256"]
    archive_digest = None if archive is None else _sha256(archive, "archive_sha256")
    audit_digest = _sha256(data["audit_trail_sha256"], "audit_trail_sha256")
    if not isinstance(data["strata"], dict):
        _refuse("strata must be an object")
    strata = {name: _stratum_entry(entry, name) for name, entry in data["strata"].items()}
    if set(strata) != set(universe_digests):
        _refuse("strata must name exactly the strata in universe_sha256")
    if not isinstance(data["admissions"], list):
        _refuse("admissions must be a list")
    admissions = [
        _admission(item, i, strata, walk_dir) for i, item in enumerate(data["admissions"])
    ]
    _check_strata(strata, admissions)
    _check_unique_recordings(admissions)
    _check_audit_trail(walk_dir, audit_digest)
    logger.info(
        "Walk manifest in %s: %d strata, %d admissions", walk_dir, len(strata), len(admissions)
    )
    return WalkManifest(
        protocol_sha256=protocol_sha256,
        universe_sha256=universe_digests,
        archive_sha256=archive_digest,
        audit_trail_sha256=audit_digest,
        strata=strata,
        admissions=tuple(admissions),
    )


def load_walk_manifest(path: Path, protocol_sha256: str) -> WalkManifest:
    """Read and validate a walk manifest file. See `parse_walk_manifest` for the checks.

    Args:
        path: The manifest file. Its directory holds the audit trail it is checked against.
        protocol_sha256: The loaded protocol's source digest, which the manifest must name.

    Returns:
        The validated manifest.

    Raises:
        HarnessFailure: If the file cannot be read or fails any check of `parse_walk_manifest`.
    """
    return parse_walk_manifest(read_walk_manifest_bytes(path), path.parent, protocol_sha256)


def _load_waveform(admission: Admission) -> CanonicalWaveform:
    """The canonical waveform of an admission, with its bytes re-checked against the digest."""
    canonical = CanonicalFile(
        path=admission.canonical_path,
        sha256=admission.canonical_sha256,
        samples=admission.samples,
        raw_sha256=admission.raw_sha256,
        decoded_pcm_sha256=admission.decoded_pcm_sha256,
    )
    try:
        samples = canonical.load()
    except CanonicalFileError as error:
        logger.error("Canonical file for %s failed its check: %s", admission.identifier, error)
        raise HarnessFailure(f"canonical file for {admission.identifier}: {error}") from error
    return CanonicalWaveform(samples=samples, sha256=admission.canonical_sha256)


def _verdicts(
    framed: FramedWaveform, protocol: Protocol, admission: Admission
) -> tuple[FrameVerdict, ...]:
    """Every candidate frame's verdict, from the same verdict function the walk uses.

    Raises:
        HarnessFailure: If no frame is eligible. The walk rejects such a recording, so an
            admission without one is inconsistent and the run aborts.
    """
    floors = frame_floors(framed, protocol.eligibility)
    verdicts = frame_verdicts(framed, floors, protocol.metrics)
    if not any(verdict.eligible for verdict in verdicts):
        logger.error("Admission %s has no eligible frame", admission.identifier)
        raise HarnessFailure(f"admission {admission.identifier} has no eligible frame")
    return verdicts


def _baseline_record(
    admission: Admission,
    index: int,
    reference: npt.NDArray[np.float64],
    degraded: npt.NDArray[np.float32],
    protocol: Protocol,
) -> FrameRecord:
    """The degraded input scored like a system output. It is not a system and has no coverage.

    Raises:
        HarnessFailure: If conditioning rejects the input, which cannot happen for a same-length
            finite input, so a rejection means the harness is inconsistent.
    """
    outcome = score_output(reference, degraded.astype(np.float64), protocol)
    if not isinstance(outcome, FrameScore):
        logger.error("Degraded input of %s frame %d was rejected", admission.identifier, index)
        raise HarnessFailure(f"degraded input of {admission.identifier} frame {index} rejected")
    return FrameRecord(admission, index, BASELINE, outcome, None, None)


def _system_record(
    admission: Admission,
    index: int,
    system: str,
    result: SystemCallResult,
    reference: npt.NDArray[np.float64],
    protocol: Protocol,
) -> FrameRecord:
    """A system's outcome on one frame: its failure, or its score."""
    if isinstance(result, SystemFailed):
        kind = FailureKind.EXCEPTION if result.exception_type else FailureKind.OUTPUT_FORMAT
        failure = SystemFailure(result.reason)
        return FrameRecord(admission, index, system, failure, kind, result.exception_type)
    outcome = score_output(reference, result.samples, protocol)
    if isinstance(outcome, SystemFailure):
        return FrameRecord(admission, index, system, outcome, FailureKind.CONDITIONING, None)
    return FrameRecord(admission, index, system, outcome, None, None)


def _score_frame(
    admission: Admission,
    clean: npt.NDArray[np.float64],
    index: int,
    peak: float,
    context: StageContext,
) -> list[FrameRecord]:
    """Degrade one eligible frame, call each system on it, and score every outcome."""
    protocol = context.protocol
    rate = protocol.canonicalisation.resample.target_sample_rate
    degraded, reference = degrade_with_reference(
        clean, peak, admission.canonical_sha256, index, protocol
    )
    input_path = context.input_dir / f"{admission.canonical_sha256}-{index:06d}.wav"
    write_frame_input(degraded.degraded, input_path, rate)
    records = [_baseline_record(admission, index, reference, degraded.degraded, protocol)]
    identity = FrameIdentity(admission.canonical_sha256, index)
    for system in context.systems:
        result = call_system(
            system,
            input_path,
            identity=identity,
            marker_path=context.marker_path,
            sample_rate=protocol.conditioning.sample_rate,
        )
        records.append(_system_record(admission, index, system.name, result, reference, protocol))
    return records


def score_admission(admission: Admission, context: StageContext) -> AdmissionResult:
    """Score every eligible frame of one test recording, for every system and the baseline.

    Args:
        admission: The recording, from the walk manifest.
        context: The protocol, the systems, and the run's paths.

    Returns:
        Every frame's verdict, the dropped sample count, the eligible frame indices, and one
        record per eligible frame and system, baseline included.

    Raises:
        HarnessFailure: On a canonical file that fails its check, a recording with no eligible
            frame, or any failure outside a system's call.
    """
    waveform = _load_waveform(admission)
    framed = frame(waveform, context.protocol.framing)
    verdicts = _verdicts(framed, context.protocol, admission)
    eligible = [verdict for verdict in verdicts if verdict.eligible]
    peak = float(np.max(np.abs(waveform.samples)))
    records: list[FrameRecord] = []
    for verdict in eligible:
        records.extend(
            _score_frame(admission, framed.frames[verdict.index], verdict.index, peak, context)
        )
    logger.info(
        "Scored %s: %d eligible frames of %d",
        admission.identifier,
        len(eligible),
        len(framed.frames),
    )
    return AdmissionResult(
        admission=admission,
        verdicts=verdicts,
        dropped_samples=framed.dropped_samples,
        eligible=tuple(verdict.index for verdict in eligible),
        records=tuple(records),
    )


def _scored_admissions(manifest: WalkManifest) -> list[Admission]:
    """The test admissions of the complete strata, in walk order.

    Raises:
        HarnessFailure: If a complete stratum has no test admission, which the walk's
            partitions rule out.
    """
    complete = {n for n, e in manifest.strata.items() if e.status is StratumStatus.COMPLETE}
    for name in complete:
        if not any(
            a.stratum == name and a.partition is Partition.TEST for a in manifest.admissions
        ):
            logger.error("Complete stratum %s has no test admission", name)
            raise HarnessFailure(f"complete stratum {name} has no test admission")
    return [
        a for a in manifest.admissions if a.stratum in complete and a.partition is Partition.TEST
    ]


def score_manifest(manifest: WalkManifest, context: StageContext) -> list[AdmissionResult]:
    """Score every test admission of every complete stratum, in walk order."""
    results = []
    for admission in _scored_admissions(manifest):
        logger.info("Scoring recording %s of %s", admission.identifier, admission.stratum)
        results.append(score_admission(admission, context))
    return results


def _identity_json(admission: Admission) -> dict[str, object]:
    """The fields that identify a recording in every output row."""
    return {
        "stratum": admission.stratum,
        "partition": admission.partition.value,
        "position": admission.position,
        "identifier": admission.identifier,
        "canonical_sha256": admission.canonical_sha256,
    }


def _frame_json(verdict: FrameVerdict) -> dict[str, object]:
    """One candidate frame's RMS, its reasons, and the primary metrics that refuse it.

    A frame with zero energy has an RMS of minus infinity, which JSON cannot hold, so that RMS
    is written as null.
    """
    finite = math.isfinite(verdict.rms_dbfs)
    return {
        "index": verdict.index,
        "rms_dbfs": verdict.rms_dbfs if finite else None,
        "reasons": [reason.value for reason in verdict.reasons],
        "unscoreable_by": list(verdict.unscoreable_by),
    }


def population_to_json(result: AdmissionResult) -> dict[str, object]:
    """One recording's population: every candidate frame's verdict and the samples dropped.

    The eligible frames are listed separately, so a reader can check the scored population
    against every candidate frame rather than trusting the scored rows alone.
    """
    row = _identity_json(result.admission)
    row.update(
        frame_count=len(result.verdicts),
        dropped_samples=result.dropped_samples,
        eligible_frames=list(result.eligible),
        frames=[_frame_json(verdict) for verdict in result.verdicts],
    )
    return row


def write_warm_up_input(path: Path, *, samples: int, sample_rate: int) -> None:
    """Write the warm-up tone: a fixed 440 Hz and 1234.5 Hz mix, as a mono float WAV.

    Args:
        path: Where the WAV is written.
        samples: The length in samples. The run passes one admitted frame's length.
        sample_rate: The rate, in Hz.

    Raises:
        HarnessFailure: If the file cannot be written.
    """
    time = np.arange(samples) / sample_rate
    tone = np.zeros(samples, dtype=np.float64)
    for frequency, amplitude in _WARM_UP_TONES:
        tone += amplitude * np.sin(2 * np.pi * frequency * time)
    write_frame_input(tone.astype(np.float32), path, sample_rate)


def _write_warm_up_marker(path: Path, system: str) -> None:
    """Write the progress marker for a warm-up call, atomically, before the call is made.

    Raises:
        HarnessFailure: If the marker cannot be written, so the call is not made.
    """
    payload = {"system": system, "phase": WARM_UP_PHASE}
    try:
        write_atomically(path, json.dumps(payload, sort_keys=True).encode("utf-8"))
    except OSError as error:
        logger.error("Cannot write warm-up marker %s: %s", path, error)
        raise HarnessFailure(f"warm-up marker {path} could not be written: {error}") from error


def warm_up_system(
    system: System, warm_up_input: Path, marker_path: Path, sample_rate: int
) -> None:
    """Make one unscored call so the system loads its model before any frame is scored.

    The marker is written before the call, as for every denoise call. The call is outside the
    engine boundary, so there is no frame to charge a failure to. An exception the call raises is
    not caught: it propagates and aborts the run, and the marker written before the call names
    the system. An output that is not mono at `sample_rate` aborts the run as a harness failure.

    Args:
        system: The system to warm up.
        warm_up_input: The warm-up WAV, already written.
        marker_path: Where the progress marker is written before the call.
        sample_rate: The rate the output must have, in Hz.

    Raises:
        HarnessFailure: If the marker cannot be written, or the output is not mono at
            `sample_rate`.
        Exception: Whatever the warm-up call raises, unchanged.
    """
    _write_warm_up_marker(marker_path, system.name)
    logger.info("Warming up %s", system.name)
    output = system.denoise(warm_up_input)
    outcome = read_output(output, sample_rate)
    if isinstance(outcome, SystemFailed):
        logger.error("Warm-up of %s produced the wrong format: %s", system.name, outcome.reason)
        message = f"warm-up of system {system.name} produced the wrong format: {outcome.reason}"
        raise HarnessFailure(message)
    logger.info("Warm-up of %s succeeded", system.name)


def warm_up_all(
    systems: Sequence[System],
    run_dir: Path,
    protocol: Protocol,
    marker_path: Path,
    *,
    after_warm_up: Callable[[], None] | None = None,
) -> None:
    """Write the warm-up input into the run directory and warm up every system on it, in order.

    Args:
        systems: The systems, after they are built and before any frame is processed.
        run_dir: The run directory, which receives the warm-up WAV.
        protocol: The loaded protocol, for the frame length and the rates.
        marker_path: The progress marker file, written before each warm-up call.
        after_warm_up: Called once after each system's warm-up succeeds, before the next system
            is warmed up. Any exception it raises propagates and aborts the run.

    Raises:
        HarnessFailure: If any system fails its warm-up, or a marker cannot be written.
    """
    path = run_dir / WARM_UP_NAME
    write_warm_up_input(
        path,
        samples=protocol.framing.frame_samples,
        sample_rate=protocol.canonicalisation.resample.target_sample_rate,
    )
    for system in systems:
        warm_up_system(system, path, marker_path, protocol.conditioning.sample_rate)
        if after_warm_up is not None:
            after_warm_up()


def record_to_json(record: FrameRecord) -> dict[str, object]:
    """One record as a JSON-ready mapping. Field names follow the protocol's vocabulary."""
    admission = record.admission
    outcome = record.outcome
    row: dict[str, object] = {
        **_identity_json(admission),
        "frame_index": record.frame_index,
        "system": record.system,
        "outcome": "scored" if isinstance(outcome, FrameScore) else "system_failure",
        "failure_kind": None if record.failure_kind is None else record.failure_kind.value,
        "exception_type": record.exception_type,
        "failure_reason": outcome.reason if isinstance(outcome, SystemFailure) else None,
    }
    if isinstance(outcome, FrameScore):
        row["results"] = {
            name: {"value": result.value, "capped": result.capped}
            for name, result in outcome.results.items()
        }
        row["shift_tolerant"] = {
            name: {"value": result.value, "capped": result.capped}
            for name, result in outcome.shift_tolerant.items()
        }
        row["reference_samples_discarded"] = outcome.reference_samples_discarded
        row["estimate_samples_discarded"] = outcome.estimate_samples_discarded
    return row


def _stratum_summary(
    admissions: Sequence[AdmissionResult], records: Sequence[FrameRecord]
) -> dict[str, object]:
    """One complete stratum's coverage per system, and its failure counts by kind."""
    expected = {a.admission.canonical_sha256: a.eligible for a in admissions}
    coverage_by_system: dict[str, float] = {}
    failures: dict[str, dict[str, int]] = {}
    for system in SYSTEMS:
        own = [r for r in records if r.system == system]
        results = [
            SystemFrameResult(r.admission.canonical_sha256, r.frame_index, r.outcome) for r in own
        ]
        coverage_by_system[system] = coverage(expected, results)
        counts = Counter(r.failure_kind.value for r in own if r.failure_kind is not None)
        failures[system] = dict(sorted(counts.items()))
    return {
        "status": StratumStatus.COMPLETE.value,
        "scored": True,
        "test_recordings": len(admissions),
        "eligible_frames": sum(len(a.eligible) for a in admissions),
        "coverage": coverage_by_system,
        "failure_kinds": failures,
    }


def build_summary(
    manifest: WalkManifest,
    manifest_sha256: str,
    results: Sequence[AdmissionResult],
    *,
    results_sha256: str,
    population_sha256: str,
) -> dict[str, object]:
    """The stage's summary: per stratum, coverage per system, or a note that it was not scored.

    Args:
        manifest: The validated walk manifest the run scored.
        manifest_sha256: The SHA-256 of the manifest bytes the run parsed.
        results: Every scored recording.
        results_sha256: The SHA-256 of the exact bytes of `systems_results.jsonl` as written.
        population_sha256: The SHA-256 of the exact bytes of `systems_population.jsonl`.
    """
    strata: dict[str, object] = {}
    for name, entry in manifest.strata.items():
        if entry.status is not StratumStatus.COMPLETE:
            strata[name] = {"status": entry.status.value, "scored": False}
            continue
        own = [r for r in results if r.admission.stratum == name]
        records = [record for result in own for record in result.records]
        strata[name] = _stratum_summary(own, records)
    return {
        "protocol_sha256": manifest.protocol_sha256,
        "walk_manifest_sha256": manifest_sha256,
        "systems_results_sha256": results_sha256,
        "systems_population_sha256": population_sha256,
        "systems": list(SYSTEMS),
        "baseline": BASELINE,
        "strata": strata,
    }
