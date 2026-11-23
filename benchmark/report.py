"""Publication gates and the paired comparison, per stratum, read from the walk and systems outputs.

The report reads a finished walk directory and a finished systems directory, and writes nothing.
Protocol v13 reports each stratum on its own and never pools them, so this module evaluates the
seven `publication_gates.required` gates per stratum: each gate is decided from one stratum's
walk and systems outputs. That per-stratum evaluation is a reading of v12's per-stratum
reporting, made here because the protocol names the gates without naming the unit they apply to.

The expected population is recomputed here from the canonical files, never taken from
`systems_population.jsonl`. That file is compared with the recomputation, and a mismatch fails the
frames gate.

Each stratum ends in one of three outcomes:

- comparison: all seven gates hold. The paired comparison is published.
- reliability finding: one or both coverage gates fail, and no other gate does. Per-system coverage
  and failure counts by kind are published, and no metric value is. A per-system mean over the
  frames that survived would reward failure, which the protocol forbids.
- withheld: a stratum is short, or any gate other than a coverage gate fails. Only the reasons are
  published.

Input integrity is checked before any stratum is evaluated. The systems summary must record the
SHA-256 of the exact bytes of `systems_results.jsonl`, `systems_population.jsonl` and
`walk_manifest.json` that the report read. A missing key or a differing digest means the report is
not reading the outputs the systems stage produced, so every stratum is withheld with the reasons.
No gate is attributed to this check, because the protocol's gates describe a stratum's evidence,
and this check concerns the run's inputs. The protocol digest is not part of it: gate a covers the
protocol.

The walk manifest is read by this module's own reader, not the systems stage's. The walk writes
`audit_trail_sha256` from PR #30 on, and the report requires it: without the key the training
screen gate fails, so the stratum is withheld. The screen gate also checks every manifest
admission against the walk's own record. The partition must equal the walk's partition plan at the
admission's position, the cluster must equal the one the pinned universe gives for the identifier,
and the admitted audit line must carry the same digests, samples, partition and position. The
protocol names the gates without naming these checks, so they are part of the screen gate here
rather than an eighth gate. A stratum with no pinned digest in `UNIVERSE_SHA256` has no universe to
check its clusters against. The universe gate already withholds that stratum, so the cluster check
is skipped for it and adds no reason.

A canonical file that fails its check during the report aborts the whole report with `ReportError`.
A failed check means the recorded evidence no longer matches the file on disk, so no stratum is
published from that run. A short stratum's canonical files are not read, so they are not checked.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from benchmark.aggregation import (
    AggregationError,
    BaselineSummary,
    MetricSummary,
    summarise,
)
from benchmark.candidate_universe import (
    UNIVERSE_SHA256,
    CandidateUniverseError,
    load_universe,
)
from benchmark.canonicalisation import CanonicalWaveform
from benchmark.eligibility import frame_floors
from benchmark.eligibility_outcomes import frame_verdicts
from benchmark.frame_scoring import FrameOutcome, FrameScore, SystemFailure
from benchmark.framing import frame
from benchmark.metrics import MetricResult
from benchmark.protocol import Protocol
from benchmark.runner import CanonicalFile, CanonicalFileError
from benchmark.scoring import (
    BASELINE,
    OPGAN,
    SYSTEMS,
    UVR,
    PopulationError,
    SystemFrameResult,
    coverage,
    population_gates,
    recording_scores,
)
from benchmark.systems_pipeline import (
    Admission,
    AdmissionResult,
    Partition,
    StratumEntry,
    StratumStatus,
    population_to_json,
)
from benchmark.walk import partition_plan

__all__ = [
    "AUDIT_NAME",
    "MANIFEST_NAME",
    "POPULATION_NAME",
    "RESULTS_NAME",
    "SUMMARY_NAME",
    "Counts",
    "Digests",
    "Gate",
    "Outcome",
    "ReportError",
    "RunReport",
    "StratumReport",
    "build_report",
]

logger = logging.getLogger(__name__)

MANIFEST_NAME = "walk_manifest.json"
AUDIT_NAME = "audit_trail.jsonl"
SUMMARY_NAME = "systems_summary.json"
RESULTS_NAME = "systems_results.jsonl"
POPULATION_NAME = "systems_population.jsonl"
AUDIT_KEY = "audit_trail_sha256"
RESULTS_KEY = "systems_results_sha256"
POPULATION_KEY = "systems_population_sha256"
MANIFEST_KEY = "walk_manifest_sha256"

# The systems stage writes these outcome and failure values (systems_pipeline.record_to_json).
_SCORED = "scored"
_SYSTEM_FAILURE = "system_failure"
# The walk writes these outcome values to each audit line (walk.Admitted and AwaitingAdjudication).
_ADMITTED = "admitted"
_AWAITING_ADJUDICATION = "awaiting_adjudication"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_REQUIRED_MANIFEST_KEYS = frozenset(
    {"protocol_sha256", "universe_sha256", "archive_sha256", "strata", "admissions"}
)
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
_STRATUM_KEYS = frozenset({"status", "admitted"})


class ReportError(Exception):
    """Raised when an input file is missing, unreadable, or not in the shape the report reads."""


class Gate(StrEnum):
    """The seven publication gates, by their protocol names."""

    PROTOCOL = "loaded_protocol_hash_matches_artifact"
    UNIVERSE = "candidate_universe_hash_verified"
    SCREEN = "training_performance_screen_complete"
    RECORDINGS = "population_ids_equal_across_systems"
    FRAMES = "frame_population_ids_equal_across_systems"
    COVERAGE_OPGAN = "coverage_opgan_complete"
    COVERAGE_UVR = "coverage_uvr_complete"


_COVERAGE_GATES = frozenset({Gate.COVERAGE_OPGAN, Gate.COVERAGE_UVR})
_COVERAGE_BY_SYSTEM = {OPGAN: Gate.COVERAGE_OPGAN, UVR: Gate.COVERAGE_UVR}


class Outcome(StrEnum):
    """What a stratum publishes."""

    COMPARISON = "comparison"
    RELIABILITY_FINDING = "reliability_finding"
    WITHHELD = "withheld"


@dataclass(frozen=True)
class Counts:
    """A stratum's test recordings, their performer clusters, and their eligible frames."""

    recordings: int
    clusters: int
    eligible_frames: int


@dataclass(frozen=True)
class StratumReport:
    """One stratum's outcome. Numbers are present only for a comparison or a reliability finding.

    `headline` holds the three co-primary metrics in protocol order, or nothing. `diagnostic` and
    `baseline` are never headlines.
    """

    stratum: str
    outcome: Outcome
    gates: Mapping[str, bool]
    reasons: tuple[str, ...]
    counts: Counts | None = None
    coverage: Mapping[str, float] | None = None
    failure_counts: Mapping[str, Mapping[str, int]] | None = None
    headline: tuple[MetricSummary, ...] = ()
    diagnostic: tuple[MetricSummary, ...] = ()
    baseline: tuple[BaselineSummary, ...] = ()


@dataclass(frozen=True)
class Digests:
    """The digests the report names: the protocol, both universes, the archive, and three files."""

    protocol_sha256: str
    universe_sha256: Mapping[str, str | None]
    archive_sha256: str | None
    walk_manifest_sha256: str
    audit_trail_sha256: str
    systems_results_sha256: str


@dataclass(frozen=True)
class RunReport:
    """The whole report: the protocol's claim scope, the digests, and one entry per stratum."""

    protocol_version: int
    claim_scope_note: str
    digests: Digests
    strata: tuple[StratumReport, ...]


@dataclass(frozen=True)
class Manifest:
    """The walk manifest, validated by this module's reader, with the digest of its file."""

    protocol_sha256: str
    universe_sha256: Mapping[str, str]
    archive_sha256: str | None
    strata: Mapping[str, StratumEntry]
    admissions: tuple[Admission, ...]
    audit_trail_sha256: str | None
    file_sha256: str


@dataclass(frozen=True)
class AuditTrail:
    """The audit trail's digest and its lines. A line that is not a JSON object is None."""

    sha256: str
    lines: tuple[dict[str, Any] | None, ...]


@dataclass(frozen=True)
class Inputs:
    """Everything one report reads: the walk, the systems outputs, and the protocol."""

    protocol: Protocol
    manifest: Manifest
    audit: AuditTrail
    summary: Mapping[str, Any]
    results: tuple[dict[str, Any], ...]
    population: tuple[dict[str, Any], ...]
    results_sha256: str
    population_sha256: str


@dataclass(frozen=True)
class Population:
    """One stratum's recomputed population, its system results, and the four population verdicts.

    A verdict is None when its gate holds and the reason otherwise.
    """

    expected: Mapping[str, Sequence[int]]
    clusters: Mapping[str, str]
    results: Mapping[str, Sequence[SystemFrameResult]]
    failure_counts: Mapping[str, Mapping[str, int]]
    verdicts: Mapping[Gate, str | None]


def _read_bytes(path: Path) -> bytes:
    """A file's bytes, or a report error naming the file."""
    try:
        return path.read_bytes()
    except OSError as error:
        logger.error("Cannot read %s: %s", path, error)
        raise ReportError(f"{path} cannot be read: {error}") from error


def _digest_of(data: bytes) -> str:
    """The SHA-256 of some bytes, as lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def _json_object(data: bytes, where: str) -> dict[str, Any]:
    """The JSON object in `data`, or a report error."""
    try:
        value = json.loads(data.decode("utf-8"))
    except ValueError as error:
        logger.error("%s is not UTF-8 JSON: %s", where, error)
        raise ReportError(f"{where} is not UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ReportError(f"{where} must be a JSON object")
    return value


def _exact(data: object, keys: frozenset[str], where: str) -> dict[str, Any]:
    """An object with exactly the given keys."""
    if not isinstance(data, dict) or set(data) != keys:
        raise ReportError(f"{where} must hold exactly the keys {sorted(keys)}")
    return data


def _sha256(value: object, where: str) -> str:
    """A lowercase hex SHA-256 digest."""
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ReportError(f"{where} must be 64 lowercase hex digits")
    return value


def _text(value: object, where: str) -> str:
    """A string."""
    if not isinstance(value, str):
        raise ReportError(f"{where} must be a string")
    return value


def _integer(value: object, where: str) -> int:
    """An integer that is not a bool."""
    if type(value) is not int:
        raise ReportError(f"{where} must be an integer")
    return value


def _object(data: object, where: str) -> dict[str, Any]:
    """A JSON object, or a report error naming where it was expected."""
    if not isinstance(data, dict):
        raise ReportError(f"{where} must be an object")
    return data


def _stratum_entry(data: object, where: str) -> StratumEntry:
    """One stratum's status and admitted count."""
    fields = _exact(data, _STRATUM_KEYS, where)
    try:
        status = StratumStatus(_text(fields["status"], f"{where}.status"))
    except ValueError as error:
        raise ReportError(f"{where}.status is not a status the walk writes") from error
    return StratumEntry(status, _integer(fields["admitted"], f"{where}.admitted"))


def _admission(data: object, where: str, strata: Mapping[str, StratumEntry]) -> Admission:
    """One admission, checked against the strata it must belong to."""
    fields = _exact(data, _ADMISSION_KEYS, where)
    stratum = _text(fields["stratum"], f"{where}.stratum")
    if stratum not in strata:
        raise ReportError(f"{where}.stratum names a stratum the manifest does not list")
    partition = _text(fields["partition"], f"{where}.partition")
    try:
        parsed = Partition(partition)
    except ValueError as error:
        raise ReportError(f"{where}.partition is not a partition the walk writes") from error
    canonical_path = _text(fields["canonical_path"], f"{where}.canonical_path")
    if not Path(canonical_path).is_absolute():
        raise ReportError(f"{where}.canonical_path must be absolute")
    return Admission(
        stratum=stratum,
        partition=parsed,
        position=_integer(fields["position"], f"{where}.position"),
        identifier=_text(fields["identifier"], f"{where}.identifier"),
        cluster=_text(fields["cluster"], f"{where}.cluster"),
        canonical_path=Path(canonical_path),
        canonical_sha256=_sha256(fields["canonical_sha256"], f"{where}.canonical_sha256"),
        samples=_integer(fields["samples"], f"{where}.samples"),
        raw_sha256=_sha256(fields["raw_sha256"], f"{where}.raw_sha256"),
        decoded_pcm_sha256=_sha256(fields["decoded_pcm_sha256"], f"{where}.decoded_pcm_sha256"),
    )


def _check_numbering(admissions: Sequence[Admission], strata: Mapping[str, StratumEntry]) -> None:
    """Each stratum's admitted count equals its admissions, numbered 1 to that count in order."""
    for name, entry in strata.items():
        positions = [a.position for a in admissions if a.stratum == name]
        if positions != list(range(1, entry.admitted + 1)):
            raise ReportError(f"the admissions of {name} do not match its admitted count")


def _read_manifest(path: Path) -> Manifest:
    """The walk manifest, read by this module's own reader.

    Raises:
        ReportError: If the file cannot be read or does not have the walk's shape. The optional
            audit digest key is accepted, and the screen gate decides whether it matches.
    """
    data = _read_bytes(path)
    top = _json_object(data, str(path))
    keys = set(top)
    if not _REQUIRED_MANIFEST_KEYS <= keys <= _REQUIRED_MANIFEST_KEYS | {AUDIT_KEY}:
        raise ReportError(f"{path} must hold the walk manifest's keys, and optionally {AUDIT_KEY}")
    strata_data = _object(top["strata"], "strata")
    strata = {name: _stratum_entry(entry, f"strata.{name}") for name, entry in strata_data.items()}
    universe = _object(top["universe_sha256"], "universe_sha256")
    if set(universe) != set(strata):
        raise ReportError("the universe digests must name exactly the manifest's strata")
    archive = top["archive_sha256"]
    admissions_data = top["admissions"]
    if not isinstance(admissions_data, list):
        raise ReportError("admissions must be a list")
    admissions = tuple(
        _admission(item, f"admissions[{i}]", strata) for i, item in enumerate(admissions_data)
    )
    _check_numbering(admissions, strata)
    audit_digest = top.get(AUDIT_KEY)
    return Manifest(
        protocol_sha256=_sha256(top["protocol_sha256"], "protocol_sha256"),
        universe_sha256={
            name: _sha256(v, f"universe_sha256.{name}") for name, v in universe.items()
        },
        archive_sha256=None if archive is None else _sha256(archive, "archive_sha256"),
        strata=strata,
        admissions=admissions,
        audit_trail_sha256=None if audit_digest is None else _sha256(audit_digest, AUDIT_KEY),
        file_sha256=_digest_of(data),
    )


def _audit_line(raw: bytes) -> dict[str, Any] | None:
    """One audit line as an object, or None if it is not one."""
    try:
        value = json.loads(raw.decode("utf-8"))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _read_audit(path: Path) -> AuditTrail:
    """The audit trail's digest and lines. A malformed line is kept as None for the screen gate."""
    data = _read_bytes(path)
    lines = tuple(_audit_line(raw) for raw in data.splitlines())
    return AuditTrail(_digest_of(data), lines)


def _read_summary(path: Path) -> dict[str, Any]:
    """The systems summary: it must name a protocol digest and list its strata."""
    summary = _json_object(_read_bytes(path), str(path))
    if "protocol_sha256" not in summary or "strata" not in summary:
        raise ReportError(f"{path} must name protocol_sha256 and strata")
    _sha256(summary["protocol_sha256"], "systems summary protocol_sha256")
    return summary


def _read_rows(path: Path) -> tuple[tuple[dict[str, Any], ...], str]:
    """A JSON-lines file's rows, each an object, and the digest of the file."""
    data = _read_bytes(path)
    rows = tuple(
        _json_object(raw, f"{path} line {number}")
        for number, raw in enumerate(data.splitlines(), start=1)
    )
    return rows, _digest_of(data)


def read_inputs(walk_dir: Path, systems_dir: Path, protocol: Protocol) -> Inputs:
    """Read the walk and systems outputs the report evaluates.

    Raises:
        ReportError: If a file is missing, unreadable, or not in the shape the report reads.
    """
    results, results_sha256 = _read_rows(systems_dir / RESULTS_NAME)
    population, population_sha256 = _read_rows(systems_dir / POPULATION_NAME)
    return Inputs(
        protocol=protocol,
        manifest=_read_manifest(walk_dir / MANIFEST_NAME),
        audit=_read_audit(walk_dir / AUDIT_NAME),
        summary=_read_summary(systems_dir / SUMMARY_NAME),
        results=results,
        population=population,
        results_sha256=results_sha256,
        population_sha256=population_sha256,
    )


def _metric_results(table: object, where: str) -> dict[str, MetricResult]:
    """A frame's metric results as the systems stage writes them: value and capped flag."""
    entries = _object(table, where)
    return {
        str(name): MetricResult(float(entry["value"]), bool(entry["capped"]))
        for name, entry in entries.items()
    }


def _frame_outcome(row: dict[str, Any]) -> FrameOutcome:
    """A scored frame, or a system failure, as the systems stage wrote it.

    Raises:
        ValueError: If the outcome is neither of the two the systems stage writes.
    """
    if row["outcome"] == _SCORED:
        return FrameScore(
            _metric_results(row["results"], "results"),
            int(row["reference_samples_discarded"]),
            int(row["estimate_samples_discarded"]),
            _metric_results(row["shift_tolerant"], "shift_tolerant"),
        )
    if row["outcome"] == _SYSTEM_FAILURE:
        return SystemFailure(str(row["failure_reason"]))
    raise ValueError(f"unknown outcome {row['outcome']!r}")


def _system_result(row: dict[str, Any]) -> SystemFrameResult:
    """One systems-results row as the scoring module's frame result.

    Raises:
        ReportError: If the row is not in the shape the systems stage writes.
    """
    try:
        return SystemFrameResult(
            str(row["canonical_sha256"]), int(row["frame_index"]), _frame_outcome(row)
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReportError("a systems result row is not in the shape the stage writes") from error


def _system_results(
    stratum: str, rows: Sequence[dict[str, Any]]
) -> tuple[dict[str, list[SystemFrameResult]], dict[str, dict[str, int]]]:
    """One stratum's results per system and baseline, and each system's failures by kind."""
    results: dict[str, list[SystemFrameResult]] = {name: [] for name in (*SYSTEMS, BASELINE)}
    failures: dict[str, Counter[str]] = {name: Counter() for name in SYSTEMS}
    for row in rows:
        if row.get("stratum") != stratum:
            continue
        system = row.get("system")
        if system not in results:
            raise ReportError(f"the systems results name an unknown system {system!r}")
        results[system].append(_system_result(row))
        if row.get("outcome") == _SYSTEM_FAILURE and system in failures:
            failures[system][str(row.get("failure_kind"))] += 1
    counts = {system: dict(sorted(found.items())) for system, found in failures.items()}
    return results, counts


def _test_admissions(stratum: str, manifest: Manifest) -> tuple[Admission, ...]:
    """The test admissions of one stratum, in the manifest's order, which is admission order."""
    return tuple(
        a for a in manifest.admissions if a.stratum == stratum and a.partition is Partition.TEST
    )


def _recompute(admission: Admission, protocol: Protocol) -> AdmissionResult:
    """One recording's population, recomputed from its canonical file with the walk's functions.

    Raises:
        CanonicalFileError: If the canonical file fails its check.
    """
    canonical = CanonicalFile(
        path=admission.canonical_path,
        sha256=admission.canonical_sha256,
        samples=admission.samples,
        raw_sha256=admission.raw_sha256,
        decoded_pcm_sha256=admission.decoded_pcm_sha256,
    )
    samples = canonical.load()
    framed = frame(
        CanonicalWaveform(samples=samples, sha256=admission.canonical_sha256), protocol.framing
    )
    verdicts = frame_verdicts(framed, frame_floors(framed, protocol.eligibility), protocol.metrics)
    eligible = tuple(verdict.index for verdict in verdicts if verdict.eligible)
    return AdmissionResult(admission, verdicts, framed.dropped_samples, eligible, records=())


def _population_rows_match(
    stratum: str, recomputed: Sequence[AdmissionResult], rows: Sequence[dict[str, Any]]
) -> bool:
    """Whether systems_population.jsonl lists exactly the recomputed population, row for row."""
    expected = {r.admission.canonical_sha256: population_to_json(r) for r in recomputed}
    listed = [row for row in rows if row.get("stratum") == stratum]
    actual = {row.get("canonical_sha256"): row for row in listed}
    return len(actual) == len(listed) and actual == expected


def _verdict(held: bool, reason: str) -> str | None:
    """None when a gate holds, otherwise the reason it does not."""
    return None if held else reason


def _unevaluable(reason: str) -> Population:
    """The population of a stratum whose frames cannot be recomputed: every population gate fails."""
    verdicts: dict[Gate, str | None] = {
        Gate.RECORDINGS: reason,
        Gate.FRAMES: reason,
        Gate.COVERAGE_OPGAN: reason,
        Gate.COVERAGE_UVR: reason,
    }
    return Population({}, {}, {}, {}, verdicts)


def _frames_verdict(frame_ids_held: bool, rows_match: bool) -> str | None:
    """The frames gate: the frame identifiers agree, and the population file agrees."""
    if not frame_ids_held:
        return "the frame identifiers differ across systems or from the recomputed eligible frames"
    return _verdict(rows_match, "systems_population.jsonl differs from the recomputed population")


def _population(stratum: str, inputs: Inputs) -> Population:
    """One stratum's population: recomputed, checked against the systems results and the file."""
    admissions = _test_admissions(stratum, inputs.manifest)
    if not admissions:
        return _unevaluable("the walk lists no test recording for this stratum")
    try:
        recomputed = [_recompute(a, inputs.protocol) for a in admissions]
    except CanonicalFileError as error:
        logger.error("Canonical file of stratum %s failed its check: %s", stratum, error)
        raise ReportError(
            f"a canonical file of stratum {stratum} failed its check: {error}"
        ) from error
    if any(not result.eligible for result in recomputed):
        return _unevaluable("a test recording has no eligible frame in the recomputed population")
    expected = {r.admission.canonical_sha256: r.eligible for r in recomputed}
    clusters = {a.canonical_sha256: a.cluster for a in admissions}
    results, failure_counts = _system_results(stratum, inputs.results)
    gates = population_gates(expected, results, inputs.protocol)
    rows_match = _population_rows_match(stratum, recomputed, inputs.population)
    verdicts: dict[Gate, str | None] = {
        Gate.RECORDINGS: _verdict(
            gates[Gate.RECORDINGS.value],
            "the recording identifiers differ across systems or from the recomputed recordings",
        ),
        Gate.FRAMES: _frames_verdict(gates[Gate.FRAMES.value], rows_match),
        Gate.COVERAGE_OPGAN: _verdict(
            gates[Gate.COVERAGE_OPGAN.value], "a frame has no valid output from OpGAN"
        ),
        Gate.COVERAGE_UVR: _verdict(
            gates[Gate.COVERAGE_UVR.value], "a frame has no valid output from UVR"
        ),
    }
    return Population(expected, clusters, results, failure_counts, verdicts)


def _protocol_verdict(
    manifest: Manifest, summary: Mapping[str, Any], protocol: Protocol
) -> str | None:
    """Gate a: the walk manifest and the systems summary name the loaded protocol's digest."""
    digest = protocol.source_sha256
    held = manifest.protocol_sha256 == digest and summary["protocol_sha256"] == digest
    return _verdict(held, "the walk manifest or the systems summary names another protocol")


def _universe_verdict(manifest: Manifest, stratum: str) -> str | None:
    """Gate b: the walk manifest's universe digest equals the pinned digest for the stratum."""
    pinned = UNIVERSE_SHA256.get(stratum)
    held = pinned is not None and manifest.universe_sha256.get(stratum) == pinned
    return _verdict(held, "the walk manifest's universe digest is not the pinned one")


def _screen_of(line: dict[str, Any]) -> object:
    """The screen result an audit line records, or None."""
    details = line.get("details")
    return details.get("screen") if isinstance(details, dict) else None


def _stratum_admissions(stratum: str, manifest: Manifest) -> tuple[Admission, ...]:
    """Every admission of one stratum, in every partition, in the manifest's order."""
    return tuple(a for a in manifest.admissions if a.stratum == stratum)


def _partition_reason(admission: Admission, plan: Sequence[str]) -> str | None:
    """The walk's partition plan gives each admission position its partition, or None if it holds."""
    if not 1 <= admission.position <= len(plan):
        return (
            f"{admission.identifier} is at position {admission.position}, outside the walk's "
            f"plan of {len(plan)} positions"
        )
    expected = plan[admission.position - 1]
    if admission.partition.value != expected:
        return (
            f"{admission.identifier} is in partition {admission.partition.value}, but the walk's "
            f"plan gives {expected} at position {admission.position}"
        )
    return None


def _cluster_reason(admission: Admission, clusters: Mapping[str, str]) -> str | None:
    """The cluster the pinned universe gives the admission's identifier, or None if it holds."""
    pinned = clusters.get(admission.identifier)
    if pinned is None:
        return f"{admission.identifier} is not in the pinned candidate universe"
    if pinned != admission.cluster:
        return (
            f"{admission.identifier} is in cluster {admission.cluster}, but the pinned universe "
            f"gives {pinned}"
        )
    return None


def _audit_reasons(admission: Admission, admitted: Sequence[dict[str, Any]]) -> list[str]:
    """The admitted audit line of an admission must be unique and carry the manifest's values."""
    matches = [
        line
        for line in admitted
        if line.get("identifier") == admission.identifier
        and line.get("admission_position") == admission.position
    ]
    if len(matches) != 1:
        return [
            f"{admission.identifier} has {len(matches)} admitted audit lines at position "
            f"{admission.position}, not one"
        ]
    line = matches[0]
    reasons = []
    if line.get("partition") != admission.partition.value:
        reasons.append(f"the audit line of {admission.identifier} names another partition")
    details = line.get("details")
    if not isinstance(details, dict):
        return [*reasons, f"the audit line of {admission.identifier} has no details object"]
    expected = {
        "raw_sha256": admission.raw_sha256,
        "decoded_pcm_sha256": admission.decoded_pcm_sha256,
        "samples": admission.samples,
        "canonical_sha256": admission.canonical_sha256,
    }
    for key, value in expected.items():
        if key not in details:
            reasons.append(f"the audit line of {admission.identifier} has no details.{key}")
        elif details[key] != value:
            reasons.append(
                f"details.{key} in the audit line of {admission.identifier} differs from the manifest"
            )
    return reasons


def _plan(stratum: str, protocol: Protocol) -> tuple[str, ...]:
    """The walk's partition plan for one stratum: the partition of each admission position."""
    by_name = {candidate.name: candidate for candidate in protocol.selection.strata}
    return partition_plan(by_name[stratum])


def _count_reasons(stratum: str, inputs: Inputs) -> list[str]:
    """A complete stratum admits exactly one recording per position of its partition plan.

    Gate c runs only for complete strata, so a short stratum never reaches this check.
    """
    expected = len(_plan(stratum, inputs.protocol))
    actual = len(_stratum_admissions(stratum, inputs.manifest))
    if actual == expected:
        return []
    return [
        f"the walk marks {stratum} complete with {actual} admissions, but its partition plan "
        f"has {expected} positions"
    ]


def _admission_reasons(
    stratum: str,
    inputs: Inputs,
    admitted: Sequence[dict[str, Any]],
    clusters: Mapping[str, str] | None,
) -> list[str]:
    """The per-admission checks of one stratum: partition, cluster, and the admitted audit line.

    The cluster check is skipped when `clusters` is None, which is the case for a stratum with no
    pinned universe digest. The universe gate withholds such a stratum.
    """
    plan = _plan(stratum, inputs.protocol)
    reasons: list[str] = []
    for admission in _stratum_admissions(stratum, inputs.manifest):
        partition_reason = _partition_reason(admission, plan)
        if partition_reason is not None:
            reasons.append(partition_reason)
        if clusters is not None:
            cluster_reason = _cluster_reason(admission, clusters)
            if cluster_reason is not None:
                reasons.append(cluster_reason)
        reasons.extend(_audit_reasons(admission, admitted))
    return reasons


def _screen_reasons(stratum: str, inputs: Inputs, clusters: Mapping[str, str] | None) -> list[str]:
    """Gate c: the audit trail is the one the manifest names, and every admission matches it.

    The admitted identifiers in the stratum's audit lines must equal the manifest's admissions of
    the stratum, across all partitions, so an admission with no audit line cannot pass. Each
    admission is then checked against the walk's partition plan, the pinned universe's cluster and
    its own admitted audit line. A complete stratum must also admit one recording per plan
    position. evaluate_stratum calls this gate only for complete strata.
    """
    manifest, audit = inputs.manifest, inputs.audit
    if manifest.audit_trail_sha256 is None:
        return [f"the walk manifest has no {AUDIT_KEY}, so the audit trail is not verified"]
    if manifest.audit_trail_sha256 != audit.sha256:
        return ["the audit trail's digest differs from the one the walk manifest names"]
    if any(line is None for line in audit.lines):
        return ["the audit trail has a line that is not a JSON object"]
    own = [line for line in audit.lines if line is not None and line.get("stratum") == stratum]
    if any(line.get("outcome") == _AWAITING_ADJUDICATION for line in own):
        return ["a candidate of this stratum awaits adjudication"]
    admitted = [line for line in own if line.get("outcome") == _ADMITTED]
    if any(_screen_of(line) is None for line in admitted):
        return ["an admitted candidate has no screen result"]
    listed = sorted(str(line.get("identifier")) for line in admitted)
    walked = sorted(a.identifier for a in _stratum_admissions(stratum, manifest))
    reasons = (
        [] if listed == walked else ["the audit trail's admissions differ from the manifest's"]
    )
    return (
        reasons
        + _count_reasons(stratum, inputs)
        + _admission_reasons(stratum, inputs, admitted, clusters)
    )


def _screen_verdict(stratum: str, inputs: Inputs, clusters: Mapping[str, str] | None) -> str | None:
    """Gate c as one verdict: None when it holds, otherwise its reasons joined in one line."""
    reasons = _screen_reasons(stratum, inputs, clusters)
    return _verdict(not reasons, "; ".join(reasons))


def _counts(population: Population) -> Counts:
    """The recordings, their performer clusters, and the eligible frames of one stratum."""
    return Counts(
        recordings=len(population.expected),
        clusters=len(set(population.clusters.values())),
        eligible_frames=sum(len(frames) for frames in population.expected.values()),
    )


def _coverage(population: Population) -> dict[str, float]:
    """Each system's coverage over the expected eligible frames."""
    return {system: coverage(population.expected, population.results[system]) for system in SYSTEMS}


def _short(stratum: str) -> StratumReport:
    """A short stratum publishes its reason only."""
    reason = "the stratum is short: its walk did not fill its partitions"
    return StratumReport(stratum, Outcome.WITHHELD, {}, (reason,))


def _comparison(
    stratum: str, gates: Mapping[str, bool], population: Population, protocol: Protocol
) -> StratumReport:
    """All gates hold: the paired comparison, with its counts and coverage."""
    try:
        scores = recording_scores(
            population.expected, population.results, protocol, population.clusters
        )
        aggregate = summarise(scores, protocol)
    except (PopulationError, AggregationError) as error:
        raise ReportError(f"stratum {stratum} cannot be aggregated: {error}") from error
    return StratumReport(
        stratum=stratum,
        outcome=Outcome.COMPARISON,
        gates=gates,
        reasons=(),
        counts=_counts(population),
        coverage=_coverage(population),
        headline=aggregate.primary,
        diagnostic=aggregate.diagnostic,
        baseline=aggregate.baseline,
    )


def _reliability(
    stratum: str, gates: Mapping[str, bool], reasons: tuple[str, ...], population: Population
) -> StratumReport:
    """Only coverage gates fail: coverage and failure counts by kind, and no metric value."""
    return StratumReport(
        stratum=stratum,
        outcome=Outcome.RELIABILITY_FINDING,
        gates=gates,
        reasons=reasons,
        counts=_counts(population),
        coverage=_coverage(population),
        failure_counts=population.failure_counts,
    )


def _static_verdicts(
    stratum: str, inputs: Inputs, clusters: Mapping[str, str] | None
) -> dict[Gate, str | None]:
    """Gates a, b and c, which read the manifest, the summary, the audit trail and the universe."""
    return {
        Gate.PROTOCOL: _protocol_verdict(inputs.manifest, inputs.summary, inputs.protocol),
        Gate.UNIVERSE: _universe_verdict(inputs.manifest, stratum),
        Gate.SCREEN: _screen_verdict(stratum, inputs, clusters),
    }


def evaluate_stratum(
    stratum: str, inputs: Inputs, pinned_clusters: Mapping[str, Mapping[str, str]]
) -> StratumReport:
    """One stratum's outcome: a comparison, a reliability finding, or reasons only.

    Args:
        stratum: The stratum's name, as the protocol and the manifest spell it.
        inputs: The walk and systems outputs, and the protocol.
        pinned_clusters: Each stratum's identifier-to-cluster map from its pinned universe. A
            stratum with no map has no pinned universe to check its clusters against.

    Returns:
        The stratum's report.

    Raises:
        ReportError: If a gate holds but the comparison cannot be built from the results.
    """
    entry = inputs.manifest.strata[stratum]
    if entry.status is not StratumStatus.COMPLETE:
        return _short(stratum)
    verdicts: dict[Gate, str | None] = dict(
        _static_verdicts(stratum, inputs, pinned_clusters.get(stratum))
    )
    population = _population(stratum, inputs)
    verdicts.update(population.verdicts)
    gates = {gate.value: verdict is None for gate, verdict in verdicts.items()}
    failing = {gate: verdict for gate, verdict in verdicts.items() if verdict is not None}
    reasons = tuple(f"{gate.value}: {verdict}" for gate, verdict in failing.items())
    if not failing:
        return _comparison(stratum, gates, population, inputs.protocol)
    if set(failing) <= _COVERAGE_GATES:
        return _reliability(stratum, gates, reasons, population)
    return StratumReport(stratum, Outcome.WITHHELD, gates, reasons)


def _binding_reasons(inputs: Inputs) -> list[str]:
    """Why the systems summary does not bind the files the report read, or an empty list.

    The summary must record the SHA-256 of each file's exact bytes. A key that is absent, or a
    digest that differs, means the report is not reading the outputs the systems stage wrote.
    """
    bound = (
        (RESULTS_KEY, inputs.results_sha256, RESULTS_NAME),
        (POPULATION_KEY, inputs.population_sha256, POPULATION_NAME),
        (MANIFEST_KEY, inputs.manifest.file_sha256, MANIFEST_NAME),
    )
    reasons = []
    for key, actual, name in bound:
        if key not in inputs.summary:
            reasons.append(f"the systems summary has no {key}, so {name} is not bound")
        elif inputs.summary[key] != actual:
            reasons.append(f"{name} does not match the {key} the systems summary records")
    return reasons


def _pinned_clusters(protocol: Protocol, manifest: Manifest) -> dict[str, dict[str, str]]:
    """Each stratum's identifier-to-cluster map from its pinned universe, for strata with admissions.

    A stratum with no pinned digest is left out: the universe gate withholds it.

    Raises:
        ReportError: If a pinned universe cannot be loaded or fails its check.
    """
    by_name = {candidate.name: candidate for candidate in protocol.selection.strata}
    pinned: dict[str, dict[str, str]] = {}
    for name in manifest.strata:
        if UNIVERSE_SHA256.get(name) is None or not _stratum_admissions(name, manifest):
            continue
        try:
            universe = load_universe(by_name[name])
        except CandidateUniverseError as error:
            logger.error("Pinned universe of stratum %s failed its check: %s", name, error)
            raise ReportError(f"the pinned universe of {name} failed its check") from error
        pinned[name] = {file.identifier: file.cluster for file in universe.files}
    return pinned


def build_report(walk_dir: Path, systems_dir: Path, protocol: Protocol) -> RunReport:
    """The report for one walk directory and one systems directory.

    When the systems summary does not bind the files the report read, every stratum is withheld
    with the reasons, and no stratum is evaluated. The pinned universes and the canonical files
    are then not read either.

    Args:
        walk_dir: The directory holding the walk manifest and the audit trail.
        systems_dir: The directory holding the systems stage's summary, results and population.
        protocol: The loaded protocol.

    Returns:
        One entry per stratum, in the protocol's order, and the run's digests.

    Raises:
        ReportError: If an input is missing, unreadable, or not in the shape the report reads, if a
            pinned universe fails its check, or if a canonical file fails its check.
    """
    inputs = read_inputs(walk_dir, systems_dir, protocol)
    names = [stratum.name for stratum in protocol.selection.strata]
    if set(names) != set(inputs.manifest.strata):
        raise ReportError("the walk manifest does not list the protocol's strata")
    binding = _binding_reasons(inputs)
    if binding:
        logger.error("Withholding every stratum: %s", "; ".join(binding))
        strata = tuple(StratumReport(name, Outcome.WITHHELD, {}, tuple(binding)) for name in names)
    else:
        pinned = _pinned_clusters(protocol, inputs.manifest)
        strata = tuple(evaluate_stratum(name, inputs, pinned) for name in names)
    digests = Digests(
        protocol_sha256=protocol.source_sha256,
        universe_sha256={name: UNIVERSE_SHA256.get(name) for name in names},
        archive_sha256=inputs.manifest.archive_sha256,
        walk_manifest_sha256=inputs.manifest.file_sha256,
        audit_trail_sha256=inputs.audit.sha256,
        systems_results_sha256=inputs.results_sha256,
    )
    return RunReport(
        protocol_version=protocol.protocol_version,
        claim_scope_note=protocol.reporting.claim_scope_note,
        digests=digests,
        strata=strata,
    )
