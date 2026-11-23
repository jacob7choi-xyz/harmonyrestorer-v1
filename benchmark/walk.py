"""One candidate through the selection walk's predicates, in protocol v13's order.

`examine_candidate` takes a candidate in its stratum's frozen universe, the walk state that
earlier decisions leave behind, and three injected functions: the fetch for the stratum, the
canonicalisation worker, and the duplicate screen's features. It applies the predicates in
`selection.predicate_order` and returns the first decision reached:

- `Admitted`: every predicate passed, and the candidate joins the walk state;
- `Rejected`: the first predicate it failed, with its protocol code;
- `AwaitingAdjudication`: the screen flagged it and no recorded answer applies, so the walk
  must halt;
- `RunInvalid`: a harness failure under a protocol code, so nothing may be published.

Failures the protocol does not give a code, such as an unreadable canonical file, propagate as
exceptions for the driver to record. Every outcome carries an `Evidence` record of what each
predicate compared, for the audit trail.

Rule 1 (training source identity) can match only an identifier inside the training collection's
item. The v13 strata are not that item, so the syntax predicate rejects every candidate before
rule 1 runs. Rule 1 is still evaluated, in order, so the predicate is in place if a stratum
names that item.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from benchmark.adjudication import AdjudicationRecord, answer_sets
from benchmark.atomic_write import write_atomically
from benchmark.candidate_universe import CandidateUniverse, ListedFile
from benchmark.canonicalisation import CanonicalWaveform
from benchmark.duplicate_screen import (
    Decision,
    Degenerate,
    Features,
    Flag,
    RecordingIdentity,
    Reference,
    ScreenResult,
    adjudicate,
    screen,
)
from benchmark.eligibility import frame_floors, recording_floors
from benchmark.eligibility_outcomes import (
    frame_verdicts,
    recording_floor_reasons,
    recording_frame_reasons,
)
from benchmark.fetch import (
    Attempt,
    FetchOutcome,
    ListingMismatch,
    PermanentlyUnavailable,
    RetriesExhausted,
    Staged,
)
from benchmark.framing import frame
from benchmark.protocol import Protocol, Stratum, ZenodoArchiveStratum
from benchmark.runner import Candidate, CanonicalFile, SourceRun
from benchmark.selection import (
    ExclusionRule,
    TrainingIdentities,
    identifier_well_formed,
    matches_training_source,
    walk_order,
)
from benchmark.source_outcome import RunAbortOutcome, SourceIneligibleOutcome

__all__ = [
    "Admission",
    "Admitted",
    "AdmittedRule",
    "AwaitingAdjudication",
    "Evidence",
    "Outcome",
    "Rejected",
    "RunInvalid",
    "WalkDeps",
    "WalkState",
    "examine_candidate",
]

logger = logging.getLogger(__name__)

FetchFn = Callable[[ListedFile], FetchOutcome]
CanonicaliseFn = Callable[[Candidate, Path], SourceRun]
FeatureFn = Callable[[npt.NDArray[np.float64]], tuple[Features, Degenerate]]


class AdmittedRule(StrEnum):
    """Rules 2 and 3 against the candidates admitted earlier in the walk, with v9's codes."""

    RAW_SHA256 = "ADMITTED_RAW_SHA256"
    DECODED_PCM_SHA256 = "ADMITTED_DECODED_PCM_SHA256"


@dataclass(frozen=True)
class WalkDeps:
    """The functions that reach the network, the worker and the feature code.

    Attributes:
        fetch: The stratum's fetch: the staged file, or why the candidate is excluded. It may
            raise `RetriesExhausted`.
        canonicalise: Runs the canonicalisation worker on a candidate in an output directory
            that does not exist yet, as `benchmark.runner.run_source` does.
        features: The duplicate screen's features for a canonical waveform.
    """

    fetch: FetchFn
    canonicalise: CanonicaliseFn
    features: FeatureFn


@dataclass
class Evidence:
    """What each predicate compared for one candidate, recorded whatever the outcome.

    Every field is None or empty when the walk did not reach the predicate that sets it.
    """

    attempts: tuple[Attempt, ...] = ()
    listing_compared: tuple[str, ...] = ()
    fetch_status: int | None = None
    size: int | None = None
    raw_sha256: str | None = None
    decoded_pcm_sha256: str | None = None
    samples: int | None = None
    cluster_admitted: int | None = None
    recording_rms_dbfs: float | None = None
    frames: int | None = None
    eligible_frames: int | None = None
    screen: ScreenResult | None = None
    adjudication: AdjudicationRecord | None = None

    def to_json(self) -> dict[str, Any]:
        """The evidence as JSON-serialisable values; non-finite numbers become null."""
        return {
            "attempts": [_attempt_json(a) for a in self.attempts],
            "listing_compared": list(self.listing_compared),
            "fetch_status": self.fetch_status,
            "size": self.size,
            "raw_sha256": self.raw_sha256,
            "decoded_pcm_sha256": self.decoded_pcm_sha256,
            "samples": self.samples,
            "cluster_admitted": self.cluster_admitted,
            "recording_rms_dbfs": _finite(self.recording_rms_dbfs),
            "frames": self.frames,
            "eligible_frames": self.eligible_frames,
            "screen": None if self.screen is None else _screen_json(self.screen),
            "adjudication": None if self.adjudication is None else _record_json(self.adjudication),
        }


def _attempt_json(attempt: Attempt) -> dict[str, Any]:
    """One fetch attempt as the audit trail records it."""
    return {
        "status": attempt.status,
        "error": attempt.error,
        "compared": list(attempt.compared),
        "mismatched": list(attempt.mismatched),
    }


def _finite(value: float | None) -> float | None:
    """*value* if it is a finite number, otherwise None."""
    return value if value is not None and math.isfinite(value) else None


def _identity_json(identity: RecordingIdentity) -> dict[str, str]:
    """A recording as an adjudication record and the audit trail name it."""
    return {
        "identifier": identity.identifier,
        "raw_sha256": identity.raw_sha256,
        "decoded_pcm_sha256": identity.decoded_pcm_sha256,
    }


def _screen_json(result: ScreenResult) -> dict[str, Any]:
    """The screen's maxima, flags, unscored references and degenerate counts."""
    return {
        "max_training": _finite(result.max_training),
        "max_admitted": _finite(result.max_admitted),
        "flagged": [
            {"group": flag.group.value, "score": flag.score, **_identity_json(flag.identity)}
            for flag in result.flagged
        ],
        "unscored": [_identity_json(identity) for identity in result.unscored],
        "uncertain": result.uncertain,
        "zero_variance_coefficients": result.degenerate.zero_variance_coefficients,
        "zero_norm_frames": result.degenerate.zero_norm_frames,
    }


def _record_json(record: AdjudicationRecord) -> dict[str, Any]:
    """The adjudication record that applied, with its evidence for each reference."""
    return {
        "candidate": _identity_json(record.candidate),
        "answers": [
            {
                "reference": _identity_json(reference),
                "answer": answer.value,
                "evidence": record.evidence[reference],
            }
            for reference, answer in record.answers.items()
        ],
    }


@dataclass(frozen=True)
class Admission:
    """An admitted candidate: its stratum, cluster, kept canonical file and screen reference."""

    stratum: str
    identifier: str
    cluster: str
    canonical: CanonicalFile
    reference: Reference

    @property
    def raw_sha256(self) -> str:
        """The candidate's raw SHA-256."""
        return self.reference.identity.raw_sha256

    @property
    def decoded_pcm_sha256(self) -> str:
        """The candidate's decoded-PCM SHA-256."""
        return self.reference.identity.decoded_pcm_sha256


@dataclass(frozen=True)
class Admitted:
    """Every predicate passed."""

    admission: Admission
    evidence: Evidence
    outcome: ClassVar[str] = "admitted"

    @property
    def identifier(self) -> str:
        """The admitted candidate's source identifier."""
        return self.admission.identifier


@dataclass(frozen=True)
class Rejected:
    """The first predicate the candidate failed, under its protocol code."""

    identifier: str
    code: str
    evidence: Evidence
    outcome: ClassVar[str] = "rejected"


@dataclass(frozen=True)
class AwaitingAdjudication:
    """The screen flagged the candidate and no recorded answer applies, so the walk halts."""

    identifier: str
    identity: RecordingIdentity
    flagged: tuple[Flag, ...]
    evidence: Evidence
    outcome: ClassVar[str] = "awaiting_adjudication"


@dataclass(frozen=True)
class RunInvalid:
    """A harness failure under a protocol code; the run publishes nothing."""

    identifier: str
    code: str
    detail: str
    evidence: Evidence
    outcome: ClassVar[str] = "run_invalid"


Outcome = Admitted | Rejected | AwaitingAdjudication | RunInvalid


@dataclass
class WalkState:
    """What earlier decisions leave for later candidates, across both strata.

    `admitted_raw`, `admitted_decoded` and `admitted_references` hold every admission so far,
    whichever stratum made it. `cluster_counts` is keyed by (stratum name, cluster).
    """

    training: TrainingIdentities
    training_references: Sequence[Reference]
    adjudications: Sequence[AdjudicationRecord] = ()
    admitted_raw: set[str] = field(default_factory=set)
    admitted_decoded: set[str] = field(default_factory=set)
    admitted_references: list[Reference] = field(default_factory=list)
    cluster_counts: Counter[tuple[str, str]] = field(default_factory=Counter)

    def record_admission(self, admission: Admission) -> None:
        """Add an admission to the digests, references and cluster counts later candidates see."""
        self.admitted_raw.add(admission.raw_sha256)
        self.admitted_decoded.add(admission.decoded_pcm_sha256)
        self.admitted_references.append(admission.reference)
        self.cluster_counts[(admission.stratum, admission.cluster)] += 1


def _reject(listed: ListedFile, code: str, evidence: Evidence) -> Rejected:
    """A rejection of *listed* under *code*."""
    return Rejected(listed.identifier, code, evidence)


def _identity_rules(
    protocol: Protocol, stratum: Stratum, listed: ListedFile, state: WalkState, evidence: Evidence
) -> Rejected | None:
    """The syntax, rule 1 and cluster cap predicates, all before any fetch."""
    selection = protocol.selection
    if not identifier_well_formed(listed.identifier, stratum.source):
        return _reject(listed, selection.identifier_syntax_code, evidence)
    if matches_training_source(
        listed.identifier, selection.training_collection_identifier, state.training.filenames
    ):
        return _reject(listed, ExclusionRule.SOURCE_IDENTITY.value, evidence)
    admitted = state.cluster_counts[(stratum.name, listed.cluster)]
    evidence.cluster_admitted = admitted
    if admitted >= stratum.cluster_cap:
        return _reject(listed, selection.cluster_cap_code, evidence)
    return None


def _fetch_stage(
    protocol: Protocol, listed: ListedFile, deps: WalkDeps, evidence: Evidence
) -> Staged | Rejected | RunInvalid:
    """The fetch predicate: the staged file, or the permanent, listing or retry outcome."""
    selection = protocol.selection
    try:
        result = deps.fetch(listed)
    except RetriesExhausted as error:
        logger.error("%s: %s", listed.identifier, error)
        evidence.attempts = error.attempts
        return RunInvalid(listed.identifier, selection.retries_exhausted_code, str(error), evidence)
    evidence.attempts = result.attempts
    if isinstance(result, PermanentlyUnavailable):
        evidence.fetch_status = result.status
        return _reject(listed, selection.permanent_unavailability_code, evidence)
    if isinstance(result, ListingMismatch):
        return _reject(listed, selection.listing_mismatch_code, evidence)
    evidence.listing_compared = result.compared
    evidence.size = result.size
    return result


def _raw_digest_rules(
    listed: ListedFile, staged: Staged, state: WalkState, evidence: Evidence
) -> Rejected | None:
    """Rules 2 and 3 on the raw digest, training first, then the earlier admissions.

    An oversize staged prefix has no digest, so it matches nothing here, as v11 states.
    """
    evidence.raw_sha256 = staged.sha256
    if staged.sha256 is None:
        return None
    if staged.sha256 in state.training.raw_sha256:
        return _reject(listed, ExclusionRule.RAW_SHA256.value, evidence)
    if staged.sha256 in state.admitted_raw:
        return _reject(listed, AdmittedRule.RAW_SHA256.value, evidence)
    return None


def _canonicalise_stage(
    listed: ListedFile, staged: Staged, out_dir: Path, deps: WalkDeps, evidence: Evidence
) -> CanonicalFile | Rejected | RunInvalid:
    """The source-outcome stages: the worker's canonical file, or its verdict or abort."""
    run = deps.canonicalise(Candidate(listed.identifier, staged.path, staged.sha256), out_dir)
    outcome = run.outcome
    if isinstance(outcome, RunAbortOutcome):
        logger.error("%s: run aborted with %s", listed.identifier, outcome.reason.value)
        return RunInvalid(listed.identifier, outcome.reason.value, outcome.detail, evidence)
    if isinstance(outcome, SourceIneligibleOutcome):
        return _reject(listed, outcome.primary_reason_code.value, evidence)
    return outcome


def _decoded_digest_rules(
    listed: ListedFile, canonical: CanonicalFile, state: WalkState, evidence: Evidence
) -> Rejected | None:
    """Rules 2 and 3 on the decoded-PCM digest, training first, then the earlier admissions."""
    evidence.raw_sha256 = canonical.raw_sha256
    evidence.decoded_pcm_sha256 = canonical.decoded_pcm_sha256
    evidence.samples = canonical.samples
    if canonical.decoded_pcm_sha256 in state.training.decoded_sha256:
        return _reject(listed, ExclusionRule.DECODED_PCM_SHA256.value, evidence)
    if canonical.decoded_pcm_sha256 in state.admitted_decoded:
        return _reject(listed, AdmittedRule.DECODED_PCM_SHA256.value, evidence)
    return None


def _load_waveform(canonical: CanonicalFile) -> CanonicalWaveform:
    """The canonical samples as an owned, read-only array, with the file's digest."""
    samples = np.array(canonical.load(), dtype=np.float64, copy=True)
    samples.setflags(write=False)
    return CanonicalWaveform(samples=samples, sha256=canonical.sha256)


def _eligibility_stage(
    protocol: Protocol, listed: ListedFile, canonical: CanonicalFile, evidence: Evidence
) -> CanonicalWaveform | Rejected:
    """The recording floors, then the frame verdicts, with RECORDING_WITHOUT_ELIGIBLE_FRAMES last."""
    waveform = _load_waveform(canonical)
    rate = protocol.canonicalisation.resample.target_sample_rate
    floors = recording_floors(waveform, protocol.eligibility, sample_rate=rate)
    evidence.recording_rms_dbfs = floors.rms_dbfs
    floor_reasons = recording_floor_reasons(floors)
    if floor_reasons:
        return _reject(listed, floor_reasons[0].value, evidence)
    framed = frame(waveform, protocol.framing)
    verdicts = frame_verdicts(framed, frame_floors(framed, protocol.eligibility), protocol.metrics)
    evidence.frames = len(verdicts)
    evidence.eligible_frames = sum(1 for verdict in verdicts if verdict.eligible)
    frame_reasons = recording_frame_reasons(verdicts)
    if frame_reasons:
        return _reject(listed, frame_reasons[0].value, evidence)
    return waveform


def _applicable_record(
    result: ScreenResult, identity: RecordingIdentity, records: Sequence[AdjudicationRecord]
) -> AdjudicationRecord | None:
    """The first record naming this candidate and exactly the flagged references, if any.

    This mirrors the rule `adjudicate` applies, so the audit trail can name the record that
    decided the candidate. `adjudicate` itself is what decides.
    """
    flagged = {flag.identity for flag in result.flagged}
    for record in records:
        if record.candidate == identity and set(record.answers) == flagged:
            return record
    return None


def _screen_stage(
    protocol: Protocol,
    listed: ListedFile,
    stratum_name: str,
    canonical: CanonicalFile,
    waveform: CanonicalWaveform,
    state: WalkState,
    deps: WalkDeps,
    evidence: Evidence,
) -> Admitted | Rejected | AwaitingAdjudication:
    """The duplicate screen against training, then the earlier admissions, and its decision."""
    features, degenerate = deps.features(waveform.samples)
    identity = RecordingIdentity(
        listed.identifier, canonical.raw_sha256, canonical.decoded_pcm_sha256
    )
    result = screen(
        features,
        degenerate,
        state.training_references,
        state.admitted_references,
        protocol.duplicate_detection,
    )
    evidence.screen = result
    decision = adjudicate(result, identity, answer_sets(state.adjudications))
    if not result.uncertain:
        evidence.adjudication = _applicable_record(result, identity, state.adjudications)
    if decision.decision is Decision.AWAITING_ADJUDICATION:
        return AwaitingAdjudication(listed.identifier, identity, result.flagged, evidence)
    if decision.decision is Decision.EXCLUDE and decision.code is not None:
        return _reject(listed, decision.code.value, evidence)
    admission = Admission(
        stratum=stratum_name,
        identifier=listed.identifier,
        cluster=listed.cluster,
        canonical=canonical,
        reference=Reference(identity, features),
    )
    return Admitted(admission, evidence)


def examine_candidate(
    protocol: Protocol,
    stratum: Stratum,
    listed: ListedFile,
    state: WalkState,
    out_dir: Path,
    deps: WalkDeps,
) -> Outcome:
    """Apply every predicate to one candidate and return the first decision reached.

    Args:
        protocol: The loaded protocol.
        stratum: The stratum the candidate is walked in.
        listed: The candidate, with its cluster and listing values.
        state: The walk state. It is read here and changed only by `WalkState.record_admission`.
        out_dir: An absolute directory that does not exist yet, for the worker's output.
        deps: The fetch, canonicalisation and feature functions.

    Returns:
        The candidate's outcome, with the evidence each predicate compared.

    Raises:
        CanonicalFileError: If an admitted or examined canonical file fails its own check.
        ScreenError: If the screen is given inputs it cannot answer for.
        EligibilityArithmeticError: If a recording's or frame's energy is not finite.
    """
    evidence = Evidence()
    rejected = _identity_rules(protocol, stratum, listed, state, evidence)
    if rejected is not None:
        return rejected
    staged = _fetch_stage(protocol, listed, deps, evidence)
    if not isinstance(staged, Staged):
        return staged
    rejected = _raw_digest_rules(listed, staged, state, evidence)
    if rejected is not None:
        return rejected
    canonical = _canonicalise_stage(listed, staged, out_dir, deps, evidence)
    if not isinstance(canonical, CanonicalFile):
        return canonical
    rejected = _decoded_digest_rules(listed, canonical, state, evidence)
    if rejected is not None:
        return rejected
    waveform = _eligibility_stage(protocol, listed, canonical, evidence)
    if not isinstance(waveform, CanonicalWaveform):
        return waveform
    return _screen_stage(protocol, listed, stratum.name, canonical, waveform, state, deps, evidence)


class MetadataUnavailable(RuntimeError):
    """Raised by a metadata provider when the MusicNet metadata file cannot be obtained."""


MusicNetMetadata = Callable[[], bytes]


@dataclass(frozen=True)
class ChamberPreparation:
    """The chamber stratum's archive, as the walk needs it.

    `archive_sha256` is None when the archive was not fetched as a whole, so no digest exists.
    `fetch` answers each chamber candidate, from the extracted member or the archive's failure.
    """

    archive_sha256: str | None
    fetch: FetchFn


@dataclass(frozen=True)
class WalkInputs:
    """Everything a run needs that touches the network, the worker or the disk, injected."""

    training: TrainingIdentities
    training_references: Sequence[Reference]
    adjudications: Sequence[AdjudicationRecord]
    universes: Mapping[str, CandidateUniverse]
    piano_fetch: FetchFn
    chamber: ChamberPreparation
    canonicalise: CanonicaliseFn
    features: FeatureFn
    musicnet_metadata: MusicNetMetadata


@dataclass(frozen=True)
class RunLayout:
    """Where one run writes. The root is a directory that the caller created for this run.

    Attributes:
        root: The absolute run directory.
    """

    root: Path

    def __post_init__(self) -> None:
        """Refuse a relative root, since every recorded path must be absolute."""
        if not self.root.is_absolute():
            logger.error("Run directory must be absolute: %s", self.root)
            raise ValueError(f"run directory must be absolute, got {self.root}")

    @property
    def audit_path(self) -> Path:
        """One JSON line per candidate examined."""
        return self.root / "audit_trail.jsonl"

    @property
    def manifest_path(self) -> Path:
        """The walk manifest, written only when the run completes validly."""
        return self.root / "walk_manifest.json"

    @property
    def halt_path(self) -> Path:
        """The adjudication request, written when the walk halts on a flagged candidate."""
        return self.root / "adjudication_needed.json"

    @property
    def invalid_path(self) -> Path:
        """The reason the run is invalid, written when it is."""
        return self.root / "run_invalid.json"

    @property
    def chamber_members_dir(self) -> Path:
        """Where the chamber stratum's members are extracted."""
        return self.root / "chamber_members"

    @property
    def training_work_dir(self) -> Path:
        """Where the training references' workers write."""
        return self.root / "training_work"

    def candidate_dir(self, stratum: str, position: int) -> Path:
        """The worker output directory for the candidate at *position* in *stratum*'s walk."""
        return self.root / "candidates" / f"{stratum}-{position:06d}"

    def ensure_dirs(self) -> None:
        """Create the directories the run writes into, if they are missing."""
        for directory in (
            self.root / "candidates",
            self.training_work_dir,
            self.chamber_members_dir,
        ):
            directory.mkdir(exist_ok=True)


@dataclass(frozen=True)
class PlacedAdmission:
    """An admission with its partition and its 1-based position among the stratum's admissions."""

    partition: str
    position: int
    admission: Admission


@dataclass(frozen=True)
class StratumWalk:
    """One stratum's walk: complete when its partitions filled, short when its universe ran out."""

    name: str
    status: str
    placed: tuple[PlacedAdmission, ...]


@dataclass(frozen=True)
class WalkRun:
    """A run that finished validly: every stratum walked, the manifest written."""

    strata: tuple[StratumWalk, ...]
    archive_sha256: str | None


@dataclass(frozen=True)
class WalkContext:
    """What every stratum's walk shares: the protocol, the run, the audit trail and the state."""

    protocol: Protocol
    layout: RunLayout
    audit: AuditTrail
    state: WalkState
    musicnet_metadata: MusicNetMetadata


class AuditTrail:
    """The audit trail, rewritten whole and atomically after each line, so each write replaces the file as one unit."""

    def __init__(self, path: Path) -> None:
        """Start an empty trail at *path*."""
        self._path = path
        self._lines: list[str] = []

    def append(self, record: dict[str, Any]) -> None:
        """Add one candidate's record and write the trail."""
        self._lines.append(json.dumps(record, sort_keys=True, allow_nan=False, ensure_ascii=True))
        write_atomically(self._path, ("\n".join(self._lines) + "\n").encode("utf-8"))


def partition_plan(stratum: Stratum) -> tuple[str, ...]:
    """The partition of each admission position, in partition order: position k is element k-1."""
    return tuple(name for name in stratum.partition_order for _ in range(stratum.partitions[name]))


def _fetch_for(stratum: Stratum, inputs: WalkInputs) -> FetchFn:
    """The fetch that applies to *stratum*: the archive's for chamber, the item's otherwise."""
    if isinstance(stratum, ZenodoArchiveStratum):
        return inputs.chamber.fetch
    return inputs.piano_fetch


def _code_of(outcome: Admitted | Rejected | AwaitingAdjudication | RunInvalid) -> str | None:
    """The protocol code the outcome carries, or None for an admission or a halt."""
    return outcome.code if isinstance(outcome, (Rejected, RunInvalid)) else None


def _audit_line(
    stratum: str,
    position: int,
    outcome: Admitted | Rejected | AwaitingAdjudication | RunInvalid,
    placed: PlacedAdmission | None,
) -> dict[str, Any]:
    """One candidate's audit record: where it sat in the walk, what happened, and the evidence."""
    return {
        "stratum": stratum,
        "position": position,
        "identifier": outcome.identifier,
        "outcome": outcome.outcome,
        "code": _code_of(outcome),
        "partition": None if placed is None else placed.partition,
        "admission_position": None if placed is None else placed.position,
        "details": outcome.evidence.to_json(),
    }


def _discard(out_dir: Path) -> None:
    """Remove a rejected candidate's worker output, which no later step reads."""
    if out_dir.exists():
        shutil.rmtree(out_dir)


def _metadata_section(
    musicnet: ZenodoArchiveStratum | None, identifiers: Sequence[str], provider: MusicNetMetadata
) -> dict[str, Any]:
    """The MusicNet metadata rows for the MusicNet identifiers named, or why none could be given.

    The file's MD5 is checked against the stratum's frozen value, and a mismatch is reported
    without rows rather than trusted.
    """
    if musicnet is None:
        return {"status": "not_required"}
    prefix = f"{musicnet.source}/"
    ids = sorted({int(i.removeprefix(prefix)) for i in identifiers if i.startswith(prefix)})
    if not ids:
        return {"status": "not_required"}
    try:
        data = provider()
        text = data.decode("utf-8")
    except (MetadataUnavailable, UnicodeDecodeError) as error:
        logger.error("MusicNet metadata unavailable for the halt file: %s", error)
        return {"status": "unavailable", "reason": str(error)}
    actual = hashlib.md5(data, usedforsecurity=False).hexdigest()
    if actual != musicnet.metadata_md5:
        return {"status": "md5_mismatch", "expected": musicnet.metadata_md5, "actual": actual}
    wanted = set(ids)
    rows = [row for row in csv.DictReader(io.StringIO(text)) if row.get("id", "").isdigit()]
    matching = [row for row in rows if int(row["id"]) in wanted]
    return {
        "status": "verified",
        "file": musicnet.metadata,
        "md5": actual,
        "ids": ids,
        "rows": matching,
    }


def _write_halt(
    ctx: WalkContext, stratum: Stratum, position: int, outcome: AwaitingAdjudication
) -> None:
    """Write the adjudication request: the candidate, every flagged reference, and the metadata."""
    musicnet = next(
        (s for s in ctx.protocol.selection.strata if isinstance(s, ZenodoArchiveStratum)), None
    )
    named = [outcome.identifier, *(flag.identity.identifier for flag in outcome.flagged)]
    payload = {
        "stratum": stratum.name,
        "position": position,
        "identifier": outcome.identifier,
        "candidate": _identity_json(outcome.identity),
        "flagged": [
            {"group": flag.group.value, "score": flag.score, **_identity_json(flag.identity)}
            for flag in outcome.flagged
        ],
        "musicnet_metadata": _metadata_section(musicnet, named, ctx.musicnet_metadata),
    }
    body = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True)
    write_atomically(ctx.layout.halt_path, (body + "\n").encode("utf-8"))
    logger.error(
        "Walk halted at %s position %d: %s awaits adjudication",
        stratum.name,
        position,
        outcome.identifier,
    )


def _record_invalid(layout: RunLayout, stratum: str | None, outcome: RunInvalid) -> RunInvalid:
    """Write why the run is invalid, and return the outcome unchanged."""
    payload = {
        "stratum": stratum,
        "identifier": outcome.identifier,
        "code": outcome.code,
        "detail": outcome.detail,
    }
    body = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=True)
    write_atomically(layout.invalid_path, (body + "\n").encode("utf-8"))
    logger.error("Run invalid (%s): %s", outcome.code, outcome.detail)
    return outcome


def _walk_stratum(
    ctx: WalkContext, stratum: Stratum, universe: CandidateUniverse, deps: WalkDeps
) -> StratumWalk | RunInvalid | AwaitingAdjudication:
    """Walk one stratum in its seeded order until its partitions fill or its universe runs out.

    A halt or a run invalid is returned to the caller, which stops the whole run. Its record is
    written here, where the stratum and position are known.
    """
    plan = partition_plan(stratum)
    by_id = {listed.identifier: listed for listed in universe.files}
    placed: list[PlacedAdmission] = []
    for position, identifier in enumerate(walk_order(by_id, ctx.protocol.selection.seed), start=1):
        if len(placed) == len(plan):
            break
        out_dir = ctx.layout.candidate_dir(stratum.name, position)
        outcome = examine_candidate(
            ctx.protocol, stratum, by_id[identifier], ctx.state, out_dir, deps
        )
        if isinstance(outcome, Admitted):
            entry = PlacedAdmission(plan[len(placed)], len(placed) + 1, outcome.admission)
            placed.append(entry)
            ctx.state.record_admission(outcome.admission)
            ctx.audit.append(_audit_line(stratum.name, position, outcome, entry))
            continue
        ctx.audit.append(_audit_line(stratum.name, position, outcome, None))
        if isinstance(outcome, Rejected):
            _discard(out_dir)
        elif isinstance(outcome, AwaitingAdjudication):
            _write_halt(ctx, stratum, position, outcome)
            return outcome
        else:
            return _record_invalid(ctx.layout, stratum.name, outcome)
    status = "complete" if len(placed) == len(plan) else "short"
    logger.info(
        "Stratum %s %s with %d of %d admissions", stratum.name, status, len(placed), len(plan)
    )
    return StratumWalk(stratum.name, status, tuple(placed))


def manifest_document(
    protocol: Protocol, universes: Mapping[str, CandidateUniverse], run: WalkRun
) -> dict[str, Any]:
    """The walk manifest, with exactly the keys and entry fields the systems stage reads."""
    return {
        "protocol_sha256": protocol.source_sha256,
        "universe_sha256": {s.name: universes[s.name].sha256 for s in protocol.selection.strata},
        "archive_sha256": run.archive_sha256,
        "strata": {w.name: {"status": w.status, "admitted": len(w.placed)} for w in run.strata},
        "admissions": [
            {
                "stratum": walk.name,
                "partition": entry.partition,
                "position": entry.position,
                "identifier": entry.admission.identifier,
                "cluster": entry.admission.cluster,
                "canonical_path": str(entry.admission.canonical.path),
                "canonical_sha256": entry.admission.canonical.sha256,
                "samples": entry.admission.canonical.samples,
                "raw_sha256": entry.admission.raw_sha256,
                "decoded_pcm_sha256": entry.admission.decoded_pcm_sha256,
            }
            for walk in run.strata
            for entry in walk.placed
        ],
    }


def archive_retries_invalid(
    protocol: Protocol, layout: RunLayout, error: RetriesExhausted
) -> RunInvalid:
    """Record that the MusicNet archive's fetch exhausted its retries, which invalidates the run."""
    outcome = RunInvalid(
        error.identifier,
        protocol.selection.retries_exhausted_code,
        str(error),
        Evidence(attempts=error.attempts),
    )
    return _record_invalid(layout, None, outcome)


def run_walk(
    protocol: Protocol, layout: RunLayout, inputs: WalkInputs
) -> WalkRun | RunInvalid | AwaitingAdjudication:
    """Prepare the chamber archive, walk each stratum in protocol order, and write the manifest.

    The manifest is written only when every stratum was walked without a halt or an invalid
    condition. A halt or an invalid run leaves its own record and no manifest.

    Args:
        protocol: The loaded protocol.
        layout: The run directory, already created and empty.
        inputs: The injected functions, training identities and references, and universes.

    Returns:
        The completed run, or the halt or invalid outcome that stopped it.
    """
    layout.ensure_dirs()
    ctx = WalkContext(
        protocol=protocol,
        layout=layout,
        audit=AuditTrail(layout.audit_path),
        state=WalkState(
            training=inputs.training,
            training_references=inputs.training_references,
            adjudications=inputs.adjudications,
        ),
        musicnet_metadata=inputs.musicnet_metadata,
    )
    walks: list[StratumWalk] = []
    for stratum in protocol.selection.strata:
        deps = WalkDeps(_fetch_for(stratum, inputs), inputs.canonicalise, inputs.features)
        walk = _walk_stratum(ctx, stratum, inputs.universes[stratum.name], deps)
        if not isinstance(walk, StratumWalk):
            return walk
        walks.append(walk)
    run = WalkRun(tuple(walks), inputs.chamber.archive_sha256)
    body = json.dumps(
        manifest_document(protocol, inputs.universes, run),
        indent=2,
        allow_nan=False,
        ensure_ascii=True,
    )
    write_atomically(layout.manifest_path, (body + "\n").encode("utf-8"))
    logger.info("Walk complete; manifest written to %s", layout.manifest_path)
    return run
