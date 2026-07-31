"""Records for a source that could not become a canonical waveform.

Two types, and the split is the point. A `SourceIneligibleOutcome` is an observation about
a candidate: it is recorded, published, and the run continues. A `RunAbortOutcome` stops the
run and blocks publication. Protocol v4 holds their reason codes in separate closed
vocabularies so that a harness failure is not merely discouraged from becoming a source
verdict but has no spelling as one, and these types carry that separation into the type
system: neither can be constructed with the other's reason.

There is one construction path per type and it validates. A first draft put the rules in a
`create` classmethod and left the generated constructor public, which made the rules a
convention rather than a boundary: a record with the wrong terminating stage, a duplicated
supplemental reason, and a fabricated digest constructed without complaint. Both types
declare `init=False` and define their own `__init__`, so there is no unvalidated path and no
way to supply a derived field at all.

Everything correlated is derived rather than accepted. The terminating stage follows from
the primary reason and the unevaluated stages follow from the terminating stage. A caller
that cannot supply a value cannot supply a wrong one, which matters here because a record is
the only surviving evidence about a candidate the benchmark rejected.

Ordering is normalised, not accepted. Two implementations holding the same supplemental set
and serialising it differently would produce different manifest bytes and therefore
different digests, which would break the content addressing the benchmark rests on.
Duplicates are rejected before normalisation rather than absorbed by it: a repeated code is
a caller defect, and silently collapsing it would hide the bug and double-count the
narrowing it describes.

Evidence is required in both directions. Metadata must be present exactly when inspection
succeeded and absent otherwise; a decoder diagnostic must be present exactly when the
decoder rejected the artifact. Enforcing only the forbidding direction would accept a decode
rejection carrying no diagnostic at all, which is a record that quietly knows less than it
claims.

The vocabularies below are code-declared mirrors of the protocol-owned ones.
`source_outcome_contract` checks them, and this module deliberately does not import it: the
dependency runs evidence to outcomes to contract, and a lower layer importing the verifier
would make the graph cyclic and the import order load-bearing.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from benchmark._source_validation import checked_digest, checked_identifier
from benchmark.source_evidence import (
    ObservedSourceMetadata,
    SanitisedDiagnostic,
    SourceIdentity,
    SourceOutcomeError,
)

__all__ = [
    "ABORT_SOURCE_CONTEXT",
    "DIGEST_FORBIDDEN_ABORTS",
    "EVALUATION_STAGE_ORDER",
    "REASON_PRECEDENCE",
    "REASON_STAGE",
    "AbortReason",
    "EvaluationStage",
    "IneligibleReason",
    "RunAbortOutcome",
    "SourceContextPolicy",
    "SourceIneligibleOutcome",
    "terminating_stage_for",
    "unevaluated_stages_after",
]


class EvaluationStage(StrEnum):
    """Eligibility adjudication only.

    Identity acquisition is deliberately absent: it is evidence collection, and placing it
    here would let a record mark a stage unevaluated while carrying a digest only that
    stage could have produced.
    """

    LOCAL_FILE_CHECK = "local_file_check"
    INSPECTION = "inspection"
    METADATA_VALIDATION = "metadata_validation"
    DECODE = "decode"
    DECODED_METADATA_VALIDATION = "decoded_metadata_validation"
    WAVEFORM_VALIDATION = "waveform_validation"


class IneligibleReason(StrEnum):
    """Predeclared outcomes. Recorded, published, and the run continues."""

    SOURCE_EXCEEDS_MAX_BYTES = "SOURCE_EXCEEDS_MAX_BYTES"
    SOURCE_EXCEEDS_MAX_DURATION = "SOURCE_EXCEEDS_MAX_DURATION"
    SOURCE_EXCEEDS_MAX_DECODED_BYTES = "SOURCE_EXCEEDS_MAX_DECODED_BYTES"
    SOURCE_BELOW_MIN_SAMPLE_RATE = "SOURCE_BELOW_MIN_SAMPLE_RATE"
    UNSUPPORTED_CONTAINER = "UNSUPPORTED_CONTAINER"
    UNSUPPORTED_SUBTYPE = "UNSUPPORTED_SUBTYPE"
    UNSUPPORTED_CHANNEL_COUNT = "UNSUPPORTED_CHANNEL_COUNT"
    SOURCE_INSPECTION_REJECTED = "SOURCE_INSPECTION_REJECTED"
    SOURCE_DECODE_REJECTED = "SOURCE_DECODE_REJECTED"
    SOURCE_METADATA_INCONSISTENT = "SOURCE_METADATA_INCONSISTENT"
    EMPTY_SOURCE = "EMPTY_SOURCE"
    NONFINITE_SOURCE = "NONFINITE_SOURCE"
    INVALID_SOURCE_AMPLITUDE = "INVALID_SOURCE_AMPLITUDE"


class AbortReason(StrEnum):
    """Harness failures. The run stops and publication is blocked."""

    SOURCE_HASH_MISMATCH = "SOURCE_HASH_MISMATCH"
    LOCAL_ARTIFACT_UNSTABLE = "LOCAL_ARTIFACT_UNSTABLE"
    LOCAL_ARTIFACT_NOT_REGULAR_FILE = "LOCAL_ARTIFACT_NOT_REGULAR_FILE"
    GOLDEN_FIXTURE_FAILURE = "GOLDEN_FIXTURE_FAILURE"
    ENVIRONMENT_MISMATCH = "ENVIRONMENT_MISMATCH"
    ENVIRONMENT_CAPACITY_FAILURE = "ENVIRONMENT_CAPACITY_FAILURE"
    UNEXPECTED_IO_FAILURE = "UNEXPECTED_IO_FAILURE"
    INTERNAL_CANONICALISATION_ERROR = "INTERNAL_CANONICALISATION_ERROR"


class SourceContextPolicy(StrEnum):
    """Whether an abort may name the candidate it happened to.

    Three states rather than two. A binary partition would force a harness failure that can
    occur either before or during candidate processing into one of the wrong shapes.
    """

    FORBIDDEN = "forbidden"
    REQUIRED = "required"
    OPTIONAL = "optional"


EVALUATION_STAGE_ORDER: tuple[EvaluationStage, ...] = tuple(EvaluationStage)

# The first applicable entry becomes the single counted reason. Non-decreasing in stage
# position, because a reason cannot be selected before the stage that establishes it runs.
REASON_PRECEDENCE: tuple[IneligibleReason, ...] = (
    IneligibleReason.SOURCE_EXCEEDS_MAX_BYTES,
    IneligibleReason.SOURCE_INSPECTION_REJECTED,
    IneligibleReason.UNSUPPORTED_CONTAINER,
    IneligibleReason.UNSUPPORTED_SUBTYPE,
    IneligibleReason.UNSUPPORTED_CHANNEL_COUNT,
    IneligibleReason.SOURCE_BELOW_MIN_SAMPLE_RATE,
    IneligibleReason.SOURCE_EXCEEDS_MAX_DURATION,
    IneligibleReason.SOURCE_EXCEEDS_MAX_DECODED_BYTES,
    IneligibleReason.SOURCE_DECODE_REJECTED,
    IneligibleReason.SOURCE_METADATA_INCONSISTENT,
    IneligibleReason.EMPTY_SOURCE,
    IneligibleReason.NONFINITE_SOURCE,
    IneligibleReason.INVALID_SOURCE_AMPLITUDE,
)

REASON_STAGE: Mapping[IneligibleReason, EvaluationStage] = MappingProxyType(
    {
        IneligibleReason.SOURCE_EXCEEDS_MAX_BYTES: EvaluationStage.LOCAL_FILE_CHECK,
        IneligibleReason.SOURCE_INSPECTION_REJECTED: EvaluationStage.INSPECTION,
        IneligibleReason.UNSUPPORTED_CONTAINER: EvaluationStage.METADATA_VALIDATION,
        IneligibleReason.UNSUPPORTED_SUBTYPE: EvaluationStage.METADATA_VALIDATION,
        IneligibleReason.UNSUPPORTED_CHANNEL_COUNT: EvaluationStage.METADATA_VALIDATION,
        IneligibleReason.SOURCE_BELOW_MIN_SAMPLE_RATE: EvaluationStage.METADATA_VALIDATION,
        IneligibleReason.SOURCE_EXCEEDS_MAX_DURATION: EvaluationStage.METADATA_VALIDATION,
        IneligibleReason.SOURCE_EXCEEDS_MAX_DECODED_BYTES: EvaluationStage.METADATA_VALIDATION,
        IneligibleReason.SOURCE_DECODE_REJECTED: EvaluationStage.DECODE,
        IneligibleReason.SOURCE_METADATA_INCONSISTENT: EvaluationStage.DECODED_METADATA_VALIDATION,
        IneligibleReason.EMPTY_SOURCE: EvaluationStage.WAVEFORM_VALIDATION,
        IneligibleReason.NONFINITE_SOURCE: EvaluationStage.WAVEFORM_VALIDATION,
        IneligibleReason.INVALID_SOURCE_AMPLITUDE: EvaluationStage.WAVEFORM_VALIDATION,
    }
)

# The golden fixture is decoded before any candidate is opened and an environment mismatch
# is established before the run touches audio, so naming a candidate on either would claim
# a run-global failure happened to one recording. A hash mismatch, an unstable artifact,
# and a non-regular file are always about a specific artifact and cannot omit it. The rest
# can arise either while processing a candidate or outside one, so they stay optional
# rather than being forced into a shape the evidence does not support.
ABORT_SOURCE_CONTEXT: Mapping[AbortReason, SourceContextPolicy] = MappingProxyType(
    {
        AbortReason.GOLDEN_FIXTURE_FAILURE: SourceContextPolicy.FORBIDDEN,
        AbortReason.ENVIRONMENT_MISMATCH: SourceContextPolicy.FORBIDDEN,
        AbortReason.SOURCE_HASH_MISMATCH: SourceContextPolicy.REQUIRED,
        AbortReason.LOCAL_ARTIFACT_UNSTABLE: SourceContextPolicy.REQUIRED,
        AbortReason.LOCAL_ARTIFACT_NOT_REGULAR_FILE: SourceContextPolicy.REQUIRED,
        AbortReason.ENVIRONMENT_CAPACITY_FAILURE: SourceContextPolicy.OPTIONAL,
        AbortReason.UNEXPECTED_IO_FAILURE: SourceContextPolicy.OPTIONAL,
        AbortReason.INTERNAL_CANONICALISATION_ERROR: SourceContextPolicy.OPTIONAL,
    }
)

# Protocol v4: a digest computed during a read later found unstable identifies nothing, and
# a digest present in a record reads as an established identity.
DIGEST_FORBIDDEN_ABORTS: frozenset[AbortReason] = frozenset({AbortReason.LOCAL_ARTIFACT_UNSTABLE})

_STAGE_POSITION = {stage: index for index, stage in enumerate(EVALUATION_STAGE_ORDER)}
_PRECEDENCE_POSITION = {reason: index for index, reason in enumerate(REASON_PRECEDENCE)}
_DECODER_REJECTIONS = frozenset(
    {IneligibleReason.SOURCE_INSPECTION_REJECTED, IneligibleReason.SOURCE_DECODE_REJECTED}
)


def terminating_stage_for(reason: IneligibleReason) -> EvaluationStage:
    """The stage that establishes this reason.

    Args:
        reason: The primary reason a source was found ineligible.

    Returns:
        The single stage mapped to that reason.
    """
    return REASON_STAGE[reason]


def unevaluated_stages_after(stage: EvaluationStage) -> tuple[EvaluationStage, ...]:
    """The stages never reached, in stage order.

    Derived rather than authored, and ordered rather than a set: two implementations
    serialising the same stages differently would produce different manifest bytes.

    Args:
        stage: The stage that terminated processing.

    Returns:
        The suffix of the stage order after `stage`.
    """
    return EVALUATION_STAGE_ORDER[_STAGE_POSITION[stage] + 1 :]


@dataclass(frozen=True, init=False)
class SourceIneligibleOutcome:
    """A candidate that will not be scored, and the evidence for that.

    Published as a finding. There is one construction path and it validates: the derived
    fields cannot be supplied, and every combination protocol v4 declares impossible is
    refused.
    """

    logical_source_id: str
    primary_reason_code: IneligibleReason
    terminating_evaluation_stage: EvaluationStage
    unevaluated_evaluation_stages: tuple[EvaluationStage, ...]
    supplemental_reason_codes: tuple[IneligibleReason, ...]
    identity: SourceIdentity
    protocol_sha256: str
    observed_metadata: ObservedSourceMetadata | None
    decoder_diagnostic: SanitisedDiagnostic | None

    def __init__(
        self,
        *,
        logical_source_id: str,
        primary_reason_code: IneligibleReason,
        identity: SourceIdentity,
        protocol_sha256: str,
        supplemental_reason_codes: Iterable[IneligibleReason] = (),
        observed_metadata: ObservedSourceMetadata | None = None,
        decoder_diagnostic: SanitisedDiagnostic | None = None,
    ) -> None:
        """Build a record, deriving what follows and rejecting what cannot be true.

        Args:
            logical_source_id: Identifier of the candidate within the frozen universe.
            primary_reason_code: The single counted reason.
            identity: What was established about the bytes.
            protocol_sha256: Digest of the protocol in force.
            supplemental_reason_codes: Further violations established in the same stage.
            observed_metadata: Required exactly when inspection succeeded.
            decoder_diagnostic: Required exactly when the decoder rejected the artifact,
                and obtainable only as a `SanitisedDiagnostic`, so native text cannot arrive
                here as a plain string.

        Raises:
            SourceOutcomeError: If an identifier is blank or a digest malformed, a
                supplemental reason repeats the primary or another supplemental or belongs
                to a different stage, or conditional evidence is present when the stage
                could not have produced it or absent when the stage did.
        """
        stage = terminating_stage_for(primary_reason_code)
        supplied = tuple(supplemental_reason_codes)
        if len(set(supplied)) != len(supplied):
            raise SourceOutcomeError(
                f"supplemental reasons contain a duplicate: {supplied}; a repeat is a caller "
                "defect and collapsing it silently would double-count one narrowing"
            )
        if primary_reason_code in supplied:
            raise SourceOutcomeError(
                f"{primary_reason_code!r} is the primary reason and cannot also be supplemental"
            )
        for reason in supplied:
            other = terminating_stage_for(reason)
            if other is not stage:
                raise SourceOutcomeError(
                    f"supplemental reason {reason!r} belongs to stage {other!r}, but processing "
                    f"terminated at {stage!r}; an earlier stage would already have stopped the "
                    "run and a later one was never reached"
                )

        if observed_metadata is not None and not isinstance(
            observed_metadata, ObservedSourceMetadata
        ):
            raise SourceOutcomeError(
                f"observed_metadata must be an ObservedSourceMetadata, got "
                f"{type(observed_metadata).__name__}; a bare mapping would reintroduce "
                "arbitrary keys and unvalidated values behind a frozen record"
            )
        if decoder_diagnostic is not None and not isinstance(
            decoder_diagnostic, SanitisedDiagnostic
        ):
            raise SourceOutcomeError(
                f"decoder_diagnostic must be a SanitisedDiagnostic, got "
                f"{type(decoder_diagnostic).__name__}; native text reaches a record only "
                "through sanitise_decoder_diagnostic"
            )
        _check_conditional_evidence(
            primary_reason_code, stage, observed_metadata, decoder_diagnostic
        )

        object.__setattr__(
            self, "logical_source_id", checked_identifier(logical_source_id, "logical_source_id")
        )
        object.__setattr__(self, "primary_reason_code", primary_reason_code)
        object.__setattr__(self, "terminating_evaluation_stage", stage)
        object.__setattr__(self, "unevaluated_evaluation_stages", unevaluated_stages_after(stage))
        object.__setattr__(
            self,
            "supplemental_reason_codes",
            tuple(sorted(supplied, key=lambda reason: _PRECEDENCE_POSITION[reason])),
        )
        object.__setattr__(self, "identity", identity)
        object.__setattr__(
            self, "protocol_sha256", checked_digest(protocol_sha256, "protocol_sha256")
        )
        object.__setattr__(self, "observed_metadata", observed_metadata)
        object.__setattr__(self, "decoder_diagnostic", decoder_diagnostic)


def _check_conditional_evidence(
    primary_reason_code: IneligibleReason,
    stage: EvaluationStage,
    observed_metadata: ObservedSourceMetadata | None,
    decoder_diagnostic: SanitisedDiagnostic | None,
) -> None:
    """Require each piece of evidence exactly where the stage produces it.

    Both directions, not one. Rejecting evidence the stage could not produce stops a record
    claiming work that did not happen; requiring it where the stage did produce it stops a
    record knowing less than it claims. A decode rejection with no diagnostic at all would
    otherwise be accepted, and every observed libsndfile rejection carries a message.

    Args:
        primary_reason_code: The single counted reason.
        stage: The stage that terminated processing.
        observed_metadata: Candidate metadata, if any.
        decoder_diagnostic: Sanitised decoder message, if any.

    Raises:
        SourceOutcomeError: If either piece of evidence disagrees with the stage.
    """
    inspection_succeeded = _STAGE_POSITION[stage] > _STAGE_POSITION[EvaluationStage.INSPECTION]
    if inspection_succeeded != (observed_metadata is not None):
        raise SourceOutcomeError(
            f"processing terminated at {stage!r}, so observed_metadata must be "
            f"{'present' if inspection_succeeded else 'absent'}"
        )
    decoder_rejected = primary_reason_code in _DECODER_REJECTIONS
    if decoder_rejected != (decoder_diagnostic is not None):
        raise SourceOutcomeError(
            f"{primary_reason_code!r} requires decoder_diagnostic to be "
            f"{'present' if decoder_rejected else 'absent'}"
        )


@dataclass(frozen=True, init=False)
class RunAbortOutcome:
    """A harness failure. The run stops and publication is blocked.

    Deliberately not a `SourceIneligibleOutcome` with a different reason field. The two
    vocabularies are separate in protocol v4 so a harness failure cannot be spelled as a
    source verdict, and keeping the types separate carries that into the code.
    """

    reason: AbortReason
    detail: str
    logical_source_id: str | None
    identity: SourceIdentity | None

    def __init__(
        self,
        *,
        reason: AbortReason,
        detail: str,
        logical_source_id: str | None = None,
        identity: SourceIdentity | None = None,
    ) -> None:
        """Build an abort record under the source-context policy for this reason.

        Args:
            reason: Why the run stopped.
            detail: Sanitised description for the run log.
            logical_source_id: The candidate being processed, per the policy for `reason`.
            identity: What had been established about the bytes, where that survives.

        Raises:
            SourceOutcomeError: If the detail is blank, the source context violates the
                policy for this reason, or an identity is attached to an abort whose digest
                the protocol requires be discarded.
        """
        checked_identifier(detail, "abort detail")
        policy = ABORT_SOURCE_CONTEXT[reason]
        named = logical_source_id is not None
        if policy is SourceContextPolicy.FORBIDDEN and named:
            raise SourceOutcomeError(
                f"{reason!r} is established before any candidate is opened, so naming "
                f"{logical_source_id!r} would claim a run-global failure happened to one "
                "recording"
            )
        if policy is SourceContextPolicy.REQUIRED and not named:
            raise SourceOutcomeError(
                f"{reason!r} is always about a specific artifact, so the record must name it"
            )
        if identity is not None and reason in DIGEST_FORBIDDEN_ABORTS:
            raise SourceOutcomeError(
                f"{reason!r} means the artifact changed under the read, so any digest computed "
                "during it identifies nothing and must be discarded rather than recorded"
            )
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "detail", detail)
        object.__setattr__(
            self,
            "logical_source_id",
            None
            if logical_source_id is None
            else checked_identifier(logical_source_id, "logical_source_id"),
        )
        object.__setattr__(self, "identity", identity)
