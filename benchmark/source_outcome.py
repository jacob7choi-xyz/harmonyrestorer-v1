"""Outcome records for a source that could not become a canonical waveform.

Two types, and the split is the point. A `SourceIneligibleOutcome` is an observation
about a candidate: it is recorded, published, and the run continues. A `RunAbortOutcome`
stops the run and blocks publication. Protocol v4 holds their reason codes in separate
closed vocabularies so that a harness failure is not merely discouraged from becoming a
source verdict but has no spelling as one, and these types carry that separation into
the type system: neither can be constructed with the other's reason.

**There is one construction path per type and it validates.** A first draft put the rules
in a `create` classmethod and left the generated constructor public, which made the rules a
convention rather than a boundary: a record with the wrong terminating stage, a duplicated
supplemental reason, and a fabricated digest constructed without complaint. Both types now
declare `init=False` and define their own `__init__`, so there is no unvalidated path and
no way to supply a derived field at all.

Everything correlated is derived rather than accepted. The terminating stage follows from
the primary reason, the unevaluated stages follow from the terminating stage, and the
identity status and method follow from the source size against the frozen bounds. A caller
that cannot supply a value cannot supply a wrong one, which matters here because a record
is the only surviving evidence about a candidate the benchmark rejected.

Ordering is normalised, not accepted. Two implementations holding the same supplemental
set and serialising it differently would produce different manifest bytes and therefore
different digests, which would break the content addressing the benchmark rests on.
Duplicates are rejected before normalisation rather than absorbed by it: a repeated code
is a caller defect, and silently collapsing it would hide the bug and double-count the
narrowing it describes.

Evidence is required in both directions. Metadata must be present exactly when inspection
succeeded and absent otherwise; a decoder diagnostic must be present exactly when the
decoder rejected the artifact. Enforcing only the forbidding direction would accept a
decode rejection carrying no diagnostic at all, which is a record that quietly knows less
than it claims.

The vocabularies below are code-owned, and `verify_source_outcome_contract` checks every
one of them against a loaded protocol. That check is deliberately not run at import.
Importing this module must not depend on library versions, filesystem state, or the
production config path, because that would make type checking, test collection, and
mutation runs fail for reasons unrelated to what they are testing. Drift is caught by the
test that calls the verifier, which is where a mismatch should surface.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from enum import StrEnum
from types import MappingProxyType

from benchmark.protocol import Decode, Protocol, SourceOutcomes

_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z")

# Native decoder text never reaches a published record. libsndfile puts the absolute path
# of the offending file in its very first message, so this is a live hazard rather than a
# precaution.
#
# An earlier draft accepted an arbitrary string and rejected it if it contained a path
# separator. That is a validator wearing a sanitiser's clothes, and it was wrong in both
# directions: it rejected "unsupported subtype PCM_24/32" while accepting
# "C:UsersJacobtrack.flac", a bare filename, a username, and an embedded newline. Safety
# cannot be inferred from the absence of two characters.
#
# What replaces it is a category the code already knows, because the code knows which
# operation failed, plus an optional native fragment that is published only if the whole
# message matches a closed allowlist. Anything unrecognised contributes no detail at all,
# which is the fail-closed direction.
_PUBLISHABLE_DETAILS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\AInternal psf_fseek\(\) failed\.\Z"),
    re.compile(r"\AFile contains data in an unknown format\.\Z"),
    re.compile(r"\AUnsupported encoding\.\Z"),
    re.compile(r"\ABad offset\.\Z"),
)
# Measured rather than assumed: all 26 formats and 33 subtypes soundfile reports match this
# grammar, and the longest is MPEG_LAYER_III at 14 characters. The bound is evidence-backed
# headroom over that, and exists so an unexpected decoder token cannot put an unbounded
# string into a published manifest.
_TOKEN = re.compile(r"\A[A-Z][A-Z0-9_]*\Z")
_MAX_TOKEN_CHARS = 32


class SourceOutcomeError(Exception):
    """Raised when a record would assert something that did not happen."""


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


class IdentityStatus(StrEnum):
    """Whether a whole-artifact digest was established."""

    COMPLETE_SHA256 = "complete_sha256"
    UNAVAILABLE_ABOVE_IDENTITY_STREAM_BOUND = "unavailable_above_identity_stream_bound"


class IdentityMethod(StrEnum):
    """How the digest was established, and nothing else.

    These names never mention inspection or decoding. A source rejected at inspection or
    on its subtype is never decoded, so a method naming those steps would have the record
    assert execution that did not occur. What ran is carried by the terminating stage.
    """

    BOUNDED_SINGLE_BUFFER_SHA256 = "bounded_single_buffer_sha256"
    BOUNDED_STREAMING_SHA256 = "bounded_streaming_sha256"
    NOT_COMPUTED_ABOVE_BOUND = "not_computed_above_bound"


class DiagnosticCategory(StrEnum):
    """Which operation the decoder refused.

    Always available, because the code knows which call it made. An earlier draft required
    a native message on every decoder rejection, justified by three fixtures under one
    libsndfile build. That proves availability in those cases, not universality: a
    different build, a wrapper-raised error, or an externally killed decode can carry
    nothing. The category never has to be invented.
    """

    INSPECTION_REJECTED = "decoder_rejected_during_inspection"
    DECODE_REJECTED = "decoder_rejected_during_decode"


@dataclass(frozen=True, init=False)
class SanitisedDiagnostic:
    """What a published record may say about a decoder refusal.

    Two supported construction paths, both validated, and neither able to carry arbitrary
    text. `sanitise_decoder_diagnostic` is the one callers holding a native message use;
    this constructor is equally approved and enforces the same allowlist, which is why a
    raw string cannot reach a record through either. An earlier docstring claimed the
    sanitiser was the only way, which was false about this class while remaining true about
    what can be published.

    The allowlist is a **code-owned privacy policy**, not a protocol value. Protocol v4
    requires a sanitised diagnostic and says nothing about which native fragments are safe
    to repeat. When a library changes its wording, the category is unchanged and the detail
    simply becomes absent, so nothing about a source's verdict, eligibility, or place in the
    reduction chain moves.
    """

    category: DiagnosticCategory
    detail: str | None

    def __init__(self, category: DiagnosticCategory, detail: str | None = None) -> None:
        """Accept a category and, at most, an allowlisted fragment.

        Args:
            category: Which operation the decoder refused.
            detail: A native fragment that has already matched the allowlist.

        Raises:
            SourceOutcomeError: If the detail is not one this module publishes. Arbitrary
                text is refused here rather than inspected for suspicious characters.
        """
        if detail is not None and not any(p.fullmatch(detail) for p in _PUBLISHABLE_DETAILS):
            raise SourceOutcomeError(
                f"{detail!r} is not an allowlisted publishable fragment; build diagnostics "
                "with sanitise_decoder_diagnostic rather than passing native text"
            )
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "detail", detail)


def sanitise_decoder_diagnostic(
    raw_message: str, *, category: DiagnosticCategory
) -> SanitisedDiagnostic:
    """Turn a native decoder message into something a record may carry.

    The message is published only if it matches the allowlist exactly. Anything else
    yields the category alone, which is truthful and leaks nothing.

    A draft of this also redacted the known source path and bounded the result in bytes.
    Both were unreachable and were removed rather than kept as reassurance: every allowlist
    entry is a fixed string with no variable part, so nothing that matches can carry a path
    or exceed a length, and substituting a path could only ever turn a match into a
    non-match. If the allowlist is ever widened to admit variable content, both controls
    have to come back with it, and the mutation sweep will say so by surviving.

    Args:
        raw_message: The native decoder text. The caller keeps it for the run log; it does
            not survive into the returned value unless it matched the allowlist.
        category: Which operation the decoder refused.

    Returns:
        A diagnostic safe to publish.
    """
    text = raw_message.strip()
    matched = any(pattern.fullmatch(text) for pattern in _PUBLISHABLE_DETAILS)
    return SanitisedDiagnostic(category, text if matched else None)


@dataclass(frozen=True, init=False)
class ObservedSourceMetadata:
    """Decoder-reported facts about a source, each validated in its own domain.

    A generic mapping of scalars was too weak: `bool` is an `int` in Python, so a channel
    count of `True` passed, and NaN, infinity, a blank string, and a ten-thousand character
    value all passed with it. This project has already been bitten by bool-as-int once.
    Typed fields also stop two implementations publishing different metadata keys.

    Per-field validity is not enough on its own. `duration_seconds` and
    `projected_decoded_bytes` both follow from the other fields and are therefore derived,
    because a value object whose fields are individually valid can still describe two
    different files.
    """

    container: str
    subtype: str
    channels: int
    sample_rate: int
    frames: int
    duration_seconds: float
    projected_decoded_bytes: int

    def __init__(
        self,
        *,
        container: str,
        subtype: str,
        channels: int,
        sample_rate: int,
        frames: int,
    ) -> None:
        """Validate each field, then derive everything that follows from the others.

        Args:
            container: Decoder-reported container, uppercase.
            subtype: Decoder-reported subtype, uppercase.
            channels: Channel count, positive.
            sample_rate: Sample rate in hertz, positive.
            frames: Frame count, non-negative.

        Raises:
            SourceOutcomeError: If any field is outside its domain, including a `bool`
                where an integer is required or a token over the length bound.
        """
        for name, value in (("container", container), ("subtype", subtype)):
            _checked_identifier(value, f"metadata {name}")
            if not _TOKEN.fullmatch(value):
                raise SourceOutcomeError(
                    f"metadata {name} {value!r} is not an uppercase decoder token"
                )
            if len(value) > _MAX_TOKEN_CHARS:
                raise SourceOutcomeError(
                    f"metadata {name} is {len(value)} characters, over the "
                    f"{_MAX_TOKEN_CHARS} bound; the longest token soundfile reports is 14"
                )
        object.__setattr__(self, "container", container)
        object.__setattr__(self, "subtype", subtype)
        object.__setattr__(self, "channels", _checked_int(channels, "channels", minimum=1))
        object.__setattr__(self, "sample_rate", _checked_int(sample_rate, "sample_rate", minimum=1))
        object.__setattr__(self, "frames", _checked_int(frames, "frames", minimum=0))
        # Derived, not accepted. A caller supplying all three could author 16,000 frames at
        # 16 kHz lasting 900 seconds: every field valid on its own, and together two
        # different files. That is the correlated-field defect this whole type exists to
        # remove, so duration follows from the frames and the rate exactly as the projected
        # size follows from the frames and the channels.
        object.__setattr__(self, "duration_seconds", frames / sample_rate)
        object.__setattr__(self, "projected_decoded_bytes", frames * channels * 8)


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


def identity_for_size(source_bytes: int, config: Decode) -> tuple[IdentityStatus, IdentityMethod]:
    """The status and method the frozen bounds permit at this size.

    A pure derivation, shared by every caller, so the status and the method can never be
    chosen independently and disagree.

    Args:
        source_bytes: Observed size of the local artifact.
        config: The frozen `canonicalisation.decode` section.

    Returns:
        The status and method for that size.
    """
    if source_bytes > config.max_identity_stream_bytes:
        return (
            IdentityStatus.UNAVAILABLE_ABOVE_IDENTITY_STREAM_BOUND,
            IdentityMethod.NOT_COMPUTED_ABOVE_BOUND,
        )
    if source_bytes > config.max_source_bytes:
        return IdentityStatus.COMPLETE_SHA256, IdentityMethod.BOUNDED_STREAMING_SHA256
    return IdentityStatus.COMPLETE_SHA256, IdentityMethod.BOUNDED_SINGLE_BUFFER_SHA256


def _checked_digest(value: str, field: str) -> str:
    """Require exactly 64 lowercase hexadecimal characters.

    Args:
        value: The candidate digest.
        field: Field name, for the error message.

    Returns:
        The value unchanged.

    Raises:
        SourceOutcomeError: If it is not a SHA-256 digest in canonical form. Case and
            surrounding whitespace are rejected rather than normalised, because a record
            that accepts two spellings of one digest breaks content addressing.
    """
    if not _SHA256.fullmatch(value):
        raise SourceOutcomeError(
            f"{field} must be 64 lowercase hexadecimal characters, got {value!r}"
        )
    return value


def _checked_int(value: int, field: str, *, minimum: int) -> int:
    """Require a plain integer at or above a floor.

    Args:
        value: The candidate.
        field: Field name, for the error message.
        minimum: Smallest permitted value.

    Returns:
        The value unchanged.

    Raises:
        SourceOutcomeError: If it is a `bool` or below the floor. `bool` is rejected
            explicitly because `isinstance(True, int)` is true, so a channel count of
            `True` would otherwise be recorded as one channel.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise SourceOutcomeError(f"{field} must be a plain integer, got {value!r}")
    if value < minimum:
        raise SourceOutcomeError(f"{field} must be at least {minimum}, got {value}")
    return value


def _checked_finite(value: float, field: str) -> float:
    """Require a finite, non-negative real number.

    Args:
        value: The candidate.
        field: Field name, for the error message.

    Returns:
        The value unchanged.

    Raises:
        SourceOutcomeError: If it is a `bool`, not a number, non-finite, or negative. NaN
            and infinity are refused because a record carrying either states nothing while
            appearing to state something.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SourceOutcomeError(f"{field} must be a real number, got {value!r}")
    if not math.isfinite(value) or value < 0:
        raise SourceOutcomeError(f"{field} must be finite and non-negative, got {value!r}")
    return float(value)


def _checked_identifier(value: str, field: str) -> str:
    """Require a non-blank identifier with no surrounding whitespace.

    Args:
        value: The candidate identifier.
        field: Field name, for the error message.

    Returns:
        The value unchanged.

    Raises:
        SourceOutcomeError: If it is empty, blank, or padded. Blank-as-present is how a
            record ends up naming nothing while appearing complete.
    """
    if not value or not value.strip() or value.strip() != value:
        raise SourceOutcomeError(f"{field} must be non-blank and unpadded, got {value!r}")
    return value


@dataclass(frozen=True, init=False)
class SourceIdentity:
    """What was established about the bytes, and how.

    `status` and `method` are derived from the size against the frozen bounds and cannot be
    supplied, so the two can never disagree with each other or with the size. `source_bytes`
    is carried because the method cannot be audited later without it.
    """

    source_bytes: int
    status: IdentityStatus
    method: IdentityMethod
    sha256: str | None

    def __init__(self, source_bytes: int, sha256: str | None, config: Decode) -> None:
        """Derive the status and method, then validate the digest against them.

        Args:
            source_bytes: Observed size of the local artifact.
            sha256: The digest, or None above the streaming bound.
            config: The frozen `canonicalisation.decode` section.

        Raises:
            SourceOutcomeError: If the size is negative, a digest is supplied above the
                streaming bound, a digest is missing at or below it, or the digest is not
                in canonical form.
        """
        _checked_int(source_bytes, "source_bytes", minimum=0)
        status, method = identity_for_size(source_bytes, config)
        complete = status is IdentityStatus.COMPLETE_SHA256
        if complete and sha256 is None:
            raise SourceOutcomeError(
                f"a {source_bytes}-byte source is within the identity stream bound, so a "
                "digest is required"
            )
        if not complete and sha256 is not None:
            raise SourceOutcomeError(
                f"a {source_bytes}-byte source is above the identity stream bound, so no "
                "digest can have been computed for it"
            )
        object.__setattr__(self, "source_bytes", source_bytes)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "method", method)
        object.__setattr__(
            self, "sha256", None if sha256 is None else _checked_digest(sha256, "sha256")
        )


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
            self, "logical_source_id", _checked_identifier(logical_source_id, "logical_source_id")
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
            self, "protocol_sha256", _checked_digest(protocol_sha256, "protocol_sha256")
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
        _checked_identifier(detail, "abort detail")
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
            else _checked_identifier(logical_source_id, "logical_source_id"),
        )
        object.__setattr__(self, "identity", identity)


def verify_source_outcome_contract(protocol: Protocol) -> None:
    """Check every vocabulary, ordering, and mapping here against the protocol in force.

    Called explicitly rather than at import. Importing this module must not depend on
    library versions, filesystem state, or the production config path, or type checking
    and test collection would fail for reasons unrelated to what they are checking. The
    test suite calls this, which is where drift should surface.

    Args:
        protocol: A loaded, validated protocol.

    Raises:
        SourceOutcomeError: If anything this module owns disagrees with it.
    """
    outcomes = protocol.source_outcomes
    for name, mine, theirs in (
        ("evaluation_stage_order", EVALUATION_STAGE_ORDER, outcomes.evaluation_stage_order),
        ("reason_precedence", REASON_PRECEDENCE, outcomes.reason_precedence),
        ("abort_reasons", tuple(AbortReason), outcomes.abort_reasons),
        ("identity_status_values", tuple(IdentityStatus), outcomes.identity_status_values),
        ("identity_method_values", tuple(IdentityMethod), outcomes.identity_method_values),
    ):
        if tuple(str(value) for value in mine) != tuple(theirs):
            raise SourceOutcomeError(f"{name} disagrees with the protocol: {mine!r} vs {theirs!r}")

    if sorted(str(reason) for reason in IneligibleReason) != sorted(outcomes.ineligible_reasons):
        raise SourceOutcomeError("ineligible reason vocabulary disagrees with the protocol")
    if {str(k): str(v) for k, v in REASON_STAGE.items()} != dict(outcomes.reason_stage):
        raise SourceOutcomeError("reason-to-stage mapping disagrees with the protocol")
    # Completeness only. The policy itself is code-owned rather than loaded: protocol v4
    # scopes its record schema to source-ineligible outcomes and leaves abort-record
    # structure to the runner, so which aborts may name a candidate is a derivation from
    # v4's meanings rather than a value it states. Checking coverage here stops a new abort
    # reason arriving with no policy at all; it does not claim v4 authorises this map.
    if set(ABORT_SOURCE_CONTEXT) != set(AbortReason):
        raise SourceOutcomeError("the abort source-context policy does not cover every reason")

    _verify_record_shape(outcomes)
    _verify_identity_bounds(protocol.canonicalisation.decode)


def _verify_record_shape(outcomes: SourceOutcomes) -> None:
    """Check that this module carries every field and constraint the protocol names.

    Args:
        outcomes: The frozen `source_outcomes` section.

    Raises:
        SourceOutcomeError: If a required core field has no home here, a conditional field
            is not enforced, or a record constraint has no counterpart.
    """
    core = set(outcomes.record_required_core)
    present = {field.name for field in fields(SourceIneligibleOutcome)}
    # `identity_status`, `identity_method`, and `local_sha256` live on the identity object
    # rather than flattened onto the record, so they are satisfied by that field.
    present |= {"identity_status", "identity_method", "local_sha256"}
    missing = core - present
    if missing:
        raise SourceOutcomeError(f"record is missing required core fields: {sorted(missing)}")

    conditional = dict(outcomes.record_conditional)
    expected_conditions = {
        "local_sha256": "when_identity_complete_sha256",
        "observed_metadata": "when_inspection_succeeded",
        "decoder_diagnostic": "when_decoder_rejected",
        "supplemental_reason_codes": "when_additional_reasons_established",
    }
    if conditional != expected_conditions:
        raise SourceOutcomeError(f"record conditional fields disagree: {conditional!r}")

    unenforced = set(outcomes.record_constraints) - _ENFORCED_CONSTRAINTS
    if unenforced:
        raise SourceOutcomeError(
            f"record constraints with no enforcement here: {sorted(unenforced)}"
        )


def _verify_identity_bounds(decode: Decode) -> None:
    """Check that the derivation agrees with the frozen bounds at every boundary.

    Args:
        decode: The frozen `canonicalisation.decode` section.

    Raises:
        SourceOutcomeError: If a boundary size selects the wrong method.
    """
    expected = (
        (decode.max_source_bytes, IdentityMethod.BOUNDED_SINGLE_BUFFER_SHA256),
        (decode.max_source_bytes + 1, IdentityMethod.BOUNDED_STREAMING_SHA256),
        (decode.max_identity_stream_bytes, IdentityMethod.BOUNDED_STREAMING_SHA256),
        (decode.max_identity_stream_bytes + 1, IdentityMethod.NOT_COMPUTED_ABOVE_BOUND),
    )
    for size, method in expected:
        if identity_for_size(size, decode)[1] is not method:
            raise SourceOutcomeError(f"identity derivation disagrees with the bounds at {size}")


# Every constraint protocol v4 states about a record, and where it is enforced. A
# constraint here with no enforcement is documentation, so the verifier refuses one it does
# not recognise rather than letting it pass unnoticed.
_ENFORCED_CONSTRAINTS = frozenset(
    {
        "primary_reason_code_in_ineligible_reasons",
        "supplemental_reason_codes_subset_of_ineligible_reasons",
        "primary_reason_code_not_in_supplemental_reason_codes",
        "supplemental_reason_codes_unique",
        "supplemental_reason_codes_share_primary_reason_stage",
        "supplemental_reason_codes_ordered_by_reason_precedence",
        "unevaluated_evaluation_stages_ordered_by_evaluation_stage_order",
        "identity_status_consistent_with_identity_method",
        "local_sha256_present_iff_identity_complete",
        "identity_method_consistent_with_source_byte_bounds",
        "digest_discarded_when_instability_detected",
    }
)
