"""Evidence about a source artifact: what was established, and how.

The lowest layer of the source-outcome model. Everything here describes a single artifact
and validates itself; nothing here knows what an evaluation stage or an ineligibility
reason is. `source_outcome` builds records out of these, and `source_outcome_contract`
checks both against the frozen protocol. The dependency runs one way only.

`SourceOutcomeError` lives here rather than in `source_outcome` because it is raised by the
lowest layer and the higher ones re-raise it. The name predates the split and is kept
deliberately: renaming it would change the exception a caller catches, which is a behaviour
change and not this commit's business.

These types may take frozen protocol sections as arguments, and never load the protocol
themselves. Importing this module must not depend on library versions, filesystem state, or
the production config path, because that would make type checking, test collection, and
mutation runs fail for reasons unrelated to what they are testing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from benchmark._source_validation import (
    SourceOutcomeError,
    checked_digest,
    checked_identifier,
    checked_int,
)
from benchmark.protocol import Decode

__all__ = [
    "DiagnosticCategory",
    "IdentityMethod",
    "IdentityStatus",
    "ObservedSourceMetadata",
    "SanitisedDiagnostic",
    "SourceIdentity",
    "SourceOutcomeError",
    "identity_for_size",
    "sanitise_decoder_diagnostic",
]

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


# Re-exported so callers of this module keep importing it from where they always did. This
# is the same class object, not an alias to a differently named one.


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
            checked_identifier(value, f"metadata {name}")
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
        object.__setattr__(self, "channels", checked_int(channels, "channels", minimum=1))
        object.__setattr__(self, "sample_rate", checked_int(sample_rate, "sample_rate", minimum=1))
        object.__setattr__(self, "frames", checked_int(frames, "frames", minimum=0))
        # Derived, not accepted. A caller supplying all three could author 16,000 frames at
        # 16 kHz lasting 900 seconds: every field valid on its own, and together two
        # different files. That is the correlated-field defect this whole type exists to
        # remove, so duration follows from the frames and the rate exactly as the projected
        # size follows from the frames and the channels.
        object.__setattr__(self, "duration_seconds", frames / sample_rate)
        object.__setattr__(self, "projected_decoded_bytes", frames * channels * 8)


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
        checked_int(source_bytes, "source_bytes", minimum=0)
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
            self, "sha256", None if sha256 is None else checked_digest(sha256, "sha256")
        )
