"""Decoding of an inspected source, and the checks on what the decoder returned.

`benchmark.source_inspection` reads the container's description and checks it against the
frozen bounds without asking for samples. This module runs only after those checks pass:
it decodes the same buffer, confirms the array matches what inspection reported, and
validates the samples. The result is the float64 array canonicalisation starts from, or
the ineligibility record for the stage that stopped it.

**The allocation is sized by checked metadata.** soundfile opens the buffer again and
would otherwise size its array from that second open's frame count. Passing the inspected
frame count as `frames` caps the read at the number `metadata_violations` checked, because
soundfile 0.14.0 reads the smaller of the requested and the available frames
(`SoundFile._check_frames`).

**The stage decides the reason, never the decoder's message.** A decoder error during
decode is `SOURCE_DECODE_REJECTED` whatever its text says, because v4 treats the native
message as provenance rather than classification. A header that overstates the sample
count is therefore a decode rejection, not an inconsistency, when the decoder raises on it.

**What the decoder was seen to do.** With soundfile 0.14.0 and libsndfile 1.2.2 on macOS,
truncated buffers, overwritten bytes, and a header claiming more samples than the stream
holds all raised `LibsndfileError` during decode. A header claiming fewer samples decoded
exactly that many and dropped the rest without an error. No buffer tried produced a
decoded shape that disagreed with inspection, an empty array, a non-finite sample, or a
sample outside [-1.0, 1.0]. Those checks are here because v4 requires them and because the
observations come from one platform and a limited set of buffers.

A decode rejection may be charged to the source only after the run-level preconditions
hold, the golden fixture among them. That ordering belongs to the runner, which is planned
and not yet written. A `MemoryError` is not caught here; the planned runner maps it to
`ENVIRONMENT_CAPACITY_FAILURE`, which aborts rather than recording a source verdict.
"""

from __future__ import annotations

import io
import logging
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import soundfile as sf

from benchmark.protocol import Canonicalisation
from benchmark.source_acquisition import ArtifactBuffered
from benchmark.source_evidence import (
    DiagnosticCategory,
    ObservedSourceMetadata,
    sanitise_decoder_diagnostic,
)
from benchmark.source_inspection import InspectedSource
from benchmark.source_outcome import IneligibleReason, SourceIneligibleOutcome

__all__ = ["decode_source", "decoded_metadata_matches", "waveform_violations"]

logger = logging.getLogger(__name__)


def decode_source(
    artifact: ArtifactBuffered,
    inspected: InspectedSource,
    *,
    logical_source_id: str,
    protocol_sha256: str,
    config: Canonicalisation,
) -> npt.NDArray[np.float64] | SourceIneligibleOutcome:
    """Decode the inspected buffer and validate the samples.

    Args:
        artifact: The acquired artifact whose buffer `inspect_source` admitted.
        inspected: What inspection reported for that buffer.
        logical_source_id: Identifier of the candidate, for the record and the log.
        protocol_sha256: Digest of the protocol in force, for the record.
        config: The frozen `canonicalisation` section.

    Returns:
        The decoded samples, shaped (frames, channels), otherwise the ineligibility record.
    """
    metadata = inspected.metadata
    samples: npt.NDArray[np.float64]
    sample_rate: int
    try:
        samples, sample_rate = sf.read(
            io.BytesIO(artifact.data),
            frames=metadata.frames,
            # The loader refuses any dtype but float64 (protocol.py, _DECODE_DTYPE).
            dtype=cast(Literal["float64"], config.decode.dtype),
            always_2d=config.decode.always_2d,
        )
    except sf.LibsndfileError as error:
        return _decode_rejected(artifact, error, metadata, logical_source_id, protocol_sha256)
    if not decoded_metadata_matches(samples, sample_rate, metadata):
        violations: tuple[IneligibleReason, ...] = (IneligibleReason.SOURCE_METADATA_INCONSISTENT,)
    else:
        violations = waveform_violations(samples, config)
    if violations:
        return _ineligible(artifact, violations, metadata, logical_source_id, protocol_sha256)
    logger.debug("Source %s decoded: %d frames", logical_source_id, metadata.frames)
    return samples


def decoded_metadata_matches(
    samples: npt.NDArray[np.float64], sample_rate: int, metadata: ObservedSourceMetadata
) -> bool:
    """Whether the decoded array and rate are the ones inspection reported.

    Args:
        samples: The decoded array, two-dimensional.
        sample_rate: The rate the decoder returned with it.
        metadata: What inspection reported.

    Returns:
        True when frames, channels, and rate all agree.
    """
    return (
        samples.shape == (metadata.frames, metadata.channels)
        and sample_rate == metadata.sample_rate
    )


def waveform_violations(
    samples: npt.NDArray[np.float64], config: Canonicalisation
) -> tuple[IneligibleReason, ...]:
    """Every waveform-stage reason the samples violate, in v4's precedence order.

    An empty array returns at once: finiteness and amplitude cannot be violated by no
    samples, and `min` raises on an empty array.

    Each check is a reduction to one value. Measured with `tracemalloc` on a contiguous
    array of 4,000,000 samples, each reduction peaked below 1 KB, where
    `np.isfinite(samples).all()` peaked at about one byte per sample.

    v4 does not say how a non-finite sample relates to the amplitude range. Here NaN fails
    finiteness only, because it has no position to compare, while an infinity fails both,
    because it lies outside [-1.0, 1.0]. The question is recorded for the next protocol
    amendment.

    Args:
        samples: The decoded array, already shown to match inspection.
        config: The frozen `canonicalisation` section.

    Returns:
        The violated reasons, empty when the samples may be transformed.
    """
    if samples.size == 0:
        return (IneligibleReason.EMPTY_SOURCE,)
    reasons: list[IneligibleReason] = []
    # min and max propagate NaN, and an infinity is an extreme, so these two values are
    # finite exactly when every sample is.
    if not (np.isfinite(samples.min()) and np.isfinite(samples.max())):
        reasons.append(IneligibleReason.NONFINITE_SOURCE)
    # fmin and fmax skip NaN. The bounds are inclusive, which the loader enforces, so only
    # a value strictly beyond one violates.
    bounds = config.source_validation
    if (
        np.fmin.reduce(samples, axis=None) < bounds.amplitude_min
        or np.fmax.reduce(samples, axis=None) > bounds.amplitude_max
    ):
        reasons.append(IneligibleReason.INVALID_SOURCE_AMPLITUDE)
    return tuple(reasons)


def _ineligible(
    artifact: ArtifactBuffered,
    violations: tuple[IneligibleReason, ...],
    metadata: ObservedSourceMetadata,
    logical_source_id: str,
    protocol_sha256: str,
) -> SourceIneligibleOutcome:
    """The record for decoded samples that failed a check, the first reason primary."""
    logger.info(
        "Source %s is ineligible after decode: %s", logical_source_id, ", ".join(violations)
    )
    return SourceIneligibleOutcome(
        logical_source_id=logical_source_id,
        primary_reason_code=violations[0],
        supplemental_reason_codes=violations[1:],
        identity=artifact.identity,
        protocol_sha256=protocol_sha256,
        observed_metadata=metadata,
    )


def _decode_rejected(
    artifact: ArtifactBuffered,
    error: sf.LibsndfileError,
    metadata: ObservedSourceMetadata,
    logical_source_id: str,
    protocol_sha256: str,
) -> SourceIneligibleOutcome:
    """The record for a buffer inspection described and the decoder then refused.

    The raw message stays in the run log; the record carries only what the sanitiser's
    allowlist permits. Kept apart from `source_inspection._inspection_rejected` because
    the reason, the category, and the presence of metadata all differ.
    """
    logger.info("Decode rejected %s: %s", logical_source_id, error.error_string)
    return SourceIneligibleOutcome(
        logical_source_id=logical_source_id,
        primary_reason_code=IneligibleReason.SOURCE_DECODE_REJECTED,
        identity=artifact.identity,
        protocol_sha256=protocol_sha256,
        observed_metadata=metadata,
        decoder_diagnostic=sanitise_decoder_diagnostic(
            error.error_string, category=DiagnosticCategory.DECODE_REJECTED
        ),
    )
