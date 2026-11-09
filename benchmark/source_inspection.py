"""Inspection of a buffered source, and the checks that run before samples are requested.

It reads the container's own description of the audio from the exact buffer acquisition
hashed, checks that description against the frozen bounds, and only then projects how
much memory the rest of canonicalisation will hold. This module calls `soundfile.info`
only; it never asks soundfile for samples.

**The decoder's report is the authority, never the filename.** Protocol v4 freezes
`metadata_authority` as `decoder_reported`, and its note records that a file named `.wav`
holding FLAC was reported as FLAC. The buffer is wrapped in `io.BytesIO`, so no path or
extension reaches the decoder from this module.

**Metadata is checked before it is multiplied.** Frames, channels, and sample rate come
from the container and are therefore controlled by whoever wrote the file. They are
compared against the frozen bounds first, and the live-memory projection is computed only
for metadata that passed, per v4's `metadata_validation`.

**Container before subtype, as a pair.** v4 admits (container, subtype) pairs rather than
two independent sets. A subtype is judged only inside a container some pair admits, so a
WAV file records `UNSUPPORTED_CONTAINER` alone. v4 does not say whether a subtype under an
unadmitted container is also "established" as unsupported; this module takes the reading
that it is not evaluated, and the question is recorded for the next protocol amendment.

**A sample count of "unknown" is believed, and excludes the file.** FLAC permits a header
whose total sample count is zero, meaning unknown. With libsndfile 1.2.2 from the locked
soundfile wheel, on macOS, such a file was reported as 9,223,372,036,854,775,807 frames;
other platforms were not checked. Under `decoder_reported` authority that is the file's
length, so it fails the duration and decoded-size bounds and is excluded before samples
are requested, which fails closed. The published reason then says the
source is too long when it may be short; the observed metadata carries the sentinel value,
so the record shows why. Whether v4 should name this case is recorded for the next
protocol amendment rather than decided here.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import soundfile as sf

from benchmark.protocol import Canonicalisation
from benchmark.source_acquisition import ArtifactBuffered
from benchmark.source_evidence import (
    DiagnosticCategory,
    ObservedSourceMetadata,
    sanitise_decoder_diagnostic,
)
from benchmark.source_outcome import IneligibleReason, SourceIneligibleOutcome

__all__ = [
    "FLOAT64_BYTES",
    "InspectedSource",
    "canonical_length",
    "inspect_source",
    "metadata_violations",
]

logger = logging.getLogger(__name__)

# Every array canonicalisation holds is float64, which `ObservedSourceMetadata` already
# assumes for `projected_decoded_bytes`.
FLOAT64_BYTES = 8


@dataclass(frozen=True)
class InspectedSource:
    """A source whose metadata passed every check in this module.

    `canonical_samples` is the exact length the resampler must produce; the resampling
    step, not yet written, is planned to assert it. `projected_live_bytes` is v4's
    `live_bytes_rule` for this source, kept for the planned comparison against measured
    peak memory.
    """

    metadata: ObservedSourceMetadata
    canonical_samples: int
    projected_live_bytes: int


def canonical_length(frames: int, source_rate: int, target_rate: int) -> int:
    """v4's `output_length_rule`, ceil(frames * target_rate / source_rate), in integers.

    Integer arithmetic because the rule is stated exactly, and in floating point a product
    that should be a whole number could in principle round past it. librosa 0.11.0
    computes its own length as `int(np.ceil(n * (float(target) / source)))`. The two
    agreed at every exact-integer point up to the 3600 s cap for the 18 sample rates
    checked (449,308,800 points), and no disagreement was found. That is evidence for
    those rates, not a proof for every rate, so the resampling step is planned to assert
    the length it receives.

    Args:
        frames: Source frame count, non-negative.
        source_rate: Source sample rate in hertz, positive.
        target_rate: Canonical sample rate in hertz, positive.

    Returns:
        The number of canonical samples.
    """
    return -(-frames * target_rate // source_rate)


def metadata_violations(
    metadata: ObservedSourceMetadata, config: Canonicalisation
) -> tuple[IneligibleReason, ...]:
    """Every metadata-stage reason this source violates, in v4's precedence order.

    All of them run, because v4 records every reason established before processing ends.
    They are evaluated over integers only, so each bound is compared exactly.

    Args:
        metadata: Decoder-reported facts about the source.
        config: The frozen `canonicalisation` section.

    Returns:
        The violated reasons, empty when the source may be decoded.
    """
    pairs = config.decode.source_formats
    reasons: list[IneligibleReason] = []
    if metadata.container not in {container for container, _ in pairs}:
        reasons.append(IneligibleReason.UNSUPPORTED_CONTAINER)
    elif (metadata.container, metadata.subtype) not in pairs:
        reasons.append(IneligibleReason.UNSUPPORTED_SUBTYPE)
    if not config.downmix.channels_min <= metadata.channels <= config.downmix.channels_max:
        reasons.append(IneligibleReason.UNSUPPORTED_CHANNEL_COUNT)
    if metadata.sample_rate < config.source_validation.min_sample_rate:
        reasons.append(IneligibleReason.SOURCE_BELOW_MIN_SAMPLE_RATE)
    # frames / rate > limit, rearranged so no division rounds the comparison.
    if metadata.frames > config.decode.max_source_duration_s * metadata.sample_rate:
        reasons.append(IneligibleReason.SOURCE_EXCEEDS_MAX_DURATION)
    if metadata.projected_decoded_bytes > config.decode.max_decoded_bytes:
        reasons.append(IneligibleReason.SOURCE_EXCEEDS_MAX_DECODED_BYTES)
    return tuple(reasons)


def inspect_source(
    artifact: ArtifactBuffered,
    *,
    logical_source_id: str,
    protocol_sha256: str,
    config: Canonicalisation,
) -> InspectedSource | SourceIneligibleOutcome:
    """Read the source's metadata from its buffer and check it before any decode.

    Args:
        artifact: The acquired artifact. Its `data` is the exact byte sequence its digest
            identifies, and the only thing inspected.
        logical_source_id: Identifier of the candidate, for the record and the log.
        protocol_sha256: Digest of the protocol in force, for the record.
        config: The frozen `canonicalisation` section.

    Returns:
        `InspectedSource` when decoding may proceed, otherwise the ineligibility record.
    """
    try:
        info = sf.info(io.BytesIO(artifact.data))
    except sf.LibsndfileError as error:
        return _inspection_rejected(artifact, error, logical_source_id, protocol_sha256)
    metadata = ObservedSourceMetadata(
        container=info.format,
        subtype=info.subtype,
        channels=info.channels,
        sample_rate=info.samplerate,
        frames=info.frames,
    )
    violations = metadata_violations(metadata, config)
    if violations:
        logger.info(
            "Source %s is ineligible at metadata validation: %s",
            logical_source_id,
            ", ".join(violations),
        )
        return SourceIneligibleOutcome(
            logical_source_id=logical_source_id,
            primary_reason_code=violations[0],
            supplemental_reason_codes=violations[1:],
            identity=artifact.identity,
            protocol_sha256=protocol_sha256,
            observed_metadata=metadata,
        )
    return _admitted(artifact, metadata, logical_source_id, config)


def _inspection_rejected(
    artifact: ArtifactBuffered,
    error: sf.LibsndfileError,
    logical_source_id: str,
    protocol_sha256: str,
) -> SourceIneligibleOutcome:
    """The record for bytes the decoder could not describe.

    The raw message stays in the run log; the record carries only what the sanitiser's
    allowlist permits.
    """
    logger.info("Inspection rejected %s: %s", logical_source_id, error.error_string)
    return SourceIneligibleOutcome(
        logical_source_id=logical_source_id,
        primary_reason_code=IneligibleReason.SOURCE_INSPECTION_REJECTED,
        identity=artifact.identity,
        protocol_sha256=protocol_sha256,
        decoder_diagnostic=sanitise_decoder_diagnostic(
            error.error_string, category=DiagnosticCategory.INSPECTION_REJECTED
        ),
    )


def _admitted(
    artifact: ArtifactBuffered,
    metadata: ObservedSourceMetadata,
    logical_source_id: str,
    config: Canonicalisation,
) -> InspectedSource:
    """Project the live memory for metadata that has already passed every bound.

    No comparison against `max_projected_live_bytes` follows, deliberately. The source
    bytes are within `max_source_bytes`, the decoded array within `max_decoded_bytes`, the
    downmix is at most the decoded array, and the canonical waveform is at most the downmix
    because the rate floor equals the target rate. The loader refuses any protocol whose
    `max_source_bytes + 3 * max_decoded_bytes` exceeds the budget, so a source that reached
    here cannot exceed it under any protocol `load_protocol` accepts. A comparison that no
    such input can fail would add a branch no test can reach.
    """
    canonical = canonical_length(
        metadata.frames, metadata.sample_rate, config.resample.target_sample_rate
    )
    projected = (
        artifact.identity.source_bytes
        + metadata.projected_decoded_bytes
        + metadata.frames * FLOAT64_BYTES
        + canonical * FLOAT64_BYTES
    )
    logger.debug(
        "Source %s admitted for decode: %d canonical samples, %d projected bytes",
        logical_source_id,
        canonical,
        projected,
    )
    return InspectedSource(metadata, canonical, projected)
