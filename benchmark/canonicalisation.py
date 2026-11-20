"""The canonical waveform: downmix, resample, signed zero, validation, and its digest.

This is v4's transformation after the source has been decoded and validated, in the frozen
order: downmix, resample, normalise signed zero, validate the canonical result. The
waveform is then made read-only and hashed, so the bytes the digest identifies are the
bytes returned.

**Every array written in place is one this module allocated.** librosa returns its input
unchanged when the source rate already equals the target, and signed-zero normalisation
writes in place, so a one-channel downmix is a copy rather than a view of the decoded
array. Otherwise the decoded array the caller holds would be changed underneath it. When
librosa's result is a view of another array, which happened for some input lengths, the
normalised waveform is written to a new array instead, so the returned waveform owns its
memory and its read-only flag cannot be bypassed through a writable base.

**A failure here is treated as the harness's, not the source's.** By this point the source was
decoded, matched its metadata, and passed the finite and amplitude checks, so the
ineligibility vocabulary has no stage left to name. A resampled length that differs from
v4's integer rule, a resampler refusal, or a canonical waveform that fails validation is
returned as `INTERNAL_CANONICALISATION_ERROR`, which aborts the run. v4 names the length
assertion but does not say which record a failed canonical validation produces; this
module takes the only code that fits, and protocol v5 states that as the rule.

Temporaries outside v4's live-bytes rule remain: librosa's input check builds one boolean
per sample, and normalising a view allocates a second canonical-sized array. Both are left
to the planned peak-memory measurement.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import librosa
import numpy as np
import numpy.typing as npt

from benchmark.protocol import Canonicalisation
from benchmark.source_inspection import InspectedSource
from benchmark.source_outcome import AbortReason, RunAbortOutcome

__all__ = ["CanonicalWaveform", "canonical_violation", "canonicalise", "downmix"]

logger = logging.getLogger(__name__)

# v4 hashes contiguous little-endian IEEE-754 float64 samples with no header.
_HASH_SAMPLE_FORMAT = "<f8"


@dataclass(frozen=True)
class CanonicalWaveform:
    """A canonical waveform and the SHA-256 of its bytes.

    `samples` is one-dimensional float64, owns its memory, and is read-only, so the
    digest keeps identifying it.
    """

    samples: npt.NDArray[np.float64]
    sha256: str


def canonicalise(
    decoded: npt.NDArray[np.float64],
    inspected: InspectedSource,
    *,
    logical_source_id: str,
    config: Canonicalisation,
) -> CanonicalWaveform | RunAbortOutcome:
    """Transform a validated decoded source into its canonical waveform.

    Args:
        decoded: The array `decode_source` returned, shaped (frames, channels).
        inspected: What inspection reported for the same source.
        logical_source_id: Identifier of the candidate, for an abort record and the log.
        config: The frozen `canonicalisation` section.

    Returns:
        The canonical waveform and its digest, or the abort record for a harness failure.
    """
    channels = config.downmix
    if decoded.ndim != 2 or not channels.channels_min <= decoded.shape[1] <= channels.channels_max:
        return _internal_error(logical_source_id, f"downmix received shape {decoded.shape}")
    resampled = _resampled(downmix(decoded), inspected, logical_source_id, config)
    if isinstance(resampled, RunAbortOutcome):
        return resampled
    # -0.0 + 0.0 is +0.0 under round-to-nearest, and every other value is unchanged.
    canonical = np.add(resampled, 0.0, out=resampled if resampled.flags.owndata else None)
    violation = canonical_violation(canonical)
    if violation is not None:
        return _internal_error(logical_source_id, f"canonical validation failed: {violation}")
    canonical.flags.writeable = False
    digest = hashlib.sha256(
        np.ascontiguousarray(canonical, dtype=_HASH_SAMPLE_FORMAT).data
    ).hexdigest()
    logger.debug(
        "Source %s canonical: %d samples, sha256 %s", logical_source_id, canonical.size, digest
    )
    return CanonicalWaveform(canonical, digest)


def _resampled(
    mono: npt.NDArray[np.float64],
    inspected: InspectedSource,
    logical_source_id: str,
    config: Canonicalisation,
) -> npt.NDArray[np.float64] | RunAbortOutcome:
    """Resample with every argument named, then hold the length to v4's integer rule."""
    resample = config.resample
    try:
        resampled: npt.NDArray[np.float64] = librosa.resample(
            mono,
            orig_sr=inspected.metadata.sample_rate,
            target_sr=resample.target_sample_rate,
            res_type=resample.res_type,
            fix=resample.fix,
            scale=resample.scale,
            axis=resample.axis,
        )
    except librosa.util.exceptions.ParameterError as error:
        return _internal_error(
            logical_source_id, f"the resampler refused a validated source: {error}"
        )
    if resampled.shape != (inspected.canonical_samples,):
        return _internal_error(
            logical_source_id,
            f"resampled shape {resampled.shape}, expected ({inspected.canonical_samples},)",
        )
    return resampled


def downmix(decoded: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """v4's downmix, into a new array the caller owns.

    One channel is copied unchanged. Two channels become (left + right) / 2.0 in float64,
    computed in place on the sum, which is the same arithmetic as the explicit expression
    without a second temporary.

    Args:
        decoded: Shaped (frames, 1) or (frames, 2); other shapes are the caller's to refuse.

    Returns:
        The mono signal, one-dimensional.
    """
    if decoded.shape[1] == 1:
        return decoded[:, 0].copy()
    mono = np.add(decoded[:, 0], decoded[:, 1])
    np.divide(mono, 2.0, out=mono)
    return mono


def canonical_violation(canonical: npt.NDArray[np.float64]) -> str | None:
    """The first way the canonical waveform fails v4's validation, if any.

    float64, one dimension, non-empty, and finite, with no amplitude bound: band-limited
    resampling overshoots full scale, and forcing it back would need an operation v4
    forbids. Finiteness uses `min` and `max`, which propagate NaN and reach an infinity,
    so no per-sample array is built.

    Args:
        canonical: The waveform after signed-zero normalisation.

    Returns:
        A description of the first failure, or None when the waveform is valid.
    """
    if canonical.dtype != np.float64:
        return f"dtype {canonical.dtype}"
    if canonical.ndim != 1:
        return f"{canonical.ndim} dimensions"
    if canonical.size == 0:
        return "no samples"
    if not (np.isfinite(canonical.min()) and np.isfinite(canonical.max())):
        return "non-finite samples"
    return None


def _internal_error(logical_source_id: str, detail: str) -> RunAbortOutcome:
    """An abort for a failure of the transformation itself."""
    logger.error("Canonicalisation of %s failed: %s", logical_source_id, detail)
    return RunAbortOutcome(
        reason=AbortReason.INTERNAL_CANONICALISATION_ERROR,
        detail=detail,
        logical_source_id=logical_source_id,
    )
