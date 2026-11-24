"""ITU-R BS.1770-4 integrated loudness, gain to a target, and the frozen clip-matching rule.

The meter is the standard's two-stage K-weighting filter (a high shelf, then a high-pass), with
400 ms gating blocks that overlap by 75 percent, an absolute gate at -70 LKFS and a relative
gate 10 LU below the level of the blocks that pass the absolute gate.

The K-weighting coefficients are derived for each sample rate with the bilinear transform from
the analog prototype. The tests check the derived 48 kHz coefficients against the published
BS.1770-4 table to 1e-6, and no other rate is checked against a reference table.

Each block is a mean of squares, so the loudness is a power measure in decibels. Sample peaks are
measured on the presented samples, and the standard's true-peak measure is not used.

> **Note:** The clip-matching rule reads the target, the sample-peak ceiling and the lowering
> step from the frozen protocol. Nothing in this module fixes those values.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import lfilter

from benchmark.listening.protocol import Loudness

__all__ = [
    "Biquad",
    "ClipMatch",
    "LoudnessError",
    "gain_to",
    "integrated_loudness",
    "k_weighting",
    "match_clip",
]

logger = logging.getLogger(__name__)

# Constants defined by ITU-R BS.1770-4 itself, not study settings.
_OFFSET_LKFS = -0.691
_BLOCK_S = 0.4
_HOP_S = 0.1
_ABSOLUTE_GATE_LKFS = -70.0
_RELATIVE_GATE_LU = -10.0
# A guard, not a protocol value: lowering stops with an error rather than looping forever.
_MAX_LOWERING_STEPS = 10_000

# The analog prototype of the K-weighting stages, as libebur128 and pyloudnorm state it.
_SHELF_F0_HZ = 1681.974450955533
_SHELF_GAIN_DB = 3.999843853973347
_SHELF_Q = 0.7071752369554196
_SHELF_VB_EXPONENT = 0.4996667741545416
_HIGHPASS_F_HZ = 38.13547087602444
_HIGHPASS_Q = 0.5003270373238773


class LoudnessError(ValueError):
    """Raised when a signal cannot be measured, or the matching rule cannot be applied."""


@dataclass(frozen=True)
class Biquad:
    """One second-order section in the direct form of `scipy.signal.lfilter`.

    Attributes:
        b: The numerator coefficients b0, b1, b2.
        a: The denominator coefficients a0 (always 1.0 here), a1, a2.
    """

    b: tuple[float, float, float]
    a: tuple[float, float, float]


@dataclass(frozen=True)
class ClipMatch:
    """The outcome of the clip-matching rule for one clip.

    Attributes:
        target_lkfs: The final target, after every lowering step.
        lowered_steps: How many times the target was lowered by the step.
        loudness: Each condition's measured integrated loudness, in LKFS.
        gains: Each condition's linear gain, applied to the samples.
        peaks_after_gain_dbfs: Each condition's sample peak after its gain, in dBFS.
    """

    target_lkfs: float
    lowered_steps: int
    loudness: Mapping[str, float]
    gains: Mapping[str, float]
    peaks_after_gain_dbfs: Mapping[str, float]


def _shelf(rate: int) -> Biquad:
    """The K-weighting high-shelf stage, bilinear-transformed for *rate*."""
    k = math.tan(math.pi * _SHELF_F0_HZ / rate)
    vh = 10.0 ** (_SHELF_GAIN_DB / 20.0)
    vb = vh**_SHELF_VB_EXPONENT
    a0 = 1.0 + k / _SHELF_Q + k * k
    b = (
        (vh + vb * k / _SHELF_Q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / _SHELF_Q + k * k) / a0,
    )
    a = (1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / _SHELF_Q + k * k) / a0)
    return Biquad(b, a)


def _highpass(rate: int) -> Biquad:
    """The K-weighting high-pass stage, bilinear-transformed for *rate*."""
    k = math.tan(math.pi * _HIGHPASS_F_HZ / rate)
    a0 = 1.0 + k / _HIGHPASS_Q + k * k
    a = (1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / _HIGHPASS_Q + k * k) / a0)
    return Biquad((1.0, -2.0, 1.0), a)


def k_weighting(rate: int) -> tuple[Biquad, Biquad]:
    """The two K-weighting stages in filter order, for one sample rate.

    Args:
        rate: The sample rate in hertz, a positive integer.

    Returns:
        The high-shelf stage, then the high-pass stage.

    Raises:
        LoudnessError: If the rate is not positive.
    """
    if rate <= 0:
        logger.error("Loudness meter refused sample rate %d", rate)
        raise LoudnessError(f"sample rate must be positive, got {rate}")
    return _shelf(rate), _highpass(rate)


def _validated(samples: npt.ArrayLike, rate: int) -> npt.NDArray[np.float64]:
    """The samples as one-dimensional float64, refused when they cannot be measured.

    Raises:
        LoudnessError: If the array is not one-dimensional, holds a NaN or infinity, or is
            shorter than one 400 ms block at *rate*.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim != 1:
        logger.error("Loudness meter refused an array of shape %s", x.shape)
        raise LoudnessError(f"loudness needs mono samples, got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        logger.error("Loudness meter refused %d samples with NaN or infinity", x.size)
        raise LoudnessError("loudness is undefined for samples that are not finite")
    block = _block_length(rate)
    if x.size < block:
        logger.error("Loudness meter refused %d samples, under one %d-sample block", x.size, block)
        raise LoudnessError(f"signal has {x.size} samples, fewer than one {block}-sample block")
    return x


def _block_length(rate: int) -> int:
    """The 400 ms block in samples, rounded to a whole number."""
    return round(_BLOCK_S * rate)


def _block_hop(rate: int) -> int:
    """The step between block starts in samples: 100 ms, which is 75 percent overlap."""
    return round(_HOP_S * rate)


def _k_weight(x: npt.NDArray[np.float64], rate: int) -> npt.NDArray[np.float64]:
    """The signal after both K-weighting stages."""
    weighted = x
    for stage in k_weighting(rate):
        weighted = lfilter(stage.b, stage.a, weighted)
    return np.asarray(weighted, dtype=np.float64)


def _block_loudness(powers: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Each block's loudness in LKFS. A block with zero power is minus infinity."""
    with np.errstate(divide="ignore"):
        return np.asarray(_OFFSET_LKFS + 10.0 * np.log10(powers), dtype=np.float64)


def integrated_loudness(samples: npt.ArrayLike, rate: int) -> float:
    """The gated integrated loudness of a mono signal, in LKFS.

    The signal is K-weighted, cut into 400 ms blocks every 100 ms, and each block's power is the
    mean square of its weighted samples. Blocks below -70 LKFS are dropped. Blocks more than
    10 LU below the mean of the remaining blocks are dropped too. The loudness is the mean power
    of what is left, in decibels, with the -0.691 offset.

    Args:
        samples: Mono samples. Must be finite and hold at least one block.
        rate: The sample rate in hertz.

    Returns:
        The integrated loudness in LKFS.

    Raises:
        LoudnessError: If the samples are not finite, not mono, shorter than one block, or no
            block passes the absolute gate (a silent or near-silent signal).
    """
    x = _validated(samples, rate)
    weighted = _k_weight(x, rate)
    windows = np.lib.stride_tricks.sliding_window_view(weighted, _block_length(rate))
    powers = np.mean(windows[:: _block_hop(rate)] ** 2, axis=1)
    loudness = _block_loudness(powers)
    above_absolute = loudness > _ABSOLUTE_GATE_LKFS
    if not np.any(above_absolute):
        logger.error("Loudness meter: no block of %d samples passes the absolute gate", x.size)
        raise LoudnessError("no block passes the absolute gate, so loudness is undefined")
    relative_threshold = (
        _OFFSET_LKFS + 10.0 * math.log10(float(np.mean(powers[above_absolute])))
    ) + _RELATIVE_GATE_LU
    kept = powers[above_absolute & (loudness > relative_threshold)]
    return _OFFSET_LKFS + 10.0 * math.log10(float(np.mean(kept)))


def gain_to(target_lkfs: float, loudness: float) -> float:
    """The linear gain that moves a measured loudness to a target.

    Args:
        target_lkfs: The target loudness in LKFS.
        loudness: The measured loudness in LKFS.

    Returns:
        The amplitude gain, 10 to the power of (target minus loudness) over 20.
    """
    return float(10.0 ** ((target_lkfs - loudness) / 20.0))


def _peak_dbfs(peak: float) -> float:
    """A linear peak in dBFS. A zero peak is minus infinity, which never exceeds a ceiling."""
    if peak == 0.0:
        return -math.inf
    return 20.0 * math.log10(peak)


def match_clip(windows: Mapping[str, npt.ArrayLike], rate: int, rule: Loudness) -> ClipMatch:
    """Gain every condition of one clip to a common target, lowering it until no peak exceeds.

    Each condition is first gained to the protocol's target. If any condition's sample peak
    after gain is above the ceiling, the target is lowered by the protocol's step for all the
    conditions together, and the check is repeated. The peak test is strictly greater than
    the ceiling, so a peak exactly at the ceiling is accepted.

    Args:
        windows: Each condition's window, mono, at *rate*.
        rate: The sample rate of every window, in hertz.
        rule: The protocol's loudness section: target, ceiling and step.

    Returns:
        The final target, the number of lowering steps, and each condition's loudness, gain and
        peak after gain.

    Raises:
        LoudnessError: If there are no windows, the step is not positive, or a window cannot be
            measured (see `integrated_loudness`).
    """
    if not windows:
        logger.error("Clip matching received no conditions")
        raise LoudnessError("clip matching needs at least one condition")
    if rule.target_step_lu <= 0.0:
        logger.error("Clip matching refused a non-positive step %s", rule.target_step_lu)
        raise LoudnessError("the target lowering step must be positive")
    arrays = {name: _validated(samples, rate) for name, samples in windows.items()}
    measured = {name: integrated_loudness(array, rate) for name, array in arrays.items()}
    peaks = {name: float(np.max(np.abs(array))) for name, array in arrays.items()}
    steps = 0
    while steps <= _MAX_LOWERING_STEPS:
        target = rule.target_lkfs - steps * rule.target_step_lu
        gains = {name: gain_to(target, measured[name]) for name in arrays}
        after = {name: _peak_dbfs(peaks[name] * gains[name]) for name in arrays}
        if all(peak <= rule.sample_peak_ceiling_dbfs for peak in after.values()):
            logger.debug("Clip matched at target %.2f LKFS after %d steps", target, steps)
            return ClipMatch(target, steps, measured, gains, after)
        steps += 1
    logger.error("Clip matching did not reach the peak ceiling in %d steps", _MAX_LOWERING_STEPS)
    raise LoudnessError(
        f"no target within {_MAX_LOWERING_STEPS} lowering steps keeps peaks under the ceiling"
    )
