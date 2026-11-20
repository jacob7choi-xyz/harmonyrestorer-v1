"""The duration and loudness floors protocol v5 sets for recordings and frames.

v5 makes a recording eligible when it decodes cleanly, is at least
`recording_min_duration_s` long, has RMS above `recording_rms_dbfs_min`, and passes
disjointness, and makes a frame eligible when its RMS is above `frame_rms_dbfs_min`. This
module measures the duration and the two loudness floors only. Decoding has already been
checked upstream, and disjointness belongs to the duplicate screen, so the results here are
named for the floors rather than for eligibility.

**Measured facts, not reason codes.** v5 froze no reason codes for these narrowings. v6
names them, each list in precedence order with the first failure primary and both
recording floors always evaluated (`eligibility.recording_ineligible_reasons` and
`eligibility.frame_ineligible_reasons`). The records here still carry the sample count,
each RMS in dBFS, and one boolean per floor, and both recording floors are always
evaluated, which is what that rule needs. Mapping these facts to codes belongs to the
code that records outcomes, which does not exist yet.

**The RMS is computed exactly as v5 defines it.** Energy is `numpy.mean(x * x)` over the
float64 samples, then `20 * log10(sqrt(energy))`, with no epsilon. Zero energy is negative
infinity, which is below every floor; it is returned directly because `log10(0)` warns.
Each frame gets its own `numpy.mean` call, the rule's literal form. A reduction along an
axis of the frame matrix matched it on one machine, but NumPy does not document that it
always sums in the same order.

**A non-finite energy raises.** An overflowing sum would give infinite dBFS, which is
above every floor, so returning it would fail open. v5 makes this a fail-closed harness
guard. Canonical validation refuses non-finite samples, and v5 states that the energy
sum's overflow is unreachable by a margin of 1.1e154, so the guard is not expected to fire
on a canonical waveform.

`x * x` over the complete waveform allocates a temporary of the waveform's size, about
460.8 MB at the 3600 s source cap. Summing in chunks would change the order v5 pins, so the
temporary stays, and peak memory has to be measured with it in place.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from benchmark.canonicalisation import CanonicalWaveform
from benchmark.framing import FramedWaveform
from benchmark.protocol import Eligibility

__all__ = [
    "EligibilityArithmeticError",
    "FrameFloors",
    "RecordingFloors",
    "frame_floors",
    "recording_floors",
    "rms_dbfs",
]

logger = logging.getLogger(__name__)


class EligibilityArithmeticError(ArithmeticError):
    """The energy of a waveform or frame was not finite, a harness failure in v5."""


@dataclass(frozen=True)
class RecordingFloors:
    """A recording measured against v5's duration floor and recording loudness floor.

    `meets_min_duration` is `canonical_samples >= recording_min_duration_s * sample_rate`
    in integer arithmetic, and `above_rms_floor` is `rms_dbfs > recording_rms_dbfs_min`.
    """

    canonical_samples: int
    meets_min_duration: bool
    rms_dbfs: float
    above_rms_floor: bool


@dataclass(frozen=True)
class FrameFloors:
    """Every candidate frame measured against v5's frame loudness floor.

    Row `i` of each array describes frame `i`, quiet frames included, so the population
    considered before filtering can be shown rather than asserted. Both arrays are
    read-only, because `frozen=True` does not protect an array's buffer.
    """

    rms_dbfs: npt.NDArray[np.float64]
    above_rms_floor: npt.NDArray[np.bool_]


def rms_dbfs(samples: npt.NDArray[np.float64]) -> float:
    """The RMS of the samples in dBFS, as v5's `rms_definition` states it.

    Args:
        samples: Finite float64 samples, one-dimensional.

    Returns:
        `20 * log10(sqrt(mean(x * x)))`, or negative infinity when the energy is zero.

    Raises:
        EligibilityArithmeticError: If the energy is not finite.
    """
    # The guard below reports an overflow, so NumPy's own overflow warning adds nothing.
    with np.errstate(over="ignore"):
        energy = float(np.mean(samples * samples))
    if not math.isfinite(energy):
        logger.error("Energy over %d samples is %s, not finite", samples.size, energy)
        raise EligibilityArithmeticError(f"energy over {samples.size} samples is {energy}")
    if energy == 0.0:
        return -math.inf
    return float(20.0 * np.log10(np.sqrt(energy)))


def recording_floors(
    waveform: CanonicalWaveform, config: Eligibility, *, sample_rate: int
) -> RecordingFloors:
    """Measure the complete canonical waveform against the recording floors.

    The RMS is taken over every canonical sample, including the trailing remainder that
    framing drops, because v5 defines recording RMS over the complete waveform.

    Args:
        waveform: The recording's canonical waveform.
        config: The frozen `eligibility` section.
        sample_rate: The canonical rate, `canonicalisation.resample.target_sample_rate`.

    Returns:
        The sample count, the RMS, and the verdict on each floor.

    Raises:
        EligibilityArithmeticError: If the energy is not finite.
    """
    samples = waveform.samples
    level = rms_dbfs(samples)
    floors = RecordingFloors(
        canonical_samples=samples.size,
        meets_min_duration=samples.size >= config.recording_min_duration_s * sample_rate,
        rms_dbfs=level,
        above_rms_floor=level > config.recording_rms_dbfs_min,
    )
    logger.debug(
        "Recording %s: %d samples, %.2f dBFS, duration floor %s, loudness floor %s",
        waveform.sha256,
        floors.canonical_samples,
        floors.rms_dbfs,
        floors.meets_min_duration,
        floors.above_rms_floor,
    )
    return floors


def frame_floors(framed: FramedWaveform, config: Eligibility) -> FrameFloors:
    """Measure every complete frame against the frame loudness floor.

    Args:
        framed: The recording's complete frames.
        config: The frozen `eligibility` section.

    Returns:
        Each frame's RMS and its verdict, in frame order.

    Raises:
        EligibilityArithmeticError: If any frame's energy is not finite.
    """
    levels = np.array([rms_dbfs(row) for row in framed.frames], dtype=np.float64)
    above = levels > config.frame_rms_dbfs_min
    levels.flags.writeable = False
    above.flags.writeable = False
    logger.debug("Frames: %d measured, %d above the loudness floor", levels.size, int(above.sum()))
    return FrameFloors(rms_dbfs=levels, above_rms_floor=above)
