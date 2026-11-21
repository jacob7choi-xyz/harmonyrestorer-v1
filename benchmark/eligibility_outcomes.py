"""Eligibility codes for a recording and its frames, from the measured floors (protocol v6).

`benchmark.eligibility` measures the floors and leaves the codes to the code that records
outcomes. This module is that mapping. v6 names the codes and their precedence, and the
loader requires both lists exactly, in order:

- a recording: `RECORDING_BELOW_MIN_DURATION`, `RECORDING_BELOW_RMS_FLOOR`,
  `RECORDING_WITHOUT_ELIGIBLE_FRAMES`;
- a frame: `FRAME_BELOW_RMS_FLOOR`, `FRAME_NOT_SCOREABLE_BY_PRIMARY_METRIC`.

The first failure in list order is the primary reason and any others are supplemental.
Both recording floors are always evaluated, and both frame checks are always evaluated.
`RECORDING_WITHOUT_ELIGIBLE_FRAMES` is evaluated only for a recording that cleared both
floors, once its frames are cut, so it is never supplemental to a floor code; the two are
therefore separate functions here.

**Metric applicability is checked by scoring the frame against itself.** v6 makes a frame
ineligible for every metric when any primary metric cannot score it as a clean reference.
Each metric's refusals are defined in `benchmark.metrics`, and the checks it applies to an
estimate (one-dimensional, non-empty, finite, equal length) are ones the reference itself
passes or fails identically. Calling each primary metric with the frame as both reference
and estimate therefore raises exactly when the frame is not a valid reference, and the
refusal rules stay in one place rather than being restated here. The frame checked is the
unscaled canonical frame, as v7 keeps eligibility on the unscaled waveform.

This decides eligibility for the frame as cut, and no more. The reference that is later
scored is that frame times the training-level gain, possibly truncated by conditioning, and
a frame that passes here can still be refused there: a frame that is constant apart from its
last 130 samples, scored against an output 130 samples shorter, is one. `frame_scoring`
treats that case as a harness abort; no protocol version up to v8 names it.

Every candidate frame gets a verdict, eligible or not, so the population before filtering
is recorded rather than asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from benchmark.eligibility import FrameFloors, RecordingFloors
from benchmark.framing import FramedWaveform
from benchmark.metrics import PRIMARY_METRICS, MetricInputError
from benchmark.protocol import Metrics

__all__ = [
    "FrameReason",
    "FrameVerdict",
    "RecordingReason",
    "frame_verdicts",
    "recording_floor_reasons",
    "recording_frame_reasons",
    "unscoreable_metrics",
]


class RecordingReason(StrEnum):
    """v6's recording codes, in precedence order."""

    BELOW_MIN_DURATION = "RECORDING_BELOW_MIN_DURATION"
    BELOW_RMS_FLOOR = "RECORDING_BELOW_RMS_FLOOR"
    WITHOUT_ELIGIBLE_FRAMES = "RECORDING_WITHOUT_ELIGIBLE_FRAMES"


class FrameReason(StrEnum):
    """v6's frame codes, in precedence order."""

    BELOW_RMS_FLOOR = "FRAME_BELOW_RMS_FLOOR"
    NOT_SCOREABLE_BY_PRIMARY_METRIC = "FRAME_NOT_SCOREABLE_BY_PRIMARY_METRIC"


@dataclass(frozen=True)
class FrameVerdict:
    """One candidate frame: its RMS, its reasons in precedence order, and which metrics refused it.

    `reasons` is empty exactly when the frame is eligible; its first entry is the primary
    reason and the rest are supplemental.
    """

    index: int
    rms_dbfs: float
    reasons: tuple[FrameReason, ...]
    unscoreable_by: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        """Whether the frame is in the scored population."""
        return not self.reasons


def recording_floor_reasons(floors: RecordingFloors) -> tuple[RecordingReason, ...]:
    """The recording floor codes a recording fails, in precedence order; empty if none.

    Both floors are evaluated, so a recording that fails both carries the duration code as
    primary and the loudness code as supplemental.
    """
    reasons = []
    if not floors.meets_min_duration:
        reasons.append(RecordingReason.BELOW_MIN_DURATION)
    if not floors.above_rms_floor:
        reasons.append(RecordingReason.BELOW_RMS_FLOOR)
    return tuple(reasons)


def unscoreable_metrics(frame: npt.NDArray[np.float64], metrics: Metrics) -> tuple[str, ...]:
    """The primary metrics that refuse this frame as a clean reference, in protocol order.

    Args:
        frame: An unscaled canonical frame.
        metrics: The frozen `metrics` section.

    Returns:
        The names of the refusing metrics; empty when every primary metric accepts it.
    """
    refusing = []
    for name in metrics.primary:
        try:
            PRIMARY_METRICS[name](frame, frame, getattr(metrics, name))
        except MetricInputError:
            refusing.append(name)
    return tuple(refusing)


def frame_verdicts(
    framed: FramedWaveform, floors: FrameFloors, metrics: Metrics
) -> tuple[FrameVerdict, ...]:
    """A verdict for every candidate frame, eligible or not, in frame order.

    Args:
        framed: The recording's complete frames.
        floors: The same frames measured against the frame loudness floor.
        metrics: The frozen `metrics` section.

    Returns:
        One verdict per frame, with both frame checks evaluated for each.

    Raises:
        ValueError: If the floors were not measured on the same number of frames.
    """
    measured = (len(floors.above_rms_floor), len(floors.rms_dbfs))
    if measured != (len(framed.frames),) * 2:
        raise ValueError(f"floors cover {measured} frames, the recording has {len(framed.frames)}")
    verdicts = []
    for index, row in enumerate(framed.frames):
        reasons = []
        if not floors.above_rms_floor[index]:
            reasons.append(FrameReason.BELOW_RMS_FLOOR)
        refusing = unscoreable_metrics(row, metrics)
        if refusing:
            reasons.append(FrameReason.NOT_SCOREABLE_BY_PRIMARY_METRIC)
        verdicts.append(
            FrameVerdict(index, float(floors.rms_dbfs[index]), tuple(reasons), refusing)
        )
    return tuple(verdicts)


def recording_frame_reasons(verdicts: tuple[FrameVerdict, ...]) -> tuple[RecordingReason, ...]:
    """`RECORDING_WITHOUT_ELIGIBLE_FRAMES` when no frame is eligible, otherwise nothing.

    Called only for a recording that cleared both recording floors, so the code is never
    supplemental to a floor code.
    """
    if any(verdict.eligible for verdict in verdicts):
        return ()
    return (RecordingReason.WITHOUT_ELIGIBLE_FRAMES,)
