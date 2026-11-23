"""One system's output on one eligible frame: scored, a system failure, or a harness abort.

Protocol v6's scoring rule decides whose fault a failure is. A failure inside one system's
denoise call for a frame, an exception or an output that conditioning rejects, is a system
failure for that frame: the frame stays in the population and lowers that system's
coverage. A failure outside the call is a harness failure and aborts the run. This module
covers what happens after the call returns:

- `reference_frame` is the clean reference the metrics compare against: the canonical
  frame times the training-level gain (protocol v7), in float64.
- `score_output` conditions the output against that reference (protocol v6) and scores
  every primary metric, in protocol order. An output conditioning rejects is returned as a
  `SystemFailure`. A malformed reference raises `ConditioningError` from conditioning, and a
  metric refusing a pair that conditioned successfully raises `FrameScoringError`; both are
  the harness's.

**Why a metric's refusal after conditioning is the harness's.** Conditioning returns a pair
only when the output is one-dimensional, non-empty, finite, and as long as the reference, so
every check a metric applies to an estimate has already passed. What is left is the
reference side. Eligibility checked that side on the full unscaled frame; the reference
scored here is that frame times the gain and possibly truncated by up to the conditioning
tolerance, so a refusal can only come from the reference, which the harness produced. No
protocol version up to v8 names this case.

> **Note:** The reference is `frame * gain` in float64. Degradation starts from the same
> product rounded to float32 (`benchmark.degradation.degrade_frame`). v7 says the metrics
> compare against "the canonical frame times the same gain" without naming a precision.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from benchmark.conditioning import InvalidSystemOutput, condition_pair
from benchmark.metrics import PRIMARY_METRICS, MetricInputError, MetricResult, shift_tolerant
from benchmark.protocol import Protocol

__all__ = ["FrameScore", "FrameScoringError", "SystemFailure", "reference_frame", "score_output"]


class FrameScoringError(Exception):
    """Raised when a frame cannot be scored for a reason that belongs to the harness."""


@dataclass(frozen=True)
class FrameScore:
    """Every primary metric for one system's valid output on one frame.

    `results` holds the metrics in protocol order. The discard counts record what
    conditioning truncated from each side, so the manifest shows how the pair was formed.
    `shift_tolerant` holds v13's diagnostic for each sample-wise metric it names; it sits
    beside `results` and never replaces a primary score.
    """

    results: Mapping[str, MetricResult]
    reference_samples_discarded: int
    estimate_samples_discarded: int
    shift_tolerant: Mapping[str, MetricResult] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemFailure:
    """A system failed this frame: an exception in its call, or an output conditioning rejects.

    The frame stays in the population and counts against this system's coverage.
    """

    reason: str


FrameOutcome = FrameScore | SystemFailure


def reference_frame(frame: npt.NDArray[np.float64], gain: float) -> npt.NDArray[np.float64]:
    """The clean reference for a frame: its canonical samples times the training-level gain."""
    reference: npt.NDArray[np.float64] = np.asarray(frame, dtype=np.float64) * gain
    reference.flags.writeable = False
    return reference


def score_output(
    reference: npt.NDArray[np.float64], estimate: npt.NDArray[np.float64], protocol: Protocol
) -> FrameOutcome:
    """Condition a system's output against the reference and score every primary metric.

    Each sample-wise metric is also scored with v13's shift-tolerant diagnostic. The degraded
    input is scored through this same function as v13's input baseline.

    Args:
        reference: The frame's clean reference, from `reference_frame`.
        estimate: The system's output for the frame, read from its output file.
        protocol: The loaded protocol.

    Returns:
        The scores, or a `SystemFailure` when conditioning rejects the output.

    Raises:
        ConditioningError: If the reference is malformed.
        FrameScoringError: If a metric refuses a pair that conditioned successfully.
    """
    conditioned = condition_pair(reference, estimate, protocol.conditioning)
    if isinstance(conditioned, InvalidSystemOutput):
        return SystemFailure(conditioned.reason)
    results = {}
    for name in protocol.metrics.primary:
        try:
            results[name] = PRIMARY_METRICS[name](
                conditioned.reference, conditioned.estimate, getattr(protocol.metrics, name)
            )
        except MetricInputError as error:
            raise FrameScoringError(
                f"{name} refused a pair that conditioned successfully, so the reference is at "
                f"fault: {error}"
            ) from error
    return FrameScore(
        results=results,
        reference_samples_discarded=conditioned.reference_samples_discarded,
        estimate_samples_discarded=conditioned.estimate_samples_discarded,
        shift_tolerant=_shift_tolerant(conditioned.reference, conditioned.estimate, protocol),
    )


def _shift_tolerant(
    reference: npt.NDArray[np.float64], estimate: npt.NDArray[np.float64], protocol: Protocol
) -> dict[str, MetricResult]:
    """v13's diagnostic for each metric `metrics.shift_tolerant` names, on a conditioned pair.

    Raises:
        FrameScoringError: If a shifted pair is refused, which the reference is blamed for,
            as for the primary metrics.
    """
    shift = protocol.metrics.shift_tolerant
    scores = {}
    for name in shift.metrics:
        try:
            scores[name] = shift_tolerant(
                PRIMARY_METRICS[name],
                reference,
                estimate,
                getattr(protocol.metrics, name),
                shift.max_shift_samples,
            )
        except MetricInputError as error:
            raise FrameScoringError(
                f"{name} refused a shifted pair that conditioned successfully: {error}"
            ) from error
    return scores
