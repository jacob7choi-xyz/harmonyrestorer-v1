"""Frame metrics to recording means to the paired comparison, as protocol v6 aggregates them.

1. Within a recording, the unweighted arithmetic mean of its eligible frames' values.
2. Across recordings, the unweighted macro mean and the median; with an even number of
   recordings the median is the mean of the middle two.
3. A percentile bootstrap over recordings, never frames: one
   `numpy.random.Generator(PCG64(seed))` draws one index array per iteration,
   `integers(0, n, size=n)`, in iteration order, and the same array is applied to every
   estimand. The bounds are `numpy.percentile` of the iteration means with
   `method="linear"`.
4. Each metric's paired effect is formed per recording before resampling, in the direction
   the protocol gives it, so a positive effect always favours OpGAN.

The three primary metrics are summarised together and in protocol order, because v6
reports all three or none.

> **Note:** The order of the recordings is an input, not a default. The index arrays pick
> positions, so the same draws over a different ordering give different intervals, and no
> protocol version up to v8 states the order. The caller passes the recordings in the order
> the run is to use, and the amendment before candidates are enumerated is expected to
> freeze it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import numpy.typing as npt

from benchmark.protocol import Protocol

__all__ = [
    "AggregationError",
    "Interval",
    "MetricSummary",
    "RecordingScores",
    "bootstrap_means",
    "median",
    "summarise",
]

Values = npt.NDArray[np.float64]

_PAIRED_EFFECTS: dict[str, Callable[[Values, Values], Values]] = {
    "opgan_minus_uvr": lambda opgan, uvr: opgan - uvr,
    "uvr_minus_opgan": lambda opgan, uvr: uvr - opgan,
}


class AggregationError(ValueError):
    """Raised when the scores cannot be aggregated as the protocol requires."""


@dataclass(frozen=True)
class RecordingScores:
    """One recording's eligible-frame values for each primary metric, per system.

    Frame `i` of a metric must be the same frame for both systems; the population checks
    that establish that belong to scoring and run before aggregation.
    """

    recording_id: str
    opgan: Mapping[str, Sequence[float]]
    uvr: Mapping[str, Sequence[float]]


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval."""

    low: float
    high: float


@dataclass(frozen=True)
class MetricSummary:
    """One primary metric across recordings, for each system and for the paired effect."""

    metric: str
    recordings: int
    opgan_mean: float
    opgan_median: float
    opgan_interval: Interval
    uvr_mean: float
    uvr_median: float
    uvr_interval: Interval
    paired_effect: str
    paired_mean: float
    paired_median: float
    paired_interval: Interval


def median(values: Values) -> float:
    """The median; with an even count, the mean of the middle two, as v6 states.

    `numpy.median` takes the mean of the two middle values for an even count, which is the
    rule v6 names, so this delegates to it.
    """
    return float(np.median(values))


def bootstrap_means(estimands: Values, protocol: Protocol) -> Values:
    """Each estimand's mean under every bootstrap draw, shaped (estimands, iterations).

    Args:
        estimands: One row per estimand, one column per recording, in run order.
        protocol: The loaded protocol.

    Returns:
        The iteration means, one row per estimand.
    """
    config = protocol.aggregation.bootstrap
    count = estimands.shape[1]
    generator = np.random.Generator(np.random.PCG64(config.seed))
    means = np.empty((estimands.shape[0], config.iterations))
    for iteration in range(config.iterations):
        draw = generator.integers(0, count, size=count)
        # One `mean` call per estimand, the literal form. Reducing the stacked matrix along
        # an axis gave one row a result one unit in the last place away from reducing that
        # row alone (seen when test_the_intervals_are_linear_percentiles_of_the_shared_draws
        # failed against the stacked form), so an estimand's interval would otherwise depend
        # on which other estimands sit beside it.
        for row in range(estimands.shape[0]):
            means[row, iteration] = estimands[row][draw].mean()
    return means


def _interval(means: Values, confidence: float) -> Interval:
    """The linear-interpolation percentile bounds at the protocol's confidence."""
    # Decimal arithmetic on the written confidence: in binary floating point,
    # 100 * (1 - 0.95) / 2 is 2.500000000000002, not the 2.5 the protocol names, and on
    # some draws that moved the lower bound by a unit in the last place.
    tail = (Decimal(1) - Decimal(repr(confidence))) * 50
    low, high = np.percentile(means, [float(tail), float(100 - tail)], method="linear")
    return Interval(float(low), float(high))


def _recording_means(recordings: Sequence[RecordingScores], metric: str) -> tuple[Values, Values]:
    """Each recording's mean frame value for one metric, for OpGAN and for UVR."""
    opgan, uvr = [], []
    for scores in recordings:
        first, second = scores.opgan[metric], scores.uvr[metric]
        if len(first) != len(second) or not first:
            raise AggregationError(
                f"recording {scores.recording_id} has {len(first)} OpGAN and {len(second)} UVR "
                f"values for {metric}; both systems need the same non-empty frame population"
            )
        opgan.append(np.mean(np.asarray(first, dtype=np.float64)))
        uvr.append(np.mean(np.asarray(second, dtype=np.float64)))
    return np.asarray(opgan, dtype=np.float64), np.asarray(uvr, dtype=np.float64)


def summarise(
    recordings: Sequence[RecordingScores], protocol: Protocol
) -> tuple[MetricSummary, ...]:
    """Summarise every primary metric, in protocol order, with one shared set of draws.

    Args:
        recordings: The scored recordings, in the order the run uses for the bootstrap.
        protocol: The loaded protocol.

    Returns:
        One summary per primary metric.

    Raises:
        AggregationError: If there are no recordings, an identifier repeats, or a recording
            has unequal or empty frame values for the two systems.
    """
    identifiers = [scores.recording_id for scores in recordings]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise AggregationError(f"recordings must be non-empty and unique, got {identifiers}")
    metrics = protocol.metrics.primary
    rows = []
    for metric in metrics:
        opgan, uvr = _recording_means(recordings, metric)
        effect = getattr(protocol.metrics, metric).paired_effect
        rows.extend([opgan, uvr, _PAIRED_EFFECTS[effect](opgan, uvr)])
    estimands = np.vstack(rows)
    means = bootstrap_means(estimands, protocol)
    confidence = protocol.aggregation.bootstrap.confidence
    summaries = []
    for index, metric in enumerate(metrics):
        o, u, p = (3 * index + offset for offset in range(3))
        summaries.append(
            MetricSummary(
                metric=metric,
                recordings=len(identifiers),
                opgan_mean=float(estimands[o].mean()),
                opgan_median=median(estimands[o]),
                opgan_interval=_interval(means[o], confidence),
                uvr_mean=float(estimands[u].mean()),
                uvr_median=median(estimands[u]),
                uvr_interval=_interval(means[u], confidence),
                paired_effect=getattr(protocol.metrics, metric).paired_effect,
                paired_mean=float(estimands[p].mean()),
                paired_median=median(estimands[p]),
                paired_interval=_interval(means[p], confidence),
            )
        )
    return tuple(summaries)
