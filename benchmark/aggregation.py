"""Frame metrics to recording means to the paired comparison, as protocol v13 aggregates them.

1. Within a recording, the unweighted arithmetic mean of its eligible frames' values.
2. Across recordings, the unweighted macro mean and the median; with an even number of
   recordings the median is the mean of the middle two.
3. A percentile bootstrap over performer clusters, never frames or single recordings. The
   clusters are ordered by their first test admission, which the caller states by passing the
   recordings in admission order. One `numpy.random.Generator(PCG64(seed))` draws one index
   array per iteration, `integers(0, k, size=k)` over the `k` clusters, in iteration order, and
   that array serves every estimand. An iteration's statistic is the mean over every recording
   of every drawn cluster, so a cluster drawn twice counts twice. The bounds are
   `numpy.percentile` of the iteration means with `method="linear"`.
4. Each metric's paired effect is formed per recording before resampling, in the direction
   the protocol gives it, so a positive value means OpGAN scored higher on that metric.
5. The shift-tolerant diagnostic (v13) is summarised exactly as the primary metrics are, from
   the same draws. The input baseline (v13) is summarised from the same draws too, with no
   paired effect, because it is not a system.

Until this change the code drew recordings, while protocol v12 and v13 state performer clusters.
Recordings by one performer share instrument, room and engineer, so drawing recordings treats
dependent recordings as independent and makes the interval too narrow. The cluster draw follows
the protocol.

The three primary metrics are summarised together and in protocol order, because the protocol
reports all three or none.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from benchmark.protocol import Protocol

__all__ = [
    "Aggregate",
    "AggregationError",
    "BaselineSummary",
    "Interval",
    "MetricSummary",
    "RecordingScores",
    "Role",
    "Series",
    "SystemValues",
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


class Series(StrEnum):
    """Which family of metrics an estimand belongs to: the primary three or the diagnostic."""

    PRIMARY = "primary"
    DIAGNOSTIC = "diagnostic"


class Role(StrEnum):
    """What an estimand row is: one system's mean, the paired effect, or the input baseline."""

    OPGAN = "opgan"
    UVR = "uvr"
    PAIRED = "paired"
    BASELINE = "baseline"


_Key = tuple[Series, str, Role]


@dataclass(frozen=True)
class SystemValues:
    """One source's per-frame values for one recording, keyed by metric name.

    `primary` holds the primary metrics and `shift_tolerant` the diagnostic metrics. Each
    sequence lists the source's value on every eligible frame of the recording, in frame order.
    """

    primary: Mapping[str, Sequence[float]]
    shift_tolerant: Mapping[str, Sequence[float]]


@dataclass(frozen=True)
class RecordingScores:
    """One recording's values for OpGAN, UVR and the degraded input, with its performer cluster.

    Frame `i` of a metric must be the same frame for all three sources; the population checks
    that establish that belong to scoring and run before aggregation.
    """

    recording_id: str
    cluster: str
    opgan: SystemValues
    uvr: SystemValues
    baseline: SystemValues


@dataclass(frozen=True)
class Interval:
    """A percentile bootstrap interval."""

    low: float
    high: float


@dataclass(frozen=True)
class MetricSummary:
    """One metric across recordings, for each system and for the paired effect."""

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


@dataclass(frozen=True)
class BaselineSummary:
    """The degraded input's mean, median and interval for one metric. It has no paired effect."""

    series: Series
    metric: str
    recordings: int
    mean: float
    median: float
    interval: Interval


@dataclass(frozen=True)
class Aggregate:
    """Every summary for one stratum: the primary metrics, the diagnostic, and the baseline.

    `primary` and `diagnostic` are in protocol order. `baseline` lists the primary metrics and then
    the diagnostic metrics, each with its series.
    """

    primary: tuple[MetricSummary, ...]
    diagnostic: tuple[MetricSummary, ...]
    baseline: tuple[BaselineSummary, ...]


def median(values: Values) -> float:
    """The median; with an even count, the mean of the middle two, as v6 states.

    `numpy.median` takes the mean of the two middle values for an even count, which is the
    rule v6 names, so this delegates to it.
    """
    return float(np.median(values))


def _series_metrics(protocol: Protocol) -> tuple[tuple[Series, tuple[str, ...]], ...]:
    """The metric names of each series, in protocol order."""
    return (
        (Series.PRIMARY, protocol.metrics.primary),
        (Series.DIAGNOSTIC, protocol.metrics.shift_tolerant.metrics),
    )


def _cluster_members(recordings: Sequence[RecordingScores]) -> list[list[int]]:
    """Each cluster's recording positions, clusters in order of first appearance."""
    members: dict[str, list[int]] = {}
    for position, scores in enumerate(recordings):
        members.setdefault(scores.cluster, []).append(position)
    return list(members.values())


def _selected_recordings(
    columns: Sequence[npt.NDArray[np.intp]], draw: npt.NDArray[np.intp]
) -> npt.NDArray[np.intp]:
    """The recording positions of every drawn cluster, in draw order, with repeats kept.

    A cluster drawn twice contributes its recordings twice, as the protocol's statistic states.
    """
    return np.concatenate([columns[cluster] for cluster in draw])


def bootstrap_means(
    estimands: Values, members: Sequence[Sequence[int]], protocol: Protocol
) -> Values:
    """Each estimand's mean under every bootstrap draw, shaped (estimands, iterations).

    Args:
        estimands: One row per estimand, one column per recording, in run order.
        members: The recording columns of each cluster, clusters in run order.
        protocol: The loaded protocol.

    Returns:
        The iteration means, one row per estimand.
    """
    config = protocol.aggregation.bootstrap
    clusters = len(members)
    columns = [np.asarray(column, dtype=np.intp) for column in members]
    generator = np.random.Generator(np.random.PCG64(config.seed))
    means = np.empty((estimands.shape[0], config.iterations))
    for iteration in range(config.iterations):
        draw = generator.integers(0, clusters, size=clusters)
        selected = _selected_recordings(columns, draw)
        # One `mean` call per estimand, the literal form. Reducing the stacked matrix along
        # an axis gave one row a result one unit in the last place away from reducing that
        # row alone, so an estimand's interval would otherwise depend on which other
        # estimands sit beside it.
        for row in range(estimands.shape[0]):
            means[row, iteration] = estimands[row][selected].mean()
    return means


def _interval(means: Values, confidence: float) -> Interval:
    """The linear-interpolation percentile bounds at the protocol's confidence."""
    # Decimal arithmetic on the written confidence: in binary floating point,
    # 100 * (1 - 0.95) / 2 is 2.500000000000002, not the 2.5 the protocol names.
    tail = (Decimal(1) - Decimal(repr(confidence))) * 50
    low, high = np.percentile(means, [float(tail), float(100 - tail)], method="linear")
    return Interval(float(low), float(high))


def _metric_values(values: SystemValues, series: Series, metric: str) -> Sequence[float]:
    """One source's frame values for one metric of one series.

    Raises:
        AggregationError: If the recording has no values for the metric.
    """
    table = values.primary if series is Series.PRIMARY else values.shift_tolerant
    if metric not in table:
        raise AggregationError(f"a recording has no {series.value} values for {metric}")
    return table[metric]


def _recording_means(
    recordings: Sequence[RecordingScores], series: Series, metric: str
) -> tuple[Values, Values, Values]:
    """Each recording's mean frame value for OpGAN, UVR and the baseline, for one metric.

    Raises:
        AggregationError: If the three sources do not hold the same non-empty frame count.
    """
    columns: tuple[list[float], list[float], list[float]] = ([], [], [])
    for scores in recordings:
        sources = (scores.opgan, scores.uvr, scores.baseline)
        frames = [_metric_values(source, series, metric) for source in sources]
        counts = {len(values) for values in frames}
        if len(counts) != 1 or 0 in counts:
            raise AggregationError(
                f"recording {scores.recording_id} has frame counts {sorted(counts)} for {metric}; "
                "every source needs the same non-empty frame population"
            )
        for column, values in zip(columns, frames, strict=True):
            column.append(float(np.mean(np.asarray(values, dtype=np.float64))))
    opgan, uvr, baseline = (np.asarray(column, dtype=np.float64) for column in columns)
    return opgan, uvr, baseline


def _estimands(
    recordings: Sequence[RecordingScores], protocol: Protocol
) -> tuple[Values, dict[_Key, int]]:
    """Every estimand row, in protocol order, and the row each (series, metric, role) names."""
    rows: list[Values] = []
    index: dict[_Key, int] = {}
    for series, metrics in _series_metrics(protocol):
        for metric in metrics:
            opgan, uvr, baseline = _recording_means(recordings, series, metric)
            effect = _PAIRED_EFFECTS[getattr(protocol.metrics, metric).paired_effect]
            for role, row in (
                (Role.OPGAN, opgan),
                (Role.UVR, uvr),
                (Role.PAIRED, effect(opgan, uvr)),
                (Role.BASELINE, baseline),
            ):
                index[(series, metric, role)] = len(rows)
                rows.append(row)
    return np.vstack(rows), index


@dataclass(frozen=True)
class _Draws:
    """The estimands, their iteration means, and the row index that ties them together."""

    estimands: Values
    means: Values
    index: Mapping[_Key, int]
    recordings: int
    confidence: float

    def estimate(self, series: Series, metric: str, role: Role) -> tuple[float, float, Interval]:
        """The mean, the median and the interval of one estimand."""
        row = self.index[(series, metric, role)]
        values = self.estimands[row]
        return float(values.mean()), median(values), _interval(self.means[row], self.confidence)


def _metric_summary(draws: _Draws, series: Series, metric: str, effect: str) -> MetricSummary:
    """One metric's summary for OpGAN, UVR and the paired effect."""
    o_mean, o_median, o_interval = draws.estimate(series, metric, Role.OPGAN)
    u_mean, u_median, u_interval = draws.estimate(series, metric, Role.UVR)
    p_mean, p_median, p_interval = draws.estimate(series, metric, Role.PAIRED)
    return MetricSummary(
        metric=metric,
        recordings=draws.recordings,
        opgan_mean=o_mean,
        opgan_median=o_median,
        opgan_interval=o_interval,
        uvr_mean=u_mean,
        uvr_median=u_median,
        uvr_interval=u_interval,
        paired_effect=effect,
        paired_mean=p_mean,
        paired_median=p_median,
        paired_interval=p_interval,
    )


def _baseline_summary(draws: _Draws, series: Series, metric: str) -> BaselineSummary:
    """The degraded input's summary for one metric. It has no paired effect."""
    mean, med, interval = draws.estimate(series, metric, Role.BASELINE)
    return BaselineSummary(series, metric, draws.recordings, mean, med, interval)


def _check_recordings(recordings: Sequence[RecordingScores]) -> None:
    """Refuse an empty or repeated set of recordings, or a recording with no cluster.

    Raises:
        AggregationError: If there are no recordings, an identifier repeats, or a cluster is empty.
    """
    identifiers = [scores.recording_id for scores in recordings]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise AggregationError(f"recordings must be non-empty and unique, got {identifiers}")
    if any(not scores.cluster for scores in recordings):
        raise AggregationError("every recording needs a non-empty performer cluster")


def summarise(recordings: Sequence[RecordingScores], protocol: Protocol) -> Aggregate:
    """Summarise every metric and the baseline, with one shared set of performer-cluster draws.

    Args:
        recordings: The scored recordings, in admission order. The order fixes the clusters'
            order, and so the intervals.
        protocol: The loaded protocol.

    Returns:
        The primary summaries, the diagnostic summaries, and the baseline summaries.

    Raises:
        AggregationError: If there are no recordings, an identifier repeats, a cluster is empty,
            or a recording has unequal or empty frame values across the sources.
    """
    _check_recordings(recordings)
    estimands, index = _estimands(recordings, protocol)
    members = _cluster_members(recordings)
    means = bootstrap_means(estimands, members, protocol)
    draws = _Draws(
        estimands,
        means,
        index,
        len(recordings),
        protocol.aggregation.bootstrap.confidence,
    )
    primary = tuple(
        _metric_summary(
            draws, Series.PRIMARY, metric, getattr(protocol.metrics, metric).paired_effect
        )
        for metric in protocol.metrics.primary
    )
    diagnostic = tuple(
        _metric_summary(
            draws, Series.DIAGNOSTIC, metric, getattr(protocol.metrics, metric).paired_effect
        )
        for metric in protocol.metrics.shift_tolerant.metrics
    )
    baseline = tuple(
        _baseline_summary(draws, series, metric)
        for series, metrics in _series_metrics(protocol)
        for metric in metrics
    )
    return Aggregate(primary, diagnostic, baseline)
