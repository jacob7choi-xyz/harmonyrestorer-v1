"""The post-hoc analyses that docs/benchmark-posthoc-analyses.md fixes, as pure functions.

Three analyses are computed beside the protocol's primary results. None of them changes a
primary result, and each number they produce is labelled with `LABEL`.

1. A cluster-level paired summary: each recording's paired difference, averaged within its
   performer cluster, with a 95% t-interval over the k cluster values (k - 1 degrees of freedom).
2. Log-spectral distance restricted to bins 0 to `BAND_LAST_BIN` inclusive, which is 0 to 4 kHz at
   a 16 kHz rate with a 1024-point transform. The restriction is applied after the same STFT and
   the same log, RMS and mean operations the protocol's metric uses.
3. Counts of frames at the reconstruction-SNR and SI-SNR cap, and of conditioned outputs that are
   exactly zero in every sample.

Every paired difference is oriented so that a positive value means OpGAN scored higher, which for
a lower-is-better metric means UVR's distance is larger. The functions here take values that the
caller has already read, so they perform no file I/O. `benchmark.run_posthoc` does the reading and
the consistency checks.
"""

from __future__ import annotations

import logging
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import t as student_t

from benchmark.metrics import _DB_MULTIPLIER, _validated_pair, stft_magnitude
from benchmark.protocol import LogSpectralDistance, Protocol
from benchmark.scoring import BASELINE, OPGAN, UVR

__all__ = [
    "BAND_LAST_BIN",
    "BAND_METRIC",
    "CAPPED_METRICS",
    "LABEL",
    "ClusterSummary",
    "PairedMetric",
    "PosthocError",
    "RecordingFrames",
    "SourceCounts",
    "SourceFrames",
    "SourceMeans",
    "StratumAnalysis",
    "analyse_stratum",
    "band_distance_from_magnitudes",
    "band_log_spectral_distance",
    "cluster_summary",
    "is_silent",
]

logger = logging.getLogger(__name__)

LABEL = "post-hoc, fixed in docs/benchmark-posthoc-analyses.md before the report ran"
BAND_METRIC = "band_log_spectral_distance"
RECONSTRUCTION_SNR = "reconstruction_snr"
SI_SNR = "si_snr"
LOG_SPECTRAL_DISTANCE = "log_spectral_distance"
CAPPED_METRICS = (RECONSTRUCTION_SNR, SI_SNR)
# The highest transform bin kept, inclusive. At 16 kHz with a 1024-point transform each bin is
# 15.625 Hz wide, so bin 256 is 4000 Hz, the lowest rolloff the degradation draws.
BAND_LAST_BIN = 256
_CONFIDENCE_QUANTILE = 0.975
_OPGAN_MINUS_UVR = "opgan_minus_uvr"
_UVR_MINUS_OPGAN = "uvr_minus_opgan"
_EFFECTS: Mapping[str, Callable[[float, float], float]] = {
    _OPGAN_MINUS_UVR: operator.sub,
    _UVR_MINUS_OPGAN: lambda opgan, uvr: uvr - opgan,
}
_SOURCES = (OPGAN, UVR, BASELINE)


class PosthocError(Exception):
    """Raised when a post-hoc analysis cannot be computed or its inputs do not agree."""


@dataclass(frozen=True)
class ClusterSummary:
    """The cluster-level paired summary of one paired difference.

    `mean` is the mean of the k cluster values, `sd` their sample standard deviation with k - 1 in
    the denominator, and `low` and `high` the 95% t-interval. `positive`, `negative` and `zero`
    count the cluster values by sign.
    """

    clusters: int
    mean: float
    sd: float
    low: float
    high: float
    positive: int
    negative: int
    zero: int


@dataclass(frozen=True)
class SourceMeans:
    """One source's mean and median over the test recordings of a stratum."""

    mean: float
    median: float


@dataclass(frozen=True)
class PairedMetric:
    """One metric's recording means per source, the paired difference, and its cluster summary."""

    metric: str
    effect: str
    recordings: int
    sources: Mapping[str, SourceMeans]
    paired_mean: float
    paired_median: float
    clusters: ClusterSummary


@dataclass(frozen=True)
class SourceCounts:
    """One source's eligible frames and the frames counted as capped or silent."""

    eligible_frames: int
    capped_reconstruction_snr: int
    capped_si_snr: int
    silent: int


@dataclass(frozen=True)
class SourceFrames:
    """One source's measured values on the eligible frames of one recording, in frame order.

    `primary` holds the recorded value of each primary metric. `band` holds the band-limited
    distance of each frame. `capped` holds, for each capped metric, whether the recorded value is
    censored at the cap. `silent` holds whether the conditioned output is exactly zero throughout.
    """

    primary: Mapping[str, Sequence[float]]
    band: Sequence[float]
    capped: Mapping[str, Sequence[bool]]
    silent: Sequence[bool]


@dataclass(frozen=True)
class RecordingFrames:
    """One test recording: its performer cluster, and each source's frames."""

    recording_id: str
    cluster: str
    sources: Mapping[str, SourceFrames]


@dataclass(frozen=True)
class StratumAnalysis:
    """Every post-hoc number for one stratum that the report publishes as a comparison."""

    recordings: int
    clusters: int
    primary: tuple[PairedMetric, ...]
    band: PairedMetric
    counts: Mapping[str, SourceCounts]


def cluster_summary(differences: Sequence[tuple[str, float]]) -> ClusterSummary:
    """The cluster-level paired summary, from each recording's cluster and paired difference.

    The recordings of a cluster are averaged first, so each cluster contributes one value and
    the interval is over the k clusters, not over the recordings.

    Args:
        differences: Each recording's performer cluster and its paired difference.

    Returns:
        The k, the mean of the cluster values, the sample standard deviation, the 95% t-interval,
        and the sign counts.

    Raises:
        PosthocError: If the differences name fewer than two performer clusters.
    """
    cluster_values = _cluster_means(differences)
    count = len(cluster_values)
    if count < 2:
        logger.error("A cluster summary needs at least two performer clusters, got %d", count)
        raise PosthocError(f"a cluster summary needs at least two performer clusters, got {count}")
    values = np.asarray(cluster_values, dtype=np.float64)
    mean = float(np.mean(values))
    spread = float(np.std(values, ddof=1))
    quantile = float(student_t.ppf(_CONFIDENCE_QUANTILE, count - 1))
    half_width = quantile * spread / math.sqrt(count)
    return ClusterSummary(
        clusters=count,
        mean=mean,
        sd=spread,
        low=mean - half_width,
        high=mean + half_width,
        positive=int(np.count_nonzero(values > 0.0)),
        negative=int(np.count_nonzero(values < 0.0)),
        zero=int(np.count_nonzero(values == 0.0)),
    )


def _cluster_means(differences: Sequence[tuple[str, float]]) -> list[float]:
    """Each cluster's arithmetic mean of its recordings' differences, in first-appearance order."""
    members: dict[str, list[float]] = {}
    for cluster, difference in differences:
        members.setdefault(cluster, []).append(difference)
    return [float(np.mean(values)) for values in members.values()]


def band_distance_from_magnitudes(
    clean_magnitude: npt.NDArray[np.float64],
    estimate_magnitude: npt.NDArray[np.float64],
    *,
    offset: float,
    multiplier: float,
    last_bin: int,
) -> float:
    """The log-spectral distance over bins 0 to `last_bin` inclusive, from two magnitude spectra.

    The operations match `benchmark.metrics.log_spectral_distance`: the offset is added, then the
    log is taken and scaled, the squared difference is averaged over the kept bins per frame and
    square-rooted, and the frames are averaged. Only the bins are different. Slicing to the kept
    bins before the arithmetic leaves every kept value unchanged, so with `last_bin` at the top
    bin the result equals the protocol's value exactly.

    Args:
        clean_magnitude: The reference's magnitudes, shaped (frames, bins).
        estimate_magnitude: The estimate's magnitudes, shaped like `clean_magnitude`.
        offset: The additive offset inside the log.
        multiplier: The decibel factor of the log operation.
        last_bin: The highest bin kept, inclusive.

    Returns:
        The mean over frames of the per-frame RMS difference, in decibels.

    Raises:
        ValueError: If `last_bin` is not a bin of the spectra.
    """
    bins = clean_magnitude.shape[1]
    if not 0 <= last_bin < bins:
        logger.error("last_bin %d is not a bin of a %d-bin spectrum", last_bin, bins)
        raise ValueError(f"last_bin must be in [0, {bins}), got {last_bin}")
    clean_db = multiplier * np.log10(clean_magnitude[:, : last_bin + 1] + offset)
    estimate_db = multiplier * np.log10(estimate_magnitude[:, : last_bin + 1] + offset)
    per_frame = np.sqrt(np.mean((clean_db - estimate_db) ** 2, axis=1))
    return float(np.mean(per_frame))


def band_log_spectral_distance(
    reference: npt.NDArray[np.float64],
    estimate: npt.NDArray[np.float64],
    config: LogSpectralDistance,
    *,
    last_bin: int = BAND_LAST_BIN,
) -> float:
    """The band-limited log-spectral distance of one conditioned pair, in decibels.

    Args:
        reference: The conditioned reference.
        estimate: The conditioned estimate, the same length as the reference.
        config: The frozen `metrics.log_spectral_distance` section.
        last_bin: The highest bin kept, inclusive.

    Returns:
        The distance over the kept bins.

    Raises:
        MetricInputError: If the pair is not scoreable, as for the protocol's metric.
    """
    clean, restored = _validated_pair(reference, estimate)
    return band_distance_from_magnitudes(
        stft_magnitude(clean, config),
        stft_magnitude(restored, config),
        offset=config.log_magnitude.offset,
        multiplier=_DB_MULTIPLIER[config.log_magnitude.operation],
        last_bin=last_bin,
    )


def is_silent(estimate: npt.NDArray[np.float64]) -> bool:
    """Whether a conditioned output is exactly zero in every sample. A value near zero is not."""
    return bool(np.all(estimate == 0.0))


def analyse_stratum(recordings: Sequence[RecordingFrames], protocol: Protocol) -> StratumAnalysis:
    """The three analyses for one stratum, from its test recordings.

    Args:
        recordings: Every test recording of the stratum, with each source's per-frame values.
        protocol: The loaded protocol, which names each primary metric's paired effect.

    Returns:
        The primary metrics in protocol order, the band-limited metric, and the frame counts.

    Raises:
        PosthocError: If the stratum has no recordings, or fewer than two performer clusters.
    """
    if not recordings:
        logger.error("A stratum has no test recordings to analyse")
        raise PosthocError("a stratum has no test recordings to analyse")
    primary = tuple(
        _paired_metric(recordings, metric, getattr(protocol.metrics, metric).paired_effect)
        for metric in protocol.metrics.primary
    )
    band = _paired_metric(recordings, BAND_METRIC, _UVR_MINUS_OPGAN)
    counts = {source: _counts(recordings, source) for source in _SOURCES}
    return StratumAnalysis(
        recordings=len(recordings),
        clusters=len({recording.cluster for recording in recordings}),
        primary=primary,
        band=band,
        counts=counts,
    )


def _values(frames: SourceFrames, metric: str) -> Sequence[float]:
    """One source's per-frame values for a metric, the band-limited one included."""
    return frames.band if metric == BAND_METRIC else frames.primary[metric]


def _paired_metric(recordings: Sequence[RecordingFrames], metric: str, effect: str) -> PairedMetric:
    """Each recording's source means, its paired difference, and the cluster summary of those.

    Raises:
        PosthocError: If the stratum has fewer than two performer clusters.
    """
    means = {
        source: np.asarray(
            [
                float(np.mean(_values(recording.sources[source], metric)))
                for recording in recordings
            ],
            dtype=np.float64,
        )
        for source in _SOURCES
    }
    paired = [
        _EFFECTS[effect](float(opgan), float(uvr))
        for opgan, uvr in zip(means[OPGAN], means[UVR], strict=True)
    ]
    clusters = cluster_summary(
        [(recording.cluster, value) for recording, value in zip(recordings, paired, strict=True)]
    )
    paired_values = np.asarray(paired, dtype=np.float64)
    return PairedMetric(
        metric=metric,
        effect=effect,
        recordings=len(recordings),
        sources={
            source: SourceMeans(float(np.mean(values)), float(np.median(values)))
            for source, values in means.items()
        },
        paired_mean=float(np.mean(paired_values)),
        paired_median=float(np.median(paired_values)),
        clusters=clusters,
    )


def _counts(recordings: Sequence[RecordingFrames], source: str) -> SourceCounts:
    """One source's eligible frames, capped frames per SNR metric, and silent frames."""
    frames = [recording.sources[source] for recording in recordings]
    return SourceCounts(
        eligible_frames=sum(len(item.silent) for item in frames),
        capped_reconstruction_snr=sum(sum(item.capped[RECONSTRUCTION_SNR]) for item in frames),
        capped_si_snr=sum(sum(item.capped[SI_SNR]) for item in frames),
        silent=sum(sum(item.silent) for item in frames),
    )
