"""Tests for the post-hoc analyses' pure functions: the cluster interval, the band and its mask."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from benchmark.metrics import MetricInputError, log_spectral_distance
from benchmark.posthoc import (
    BAND_LAST_BIN,
    BAND_METRIC,
    LABEL,
    PosthocError,
    RecordingFrames,
    SourceFrames,
    analyse_stratum,
    band_distance_from_magnitudes,
    band_log_spectral_distance,
    cluster_summary,
    is_silent,
)
from benchmark.protocol import Protocol, load_protocol
from benchmark.scoring import BASELINE, OPGAN, UVR

# The 97.5% Student t quantiles for 1 and 2 degrees of freedom, in closed form. For one degree of
# freedom the CDF is 1/2 + arctan(t) / pi, so t = tan(0.475 pi). For two it is
# 1/2 + t / (2 sqrt(2 + t^2)), which gives t^2 = 1.805 / 0.0975.
T_975_DF1 = 1.0 / math.tan(math.pi / 40.0)
T_975_DF2 = math.sqrt(1.805 / 0.0975)
RATE = 16000
MAX_LOW_BAND_BIN = 100
ABOVE_BAND_BIN = 400


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol, loaded once for the module."""
    return load_protocol()


def make_source(
    snr: Sequence[float],
    si_snr: Sequence[float],
    lsd: Sequence[float],
    band: Sequence[float],
    *,
    capped_snr: Sequence[bool] | None = None,
    capped_si: Sequence[bool] | None = None,
    silent: Sequence[bool] | None = None,
) -> SourceFrames:
    """One source's frames, with every flag false unless the test says otherwise."""
    count = len(snr)
    return SourceFrames(
        primary={
            "reconstruction_snr": list(snr),
            "si_snr": list(si_snr),
            "log_spectral_distance": list(lsd),
        },
        band=list(band),
        capped={
            "reconstruction_snr": list(capped_snr or [False] * count),
            "si_snr": list(capped_si or [False] * count),
        },
        silent=list(silent or [False] * count),
    )


def make_recording(
    cluster: str,
    opgan: SourceFrames,
    uvr: SourceFrames,
    baseline: SourceFrames,
) -> RecordingFrames:
    """One recording with its three sources."""
    return RecordingFrames(
        recording_id=f"{cluster}-recording",
        cluster=cluster,
        sources={OPGAN: opgan, UVR: uvr, BASELINE: baseline},
    )


def test_the_cluster_interval_matches_a_hand_computation() -> None:
    """Three cluster values 1, 2 and 4: mean 7/3, sample sd sqrt(7/3), t with 2 degrees of freedom."""
    summary = cluster_summary([("a", 1.0), ("b", 2.0), ("c", 4.0)])
    mean = 7.0 / 3.0
    sd = math.sqrt(7.0 / 3.0)
    half_width = T_975_DF2 * sd / math.sqrt(3)
    assert summary.clusters == 3
    assert summary.mean == pytest.approx(mean, abs=1e-12)
    assert summary.sd == pytest.approx(sd, abs=1e-12)
    assert summary.low == pytest.approx(mean - half_width, abs=1e-12)
    assert summary.high == pytest.approx(mean + half_width, abs=1e-12)


def test_recordings_of_one_cluster_are_averaged_before_the_interval_is_taken() -> None:
    """Clusters a (1 and 3) and b (4) give two values, 2 and 4, so the interval has 1 df."""
    summary = cluster_summary([("a", 1.0), ("a", 3.0), ("b", 4.0)])
    mean = 3.0
    sd = math.sqrt(2.0)
    half_width = T_975_DF1 * sd / math.sqrt(2)
    assert summary.clusters == 2
    assert summary.mean == pytest.approx(mean, abs=1e-12)
    assert summary.low == pytest.approx(mean - half_width, abs=1e-9)
    assert summary.high == pytest.approx(mean + half_width, abs=1e-9)


def test_the_sign_counts_separate_positive_negative_and_exact_zero() -> None:
    """Cluster values -1.0, 0.0, 2.0 and 3.0 give one negative, one zero and two positive."""
    summary = cluster_summary([("a", -1.0), ("b", 0.0), ("c", 2.0), ("d", 3.0)])
    assert (summary.positive, summary.negative, summary.zero) == (2, 1, 1)


def test_fewer_than_two_clusters_is_refused_with_a_clear_message() -> None:
    """One cluster, however many recordings it holds, has no interval, and no value is returned."""
    with pytest.raises(PosthocError, match="at least two performer clusters, got 1"):
        cluster_summary([("a", 1.0), ("a", 2.0)])
    with pytest.raises(PosthocError, match="got 0"):
        cluster_summary([])


def test_band_with_every_bin_kept_equals_the_protocol_metric_exactly(protocol: Protocol) -> None:
    """Keeping all 513 bins reproduces metrics.log_spectral_distance bit for bit."""
    config = protocol.metrics.log_spectral_distance
    rng = np.random.default_rng(11)
    reference = rng.normal(0.0, 0.1, 8000)
    estimate = reference + rng.normal(0.0, 0.02, 8000)
    full = log_spectral_distance(reference, estimate, config).value
    band = band_log_spectral_distance(reference, estimate, config, last_bin=config.n_fft // 2)
    assert band == full


def test_the_band_boundary_is_bin_256_and_matches_the_lowest_drawn_rolloff(
    protocol: Protocol,
) -> None:
    """BAND_LAST_BIN is the bin at the lowest rolloff the degradation draws: 4 kHz is bin 256."""
    rolloff_low = min(protocol.degradation.ranges.rolloff_hz)
    bin_width = protocol.conditioning.sample_rate / protocol.metrics.log_spectral_distance.n_fft
    assert BAND_LAST_BIN == math.floor(rolloff_low / bin_width) == 256


def test_a_difference_below_4_khz_moves_the_band_value(protocol: Protocol) -> None:
    """A tone in the low band changes the band-limited distance by a clearly visible amount."""
    config = protocol.metrics.log_spectral_distance
    rng = np.random.default_rng(3)
    reference = rng.normal(0.0, 0.1, RATE)
    tone = 0.1 * np.cos(2 * np.pi * MAX_LOW_BAND_BIN * np.arange(RATE) / 1024)
    assert band_log_spectral_distance(reference, reference, config) == 0.0
    assert band_log_spectral_distance(reference, reference + tone, config) > 0.1


def test_a_difference_only_above_4_khz_leaves_the_band_nearly_unchanged(protocol: Protocol) -> None:
    """A smooth tone at bin 400 (6.25 kHz) moves the full band but not the band below 4 kHz."""
    config = protocol.metrics.log_spectral_distance
    rng = np.random.default_rng(3)
    reference = rng.normal(0.0, 0.1, RATE)
    samples = np.arange(RATE)
    tone = 0.1 * np.cos(2 * np.pi * ABOVE_BAND_BIN * samples / 1024) * np.hanning(RATE)
    estimate = reference + tone
    full = log_spectral_distance(reference, estimate, config).value
    band = band_log_spectral_distance(reference, estimate, config)
    assert full > 0.1
    assert band < 1e-3


def test_the_band_refuses_a_silent_reference(protocol: Protocol) -> None:
    """A silent reference has no fidelity to measure, so the band refuses it as the metric does."""
    config = protocol.metrics.log_spectral_distance
    with pytest.raises(MetricInputError):
        band_log_spectral_distance(np.zeros(4096), np.ones(4096), config)


@pytest.fixture(scope="module")
def spectra() -> tuple[np.ndarray, np.ndarray]:
    """A clean and an estimated magnitude spectrum, shaped as 5 frames of 513 bins."""
    rng = np.random.default_rng(21)
    clean = rng.uniform(0.1, 2.0, (5, 513))
    return clean, clean * 1.05


def band_of(spectra_pair: tuple[np.ndarray, np.ndarray], last_bin: int) -> float:
    """The band distance of a magnitude pair at the protocol's offset and dB factor."""
    clean, estimate = spectra_pair
    return band_distance_from_magnitudes(
        clean, estimate, offset=1e-8, multiplier=20.0, last_bin=last_bin
    )


def test_bins_above_the_boundary_do_not_change_the_band(
    spectra: tuple[np.ndarray, np.ndarray],
) -> None:
    """Changing only bins 257 and above leaves the band value identical, and the full band moves."""
    clean, estimate = spectra
    changed = estimate.copy()
    changed[:, BAND_LAST_BIN + 1 :] *= 3.0
    assert band_of((clean, changed), BAND_LAST_BIN) == band_of((clean, estimate), BAND_LAST_BIN)
    assert band_of((clean, changed), 512) != band_of((clean, estimate), 512)


def test_bin_256_is_inside_the_band(spectra: tuple[np.ndarray, np.ndarray]) -> None:
    """Changing bin 256 alone changes the band value, so 256 is the last bin kept."""
    clean, estimate = spectra
    changed = estimate.copy()
    changed[:, BAND_LAST_BIN] *= 3.0
    assert band_of((clean, changed), BAND_LAST_BIN) != band_of((clean, estimate), BAND_LAST_BIN)


def test_bin_255_is_inside_the_band(spectra: tuple[np.ndarray, np.ndarray]) -> None:
    """Changing bin 255 alone changes the band value, so the band starts at bin 0 and runs to 256."""
    clean, estimate = spectra
    changed = estimate.copy()
    changed[:, BAND_LAST_BIN - 1] *= 3.0
    assert band_of((clean, changed), BAND_LAST_BIN) != band_of((clean, estimate), BAND_LAST_BIN)


def test_a_last_bin_the_spectrum_does_not_have_is_refused(
    spectra: tuple[np.ndarray, np.ndarray],
) -> None:
    """A last bin past the top bin is a programming error, so it raises rather than clipping."""
    clean, estimate = spectra
    with pytest.raises(ValueError, match="last_bin must be in"):
        band_of((clean, estimate), 513)


def test_silent_means_exactly_zero_in_every_sample() -> None:
    """All-zero output is silent. One sample that is merely tiny is not, so near-silence is kept."""
    assert is_silent(np.zeros(1000)) is True
    tiny = np.zeros(1000)
    tiny[500] = 1e-300
    assert is_silent(tiny) is False


def test_the_paired_difference_is_oriented_so_positive_means_opgan_scored_higher(
    protocol: Protocol,
) -> None:
    """SNR: OpGAN minus UVR. Distance: UVR minus OpGAN, so a larger UVR distance is positive."""
    recordings = [
        make_recording(
            "a",
            make_source([12.0, 12.0], [11.0, 11.0], [2.0, 2.0], [2.0, 2.0]),
            make_source([9.0, 9.0], [8.0, 8.0], [3.0, 3.0], [3.0, 3.0]),
            make_source([5.0, 5.0], [4.0, 4.0], [6.0, 6.0], [6.0, 6.0]),
        ),
        make_recording(
            "b",
            make_source([10.0], [9.0], [2.0], [2.0]),
            make_source([11.0], [10.0], [2.5], [2.5]),
            make_source([5.0], [4.0], [6.0], [6.0]),
        ),
    ]
    analysis = analyse_stratum(recordings, protocol)
    snr = next(m for m in analysis.primary if m.metric == "reconstruction_snr")
    assert snr.effect == "opgan_minus_uvr"
    # Recording means: OpGAN 12 and 10, UVR 9 and 11. Differences: 12 - 9 = 3 and 10 - 11 = -1.
    assert snr.sources[OPGAN].mean == pytest.approx(11.0)
    assert snr.sources[UVR].mean == pytest.approx(10.0)
    assert snr.paired_mean == pytest.approx(1.0)
    lsd = next(m for m in analysis.primary if m.metric == "log_spectral_distance")
    assert lsd.effect == "uvr_minus_opgan"
    # Recording means: OpGAN 2 and 2, UVR 3 and 2.5. UVR minus OpGAN: 1 and 0.5.
    assert lsd.paired_mean == pytest.approx(0.75)
    assert analysis.band.metric == BAND_METRIC
    assert analysis.band.effect == "uvr_minus_opgan"
    assert analysis.band.paired_mean == pytest.approx(0.75)
    assert (analysis.band.clusters.positive, analysis.band.clusters.negative) == (2, 0)


def test_the_counts_are_capped_and_silent_frames_per_source(protocol: Protocol) -> None:
    """Capped flags are counted per SNR metric, silent flags per source, over eligible frames."""
    recordings = [
        make_recording(
            "a",
            make_source(
                [12.0, 12.0, 12.0],
                [11.0, 11.0, 11.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 2.0],
                capped_snr=[True, False, False],
                capped_si=[True, True, False],
                silent=[False, True, False],
            ),
            make_source([9.0, 9.0], [8.0, 8.0], [3.0, 3.0], [3.0, 3.0]),
            make_source([5.0], [4.0], [6.0], [6.0]),
        ),
        make_recording(
            "b",
            make_source([10.0], [9.0], [2.0], [2.0], capped_snr=[True], silent=[True]),
            make_source([11.0], [10.0], [1.0], [1.0]),
            make_source([5.0, 5.0], [4.0, 4.0], [6.0, 6.0], [6.0, 6.0]),
        ),
    ]
    counts = analyse_stratum(recordings, protocol).counts
    assert counts[OPGAN].eligible_frames == 4
    assert counts[OPGAN].capped_reconstruction_snr == 2
    assert counts[OPGAN].capped_si_snr == 2
    assert counts[OPGAN].silent == 2
    assert counts[UVR].eligible_frames == 3
    assert (counts[UVR].capped_reconstruction_snr, counts[UVR].silent) == (0, 0)
    assert counts[BASELINE].eligible_frames == 3


def test_a_stratum_with_one_cluster_has_no_analysis(protocol: Protocol) -> None:
    """Every recording in one cluster leaves k = 1, so the analysis is refused and nothing is made."""
    recordings = [
        make_recording(
            "only",
            make_source([12.0], [11.0], [2.0], [2.0]),
            make_source([9.0], [8.0], [3.0], [3.0]),
            make_source([5.0], [4.0], [6.0], [6.0]),
        )
    ]
    with pytest.raises(PosthocError, match="at least two performer clusters"):
        analyse_stratum(recordings, protocol)


def test_a_stratum_with_no_recording_has_no_analysis(protocol: Protocol) -> None:
    """An empty stratum is refused, since there is nothing to summarise."""
    with pytest.raises(PosthocError, match="no test recordings"):
        analyse_stratum([], protocol)


def test_the_label_is_the_one_every_number_carries() -> None:
    """The label names the status of every number: post-hoc, and fixed before any score was read."""
    assert LABEL == "post-hoc, added after the protocol froze, before any score was read"
