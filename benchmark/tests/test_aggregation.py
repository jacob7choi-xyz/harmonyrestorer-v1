"""Tests for aggregating frame metrics into recording means and the paired bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.aggregation import (
    AggregationError,
    RecordingScores,
    bootstrap_means,
    median,
    summarise,
)
from benchmark.protocol import Protocol, load_protocol

METRICS = ("reconstruction_snr", "si_snr", "log_spectral_distance")


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol."""
    return load_protocol()


def recording(
    identifier: str, opgan: dict[str, list[float]], uvr: dict[str, list[float]]
) -> RecordingScores:
    """A recording's scores."""
    return RecordingScores(identifier, opgan, uvr)


def same_everywhere(identifier: str, opgan: list[float], uvr: list[float]) -> RecordingScores:
    """A recording whose frame values are the same for every metric."""
    return recording(identifier, dict.fromkeys(METRICS, opgan), dict.fromkeys(METRICS, uvr))


class TestMedian:
    """v6's median, including the even-count rule."""

    def test_an_odd_count_takes_the_middle_value(self) -> None:
        assert median(np.array([5.0, 1.0, 3.0])) == 3.0

    def test_an_even_count_takes_the_mean_of_the_middle_two(self) -> None:
        assert median(np.array([4.0, 1.0, 3.0, 10.0])) == 3.5


class TestBootstrapMeans:
    """One generator, one index array per iteration, applied to every estimand."""

    def test_the_draws_are_the_frozen_generator_in_iteration_order(
        self, protocol: Protocol
    ) -> None:
        config = protocol.aggregation.bootstrap
        values = np.arange(7, dtype=np.float64) ** 2
        generator = np.random.Generator(np.random.PCG64(config.seed))
        expected = [values[generator.integers(0, 7, size=7)].mean() for _ in range(3)]

        means = bootstrap_means(values[None, :], protocol)

        assert means.shape == (1, config.iterations)
        np.testing.assert_array_equal(means[0, :3], expected)

    def test_every_estimand_shares_each_draw(self, protocol: Protocol) -> None:
        values = np.random.default_rng(0).normal(size=12)
        means = bootstrap_means(np.vstack([values, values + 1.0]), protocol)
        np.testing.assert_allclose(means[1] - means[0], 1.0)


class TestSummarise:
    """Recording means, macro means, medians, intervals, and the direction of each effect."""

    def test_a_positive_effect_favours_opgan_for_every_metric(self, protocol: Protocol) -> None:
        """OpGAN is better everywhere: higher SNR and SI-SNR, lower log-spectral distance."""
        recordings = [
            recording(
                f"r{i}",
                {
                    "reconstruction_snr": [10.0 + i],
                    "si_snr": [12.0],
                    "log_spectral_distance": [2.0],
                },
                {"reconstruction_snr": [8.0 + i], "si_snr": [11.0], "log_spectral_distance": [3.5]},
            )
            for i in range(4)
        ]
        summaries = summarise(recordings, protocol)
        assert [s.metric for s in summaries] == list(METRICS)
        assert [s.paired_mean for s in summaries] == [2.0, 1.0, 1.5]
        assert [s.paired_effect for s in summaries] == [
            "opgan_minus_uvr",
            "opgan_minus_uvr",
            "uvr_minus_opgan",
        ]

    def test_frames_are_averaged_within_and_recordings_unweighted_across(
        self, protocol: Protocol
    ) -> None:
        """A recording with more frames does not weigh more in the macro mean."""
        recordings = [
            same_everywhere("long", [1.0, 2.0, 3.0, 4.0, 5.0], [0.0] * 5),
            same_everywhere("short", [9.0], [0.0]),
        ]
        snr = summarise(recordings, protocol)[0]
        assert snr.opgan_mean == 6.0
        assert snr.opgan_median == 6.0
        assert snr.recordings == 2

    def test_the_intervals_are_linear_percentiles_of_the_shared_draws(
        self, protocol: Protocol
    ) -> None:
        values = np.random.default_rng(1).normal(size=9)
        recordings = [same_everywhere(f"r{i}", [float(v)], [0.0]) for i, v in enumerate(values)]
        snr = summarise(recordings, protocol)[0]

        means = bootstrap_means(values[None, :], protocol)[0]
        low, high = np.percentile(means, [2.5, 97.5], method="linear")
        assert (snr.opgan_interval.low, snr.opgan_interval.high) == (low, high)
        assert (snr.paired_interval.low, snr.paired_interval.high) == (low, high)
        assert (snr.uvr_interval.low, snr.uvr_interval.high) == (0.0, 0.0)

    def test_a_paired_effect_is_formed_before_resampling(self, protocol: Protocol) -> None:
        """Equal per-recording differences give a zero-width interval, however the systems vary."""
        recordings = [same_everywhere(f"r{i}", [float(i) + 1.0], [float(i)]) for i in range(6)]
        snr = summarise(recordings, protocol)[0]
        assert (snr.paired_interval.low, snr.paired_interval.high) == (1.0, 1.0)
        assert snr.opgan_interval.low < snr.opgan_interval.high

    def test_the_paired_median_is_the_median_of_per_recording_effects(
        self, protocol: Protocol
    ) -> None:
        """Differences [1, -3, 4] have median 1, while the medians differ by 2 - 5 = -3."""
        pairs = [(1.0, 0.0), (2.0, 5.0), (10.0, 6.0)]
        recordings = [same_everywhere(f"r{i}", [o], [u]) for i, (o, u) in enumerate(pairs)]
        snr = summarise(recordings, protocol)[0]
        assert (snr.opgan_median, snr.uvr_median, snr.paired_median) == (2.0, 5.0, 1.0)

    @pytest.mark.parametrize(
        "recordings",
        [
            [],
            [same_everywhere("r", [1.0], [1.0]), same_everywhere("r", [1.0], [1.0])],
            [same_everywhere("r", [1.0, 2.0], [1.0])],
            [same_everywhere("r", [], [])],
        ],
        ids=["no_recordings", "repeated_identifier", "unequal_frames", "no_frames"],
    )
    def test_scores_that_cannot_be_aggregated_are_refused(
        self, protocol: Protocol, recordings: list[RecordingScores]
    ) -> None:
        with pytest.raises(AggregationError):
            summarise(recordings, protocol)
