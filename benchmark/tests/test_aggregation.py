"""Tests for aggregating frame metrics into recording means and the performer-cluster bootstrap."""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.aggregation import (
    AggregationError,
    BaselineSummary,
    RecordingScores,
    Series,
    SystemValues,
    _cluster_members,
    _selected_recordings,
    bootstrap_means,
    median,
    summarise,
)
from benchmark.protocol import Protocol, load_protocol

METRICS = ("reconstruction_snr", "si_snr", "log_spectral_distance")
SHIFT = ("reconstruction_snr", "si_snr")


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol."""
    return load_protocol()


def source(frames: list[float], shift: list[float] | None = None) -> SystemValues:
    """One source's values: the same frames for every primary metric, and the shift diagnostic."""
    return SystemValues(
        dict.fromkeys(METRICS, frames), dict.fromkeys(SHIFT, frames if shift is None else shift)
    )


def recording(
    identifier: str,
    cluster: str,
    opgan: list[float],
    uvr: list[float],
    baseline: list[float] | None = None,
) -> RecordingScores:
    """A recording whose baseline defaults to the UVR values."""
    return RecordingScores(
        identifier,
        cluster,
        source(opgan),
        source(uvr),
        source(uvr if baseline is None else baseline),
    )


def one_per_cluster(values: list[float]) -> list[RecordingScores]:
    """Recordings whose values differ, each in a cluster of its own, with OpGAN equal to UVR."""
    return [recording(f"r{i}", f"c{i}", [v], [v]) for i, v in enumerate(values)]


class TestMedian:
    """v6's median, including the even-count rule."""

    def test_an_odd_count_takes_the_middle_value(self) -> None:
        assert median(np.array([5.0, 1.0, 3.0])) == 3.0

    def test_an_even_count_takes_the_mean_of_the_middle_two(self) -> None:
        assert median(np.array([4.0, 1.0, 3.0, 10.0])) == 3.5


class TestClusterMembers:
    """Clusters are numbered by their first appearance, which is the first test admission."""

    def test_clusters_follow_first_appearance_not_their_names(self) -> None:
        """The first recording is in cluster "y", so "y" comes before "x" although "x" sorts first."""
        recordings = [
            recording("a", "y", [1.0], [1.0]),
            recording("b", "x", [1.0], [1.0]),
            recording("c", "y", [1.0], [1.0]),
        ]
        assert _cluster_members(recordings) == [[0, 2], [1]]


class TestSelectedRecordings:
    """A drawn cluster contributes every recording it holds, and a repeated draw repeats them."""

    def test_a_cluster_drawn_twice_counts_twice(self) -> None:
        columns = [np.array([0], dtype=np.intp), np.array([1, 2], dtype=np.intp)]
        draw = np.array([0, 0, 1])
        np.testing.assert_array_equal(_selected_recordings(columns, draw), [0, 0, 1, 2])

    def test_the_mean_over_a_repeated_cluster_is_weighted_by_its_repeats(self) -> None:
        """Clusters [0] and [1] with values 0 and 10: drawing cluster 0 twice and 1 once gives
        (0 + 0 + 10) / 3, not the 5 that counting each drawn cluster once would give."""
        columns = [np.array([0], dtype=np.intp), np.array([1], dtype=np.intp)]
        values = np.array([0.0, 10.0])
        selected = _selected_recordings(columns, np.array([0, 0, 1]))
        assert values[selected].mean() == pytest.approx(10.0 / 3.0)


class TestBootstrapMeans:
    """One generator, one cluster index array per iteration, applied to every estimand."""

    def test_the_draws_are_the_frozen_generator_over_clusters_in_iteration_order(
        self, protocol: Protocol
    ) -> None:
        config = protocol.aggregation.bootstrap
        members = [[0, 1], [2], [3, 4, 5]]
        values = np.arange(6, dtype=np.float64) ** 2
        generator = np.random.Generator(np.random.PCG64(config.seed))
        expected = []
        for _ in range(3):
            draw = generator.integers(0, 3, size=3)
            selected = [i for cluster in draw for i in members[cluster]]
            expected.append(values[selected].mean())

        means = bootstrap_means(values[None, :], members, protocol)

        assert means.shape == (1, config.iterations)
        np.testing.assert_allclose(means[0, :3], expected, rtol=0, atol=1e-12)

    def test_the_first_two_draws_of_the_frozen_seed_are_pinned(self, protocol: Protocol) -> None:
        """Three singleton clusters with values 10, 20, 30. The seed's first two draws of
        integers(0, 3, size=3) are [0, 2, 0] and [2, 2, 1], which give the means
        (10 + 30 + 10) / 3 and (30 + 30 + 20) / 3."""
        means = bootstrap_means(np.array([[10.0, 20.0, 30.0]]), [[0], [1], [2]], protocol)
        np.testing.assert_allclose(means[0, :2], [50.0 / 3.0, 80.0 / 3.0], rtol=0, atol=1e-12)

    def test_a_single_cluster_gives_its_own_mean_on_every_iteration(
        self, protocol: Protocol
    ) -> None:
        means = bootstrap_means(np.array([[1.0, 2.0, 6.0]]), [[0, 1, 2]], protocol)
        np.testing.assert_array_equal(
            means[0], np.full(protocol.aggregation.bootstrap.iterations, 3.0)
        )

    def test_every_iteration_is_a_cluster_level_mean(self, protocol: Protocol) -> None:
        """Clusters {0, 1, 2} and {3} with values 0, 0, 0, 10. A cluster draw can only give the
        means of [0, 0, 0] (cluster 0 drawn twice), of [10] (cluster 1 twice), or of the four
        recordings (one of each). A recording-level draw could also give 5, 7.5 and others."""
        values = np.array([[0.0, 0.0, 0.0, 10.0]])
        means = bootstrap_means(values, [[0, 1, 2], [3]], protocol)[0]
        assert set(np.round(means, 9)) == {0.0, 2.5, 10.0}

    def test_every_estimand_shares_each_draw(self, protocol: Protocol) -> None:
        values = np.random.default_rng(0).normal(size=12)
        members = [[0, 1], [2, 3, 4], [5], [6, 7, 8], [9, 10, 11]]
        means = bootstrap_means(np.vstack([values, values + 1.0]), members, protocol)
        np.testing.assert_allclose(means[1] - means[0], 1.0, rtol=0, atol=1e-12)


class TestSummarise:
    """Recording means, macro means, medians, intervals, and the direction of each effect."""

    def test_a_positive_effect_favours_opgan_for_every_metric(self, protocol: Protocol) -> None:
        """OpGAN is better everywhere: higher SNR and SI-SNR, lower log-spectral distance."""
        recordings = [
            RecordingScores(
                f"r{i}",
                f"c{i}",
                SystemValues(
                    {
                        "reconstruction_snr": [10.0 + i],
                        "si_snr": [12.0],
                        "log_spectral_distance": [2.0],
                    },
                    {"reconstruction_snr": [10.0 + i], "si_snr": [12.0]},
                ),
                SystemValues(
                    {
                        "reconstruction_snr": [8.0 + i],
                        "si_snr": [11.0],
                        "log_spectral_distance": [3.5],
                    },
                    {"reconstruction_snr": [8.0 + i], "si_snr": [11.0]},
                ),
                source([0.0]),
            )
            for i in range(4)
        ]
        aggregate = summarise(recordings, protocol)
        assert [s.metric for s in aggregate.primary] == list(METRICS)
        assert [s.paired_mean for s in aggregate.primary] == [2.0, 1.0, 1.5]
        assert [s.paired_effect for s in aggregate.primary] == [
            "opgan_minus_uvr",
            "opgan_minus_uvr",
            "uvr_minus_opgan",
        ]

    def test_frames_are_averaged_within_and_recordings_unweighted_across(
        self, protocol: Protocol
    ) -> None:
        """A recording with more frames does not weigh more in the macro mean."""
        recordings = [
            recording("long", "a", [1.0, 2.0, 3.0, 4.0, 5.0], [0.0] * 5),
            recording("short", "b", [9.0], [0.0]),
        ]
        snr = summarise(recordings, protocol).primary[0]
        assert snr.opgan_mean == 6.0
        assert snr.opgan_median == 6.0
        assert snr.recordings == 2

    def test_the_intervals_are_linear_percentiles_of_the_shared_draws(
        self, protocol: Protocol
    ) -> None:
        """The bounds are the 2.5th and 97.5th percentiles the protocol names, taken from the
        same cluster draws as the means. Seeds 2, 4, 8 and 9 gave a different last bit when the
        lower bound was computed in floating point as 2.500000000000002, so the test covers ten
        seeds rather than one."""
        for seed in range(1, 11):
            values = np.random.default_rng(seed).normal(size=9)
            snr = summarise(one_per_cluster([float(v) for v in values]), protocol).primary[0]

            means = bootstrap_means(values[None, :], [[i] for i in range(9)], protocol)[0]
            low, high = np.percentile(means, [2.5, 97.5], method="linear")
            assert (snr.opgan_interval.low, snr.opgan_interval.high) == (low, high), seed
            assert (snr.paired_interval.low, snr.paired_interval.high) == (0.0, 0.0), seed

    def test_a_paired_effect_is_formed_before_resampling(self, protocol: Protocol) -> None:
        """Equal per-recording differences give a zero-width interval, however the systems vary."""
        recordings = [recording(f"r{i}", f"c{i}", [float(i) + 1.0], [float(i)]) for i in range(6)]
        snr = summarise(recordings, protocol).primary[0]
        assert (snr.paired_interval.low, snr.paired_interval.high) == (1.0, 1.0)
        assert snr.opgan_interval.low < snr.opgan_interval.high

    def test_the_paired_median_is_the_median_of_per_recording_effects(
        self, protocol: Protocol
    ) -> None:
        """Differences [1, -3, 4] have median 1, while the medians differ by 2 - 5 = -3."""
        pairs = [(1.0, 0.0), (2.0, 5.0), (10.0, 6.0)]
        recordings = [recording(f"r{i}", f"c{i}", [o], [u]) for i, (o, u) in enumerate(pairs)]
        snr = summarise(recordings, protocol).primary[0]
        assert (snr.opgan_median, snr.uvr_median, snr.paired_median) == (2.0, 5.0, 1.0)

    def test_the_diagnostic_is_summarised_from_the_same_draws(self, protocol: Protocol) -> None:
        """A diagnostic that is the primary value plus one shifts every interval by exactly one,
        which holds only if both were resampled with the same cluster draws."""
        recordings = [
            RecordingScores(
                f"r{i}",
                f"c{i % 3}",
                SystemValues(
                    {m: [float(i)] for m in METRICS},
                    {"reconstruction_snr": [float(i)], "si_snr": [float(i) + 1.0]},
                ),
                source([0.0]),
                source([0.0]),
            )
            for i in range(7)
        ]
        aggregate = summarise(recordings, protocol)
        primary = next(s for s in aggregate.primary if s.metric == "si_snr")
        diagnostic = aggregate.diagnostic[1]
        assert diagnostic.metric == "si_snr"
        assert diagnostic.paired_effect == "opgan_minus_uvr"
        assert diagnostic.paired_mean == pytest.approx(primary.paired_mean + 1.0)
        assert diagnostic.paired_interval.low == pytest.approx(primary.paired_interval.low + 1.0)
        assert diagnostic.paired_interval.high == pytest.approx(primary.paired_interval.high + 1.0)

    def test_the_diagnostic_is_listed_for_the_shift_tolerant_metrics_only(
        self, protocol: Protocol
    ) -> None:
        aggregate = summarise(one_per_cluster([1.0, 2.0]), protocol)
        assert [s.metric for s in aggregate.diagnostic] == list(SHIFT)

    def test_the_baseline_has_no_paired_effect_and_uses_the_same_draws(
        self, protocol: Protocol
    ) -> None:
        """The baseline is the UVR values plus two: its interval moves by two, with no paired
        field to read. Primary first, then the diagnostic metrics, each with its series."""
        recordings = [
            recording(f"r{i}", f"c{i}", [float(i)], [float(i)], [float(i) + 2.0]) for i in range(5)
        ]
        aggregate = summarise(recordings, protocol)
        assert [(b.series, b.metric) for b in aggregate.baseline] == [
            (Series.PRIMARY, "reconstruction_snr"),
            (Series.PRIMARY, "si_snr"),
            (Series.PRIMARY, "log_spectral_distance"),
            (Series.DIAGNOSTIC, "reconstruction_snr"),
            (Series.DIAGNOSTIC, "si_snr"),
        ]
        baseline = aggregate.baseline[2]
        assert isinstance(baseline, BaselineSummary)
        assert not hasattr(baseline, "paired_mean")
        assert baseline.mean == pytest.approx(aggregate.primary[2].uvr_mean + 2.0)
        assert baseline.interval.low == pytest.approx(aggregate.primary[2].uvr_interval.low + 2.0)
        assert baseline.interval.high == pytest.approx(aggregate.primary[2].uvr_interval.high + 2.0)

    @pytest.mark.parametrize(
        "recordings",
        [
            [],
            [recording("r", "c", [1.0], [1.0]), recording("r", "d", [1.0], [1.0])],
            [recording("r", "c", [1.0, 2.0], [1.0])],
            [recording("r", "c", [], [])],
            [recording("r", "", [1.0], [1.0])],
            [recording("r", "c", [1.0], [1.0], [1.0, 2.0])],
        ],
        ids=[
            "no_recordings",
            "repeated_identifier",
            "unequal_frames",
            "no_frames",
            "empty_cluster",
            "baseline_frames_differ",
        ],
    )
    def test_scores_that_cannot_be_aggregated_are_refused(
        self, protocol: Protocol, recordings: list[RecordingScores]
    ) -> None:
        with pytest.raises(AggregationError):
            summarise(recordings, protocol)

    def test_a_missing_shift_tolerant_value_is_refused(self, protocol: Protocol) -> None:
        """A recording must carry every diagnostic metric the protocol names."""
        broken = RecordingScores(
            "r",
            "c",
            SystemValues({m: [1.0] for m in METRICS}, {"si_snr": [1.0]}),
            source([1.0]),
            source([1.0]),
        )
        with pytest.raises(AggregationError, match="reconstruction_snr"):
            summarise([broken], protocol)
