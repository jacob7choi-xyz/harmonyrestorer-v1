"""Tests for the population gates, coverage, and building aggregation input."""

from __future__ import annotations

import pytest

from benchmark.frame_scoring import FrameScore, SystemFailure
from benchmark.metrics import MetricResult
from benchmark.protocol import Protocol, load_protocol
from benchmark.scoring import (
    BASELINE,
    OPGAN,
    UVR,
    PopulationError,
    SystemFrameResult,
    coverage,
    population_gates,
    recording_scores,
)

EXPECTED = {"a": [0, 2], "b": [1]}
CLUSTERS = {"a": "performer-x", "b": "performer-y"}
ALL_GATES = {
    "population_ids_equal_across_systems",
    "frame_population_ids_equal_across_systems",
    "coverage_opgan_complete",
    "coverage_uvr_complete",
}


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol."""
    return load_protocol()


def scored(value: float) -> FrameScore:
    """A frame scored at `value` on every primary metric and on both shift-tolerant metrics."""
    result = MetricResult(value, False)
    return FrameScore(
        {"reconstruction_snr": result, "si_snr": result, "log_spectral_distance": result},
        0,
        0,
        {"reconstruction_snr": result, "si_snr": result},
    )


def results_for(base: float, *, skip: tuple[str, int] | None = None) -> list[SystemFrameResult]:
    """A scored result for every expected frame, valued `base + frame index`."""
    return [
        SystemFrameResult(recording, index, scored(base + index))
        for recording, indices in EXPECTED.items()
        for index in indices
        if (recording, index) != skip
    ]


class TestCoverage:
    """Valid outputs over the expected eligible frames, never over the results."""

    def test_full_coverage(self) -> None:
        assert coverage(EXPECTED, results_for(0.0)) == 1.0

    def test_a_system_failure_stays_in_the_denominator(self) -> None:
        results = results_for(0.0)
        results[0] = SystemFrameResult("a", 0, SystemFailure("exception in denoise"))
        assert coverage(EXPECTED, results) == pytest.approx(2 / 3)

    def test_a_missing_result_lowers_coverage(self) -> None:
        assert coverage(EXPECTED, results_for(0.0, skip=("b", 1))) == pytest.approx(2 / 3)

    def test_results_outside_the_population_and_repeats_do_not_raise_it(self) -> None:
        results = results_for(0.0, skip=("b", 1))
        results += [SystemFrameResult("c", 0, scored(1.0)), results[0]]
        assert coverage(EXPECTED, results) == pytest.approx(2 / 3)

    def test_an_empty_population_is_refused(self) -> None:
        with pytest.raises(PopulationError):
            coverage({"a": []}, [])


class TestPopulationGates:
    """Each gate by its protocol name, checked for both systems against the frozen set."""

    def test_identical_complete_populations_pass_every_gate(self, protocol: Protocol) -> None:
        gates = population_gates(
            EXPECTED, {OPGAN: results_for(0.0), UVR: results_for(1.0)}, protocol
        )
        assert gates == dict.fromkeys(ALL_GATES, True)

    def test_the_gate_names_are_the_protocol_names(self, protocol: Protocol) -> None:
        assert set(protocol.publication_gates.required) >= ALL_GATES

    def test_a_missing_recording_fails_both_population_gates(self, protocol: Protocol) -> None:
        uvr = [r for r in results_for(1.0) if r.recording_id != "b"]
        gates = population_gates(EXPECTED, {OPGAN: results_for(0.0), UVR: uvr}, protocol)
        assert not gates["population_ids_equal_across_systems"]
        assert not gates["frame_population_ids_equal_across_systems"]
        assert not gates["coverage_uvr_complete"]
        assert gates["coverage_opgan_complete"]

    def test_an_extra_recording_fails_both_population_gates(self, protocol: Protocol) -> None:
        uvr = [*results_for(1.0), SystemFrameResult("c", 0, scored(1.0))]
        gates = population_gates(EXPECTED, {OPGAN: results_for(0.0), UVR: uvr}, protocol)
        assert not gates["population_ids_equal_across_systems"]
        assert not gates["frame_population_ids_equal_across_systems"]
        assert gates["coverage_uvr_complete"]

    def test_a_substituted_frame_fails_the_frame_gate_only(self, protocol: Protocol) -> None:
        """Equal counts are not enough: frame 1 of `a` in place of frame 2."""
        uvr = [r for r in results_for(1.0) if (r.recording_id, r.frame_index) != ("a", 2)]
        uvr.append(SystemFrameResult("a", 1, scored(1.0)))
        gates = population_gates(EXPECTED, {OPGAN: results_for(0.0), UVR: uvr}, protocol)
        assert gates["population_ids_equal_across_systems"]
        assert not gates["frame_population_ids_equal_across_systems"]

    def test_a_repeated_frame_fails_the_frame_gate(self, protocol: Protocol) -> None:
        uvr = [*results_for(1.0), SystemFrameResult("b", 1, scored(1.0))]
        gates = population_gates(EXPECTED, {OPGAN: results_for(0.0), UVR: uvr}, protocol)
        assert not gates["frame_population_ids_equal_across_systems"]

    def test_a_system_failure_fails_that_systems_coverage_gate(self, protocol: Protocol) -> None:
        opgan = results_for(0.0)
        opgan[1] = SystemFrameResult("a", 2, SystemFailure("output rejected"))
        gates = population_gates(EXPECTED, {OPGAN: opgan, UVR: results_for(1.0)}, protocol)
        assert not gates["coverage_opgan_complete"]
        assert gates["frame_population_ids_equal_across_systems"]

    def test_a_system_with_no_results_fails(self, protocol: Protocol) -> None:
        gates = population_gates(EXPECTED, {OPGAN: results_for(0.0)}, protocol)
        assert not gates["population_ids_equal_across_systems"]
        assert not gates["coverage_uvr_complete"]


class TestRecordingScores:
    """Aggregation input is built only when every gate holds, in the expected order."""

    def test_values_follow_the_expected_recording_and_frame_order(self, protocol: Protocol) -> None:
        uvr = list(reversed(results_for(10.0)))
        scores = recording_scores(
            EXPECTED,
            {OPGAN: results_for(0.0), UVR: uvr, BASELINE: results_for(20.0)},
            protocol,
            CLUSTERS,
        )
        assert [s.recording_id for s in scores] == ["a", "b"]
        assert [s.cluster for s in scores] == ["performer-x", "performer-y"]
        assert scores[0].opgan.primary["si_snr"] == [0.0, 2.0]
        assert scores[0].uvr.primary["log_spectral_distance"] == [10.0, 12.0]
        assert scores[1].uvr.primary["reconstruction_snr"] == [11.0]

    def test_the_baseline_and_the_shift_tolerant_values_are_carried(
        self, protocol: Protocol
    ) -> None:
        scores = recording_scores(
            EXPECTED,
            {OPGAN: results_for(0.0), UVR: results_for(1.0), BASELINE: results_for(20.0)},
            protocol,
            CLUSTERS,
        )
        assert scores[0].baseline.primary["si_snr"] == [20.0, 22.0]
        assert scores[1].opgan.shift_tolerant["reconstruction_snr"] == [1.0]
        assert set(scores[0].uvr.shift_tolerant) == {"reconstruction_snr", "si_snr"}

    def test_incomplete_coverage_withholds_the_input(self, protocol: Protocol) -> None:
        opgan = results_for(0.0)
        opgan[2] = SystemFrameResult("b", 1, SystemFailure("exception in denoise"))
        with pytest.raises(PopulationError, match="coverage_opgan_complete"):
            recording_scores(
                EXPECTED,
                {OPGAN: opgan, UVR: results_for(1.0), BASELINE: results_for(2.0)},
                protocol,
                CLUSTERS,
            )

    def test_a_baseline_with_a_gap_withholds_the_input(self, protocol: Protocol) -> None:
        """The baseline has no coverage gate, but a missing baseline frame still stops the run."""
        baseline = [r for r in results_for(2.0) if r.recording_id != "b"]
        with pytest.raises(PopulationError, match=BASELINE):
            recording_scores(
                EXPECTED,
                {OPGAN: results_for(0.0), UVR: results_for(1.0), BASELINE: baseline},
                protocol,
                CLUSTERS,
            )

    def test_a_baseline_failure_withholds_the_input(self, protocol: Protocol) -> None:
        baseline = results_for(2.0)
        baseline[0] = SystemFrameResult("a", 0, SystemFailure("exception in denoise"))
        with pytest.raises(PopulationError, match=r"no score|do not match"):
            recording_scores(
                EXPECTED,
                {OPGAN: results_for(0.0), UVR: results_for(1.0), BASELINE: baseline},
                protocol,
                CLUSTERS,
            )

    def test_a_recording_without_a_cluster_withholds_the_input(self, protocol: Protocol) -> None:
        with pytest.raises(PopulationError, match="performer cluster"):
            recording_scores(
                EXPECTED,
                {OPGAN: results_for(0.0), UVR: results_for(1.0), BASELINE: results_for(2.0)},
                protocol,
                {"a": "performer-x"},
            )

    def test_a_missing_shift_tolerant_score_withholds_the_input(self, protocol: Protocol) -> None:
        opgan = results_for(0.0)
        primary_only = scored(0.0)
        opgan[0] = SystemFrameResult(
            "a", 0, FrameScore(primary_only.results, 0, 0, shift_tolerant={})
        )
        with pytest.raises(PopulationError, match="no reconstruction_snr"):
            recording_scores(
                EXPECTED,
                {OPGAN: opgan, UVR: results_for(1.0), BASELINE: results_for(2.0)},
                protocol,
                CLUSTERS,
            )
