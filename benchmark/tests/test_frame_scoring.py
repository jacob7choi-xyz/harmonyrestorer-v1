"""Tests for scoring one system's output on one frame and attributing its failures."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from benchmark.conditioning import ConditioningError
from benchmark.degradation import degrade_frame, frame_seed
from benchmark.frame_scoring import (
    FrameScore,
    FrameScoringError,
    SystemFailure,
    degrade_with_reference,
    reference_frame,
    score_output,
)
from benchmark.metrics import PRIMARY_METRICS
from benchmark.protocol import Protocol, load_protocol


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol."""
    return load_protocol()


def music(samples: int = 32000, seed: int = 0) -> npt.NDArray[np.float64]:
    """Noise standing in for a canonical frame."""
    return np.random.default_rng(seed).uniform(-0.25, 0.25, samples)


class TestReferenceFrame:
    """The reference is the canonical frame times the gain, in float64, and read-only."""

    def test_the_reference_is_the_frame_times_the_gain(self) -> None:
        frame = music()
        reference = reference_frame(frame, 1.7)
        np.testing.assert_array_equal(reference, frame * 1.7)
        assert reference.dtype == np.float64
        assert not reference.flags.writeable


class TestDegradeWithReference:
    """The shared helper degrades and builds the reference exactly as the two steps did."""

    def test_the_helper_matches_degrading_then_building_the_reference(
        self, protocol: Protocol
    ) -> None:
        clean = music(seed=3)
        digest = "ab" * 32

        degraded, reference = degrade_with_reference(clean, 0.4, digest, 7, protocol)

        expected = degrade_frame(
            clean,
            recording_peak=0.4,
            sample_rate=protocol.canonicalisation.resample.target_sample_rate,
            ranges=protocol.degradation.ranges,
            seed=frame_seed(protocol.degradation.seed, digest, 7),
        )
        np.testing.assert_array_equal(degraded.degraded, expected.degraded)
        assert degraded.gain == expected.gain
        assert degraded.params == expected.params
        np.testing.assert_array_equal(reference, reference_frame(clean, expected.gain))
        assert not reference.flags.writeable

    def test_the_frame_index_and_digest_change_the_degradation(self, protocol: Protocol) -> None:
        clean = music(seed=3)

        first, _ = degrade_with_reference(clean, 0.4, "ab" * 32, 7, protocol)
        other_index, _ = degrade_with_reference(clean, 0.4, "ab" * 32, 8, protocol)
        other_digest, _ = degrade_with_reference(clean, 0.4, "cd" * 32, 7, protocol)

        assert not np.array_equal(first.degraded, other_index.degraded)
        assert not np.array_equal(first.degraded, other_digest.degraded)


class TestScoreOutput:
    """Valid outputs are scored; rejected outputs are the system's; reference faults abort."""

    def test_a_valid_output_is_scored_on_every_primary_metric_in_order(
        self, protocol: Protocol
    ) -> None:
        reference = reference_frame(music(), 1.0)
        estimate = reference + np.random.default_rng(1).normal(0, 0.01, reference.size)

        outcome = score_output(reference, estimate, protocol)

        assert isinstance(outcome, FrameScore)
        assert tuple(outcome.results) == protocol.metrics.primary
        for name in protocol.metrics.primary:
            expected = PRIMARY_METRICS[name](reference, estimate, getattr(protocol.metrics, name))
            assert outcome.results[name] == expected

    def test_a_short_output_within_tolerance_is_scored_on_the_common_length(
        self, protocol: Protocol
    ) -> None:
        """UVR returned 130 samples fewer on each of the 23 degraded frames measured before v7."""
        reference = reference_frame(music(), 1.0)
        estimate = reference[:-130]

        outcome = score_output(reference, estimate, protocol)

        assert isinstance(outcome, FrameScore)
        assert (outcome.reference_samples_discarded, outcome.estimate_samples_discarded) == (130, 0)
        assert outcome.results["reconstruction_snr"].capped

    @pytest.mark.parametrize(
        "estimate",
        [np.zeros(32000 - 161), np.full(32000, np.nan), np.zeros((32000, 2))],
        ids=["too_short", "non_finite", "stereo"],
    )
    def test_an_output_conditioning_rejects_is_a_system_failure(
        self, protocol: Protocol, estimate: npt.NDArray[np.float64]
    ) -> None:
        outcome = score_output(reference_frame(music(), 1.0), estimate, protocol)
        assert isinstance(outcome, SystemFailure)
        assert outcome.reason

    def test_a_malformed_reference_is_the_harness_fault(self, protocol: Protocol) -> None:
        reference = reference_frame(music(), 1.0).copy()
        reference[5] = np.inf
        with pytest.raises(ConditioningError):
            score_output(reference, music(), protocol)

    def test_a_reference_that_truncation_makes_unscoreable_is_the_harness_fault(
        self, protocol: Protocol
    ) -> None:
        """Constant except its last 130 samples: scoreable in full, constant once a 130-sample
        shorter output truncates it, which SI-SNR refuses."""
        frame = np.full(32000, 0.1)
        frame[-130:] = np.linspace(0.0, 0.2, 130)
        reference = reference_frame(frame, 1.0)
        with pytest.raises(FrameScoringError, match="si_snr"):
            score_output(reference, reference[:-130], protocol)


def delayed(signal: npt.NDArray[np.float64], samples: int) -> npt.NDArray[np.float64]:
    """*signal* delayed by *samples*, the same length, the start filled with zeros."""
    return np.concatenate([np.zeros(samples), signal[: len(signal) - samples]])


class TestShiftTolerantDiagnostic:
    """v13's diagnostic sets a one-sample delay aside and leaves the primaries unchanged."""

    def test_a_one_sample_delay_costs_the_primary_but_not_the_diagnostic(
        self, protocol: Protocol
    ) -> None:
        reference = reference_frame(music(), 1.0)

        outcome = score_output(reference, delayed(reference, 1), protocol)

        assert isinstance(outcome, FrameScore)
        assert tuple(outcome.shift_tolerant) == protocol.metrics.shift_tolerant.metrics
        assert outcome.results["reconstruction_snr"].value < 10.0
        assert outcome.shift_tolerant["reconstruction_snr"].capped
        assert outcome.shift_tolerant["si_snr"].value > outcome.results["si_snr"].value + 30.0

    def test_the_spectral_metric_has_no_diagnostic(self, protocol: Protocol) -> None:
        reference = reference_frame(music(), 1.0)
        outcome = score_output(reference, reference * 0.9, protocol)
        assert isinstance(outcome, FrameScore)
        assert "log_spectral_distance" not in outcome.shift_tolerant

    def test_a_rejected_output_has_no_diagnostic_to_report(self, protocol: Protocol) -> None:
        reference = reference_frame(music(), 1.0)
        assert isinstance(score_output(reference, reference[:-500], protocol), SystemFailure)
