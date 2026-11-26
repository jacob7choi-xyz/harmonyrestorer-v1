"""Tests for the model v2 Phase 0 measurement of v1 on real 78 noise."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from benchmark.degradation import apply_degradation, draw_params
from benchmark.measurements import real_noise_gap as gap
from benchmark.protocol import Protocol, load_protocol


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen benchmark protocol."""
    return load_protocol()


class TestScaleToSnr:
    """Scaling sets the clean-to-noise power ratio exactly."""

    @pytest.mark.parametrize("snr_db", [5.0, 20.0])
    def test_the_ratio_is_the_requested_snr(self, snr_db: float) -> None:
        rng = np.random.default_rng(0)
        clean, noise = rng.normal(0, 0.3, 32000), rng.normal(0, 0.01, 32000)
        scaled = gap.scale_to_snr(clean, noise, snr_db)
        measured = 10 * np.log10(np.mean(clean**2) / np.mean(scaled**2))
        assert measured == pytest.approx(snr_db, abs=1e-9)

    def test_silent_noise_is_refused(self) -> None:
        with pytest.raises(gap.GapError, match="silent"):
            gap.scale_to_snr(np.ones(10), np.zeros(10), 10.0)


class TestNoiseExcerpt:
    """Excerpts have the frame's length, repeat with the seed, and join pieces with crossfades."""

    def test_length_and_reproducibility(self) -> None:
        pieces = [np.full(3000, 1.0), np.full(5000, -1.0)]
        first = gap.noise_excerpt(pieces, 32000, 800, np.random.default_rng(3))
        again = gap.noise_excerpt(pieces, 32000, 800, np.random.default_rng(3))
        assert first.shape == (32000,)
        np.testing.assert_array_equal(first, again)

    def test_joins_ramp_between_pieces_instead_of_jumping(self) -> None:
        pieces = [np.full(3000, 1.0), np.full(5000, -1.0)]
        run = gap.noise_excerpt(pieces, 32000, 800, np.random.default_rng(3))
        # A linear crossfade over 800 samples moves at most 2/800 per sample between +1 and -1.
        assert np.max(np.abs(np.diff(run))) <= 2.0 / 800 + 1e-12

    def test_a_piece_no_longer_than_the_crossfade_is_refused(self) -> None:
        with pytest.raises(gap.GapError, match="longer than the crossfade"):
            gap.noise_excerpt([np.ones(800)], 32000, 800, np.random.default_rng(0))


class TestSyntheticNoise:
    """The synthetic condition is v1's hiss, crackle and hum, without low-pass or saturation."""

    def test_it_equals_v1s_degradation_with_both_switched_off(self, protocol: Protocol) -> None:
        clean = np.random.default_rng(1).normal(0, 0.1, 32000)
        noise = gap.synthetic_noise(clean, protocol, np.random.default_rng(9))
        rng = np.random.default_rng(9)
        rate = protocol.canonicalisation.resample.target_sample_rate
        params = replace(
            draw_params(rng, protocol.degradation.ranges),
            rolloff_hz=float(rate) / 2,
            enable_saturation=False,
        )
        expected = apply_degradation(clean.astype(np.float32), rate, params, rng)
        np.testing.assert_array_equal(noise, expected.astype(np.float64) - clean.astype(np.float32))
        assert np.mean(noise**2) > 0


class TestGate:
    """The gate needs the margin and an interval above zero."""

    def test_a_large_consistent_difference_passes(self, protocol: Protocol) -> None:
        result = gap.gate([4.0, 4.5, 5.0, 3.8, 4.2, 4.9], protocol)
        assert result.passed
        assert result.interval_db[0] > 0

    def test_a_difference_below_the_margin_fails(self, protocol: Protocol) -> None:
        assert not gap.gate([2.0, 2.5, 2.2, 2.4, 2.1, 2.3], protocol).passed

    def test_a_large_mean_with_an_interval_reaching_zero_fails(self, protocol: Protocol) -> None:
        result = gap.gate([30.0, -1.0, -1.0, -1.0, -1.0, -1.0], protocol)
        assert result.mean_difference_db >= gap.GATE_MARGIN_DB
        assert not result.passed


class TestRecordingInterval:
    """The interval brackets the mean of the per-recording values and is the gate's interval."""

    def test_identical_values_give_a_point_interval(self, protocol: Protocol) -> None:
        assert gap.recording_interval([1.5] * 6, protocol) == (1.5, 1.5)

    def test_it_is_the_interval_the_gate_reports(self, protocol: Protocol) -> None:
        values = [1.0, 2.5, 0.5, 3.0, 1.2, 2.2]
        low, high = gap.recording_interval(values, protocol)
        assert low < float(np.mean(values)) < high
        assert gap.gate(values, protocol).interval_db == (low, high)


class _Identity:
    """A system whose output is its input, so every improvement is zero."""

    name = "opgan"

    def denoise(self, input_path: Path) -> Path:
        return input_path


class TestScoreMixture:
    """A system that changes nothing improves SI-SNR by zero."""

    def test_identity_output_has_zero_improvement(self, protocol: Protocol, tmp_path: Path) -> None:
        run = gap.MeasurementRun(
            protocol=protocol,
            system=_Identity(),
            banks=gap.NoiseBanks(great78=[], freesound=[]),
            scratch=tmp_path,
        )
        rng = np.random.default_rng(2)
        reference = np.sin(np.arange(32000) / 7.0) * 0.3
        mixture = reference + gap.scale_to_snr(reference, rng.normal(size=32000), 10.0)
        identity = gap.FrameIdentity("a" * 64, 0)
        values = gap.score_mixture(run, identity, reference, mixture)
        assert values["si_snr_improvement_db"] == pytest.approx(0.0, abs=1e-6)


class TestMain:
    """An existing output directory is refused before anything runs."""

    def test_existing_out_dir_is_refused(self, tmp_path: Path) -> None:
        args = [
            "--walk-dir",
            "w",
            "--aalto-dir",
            "a",
            "--freesound-dir",
            "f",
            "--uvr-model-dir",
            "u",
        ]
        assert gap.main([*args, "--out-dir", str(tmp_path)]) == 1
