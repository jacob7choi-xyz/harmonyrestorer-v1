"""Tests for the duplicate screen's features and similarity, on synthetic waveforms."""

from __future__ import annotations

import dataclasses

import librosa
import numpy as np
import numpy.typing as npt
import pytest

from benchmark.duplicate_screen import Degenerate, coefficients, offsets, similarity, unit_features
from benchmark.protocol import MfccConfig, load_protocol

RATE = 16000


class TestFeatures:
    """The features, offsets and similarity the protocol freezes."""

    @pytest.fixture(scope="class")
    def config(self) -> MfccConfig:
        """The frozen MFCC configuration."""
        return load_protocol().duplicate_detection.mfcc

    @staticmethod
    def random_unit(frames: int, seed: int) -> npt.NDArray[np.float64]:
        """Unit-length random frames, standing in for normalised MFCCs."""
        values = np.random.default_rng(seed).standard_normal((frames, 20))
        unit: npt.NDArray[np.float64] = values / np.linalg.norm(values, axis=1, keepdims=True)
        return unit

    def test_offsets_are_multiples_of_the_stride_within_the_search(
        self, config: MfccConfig
    ) -> None:
        lags = offsets(config)
        assert (lags[0], lags[-1], len(lags)) == (-156, 156, 157)
        assert 0 in lags
        assert all(lag % config.alignment_stride_frames == 0 for lag in lags)

    def test_an_odd_reach_stops_at_the_last_multiple_of_the_stride(
        self, config: MfccConfig
    ) -> None:
        """At hop 512 and 16 kHz, 4.97 s reaches 155.3 frames: rounded down to 155, and 155 is
        not a multiple of 2, so the last offset is 154."""
        lags = offsets(dataclasses.replace(config, alignment_search_s=4.97))
        assert (lags[0], lags[-1]) == (-154, 154)

    def test_identical_features_score_one(self) -> None:
        a = self.random_unit(500, 0)
        assert similarity(a, a, [0], 100) == pytest.approx(1.0)

    def test_a_shift_inside_the_search_is_found(self, config: MfccConfig) -> None:
        a = self.random_unit(800, 1)
        assert similarity(a, a[40:], offsets(config), 100) == pytest.approx(1.0)
        assert similarity(a[40:], a, offsets(config), 100) == pytest.approx(1.0)

    def test_a_shift_outside_the_search_is_not_found(self, config: MfccConfig) -> None:
        a = self.random_unit(800, 2)
        score = similarity(a, a[200:], offsets(config), 100)
        assert score is not None
        assert score < 0.2

    def test_too_little_overlap_gives_no_score(self) -> None:
        a = self.random_unit(99, 3)
        assert similarity(a, a, [0], 100) is None

    def test_features_are_unit_frames_of_z_normalised_coefficients(
        self, config: MfccConfig
    ) -> None:
        t = np.arange(RATE * 2) / RATE
        waveform = 0.3 * np.sin(2 * np.pi * 440.0 * t * (1.0 + t))
        unit, degenerate = unit_features(waveform, config)
        assert unit.shape[1] == config.n_mfcc
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0)
        assert degenerate == Degenerate(0, 0)

    def test_the_protocol_s_arguments_give_the_default_call_s_bits(
        self, config: MfccConfig
    ) -> None:
        """The screen was calibrated with librosa's defaults; v9 writes each one down."""
        for seed in range(3):
            waveform = np.random.default_rng(seed).uniform(-0.5, 0.5, RATE * 3 + seed)
            default = librosa.feature.mfcc(y=waveform, sr=RATE, n_mfcc=20, hop_length=512)
            explicit = coefficients(waveform, config)
            assert explicit.dtype == default.dtype
            assert np.array_equal(explicit, default), seed

    def test_features_match_a_direct_computation(self, config: MfccConfig) -> None:
        """Centred and scaled per coefficient, then each frame scaled to unit length."""
        waveform = np.random.default_rng(5).uniform(-0.5, 0.5, RATE * 2)
        mfcc = librosa.feature.mfcc(y=waveform, sr=RATE, n_mfcc=20, hop_length=512)
        z = ((mfcc - mfcc.mean(axis=1, keepdims=True)) / mfcc.std(axis=1, keepdims=True)).T
        expected = z / np.linalg.norm(z, axis=1, keepdims=True)
        np.testing.assert_allclose(unit_features(waveform, config)[0], expected, atol=1e-12)

    def test_silence_is_counted_as_degenerate_rather_than_divided_by_zero(
        self, config: MfccConfig
    ) -> None:
        unit, degenerate = unit_features(np.zeros(RATE), config)
        assert not np.isnan(unit).any()
        assert degenerate.zero_variance_coefficients == config.n_mfcc
        assert degenerate.zero_norm_frames == len(unit)
