"""Tests for the duplicate screen: its features, similarity and verdict, on synthetic data."""

from __future__ import annotations

import dataclasses
import math

import librosa
import numpy as np
import numpy.typing as npt
import pytest

from benchmark.duplicate_screen import (
    Degenerate,
    Group,
    RecordingIdentity,
    Reference,
    ScreenError,
    coefficients,
    offsets,
    screen,
    similarity,
    unit_features,
)
from benchmark.protocol import DuplicateDetection, MfccConfig, load_protocol

RATE = 16000


def unit_frames(frames: int, seed: int) -> npt.NDArray[np.float64]:
    """Unit-length random frames, standing in for a recording's normalised MFCCs."""
    values = np.random.default_rng(seed).standard_normal((frames, 20))
    unit: npt.NDArray[np.float64] = values / np.linalg.norm(values, axis=1, keepdims=True)
    return unit


class TestFeatures:
    """The features, offsets and similarity the protocol freezes."""

    @pytest.fixture(scope="class")
    def config(self) -> MfccConfig:
        """The frozen MFCC configuration."""
        return load_protocol().duplicate_detection.mfcc

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
        a = unit_frames(500, 0)
        assert similarity(a, a, [0], 100) == pytest.approx(1.0)

    def test_a_shift_inside_the_search_is_found(self, config: MfccConfig) -> None:
        a = unit_frames(800, 1)
        assert similarity(a, a[40:], offsets(config), 100) == pytest.approx(1.0)
        assert similarity(a[40:], a, offsets(config), 100) == pytest.approx(1.0)

    def test_a_shift_outside_the_search_is_not_found(self, config: MfccConfig) -> None:
        a = unit_frames(800, 2)
        score = similarity(a, a[200:], offsets(config), 100)
        assert score is not None
        assert score < 0.2

    def test_too_little_overlap_gives_no_score(self) -> None:
        a = unit_frames(99, 3)
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


def reference(name: str, features: npt.NDArray[np.float64]) -> Reference:
    """A reference named by a file name and two digests derived from it."""
    return Reference(RecordingIdentity(name, name[0] * 64, name[-1] * 64), features)


CLEAN = Degenerate(0, 0)


class TestScreen:
    """One candidate against the training references, then the earlier admissions."""

    @pytest.fixture(scope="class")
    def config(self) -> DuplicateDetection:
        """The frozen duplicate-detection configuration."""
        return load_protocol().duplicate_detection

    def test_a_copy_is_flagged_and_an_unrelated_recording_is_not(
        self, config: DuplicateDetection
    ) -> None:
        candidate = unit_frames(400, 0)
        copy, other = (
            reference("copy.flac", candidate.copy()),
            reference("b.flac", unit_frames(400, 1)),
        )

        result = screen(candidate, CLEAN, [other, copy], [], config)

        assert [(f.group, f.identity) for f in result.flagged] == [(Group.TRAINING, copy.identity)]
        assert result.flagged[0].score == pytest.approx(1.0)
        assert result.max_training == pytest.approx(1.0)
        assert not result.uncertain

    def test_a_score_equal_to_the_threshold_is_flagged(self, config: DuplicateDetection) -> None:
        """The protocol flags at or above the threshold."""
        candidate, ref = unit_frames(400, 2), reference("r.flac", unit_frames(400, 3))
        score = screen(candidate, CLEAN, [ref], [], config).max_training
        assert score is not None

        at = screen(candidate, CLEAN, [ref], [], dataclasses.replace(config, flag_threshold=score))
        above = dataclasses.replace(config, flag_threshold=math.nextafter(score, 2.0))

        assert [f.identity for f in at.flagged] == [ref.identity]
        assert screen(candidate, CLEAN, [ref], [], above).flagged == ()

    def test_training_flags_come_before_admitted_ones(self, config: DuplicateDetection) -> None:
        candidate = unit_frames(400, 4)
        training = reference("t.flac", candidate.copy())
        admitted = reference("item/a.flac", candidate.copy())

        result = screen(candidate, CLEAN, [training], [admitted], config)

        assert [(f.group, f.identity) for f in result.flagged] == [
            (Group.TRAINING, training.identity),
            (Group.ADMITTED, admitted.identity),
        ]
        assert result.max_admitted == pytest.approx(1.0)

    def test_with_nothing_admitted_the_admitted_maximum_is_none_not_zero(
        self, config: DuplicateDetection
    ) -> None:
        result = screen(
            unit_frames(400, 5), CLEAN, [reference("t.flac", unit_frames(400, 6))], [], config
        )
        assert result.max_admitted is None
        assert result.max_training is not None

    @pytest.mark.parametrize("degenerate", [Degenerate(1, 0), Degenerate(0, 1)])
    def test_degenerate_candidate_features_are_uncertain_but_still_compared(
        self, config: DuplicateDetection, degenerate: Degenerate
    ) -> None:
        """The audit trail records both maxima even for an uncertain candidate."""
        candidate = unit_frames(400, 7)
        result = screen(candidate, degenerate, [reference("t.flac", candidate.copy())], [], config)
        assert result.uncertain
        assert result.max_training == pytest.approx(1.0)

    def test_a_comparison_with_too_little_overlap_is_uncertain(
        self, config: DuplicateDetection
    ) -> None:
        short = reference("short.flac", unit_frames(config.mfcc.min_overlap_frames - 1, 8))
        long = reference("long.flac", unit_frames(400, 9))

        result = screen(unit_frames(400, 10), CLEAN, [long], [short], config)

        assert result.unscored == (short.identity,)
        assert result.uncertain
        assert result.max_admitted is None

    def test_an_empty_training_set_is_refused_rather_than_admitting_everything(
        self, config: DuplicateDetection
    ) -> None:
        with pytest.raises(ScreenError, match="training references"):
            screen(unit_frames(400, 11), CLEAN, [], [], config)

    def test_a_recording_named_twice_is_refused(self, config: DuplicateDetection) -> None:
        twice = reference("t.flac", unit_frames(400, 12))
        with pytest.raises(ScreenError, match="distinct"):
            screen(unit_frames(400, 13), CLEAN, [twice], [twice], config)

    @pytest.mark.parametrize(
        "features",
        [
            np.zeros((400, 19)),
            np.zeros(400),
            np.zeros((400, 20), dtype=np.float32),
        ],
        ids=["wrong_coefficients", "one_dimensional", "float32"],
    )
    @pytest.mark.parametrize("side", ["candidate", "reference"])
    def test_features_of_another_shape_or_type_are_refused(
        self, config: DuplicateDetection, features: npt.NDArray[np.float64], side: str
    ) -> None:
        good = unit_frames(400, 14)
        candidate = features if side == "candidate" else good
        ref = reference("t.flac", good if side == "candidate" else features)
        with pytest.raises(ScreenError, match="float64"):
            screen(candidate, CLEAN, [ref], [], config)
