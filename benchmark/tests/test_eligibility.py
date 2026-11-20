"""Tests for the duration and loudness floors protocol v5 sets for recordings and frames."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import warnings

import numpy as np
import numpy.typing as npt
import pytest

from benchmark.canonicalisation import CanonicalWaveform
from benchmark.eligibility import (
    EligibilityArithmeticError,
    frame_floors,
    recording_floors,
    rms_dbfs,
)
from benchmark.framing import frame
from benchmark.protocol import load_protocol

PROTOCOL = load_protocol()
ELIGIBILITY = PROTOCOL.eligibility
RATE = PROTOCOL.canonicalisation.resample.target_sample_rate
FRAME = PROTOCOL.framing.frame_samples
MIN_SAMPLES = ELIGIBILITY.recording_min_duration_s * RATE


def canonical(samples: npt.ArrayLike) -> CanonicalWaveform:
    """A canonical waveform built directly: owned, read-only, with its digest."""
    owned = np.array(samples, dtype=np.float64)
    owned.flags.writeable = False
    digest = hashlib.sha256(owned.astype("<f8").tobytes(order="C")).hexdigest()
    return CanonicalWaveform(owned, digest)


def spec_dbfs(samples: npt.NDArray[np.float64]) -> float:
    """v5's rms_definition written out from its text, for a non-zero energy."""
    return float(20.0 * np.log10(np.sqrt(np.mean(samples * samples))))


class TestRmsDbfs:
    """The RMS in dBFS, exactly as v5 defines it."""

    def test_a_constant_tenth_is_minus_twenty_dbfs(self) -> None:
        assert rms_dbfs(np.full(FRAME, 0.1)) == pytest.approx(-20.0, abs=1e-12)

    def test_the_value_is_the_v5_expression_bit_for_bit(self) -> None:
        """A different reduction or 10 * log10(energy) can differ in the last bits."""
        samples = np.random.default_rng(5).uniform(-0.5, 0.5, FRAME)

        assert rms_dbfs(samples) == spec_dbfs(samples)

    def test_digital_silence_is_negative_infinity_without_a_warning(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            assert rms_dbfs(np.zeros(FRAME)) == -math.inf

    @pytest.mark.parametrize("value", [1e200, math.nan], ids=["overflow", "nan"])
    def test_a_non_finite_energy_raises(self, value: float) -> None:
        """v5's fail-closed harness guard, on arrays built directly. Canonical validation
        refuses NaN, and v5 states the energy sum's overflow is unreachable by a margin of
        1.1e154, so neither input is expected from canonicalisation. Without the guard an
        overflow would read as infinite dBFS, which is above every floor."""
        with pytest.raises(EligibilityArithmeticError, match="energy over 4 samples"):
            rms_dbfs(np.full(4, value))


class TestRecordingFloors:
    """Duration in integer samples and loudness over the complete waveform."""

    @pytest.mark.parametrize(
        ("samples", "expected"), [(MIN_SAMPLES, True), (MIN_SAMPLES - 1, False)]
    )
    def test_the_duration_floor_is_inclusive_in_samples(self, samples: int, expected: bool) -> None:
        floors = recording_floors(canonical(np.full(samples, 0.1)), ELIGIBILITY, sample_rate=RATE)

        assert floors.canonical_samples == samples
        assert floors.meets_min_duration is expected

    def test_the_loudness_floor_is_strictly_above(self) -> None:
        """A level equal to the floor is not above it; the next float below it is."""
        waveform = canonical(np.random.default_rng(6).uniform(-0.01, 0.01, MIN_SAMPLES))
        level = rms_dbfs(waveform.samples)
        at = dataclasses.replace(ELIGIBILITY, recording_rms_dbfs_min=level)
        below = dataclasses.replace(ELIGIBILITY, recording_rms_dbfs_min=math.nextafter(level, -1e9))

        assert recording_floors(waveform, at, sample_rate=RATE).above_rms_floor is False
        assert recording_floors(waveform, below, sample_rate=RATE).above_rms_floor is True

    def test_the_rms_includes_the_samples_framing_drops(self) -> None:
        """Whole frames silent and only the trailing remainder sounding: measured over the
        frames this would be negative infinity."""
        samples = np.concatenate([np.zeros(MIN_SAMPLES), np.full(FRAME // 2, 0.5)])
        waveform = canonical(samples)

        floors = recording_floors(waveform, ELIGIBILITY, sample_rate=RATE)

        assert frame(waveform, PROTOCOL.framing).dropped_samples == FRAME // 2
        assert floors.rms_dbfs == spec_dbfs(waveform.samples)
        assert math.isfinite(floors.rms_dbfs)

    def test_both_floors_are_evaluated_when_both_fail(self) -> None:
        """No precedence is frozen, so neither verdict hides the other."""
        floors = recording_floors(canonical(np.zeros(FRAME)), ELIGIBILITY, sample_rate=RATE)

        assert floors.meets_min_duration is False
        assert floors.above_rms_floor is False
        assert floors.rms_dbfs == -math.inf


class TestFrameFloors:
    """Every candidate frame is measured, quiet ones included."""

    def test_every_frame_is_recorded_with_its_verdict(self) -> None:
        samples = np.concatenate([np.full(FRAME, 0.1), np.zeros(FRAME), np.full(FRAME, 0.1)])

        floors = frame_floors(frame(canonical(samples), PROTOCOL.framing), ELIGIBILITY)

        loud = rms_dbfs(np.full(FRAME, 0.1))
        assert floors.rms_dbfs.tolist() == [loud, -math.inf, loud]
        assert floors.above_rms_floor.tolist() == [True, False, True]

    def test_each_frame_is_its_own_mean(self) -> None:
        """One numpy.mean per frame, the rule's literal form."""
        rng = np.random.default_rng(7)
        samples = np.concatenate([rng.uniform(-a, a, FRAME) for a in (0.9, 0.05, 0.003)])
        framed = frame(canonical(samples), PROTOCOL.framing)

        floors = frame_floors(framed, ELIGIBILITY)

        assert floors.rms_dbfs.tolist() == [spec_dbfs(row) for row in framed.frames]

    def test_the_loudness_floor_is_strictly_above(self) -> None:
        framed = frame(
            canonical(np.random.default_rng(8).uniform(-0.01, 0.01, FRAME)), PROTOCOL.framing
        )
        level = rms_dbfs(framed.frames[0])
        at = dataclasses.replace(ELIGIBILITY, frame_rms_dbfs_min=level)
        below = dataclasses.replace(ELIGIBILITY, frame_rms_dbfs_min=math.nextafter(level, -1e9))

        assert frame_floors(framed, at).above_rms_floor.tolist() == [False]
        assert frame_floors(framed, below).above_rms_floor.tolist() == [True]

    def test_the_results_are_read_only(self) -> None:
        floors = frame_floors(frame(canonical(np.full(FRAME, 0.1)), PROTOCOL.framing), ELIGIBILITY)

        for array in (floors.rms_dbfs, floors.above_rms_floor):
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 0
