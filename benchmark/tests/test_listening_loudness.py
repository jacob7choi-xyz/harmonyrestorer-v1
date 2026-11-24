"""Tests for the BS.1770 loudness meter and the frozen clip-matching rule."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest

from benchmark.listening.loudness import (
    LoudnessError,
    gain_to,
    integrated_loudness,
    k_weighting,
    match_clip,
)
from benchmark.listening.protocol import Loudness, load_listening_protocol

_FREQUENCY_HZ = 997.0
_RATE_48K = 48_000
_RATE_16K = 16_000
# BS.1770-4 Annex 1 table, 48 kHz: high-shelf numerator and denominator, high-pass denominator.
_PUBLISHED_SHELF_B = (1.53512485958697, -2.69169618940638, 1.19839281085285)
_PUBLISHED_SHELF_A = (1.0, -1.69065929318241, 0.73248077421585)
_PUBLISHED_HIGHPASS_A = (1.0, -1.99004745483398, 0.99007225036621)
_COEFFICIENT_TOLERANCE = 1e-6


def _sine(seconds: float, rate: int, amplitude: float = 1.0) -> npt.NDArray[np.float64]:
    """A sine at the study's test frequency, *seconds* long, at *rate*."""
    t = np.arange(round(seconds * rate)) / rate
    return amplitude * np.sin(2.0 * np.pi * _FREQUENCY_HZ * t)


def test_a_full_scale_sine_at_48_khz_reads_minus_3_01_lkfs() -> None:
    """A 997 Hz sine at amplitude 1.0 reads -3.01 LKFS within 0.05 dB at 48 kHz."""
    loudness = integrated_loudness(_sine(8.0, _RATE_48K), _RATE_48K)

    assert loudness == pytest.approx(-3.01, abs=0.05)


def test_a_full_scale_sine_at_16_khz_reads_within_0_1_db_of_minus_3_01() -> None:
    """The same sine at the presented 16 kHz reads within 0.1 dB of the 48 kHz reading."""
    loudness = integrated_loudness(_sine(8.0, _RATE_16K), _RATE_16K)

    assert loudness == pytest.approx(-3.01, abs=0.1)


def test_a_silent_half_reads_close_to_the_non_silent_half() -> None:
    """Silence is gated, so a signal with a silent half reads close to its non-silent half.

    The measured difference is 0.17 dB at 16 kHz. The blocks that straddle the boundary are
    partly silent and BS.1770 does not gate them. That explanation is an inference from the
    block arithmetic and was not measured separately.
    """
    sine = _sine(8.0, _RATE_16K)
    half_silent = sine.copy()
    half_silent[sine.size // 2 :] = 0.0
    non_silent_half = sine[: sine.size // 2]

    difference = integrated_loudness(half_silent, _RATE_16K) - integrated_loudness(
        non_silent_half, _RATE_16K
    )

    assert abs(difference) <= 0.25


def test_the_relative_gate_removes_quiet_blocks() -> None:
    """Seven seconds 20 dB below a one-second loud sine read close to the loud second alone.

    The measured reading is -3.67 LKFS against -2.97 LKFS for the loud second alone. A rough
    estimate from the block powers, not a measurement, puts the ungated reading near -12 LKFS.
    """
    loud = _sine(1.0, _RATE_16K)
    quiet = _sine(7.0, _RATE_16K, amplitude=0.1)

    mixed = integrated_loudness(np.concatenate([loud, quiet]), _RATE_16K)

    assert abs(mixed - integrated_loudness(loud, _RATE_16K)) <= 1.0


def test_a_signal_with_no_block_above_the_absolute_gate_is_refused() -> None:
    """A sine near -83 LKFS has no block above -70 LKFS, so its loudness is refused."""
    quiet = _sine(8.0, _RATE_16K, amplitude=1e-4)

    with pytest.raises(LoudnessError, match="absolute gate"):
        integrated_loudness(quiet, _RATE_16K)


def test_a_digital_silence_is_refused() -> None:
    """All-zero samples have no block above the absolute gate, so they are refused."""
    with pytest.raises(LoudnessError, match="absolute gate"):
        integrated_loudness(np.zeros(_RATE_16K), _RATE_16K)


def test_non_finite_samples_are_refused() -> None:
    """A NaN or an infinity anywhere in the signal is refused, with no loudness returned."""
    for bad_value in (math.nan, math.inf):
        samples = _sine(1.0, _RATE_16K)
        samples[100] = bad_value

        with pytest.raises(LoudnessError, match="not finite"):
            integrated_loudness(samples, _RATE_16K)


def test_a_signal_shorter_than_one_block_is_refused_and_one_block_is_not() -> None:
    """One sample under a 400 ms block is refused, and exactly one block is measured."""
    block = round(0.4 * _RATE_16K)

    with pytest.raises(LoudnessError, match="fewer than one"):
        integrated_loudness(_sine(1.0, _RATE_16K)[: block - 1], _RATE_16K)
    assert math.isfinite(integrated_loudness(_sine(1.0, _RATE_16K)[:block], _RATE_16K))


def test_the_48_khz_coefficients_equal_the_published_bs1770_table() -> None:
    """The derived 48 kHz stages match the published BS.1770-4 coefficients to 1e-6."""
    shelf, highpass = k_weighting(_RATE_48K)

    assert np.allclose(shelf.b, _PUBLISHED_SHELF_B, rtol=0.0, atol=_COEFFICIENT_TOLERANCE)
    assert np.allclose(shelf.a, _PUBLISHED_SHELF_A, rtol=0.0, atol=_COEFFICIENT_TOLERANCE)
    assert np.allclose(highpass.a, _PUBLISHED_HIGHPASS_A, rtol=0.0, atol=_COEFFICIENT_TOLERANCE)
    assert highpass.b == (1.0, -2.0, 1.0)


def test_gain_to_moves_a_loudness_to_the_target() -> None:
    """Applying gain_to to a measured loudness yields that target, and 0 dB is a gain of one."""
    assert gain_to(-23.0, -23.0) == 1.0
    assert gain_to(-20.0, -23.0) == pytest.approx(10.0 ** (3.0 / 20.0))


def _loudness_rule() -> Loudness:
    """The loudness section of the frozen protocol, which the matching rule reads."""
    return load_listening_protocol().loudness


def test_a_clip_within_the_ceiling_keeps_the_protocol_target() -> None:
    """When no peak exceeds the ceiling, the protocol's target is kept and nothing lowers."""
    rule = _loudness_rule()
    windows = {
        "original": _sine(8.0, _RATE_16K, amplitude=0.3),
        "opgan": _sine(8.0, _RATE_16K, amplitude=0.3),
        "uvr": _sine(8.0, _RATE_16K, amplitude=0.3),
    }

    match = match_clip(windows, _RATE_16K, rule)

    assert match.lowered_steps == 0
    assert match.target_lkfs == rule.target_lkfs


def test_the_peak_rule_lowers_all_three_conditions_together_and_never_clips() -> None:
    """A spike in one condition lowers the target for all three, and no peak ends above the ceiling.

    The first two conditions are quiet sines that would pass alone. Their gains are recomputed
    at the final target, so they move with the third.
    """
    rule = _loudness_rule()
    sine = _sine(8.0, _RATE_16K, amplitude=0.05)
    spiky = sine.copy()
    spiky[100] = 0.8
    windows = {"original": _sine(8.0, _RATE_16K, 0.3), "opgan": _sine(8.0, _RATE_16K, 0.3)}
    windows["uvr"] = spiky

    match = match_clip(windows, _RATE_16K, rule)

    assert match.lowered_steps > 0
    assert match.target_lkfs == pytest.approx(
        rule.target_lkfs - match.lowered_steps * rule.target_step_lu
    )
    for name in windows:
        assert match.gains[name] == pytest.approx(gain_to(match.target_lkfs, match.loudness[name]))
        assert match.peaks_after_gain_dbfs[name] <= rule.sample_peak_ceiling_dbfs
    doubled_step = match_clip(windows, _RATE_16K, _rule_with_step(rule, 0.2))
    assert doubled_step.target_lkfs == pytest.approx(
        rule.target_lkfs - doubled_step.lowered_steps * 0.2
    )
    assert doubled_step.lowered_steps < match.lowered_steps


def _rule_with_step(rule: Loudness, step: float) -> Loudness:
    """A copy of the protocol's loudness rule with a different lowering step."""
    return Loudness(
        target_lkfs=rule.target_lkfs,
        sample_peak_ceiling_dbfs=rule.sample_peak_ceiling_dbfs,
        target_step_lu=step,
        presentation_bits=rule.presentation_bits,
    )


def test_a_step_too_small_to_reach_the_ceiling_is_refused_after_the_cap() -> None:
    """A positive step too small to reach the ceiling stops with an error, not a loop."""
    rule = _rule_with_step(_loudness_rule(), 1e-9)
    sine = _sine(8.0, _RATE_16K, amplitude=0.05)
    spiky = sine.copy()
    spiky[100] = 0.8

    with pytest.raises(LoudnessError, match="lowering steps"):
        match_clip({"original": sine, "uvr": spiky}, _RATE_16K, rule)


def test_the_matching_rule_refuses_an_empty_or_non_positive_step_rule() -> None:
    """No conditions, or a step that is not positive, is refused before any gain is computed."""
    rule = _loudness_rule()
    window = _sine(8.0, _RATE_16K, 0.3)

    with pytest.raises(LoudnessError, match="at least one condition"):
        match_clip({}, _RATE_16K, rule)
    with pytest.raises(LoudnessError, match="must be positive"):
        match_clip({"original": window}, _RATE_16K, _rule_with_step(rule, 0.0))
