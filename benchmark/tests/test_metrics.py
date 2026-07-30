"""Tests for the three co-primary metrics.

Nothing here reimplements a metric. Where a closed form exists it is derived
independently, and the LSD transform is checked against configured librosa and scipy
rather than against another copy of the same loop. The question each test answers is
whether production code could be changed so a published number changes while the suite
stays green.

The oracles are deliberately decomposed. The window is checked against its equation, the
transform against an independent library, and the log and reduction stages against a
closed form computed from the magnitudes. No single oracle is asked to prove everything,
because one that covers the whole pipeline tends to be insensitive to the middle of it.
"""

from __future__ import annotations

import librosa
import numpy as np
import pytest
from scipy.signal import get_window

from benchmark import metrics
from benchmark.protocol import load_protocol

METRICS = load_protocol().metrics
SNR = METRICS.reconstruction_snr
SISNR = METRICS.si_snr
LSD = METRICS.log_spectral_distance

UNIT_SAMPLES = 32000


def dense(scale: float = 0.2, size: int = UNIT_SAMPLES, seed: int = 0) -> np.ndarray:
    """A deterministic broadband signal whose bins sit far above the spectral offset."""
    return np.random.default_rng(seed).standard_normal(size) * scale


class TestSharedValidation:
    """An invalid item raises. It never becomes a number."""

    @pytest.mark.parametrize(
        ("reference", "estimate", "expected"),
        [
            (np.zeros((2, 10)), np.zeros((2, 10)), "must be 1-D"),
            (np.ones(10), np.ones(9), "length mismatch"),
            (np.array([]), np.array([]), "is empty"),
            (np.array([1.0, np.nan]), np.ones(2), "non-finite"),
            (np.ones(2), np.array([1.0, np.inf]), "non-finite"),
            (np.zeros(10), np.ones(10), "no energy"),
        ],
    )
    def test_invalid_pairs_raise(
        self, reference: np.ndarray, estimate: np.ndarray, expected: str
    ) -> None:
        with pytest.raises(metrics.MetricInputError, match=expected):
            metrics.reconstruction_snr(reference, estimate, SNR)

    def test_non_numeric_input_raises_the_metric_error(self) -> None:
        """The boundary owns its failure type, so callers see one exception rather than
        whatever numpy happened to throw."""
        with pytest.raises(metrics.MetricInputError, match="must be numeric"):
            metrics.reconstruction_snr("abc", "def", SNR)  # type: ignore[arg-type]

    def test_a_stereo_array_is_never_flattened(self) -> None:
        """Flattening would interleave channels into something that scores as a signal."""
        stereo = np.stack([np.ones(10), np.ones(10)])
        with pytest.raises(metrics.MetricInputError, match="must be 1-D"):
            metrics.si_snr(stereo, stereo, SISNR)

    def test_integer_input_does_not_overflow(self) -> None:
        """int16 subtraction and squaring wraps silently, so conversion precedes
        arithmetic. Full-scale opposite signs are the case that would wrap."""
        reference = np.full(100, 32767, dtype=np.int16)
        estimate = np.full(100, -32768, dtype=np.int16)
        result = metrics.reconstruction_snr(reference, estimate, SNR)
        assert result.value < 1.0  # a maximally wrong estimate, not a wrapped positive


class TestReconstructionSnr:
    """Sample-wise fidelity, with two distinct censoring paths."""

    def test_known_ratio_is_exact(self) -> None:
        """Error energy exactly one hundredth of signal energy is 20 dB by definition."""
        reference = np.ones(1000)
        estimate = reference + 0.1
        result = metrics.reconstruction_snr(reference, estimate, SNR)
        assert result.value == pytest.approx(20.0, abs=1e-12)
        assert not result.capped

    def test_a_second_known_ratio_pins_the_factor_as_ten(self) -> None:
        """A single point cannot distinguish 10*log10 from 20*log10. Error energy one
        ten-thousandth of signal energy is 40 dB under 10*log10 and 80 under 20*log10."""
        reference = np.ones(1000)
        estimate = reference + 0.01
        assert metrics.reconstruction_snr(reference, estimate, SNR).value == pytest.approx(
            40.0, abs=1e-12
        )

    def test_identical_signals_are_censored_by_the_residual_floor(self) -> None:
        """The first censoring path: residual below the floor, with no ratio to compute."""
        signal = dense()
        result = metrics.reconstruction_snr(signal, signal, SNR)
        assert result.value == SNR.cap_db
        assert result.capped

    def test_a_computed_ratio_above_the_cap_is_censored_too(self) -> None:
        """The second path, distinct from the first: the residual clears the floor, so a
        ratio is computed, and that ratio exceeds the cap and is clamped."""
        reference = dense()
        estimate = reference + 1e-5 * dense(seed=1)
        result = metrics.reconstruction_snr(reference, estimate, SNR)
        assert result.capped
        assert result.value == SNR.cap_db

    def test_equal_scaling_of_both_signals_leaves_it_unchanged(self) -> None:
        """A ratio of energies, so a common factor cancels."""
        reference, estimate = dense(), dense() + 0.05 * dense(seed=1)
        base = metrics.reconstruction_snr(reference, estimate, SNR).value
        for factor in (0.01, 100.0):
            scaled = metrics.reconstruction_snr(factor * reference, factor * estimate, SNR)
            assert scaled.value == pytest.approx(base, abs=1e-9)

    def test_it_punishes_a_gain_error_that_si_snr_forgives(self) -> None:
        """The reason both are reported. This is the pair's whole discriminating power."""
        reference = dense()
        estimate = 0.3 * reference
        assert metrics.reconstruction_snr(reference, estimate, SNR).value < 10.0
        assert metrics.si_snr(reference, estimate, SISNR).capped

    def test_a_one_sample_shift_degrades_it(self) -> None:
        """No projection or filtering is permitted, which is why alignment search is
        forbidden: a shift the ear ignores must show up here."""
        reference = dense()
        shifted = np.roll(reference, 1)
        assert metrics.reconstruction_snr(reference, shifted, SNR).value < 3.0


class TestSiSnr:
    """Fidelity with global gain projected out."""

    def test_gain_invariance(self) -> None:
        reference = dense()
        estimate = reference + 0.05 * dense(seed=1)
        base = metrics.si_snr(reference, estimate, SISNR).value
        for factor in (0.3, 2.5, -1.7):
            scaled = metrics.si_snr(reference, factor * estimate, SISNR)
            assert scaled.value == pytest.approx(base, abs=1e-6)

    def test_orthogonal_error_has_a_closed_form(self) -> None:
        """An error orthogonal to the reference makes the projection analytic, so the
        expected value follows in closed form rather than from a second implementation.

        The naive epsilon-free expression is 20 dB here and the implementation returns
        19.999999997858, a 2.1e-09 dB gap that a loose tolerance would hide. Epsilon sits
        in the projection denominator and in both energies, so it scales the recovered
        target slightly below the reference and leaves a small residual along it. Keeping
        those terms makes the expectation exact instead of asymptotic.
        """
        n = 4096
        t = np.arange(n)
        reference = np.sin(2 * np.pi * 5 * t / n)
        error = np.sin(2 * np.pi * 11 * t / n)  # orthogonal over whole numbers of cycles
        alpha = 0.1

        centred_reference = reference - reference.mean()
        centred_error = error - error.mean()
        assert abs(float(np.dot(centred_reference, centred_error))) < 1e-9

        epsilon = SISNR.epsilon
        energy = float(np.dot(centred_reference, centred_reference))
        # Orthogonality kills the cross terms, so the projection is energy / (energy + eps)
        # and the residual splits into a leftover along the reference plus the error.
        scale = energy / (energy + epsilon)
        target_energy = scale**2 * energy
        residual_energy = (epsilon / (energy + epsilon)) ** 2 * energy + alpha**2 * float(
            np.sum(centred_error**2)
        )
        expected = 10.0 * np.log10((target_energy + epsilon) / (residual_energy + epsilon))

        result = metrics.si_snr(reference, reference + alpha * error, SISNR)
        assert result.value == pytest.approx(expected, abs=1e-12)

    def test_zero_mean_is_applied(self) -> None:
        """A DC offset on the estimate is removed by centring, so it must not change the
        score. Without centring it would."""
        reference = dense()
        estimate = reference + 0.05 * dense(seed=1)
        without = metrics.si_snr(reference, estimate, SISNR).value
        with_offset = metrics.si_snr(reference, estimate + 3.0, SISNR).value
        assert with_offset == pytest.approx(without, abs=1e-9)

    def test_a_shift_survives_gain_invariance(self) -> None:
        """Gain invariance must not become invariance to everything."""
        reference = dense()
        assert metrics.si_snr(reference, np.roll(reference, 40), SISNR).value < 3.0

    def test_identical_signals_are_capped(self) -> None:
        signal = dense()
        result = metrics.si_snr(signal, signal, SISNR)
        assert result.capped
        assert result.value == SISNR.cap_db

    @pytest.mark.parametrize("value", [0.5, 0.25, 0.1, 0.3, 0.7, 1.0 / 3.0])
    def test_a_constant_reference_is_invalid_whatever_its_value(self, value: float) -> None:
        """Protocol v3, and the values matter.

        A first draft tested for exactly zero energy after centring, which rejected 0.5
        and 0.25 while scoring 0.1, 0.3, 0.7, and 1/3 at 0.0 dB, because whether a
        constant array centres to exactly zero depends on whether its value is
        representable in binary. The rule is constancy, which is exact for all of them.
        """
        constant = np.full(1000, value)
        with pytest.raises(metrics.MetricInputError, match="reference is constant"):
            metrics.si_snr(constant, np.full(1000, value + 0.2), SISNR)
        with pytest.raises(metrics.MetricInputError, match="reference is constant"):
            metrics.si_snr(constant, dense(size=1000), SISNR)

    def test_a_nearly_constant_reference_is_still_valid(self) -> None:
        """The rule must not widen. One sample differing by 1e-15 leaves a real, if tiny,
        projection direction, so the item is mathematically scoreable. Excluding it would
        need a tolerance, which would be an unstated protocol parameter."""
        nearly = np.full(1000, 0.1)
        nearly[0] += 1e-15
        result = metrics.si_snr(nearly, nearly + 0.01 * dense(size=1000), SISNR)
        assert np.isfinite(result.value)

    def test_an_unrepresentable_centred_energy_fails_closed(self) -> None:
        """Out-of-domain input, not an invalid benchmark item, and the distinction matters.

        This reference is not constant, so a projection direction mathematically exists.
        Only its centred energy underflows: residues near 5e-166 square to about 1e-331.
        That is a float64 representation limit rather than a fact about SI-SNR's domain,
        which is why v3 does not make it a population rule.

        For an admitted reference the branch is unreachable by argument: a sum of
        non-negative terms is exactly zero only if every term underflows, so every residue
        would need to be under 2.2e-162, forcing near-constancy and putting the whole RMS
        in the mean. At the -45 dBFS floor that mean sits near 5.6e-3, where distinct
        doubles differ by at least one ULP of 8.7e-19, so every sample would have to be
        the same double and the constancy rule would catch it first. A floor-RMS frame with
        one-ULP variation has centred energy near 7.5e-37. Tested because the guard is live
        code, not because a benchmark frame can arrive this way.
        """
        reference = np.array([1e-150, 1e-150 + 1e-165])
        assert not np.all(reference == reference[0])
        assert float(np.sum(reference**2)) > 0.0
        assert float(np.dot(reference - reference.mean(), reference - reference.mean())) == 0.0

        with pytest.raises(metrics.MetricInputError, match="no energy after centring"):
            metrics.si_snr(reference, reference * 2, SISNR)

    def test_a_single_sample_reference_falls_under_the_same_rule(self) -> None:
        """One sample is trivially all-equal, so it is constant and the same rule rejects
        it. An earlier version excluded it with a size check, which bought nothing and
        split one condition across two branches."""
        with pytest.raises(metrics.MetricInputError, match="reference is constant"):
            metrics.si_snr(np.array([0.5]), np.array([0.7]), SISNR)

    def test_epsilon_stays_in_the_projection_denominator(self) -> None:
        """A regression pin on the frozen formula's exact shape, not an independent
        oracle: it necessarily encodes the placement it is pinning.

        Worth having because every other test is blind to it. On a normal reference the
        centred energy dwarfs epsilon, so removing it from the denominator moves the
        result below any sensible tolerance and every other assertion still passes. This
        fixture is valid but very quiet, with centred energy at about one hundredth of
        epsilon, where the two placements differ by 0.044 dB.
        """
        rng = np.random.default_rng(0)
        reference = rng.standard_normal(64) * 1e-6
        estimate = reference + 1e-7 * rng.standard_normal(64)

        centred_reference = reference - reference.mean()
        centred_estimate = estimate - estimate.mean()
        epsilon = SISNR.epsilon
        assert float(np.dot(centred_reference, centred_reference)) < epsilon

        denominator = float(np.dot(centred_reference, centred_reference)) + epsilon
        target = (float(np.dot(centred_estimate, centred_reference)) / denominator) * (
            centred_reference
        )
        residual = centred_estimate - target
        expected = 10.0 * np.log10(
            (float(np.sum(target**2)) + epsilon) / (float(np.sum(residual**2)) + epsilon)
        )

        result = metrics.si_snr(reference, estimate, SISNR)
        assert result.value == pytest.approx(expected, abs=1e-12)

    def test_a_silent_estimate_scores_rather_than_raising(self) -> None:
        """A model that emits zeros failed at restoring, which is a result and not an
        invalid item. It must produce a number so coverage stays complete."""
        result = metrics.si_snr(dense(), np.zeros(UNIT_SAMPLES), SISNR)
        assert np.isfinite(result.value)


class TestPeriodicHann:
    """The window is defined by equation; libraries are witnesses, not the definition."""

    def test_matches_the_protocol_equation(self) -> None:
        length = 1024
        n = np.arange(length)
        expected = 0.5 - 0.5 * np.cos(2 * np.pi * n / length)
        assert np.allclose(metrics.periodic_hann(length), expected, atol=0.0, rtol=0.0)

    def test_agrees_with_scipy_periodic(self) -> None:
        assert np.allclose(metrics.periodic_hann(1024), get_window("hann", 1024), atol=1e-15)

    def test_differs_from_the_symmetric_convention(self) -> None:
        """numpy.hanning divides by length-1 and answers to the same name. If these ever
        agreed, the distinction v2 was amended to freeze would not exist."""
        assert not np.allclose(metrics.periodic_hann(1024), np.hanning(1024), atol=1e-6)


class TestStftTransform:
    """Framing and transform, checked against an independent implementation."""

    def test_frame_and_bin_count_for_the_evaluation_unit(self) -> None:
        """The protocol's worked example. Framing errors usually still produce a
        plausible-looking spectrum, so shape is the cheap structural oracle."""
        assert metrics.stft_magnitude(dense(), LSD).shape == (126, 513)

    def test_agrees_with_librosa_configured_to_the_frozen_semantics(self) -> None:
        """librosa at its defaults would disagree, which is the entire reason the
        protocol names window symmetry, centring, and padding. Configured to match, it is
        an independent implementation of the same transform."""
        signal = dense()
        theirs = np.abs(
            librosa.stft(
                signal,
                n_fft=LSD.n_fft,
                win_length=LSD.win_length,
                hop_length=LSD.hop_length,
                window="hann",
                center=True,
                pad_mode="constant",
            )
        )
        assert np.allclose(metrics.stft_magnitude(signal, LSD).T, theirs, atol=1e-12)

    def test_an_unsupported_transform_is_refused(self) -> None:
        """This module implements one transform and says so, rather than passing a config
        string into a call that accepts only certain values."""
        import dataclasses

        ortho = dataclasses.replace(LSD, fft=dataclasses.replace(LSD.fft, norm="ortho"))
        with pytest.raises(NotImplementedError, match="rfft"):
            metrics.stft_magnitude(dense(), ortho)


class TestLogSpectralDistance:
    """Log and reduction stages, given a transform verified separately."""

    @staticmethod
    def offset_aware_expectation(signal: np.ndarray, alpha: float) -> float:
        """Closed form for an estimate that is `alpha` times the reference.

        Exact, unlike `|20*log10(alpha)|`, because the offset is additive: each bin
        contributes `20*log10((alpha*|X| + offset) / (|X| + offset))`, which equals
        `20*log10(alpha)` only as the offset vanishes. On a quiet fixture the naive form
        is wrong by tens of millidecibels.
        """
        magnitudes = metrics.stft_magnitude(signal, LSD)
        offset = LSD.log_magnitude.offset
        per_bin = 20.0 * np.log10((alpha * magnitudes + offset) / (magnitudes + offset))
        return float(np.mean(np.sqrt(np.mean(per_bin**2, axis=1))))

    def test_identical_signals_are_zero(self) -> None:
        assert metrics.log_spectral_distance(dense(), dense(), LSD).value == 0.0

    @pytest.mark.parametrize("alpha", [2.0, 10.0])
    def test_scaling_matches_the_offset_aware_closed_form_on_loud_content(
        self, alpha: float
    ) -> None:
        signal = dense()
        result = metrics.log_spectral_distance(signal, alpha * signal, LSD)
        assert result.value == pytest.approx(
            self.offset_aware_expectation(signal, alpha), abs=1e-12
        )

    @pytest.mark.parametrize("alpha", [2.0, 10.0])
    def test_scaling_matches_it_on_quiet_content_too(self, alpha: float) -> None:
        """The fixture that matters. Magnitudes near the offset are where additive and
        clamped semantics diverge, and where the naive closed form fails outright."""
        signal = dense(scale=1e-7)
        result = metrics.log_spectral_distance(signal, alpha * signal, LSD)
        assert result.value == pytest.approx(
            self.offset_aware_expectation(signal, alpha), abs=1e-12
        )

    def test_the_naive_closed_form_is_wrong_on_quiet_content(self) -> None:
        """Guards the guard. If this ever passed, the offset semantics would not be
        under test and `max(|X|, offset)` could be substituted unnoticed."""
        signal = dense(scale=1e-7)
        naive = abs(20.0 * np.log10(2.0))
        actual = metrics.log_spectral_distance(signal, 2.0 * signal, LSD).value
        assert abs(actual - naive) > 1e-3

    def test_it_is_blind_to_phase(self) -> None:
        """A magnitude metric. A sign flip changes every sample and no magnitude.

        Asserted as exact equality rather than a tolerance, because it is provably exact:
        IEEE negation flips one bit, padding and windowing carry the negation through
        unchanged, the transform is linear over exactly negated inputs, and the magnitude
        discards the sign. A tolerance here would weaken a bitwise-true property.
        """
        signal = dense()
        assert metrics.log_spectral_distance(signal, -signal, LSD).value == 0.0

    def test_reduction_order_is_rms_then_mean(self) -> None:
        """RMS across bins then a mean across frames is not the same statistic as the
        reverse. Built so the two differ: one frame carries all the error."""
        signal = dense()
        estimate = signal.copy()
        estimate[:1024] *= 4.0

        magnitudes_a = metrics.stft_magnitude(signal, LSD)
        magnitudes_b = metrics.stft_magnitude(estimate, LSD)
        offset = LSD.log_magnitude.offset
        difference = 20.0 * np.log10(magnitudes_b + offset) - 20.0 * np.log10(magnitudes_a + offset)
        # The protocol's order, then the swap: mean across bins first, RMS across frames.
        rms_then_mean = float(np.mean(np.sqrt(np.mean(difference**2, axis=1))))
        mean_then_rms = float(np.sqrt(np.mean(np.mean(difference, axis=1) ** 2)))
        # 0.50 against 2.28 on this fixture. Signed differences in the loud frames do not
        # cancel, so averaging bins first keeps the excursion that RMS-then-mean dilutes
        # across all 126 frames. A test where the two orders agreed would prove nothing.
        assert abs(rms_then_mean - mean_then_rms) > 1.0

        result = metrics.log_spectral_distance(signal, estimate, LSD)
        assert result.value == pytest.approx(rms_then_mean, abs=1e-12)

    def test_it_is_never_reported_as_capped(self) -> None:
        """No cap is defined for a distance, so the flag must stay false rather than
        being left uninitialised or borrowed from another metric."""
        assert not metrics.log_spectral_distance(dense(), 2 * dense(), LSD).capped

    def test_two_silent_signals_never_reach_a_score(self) -> None:
        """The trap the module docstring names: flooring both spectra gives 0 dB, which
        reads as perfect restoration. Shared validation rejects it first."""
        silence = np.zeros(UNIT_SAMPLES)
        with pytest.raises(metrics.MetricInputError, match="no energy"):
            metrics.log_spectral_distance(silence, silence, LSD)
