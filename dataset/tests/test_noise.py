"""Tests for analog noise synthesis functions."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.noise import (
    DegradationParams,
    apply_degradation,
    high_freq_rolloff,
    mains_hum,
    random_degradation,
    tape_hiss,
    tape_saturation,
    vinyl_crackle,
)

SR = 16000
LENGTH = SR  # 1 second at 16kHz


def _sine(freq: float = 440.0, length: int = LENGTH, sr: int = SR) -> np.ndarray:
    """Generate a unit-amplitude sine wave for testing."""
    t = np.arange(length, dtype=np.float32) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


class TestTapeHiss:
    """Tests for tape_hiss."""

    def test_shape_and_dtype(self) -> None:
        """Verify output has correct length and float32 dtype."""
        noise = tape_hiss(LENGTH, SR, rng=np.random.default_rng(42))
        assert noise.shape == (LENGTH,)
        assert noise.dtype == np.float32

    def test_peak_normalized(self) -> None:
        """Verify output is normalized to unit peak."""
        noise = tape_hiss(LENGTH, SR, rng=np.random.default_rng(42))
        np.testing.assert_almost_equal(np.max(np.abs(noise)), 1.0, decimal=5)

    def test_pink_has_more_low_freq_energy(self) -> None:
        """Pink noise should have more low-frequency energy than white noise."""
        rng = np.random.default_rng(42)
        pink = tape_hiss(LENGTH, SR, color="pink", rng=rng)
        rng2 = np.random.default_rng(99)
        white = tape_hiss(LENGTH, SR, color="white", rng=rng2)

        # Compare energy below 500 Hz vs total in the spectrum
        pink_spectrum = np.abs(np.fft.rfft(pink))
        white_spectrum = np.abs(np.fft.rfft(white))
        freqs = np.fft.rfftfreq(LENGTH, d=1.0 / SR)
        low_mask = freqs < 500

        pink_low_ratio = np.sum(pink_spectrum[low_mask] ** 2) / np.sum(pink_spectrum**2)
        white_low_ratio = np.sum(white_spectrum[low_mask] ** 2) / np.sum(white_spectrum**2)

        assert pink_low_ratio > white_low_ratio

    def test_reproducible_with_same_seed(self) -> None:
        """Same rng seed produces identical output."""
        a = tape_hiss(LENGTH, SR, rng=np.random.default_rng(42))
        b = tape_hiss(LENGTH, SR, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize(
        "color_a,color_b", [("pink", "brown"), ("pink", "white"), ("brown", "white")]
    )
    def test_different_colors_produce_different_output(self, color_a: str, color_b: str) -> None:
        """Different color parameters produce different noise signals."""
        a = tape_hiss(LENGTH, SR, color=color_a, rng=np.random.default_rng(42))
        b = tape_hiss(LENGTH, SR, color=color_b, rng=np.random.default_rng(42))
        assert not np.array_equal(a, b)


class TestVinylCrackle:
    """Tests for vinyl_crackle."""

    def test_shape(self) -> None:
        """Verify output has correct length."""
        crackle = vinyl_crackle(LENGTH, SR, rng=np.random.default_rng(42))
        assert crackle.shape == (LENGTH,)

    def test_zero_density_produces_silence(self) -> None:
        """Density of zero should produce all-zero output."""
        crackle = vinyl_crackle(LENGTH, SR, density=0.0, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(crackle, np.zeros(LENGTH, dtype=np.float32))

    def test_higher_density_produces_more_clicks(self) -> None:
        """Higher density should produce more non-zero samples."""
        low = vinyl_crackle(LENGTH, SR, density=0.001, rng=np.random.default_rng(42))
        high = vinyl_crackle(LENGTH, SR, density=0.01, rng=np.random.default_rng(42))

        low_nonzero = np.count_nonzero(low)
        high_nonzero = np.count_nonzero(high)
        assert high_nonzero > low_nonzero

    def test_reproducible_with_seed(self) -> None:
        """Same rng seed produces identical output."""
        a = vinyl_crackle(LENGTH, SR, rng=np.random.default_rng(42))
        b = vinyl_crackle(LENGTH, SR, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)


class TestMainsHum:
    """Tests for mains_hum."""

    def test_shape(self) -> None:
        """Verify output has correct length."""
        hum = mains_hum(LENGTH, SR, rng=np.random.default_rng(42))
        assert hum.shape == (LENGTH,)

    def test_fundamental_frequency_present(self) -> None:
        """FFT should show a peak at the fundamental frequency."""
        freq = 60.0
        # Use longer signal for better frequency resolution
        long_length = SR * 4
        hum = mains_hum(long_length, SR, freq=freq, n_harmonics=1, rng=np.random.default_rng(42))
        spectrum = np.abs(np.fft.rfft(hum))
        freqs = np.fft.rfftfreq(long_length, d=1.0 / SR)

        # Find the bin closest to 60 Hz and check it is a local maximum
        fundamental_bin = np.argmin(np.abs(freqs - freq))
        nearby = spectrum[max(0, fundamental_bin - 5) : fundamental_bin + 6]
        assert spectrum[fundamental_bin] == np.max(nearby)

    def test_harmonics_present(self) -> None:
        """FFT should show peaks at harmonic multiples of the fundamental."""
        freq = 60.0
        n_harmonics = 4
        long_length = SR * 4
        hum = mains_hum(
            long_length, SR, freq=freq, n_harmonics=n_harmonics, rng=np.random.default_rng(42)
        )
        spectrum = np.abs(np.fft.rfft(hum))
        freqs = np.fft.rfftfreq(long_length, d=1.0 / SR)

        for h in range(1, n_harmonics + 1):
            target = freq * h
            target_bin = np.argmin(np.abs(freqs - target))
            # Energy at harmonic bin should be significant relative to its neighbors
            neighbor_mean = np.mean(spectrum[max(0, target_bin - 10) : target_bin + 11])
            assert spectrum[target_bin] > neighbor_mean

    def test_reproducible_with_seed(self) -> None:
        """Same rng seed produces identical output."""
        a = mains_hum(LENGTH, SR, rng=np.random.default_rng(42))
        b = mains_hum(LENGTH, SR, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)


class TestHighFreqRolloff:
    """Tests for high_freq_rolloff."""

    def test_attenuates_above_cutoff(self) -> None:
        """Energy above cutoff should be significantly reduced."""
        cutoff = 4000.0
        # Create a signal with energy well above the cutoff
        high_freq_signal = _sine(7000.0)
        filtered = high_freq_rolloff(high_freq_signal, SR, cutoff_hz=cutoff)

        original_energy = np.mean(high_freq_signal**2)
        filtered_energy = np.mean(filtered**2)
        # 4th-order Butterworth at 7kHz with 4kHz cutoff: heavy attenuation
        assert filtered_energy < original_energy * 0.1

    def test_passthrough_when_cutoff_at_nyquist(self) -> None:
        """When cutoff >= nyquist, signal should pass through unchanged."""
        signal = _sine(440.0)
        result = high_freq_rolloff(signal, SR, cutoff_hz=SR / 2.0)
        np.testing.assert_array_equal(result, signal)

    def test_returns_same_length(self) -> None:
        """Filtered output must have the same length as input."""
        signal = _sine(440.0)
        filtered = high_freq_rolloff(signal, SR, cutoff_hz=4000.0)
        assert len(filtered) == len(signal)


class TestTapeSaturation:
    """Tests for tape_saturation."""

    def test_drive_zero_returns_input(self) -> None:
        """With drive=0, output should be identical to input."""
        signal = _sine(440.0)
        result = tape_saturation(signal, drive=0.0)
        np.testing.assert_array_equal(result, signal)

    def test_compressed_dynamics(self) -> None:
        """Saturation should reduce peak-to-RMS ratio (crest factor)."""
        signal = _sine(440.0) * 0.8
        saturated = tape_saturation(signal, drive=0.8)

        def crest_factor(s: np.ndarray) -> float:
            rms = np.sqrt(np.mean(s**2))
            return float(np.max(np.abs(s)) / rms) if rms > 0 else 0.0

        # tanh compresses peaks, so crest factor should decrease
        assert crest_factor(saturated) < crest_factor(signal)

    def test_preserves_peak_level(self) -> None:
        """Volume compensation should preserve the original peak level."""
        signal = _sine(440.0) * 0.7
        saturated = tape_saturation(signal, drive=0.5)
        np.testing.assert_almost_equal(np.max(np.abs(saturated)), np.max(np.abs(signal)), decimal=4)


class TestApplyDegradation:
    """Tests for apply_degradation."""

    def test_output_shape(self) -> None:
        """Output shape must match input shape."""
        signal = _sine(440.0)
        params = DegradationParams()
        result = apply_degradation(signal, SR, params, rng=np.random.default_rng(42))
        assert result.shape == signal.shape

    def test_output_clipped_to_valid_range(self) -> None:
        """Output must stay within [-1, 1]."""
        signal = _sine(440.0)
        params = DegradationParams(
            enable_hiss=True,
            hiss_snr_db=0.0,
            enable_crackle=True,
            crackle_amplitude=1.0,
            crackle_density=0.01,
            enable_hum=True,
            hum_amplitude=1.0,
        )
        result = apply_degradation(signal, SR, params, rng=np.random.default_rng(42))
        assert np.all(result >= -1.0)
        assert np.all(result <= 1.0)

    def test_all_effects_disabled_returns_input(self) -> None:
        """Disabling all effects should return the input unchanged."""
        signal = _sine(440.0)
        params = DegradationParams(
            enable_hiss=False,
            enable_crackle=False,
            enable_hum=False,
            enable_rolloff=False,
            enable_saturation=False,
        )
        result = apply_degradation(signal, SR, params, rng=np.random.default_rng(42))
        np.testing.assert_array_almost_equal(result, signal)

    @pytest.mark.parametrize(
        "effect_field",
        ["enable_hiss", "enable_crackle", "enable_hum", "enable_rolloff", "enable_saturation"],
    )
    def test_each_effect_independently_modifiable(self, effect_field: str) -> None:
        """Enabling a single effect should change the output vs all-disabled baseline."""
        signal = _sine(440.0) * 0.5
        base_params = DegradationParams(
            enable_hiss=False,
            enable_crackle=False,
            enable_hum=False,
            enable_rolloff=False,
            enable_saturation=False,
        )
        # Build params with one effect enabled, providing non-zero intensities
        kwargs = {
            "enable_hiss": False,
            "enable_crackle": False,
            "enable_hum": False,
            "enable_rolloff": False,
            "enable_saturation": False,
            effect_field: True,
            # Ensure the effect has non-zero intensity when enabled
            "crackle_amplitude": 0.5,
            "crackle_density": 0.01,
            "hum_amplitude": 0.1,
            "saturation_drive": 0.5,
            "rolloff_hz": 4000.0,
        }
        one_effect_params = DegradationParams(**kwargs)

        baseline = apply_degradation(signal, SR, base_params, rng=np.random.default_rng(42))
        result = apply_degradation(signal, SR, one_effect_params, rng=np.random.default_rng(42))
        assert not np.array_equal(baseline, result), f"{effect_field} had no effect"


class TestRandomDegradation:
    """Tests for random_degradation."""

    def test_reproducible_with_seed(self) -> None:
        """Same seed produces identical output and params."""
        signal = _sine(440.0)
        a_signal, a_params = random_degradation(signal, SR, seed=42)
        b_signal, b_params = random_degradation(signal, SR, seed=42)
        np.testing.assert_array_equal(a_signal, b_signal)
        assert a_params == b_params

    def test_returns_signal_and_params(self) -> None:
        """Return value is a tuple of (ndarray, DegradationParams)."""
        signal = _sine(440.0)
        result = random_degradation(signal, SR, seed=42)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], DegradationParams)

    def test_snr_within_requested_range(self) -> None:
        """Generated hiss_snr_db should fall within the given range."""
        signal = _sine(440.0)
        snr_range = (10.0, 25.0)
        for seed in range(20):
            _, params = random_degradation(signal, SR, snr_range=snr_range, seed=seed)
            assert snr_range[0] <= params.hiss_snr_db <= snr_range[1]

    def test_actual_snr_matches_requested(self) -> None:
        """Measured SNR of the degraded signal should be close to the requested hiss_snr_db.

        Uses a zero signal except for a known tone, then measures the added noise
        power to verify SNR within 1 dB tolerance.
        """
        signal = _sine(440.0) * 0.5
        snr_range = (15.0, 15.0)  # Fix SNR to a single value for precise checking
        degraded, params = random_degradation(
            signal,
            SR,
            snr_range=snr_range,
            include_crackle=False,
            include_hum=False,
            include_saturation=False,
            seed=42,
        )

        # The rolloff filter modifies the signal, so compare against the
        # rolloff-filtered version of the clean signal to isolate noise contribution
        clean_after_rolloff = high_freq_rolloff(signal.copy(), SR, cutoff_hz=params.rolloff_hz)
        noise_estimate = degraded - clean_after_rolloff

        signal_power = np.mean(clean_after_rolloff**2)
        noise_power = np.mean(noise_estimate**2)

        if signal_power > 0 and noise_power > 0:
            measured_snr = 10 * np.log10(signal_power / noise_power)
            assert (
                abs(measured_snr - params.hiss_snr_db) < 1.0
            ), f"Measured SNR {measured_snr:.1f} dB vs requested {params.hiss_snr_db:.1f} dB"
