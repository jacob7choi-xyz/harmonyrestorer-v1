"""Analog noise synthesis for training data generation.

Synthesizes realistic analog audio degradation including tape hiss,
vinyl crackle, mains hum, high-frequency rolloff, and tape saturation.
Each effect is independently controllable and combinable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Noise primitives
# ---------------------------------------------------------------------------


def tape_hiss(
    length: int,
    sr: int,
    color: str = "pink",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate frequency-shaped noise resembling analog tape hiss.

    Real tape hiss is not white noise -- it rolls off at low frequencies
    and has a characteristic "shhh" quality closer to pink noise.

    Args:
        length: Number of samples.
        sr: Sample rate.
        color: Noise color ("pink" or "brown"). Pink is typical tape hiss.
        rng: NumPy random generator for reproducibility.

    Returns:
        Noise signal normalized to unit peak.
    """
    if rng is None:
        rng = np.random.default_rng()

    white = rng.standard_normal(length).astype(np.float32)

    if color == "pink":
        # Voss-McCartney approximation: -3 dB/octave rolloff
        # Apply via FFT shaping: multiply spectrum by 1/sqrt(f)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length, d=1.0 / sr)
        freqs = np.maximum(freqs, 1.0)  # avoid division by zero across all bins
        spectrum *= 1.0 / np.sqrt(freqs)
        noise = np.fft.irfft(spectrum, n=length).astype(np.float32)
    elif color == "brown":
        # -6 dB/octave rolloff (cumulative sum of white noise)
        noise = np.cumsum(white).astype(np.float32)
        # Remove DC drift and normalize to prevent float32 overflow on long signals
        noise -= np.mean(noise)
        std = np.std(noise)
        if std > 0:
            noise /= std
    else:
        noise = white

    # Normalize to unit peak
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise /= peak

    return noise


def vinyl_crackle(
    length: int,
    sr: int,
    density: float = 0.001,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate vinyl-style impulse noise (clicks and pops).

    Models the random impulse artifacts from vinyl playback caused by
    dust, scratches, and static discharge.

    Args:
        length: Number of samples.
        sr: Sample rate.
        density: Probability of a click at each sample (0.001 = ~1 click/ms).
        rng: NumPy random generator for reproducibility.

    Returns:
        Crackle signal normalized to unit peak.
    """
    if rng is None:
        rng = np.random.default_rng()

    crackle = np.zeros(length, dtype=np.float32)

    # Random impulse positions
    click_mask = rng.random(length) < density
    num_clicks = int(click_mask.sum())

    if num_clicks == 0:
        return crackle

    # Random amplitudes (bipolar, varying intensity)
    crackle[click_mask] = rng.standard_normal(num_clicks).astype(np.float32)

    # Convolve with a short decay kernel to make clicks more realistic
    decay_len = int(sr * 0.001)  # 1ms decay
    if decay_len > 1:
        decay = np.exp(-np.linspace(0, 5, decay_len)).astype(np.float32)
        crackle = np.convolve(crackle, decay, mode="same")

    peak = np.max(np.abs(crackle))
    if peak > 0:
        crackle /= peak

    return crackle


def mains_hum(
    length: int,
    sr: int,
    freq: float = 60.0,
    n_harmonics: int = 4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate mains hum with harmonics.

    Models electrical interference from AC power lines. Includes the
    fundamental frequency and its harmonics with decreasing amplitude.

    Args:
        length: Number of samples.
        sr: Sample rate.
        freq: Fundamental frequency (60 Hz in US/Japan, 50 Hz in Europe).
        n_harmonics: Number of harmonics to include.
        rng: NumPy random generator for reproducibility.

    Returns:
        Hum signal normalized to unit peak.
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(length, dtype=np.float32) / sr
    hum = np.zeros(length, dtype=np.float32)

    for h in range(1, n_harmonics + 1):
        # Amplitude falls off as 1/h^1.5 (steeper than 1/f)
        amplitude = 1.0 / (h**1.5)
        phase = rng.uniform(0, 2 * np.pi)
        hum += amplitude * np.sin(2 * np.pi * freq * h * t + phase)

    peak = np.max(np.abs(hum))
    if peak > 0:
        hum /= peak

    return hum


def high_freq_rolloff(signal: np.ndarray, sr: int, cutoff_hz: float = 6000.0) -> np.ndarray:
    """Apply low-pass filter simulating old equipment frequency limitations.

    Analog recordings from the 1940s-60s typically had limited high-frequency
    response, rolling off above 6-10 kHz depending on the medium.

    Args:
        signal: Input audio signal.
        sr: Sample rate.
        cutoff_hz: Cutoff frequency in Hz.

    Returns:
        Filtered signal (same length as input).
    """
    nyquist = sr / 2.0
    if cutoff_hz >= nyquist:
        logger.debug("Rolloff skipped: cutoff %.0f Hz >= nyquist %.0f Hz", cutoff_hz, nyquist)
        return signal

    normalized_cutoff = cutoff_hz / nyquist
    b, a = butter(4, normalized_cutoff, btype="low")
    return lfilter(b, a, signal).astype(np.float32)


def tape_saturation(signal: np.ndarray, drive: float = 0.5) -> np.ndarray:
    """Apply soft clipping simulating analog tape saturation.

    Tape saturation compresses loud signals through magnetic hysteresis,
    adding subtle harmonic distortion. Modeled as tanh soft clipping.

    Args:
        signal: Input audio signal.
        drive: Saturation intensity (0.0 = none, 1.0 = heavy).

    Returns:
        Saturated signal.
    """
    if drive <= 0:
        return signal

    # Scale input by drive amount, apply tanh, scale back
    gain = 1.0 + drive * 4.0  # drive=1.0 gives 5x gain into tanh
    saturated = np.tanh(signal * gain).astype(np.float32)

    # Compensate for volume change
    peak_in = np.max(np.abs(signal))
    peak_out = np.max(np.abs(saturated))
    if peak_out > 0 and peak_in > 0:
        saturated *= peak_in / peak_out

    return saturated


# ---------------------------------------------------------------------------
# Composite degradation
# ---------------------------------------------------------------------------


@dataclass
class DegradationParams:
    """Parameters controlling the analog degradation applied to a signal.

    All intensities are in [0, 1] where 0 means disabled.
    """

    hiss_snr_db: float = 20.0
    hiss_color: str = "pink"
    crackle_density: float = 0.0005
    crackle_amplitude: float = 0.0
    hum_amplitude: float = 0.0
    hum_freq: float = 60.0
    rolloff_hz: float = 8000.0
    saturation_drive: float = 0.0

    # Which effects are enabled
    enable_hiss: bool = True
    enable_crackle: bool = False
    enable_hum: bool = False
    enable_rolloff: bool = True
    enable_saturation: bool = False


def apply_degradation(
    signal: np.ndarray,
    sr: int,
    params: DegradationParams,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a combination of analog degradation effects to a clean signal.

    Args:
        signal: Clean audio signal (float32, normalized to [-1, 1]).
        sr: Sample rate.
        params: Degradation parameters controlling each effect.
        rng: NumPy random generator for reproducibility.

    Returns:
        Degraded signal (same length as input).
    """
    if rng is None:
        rng = np.random.default_rng()

    degraded = signal.copy()
    length = len(signal)

    # Tape saturation first (modifies the signal itself)
    if params.enable_saturation and params.saturation_drive > 0:
        degraded = tape_saturation(degraded, drive=params.saturation_drive)

    # High-frequency rolloff
    if params.enable_rolloff and params.rolloff_hz < sr / 2:
        degraded = high_freq_rolloff(degraded, sr, cutoff_hz=params.rolloff_hz)

    # Additive noise: tape hiss
    if params.enable_hiss:
        hiss = tape_hiss(length, sr, color=params.hiss_color, rng=rng)
        # Scale hiss to target SNR relative to the degraded signal
        signal_power = np.mean(degraded**2)
        hiss_power = np.mean(hiss**2)
        if signal_power > 0 and hiss_power > 0:
            snr_linear = 10 ** (params.hiss_snr_db / 10)
            target_noise_power = signal_power / snr_linear
            hiss *= np.sqrt(target_noise_power / hiss_power)
            degraded += hiss

    # Additive noise: vinyl crackle
    if params.enable_crackle and params.crackle_amplitude > 0:
        crackle = vinyl_crackle(length, sr, density=params.crackle_density, rng=rng)
        degraded += crackle * params.crackle_amplitude

    # Additive noise: mains hum
    if params.enable_hum and params.hum_amplitude > 0:
        hum = mains_hum(length, sr, freq=params.hum_freq, rng=rng)
        degraded += hum * params.hum_amplitude

    # Clip to valid range
    return np.clip(degraded, -1.0, 1.0).astype(np.float32)


def random_degradation(
    signal: np.ndarray,
    sr: int,
    *,
    snr_range: tuple[float, float] = (5.0, 30.0),
    include_crackle: bool = True,
    include_hum: bool = True,
    include_saturation: bool = True,
    seed: int | None = None,
) -> tuple[np.ndarray, DegradationParams]:
    """Apply randomized analog degradation for data augmentation.

    Always applies tape hiss and rolloff. Optionally includes crackle,
    hum, and saturation with random probability and intensity.

    Args:
        signal: Clean audio signal.
        sr: Sample rate.
        snr_range: Min/max SNR in dB for tape hiss.
        include_crackle: Whether crackle can be randomly enabled.
        include_hum: Whether hum can be randomly enabled.
        include_saturation: Whether saturation can be randomly enabled.
        seed: Random seed for reproducibility (None = non-deterministic).

    Returns:
        Tuple of (degraded_signal, params_used).
    """
    rng = np.random.default_rng(seed)

    params = DegradationParams(
        # Always-on effects
        enable_hiss=True,
        hiss_snr_db=rng.uniform(snr_range[0], snr_range[1]),
        hiss_color=rng.choice(["pink", "brown"]),
        enable_rolloff=True,
        rolloff_hz=rng.uniform(4000.0, 8000.0),
        # Optional effects (50% chance each)
        enable_crackle=include_crackle and rng.random() > 0.5,
        crackle_density=rng.uniform(0.0002, 0.002),
        crackle_amplitude=rng.uniform(0.01, 0.1),
        enable_hum=include_hum and rng.random() > 0.7,
        hum_amplitude=rng.uniform(0.005, 0.05),
        hum_freq=rng.choice([50.0, 60.0]),
        enable_saturation=include_saturation and rng.random() > 0.6,
        saturation_drive=rng.uniform(0.1, 0.6),
    )

    degraded = apply_degradation(signal, sr, params, rng=rng)
    return degraded, params
