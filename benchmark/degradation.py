"""The noise process that generated the training pairs, ported so the benchmark can reuse it.

The benchmark degrades clean frames with fresh draws of the process that produced the
training pairs, so OpGAN is tested on unseen recordings under the noise it was trained
for. That process is not the current `dataset/noise.py`. The pairs were written on
2026-03-07 between commits `d8993b2` and `2174dce`. On a seeded random sample of 400
unclipped training pairs with crackle and hum off (200 of them with saturation on), the
hiss added to each pair peaked at `sqrt(signal_power / snr_linear)`, with the signal
computed by this module's saturation and rolloff, to float32 rounding (median relative
error 8.3e-8, maximum 1.7e-6, computed in float64; `benchmark/measurements/hiss_peak.py`).
That is `d8993b2`'s scaling: hiss normalised to unit peak,
then multiplied as though it had unit power. The current code normalises the hiss power
instead, so its hiss is louder than the training hiss for the same recorded SNR. On 100
such pairs per colour, the training hiss's median spectral slope over 100 Hz to 4 kHz was
-1.000 for pink and -1.934 for brown, against -0.999 and -1.936 for `tape_hiss`
(`benchmark/measurements/hiss_colour.py`).

**Ported, not imported.** The primitives below are copies of `dataset/noise.py` at
`d8993b2`, with the arithmetic written as it was there. Given the same random stream,
each primitive and the composite were bit-identical to `d8993b2` in 500 of 500
comparisons, and `draw_params` matched its `random_degradation` for 200 of 200 seeds
(`benchmark/measurements/d8993b2_equivalence.py`).
Importing the training module would let an edit to training code change the benchmark
without any benchmark change.

**One generator per frame.** Training drew its parameters from an unseeded
`default_rng()` and its noise from the global `np.random`, and recorded no seed. Here one
`np.random.Generator` per frame makes every draw, in a fixed order: the parameters in
`random_degradation`'s order, then the hiss, crackle, and hum samples as
`apply_degradation` requests them. The distributions are training's; the streams are new.

**Training level.** Training scaled each recording to a peak of 0.95 before cutting
frames, and crackle and hum are added at fixed amplitudes, so their loudness relative to
the music depends on that scale. A frame is degraded here after the same scaling, using
the peak of its complete canonical waveform, and the result is returned at that level
with the gain, so the caller decides the level a system sees. The protocol does not yet
freeze this process; nothing here is called by a run.
"""

from __future__ import annotations

import logging
import math
import struct
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, lfilter

__all__ = [
    "D8993B2_RANGES",
    "DegradationParams",
    "DegradedFrame",
    "TrainingNoiseRanges",
    "apply_degradation",
    "degrade_frame",
    "draw_params",
    "frame_seed",
    "high_freq_rolloff",
    "mains_hum",
    "tape_hiss",
    "tape_saturation",
    "vinyl_crackle",
]

logger = logging.getLogger(__name__)

# Fixed constants of the d8993b2 primitives, named here rather than left inline.
_CRACKLE_DECAY_S = 0.001
_CRACKLE_DECAY_SPAN = 5.0
_HUM_HARMONICS = 4
_HUM_HARMONIC_EXPONENT = 1.5
_ROLLOFF_ORDER = 4
_SATURATION_GAIN_PER_DRIVE = 4.0
_HISS_COLORS = ("pink", "brown")


@dataclass(frozen=True)
class TrainingNoiseRanges:
    """The parameter distributions of training's `random_degradation`.

    A `(low, high)` pair is drawn with `Generator.uniform`. An effect behind a threshold is
    enabled when `Generator.random()` exceeds it. `headroom` is the peak each recording was
    scaled to before its frames were cut.
    """

    hiss_snr_db: tuple[float, float]
    hiss_colors: tuple[str, ...]
    rolloff_hz: tuple[float, float]
    crackle_threshold: float
    crackle_density: tuple[float, float]
    crackle_amplitude: tuple[float, float]
    hum_threshold: float
    hum_amplitude: tuple[float, float]
    hum_freqs: tuple[float, ...]
    saturation_threshold: float
    saturation_drive: tuple[float, float]
    headroom: float


# Training's distributions: random_degradation's ranges and thresholds at d8993b2, and the
# 0.95 peak preprocess.normalize scaled each recording to. The one copy in the code until a
# protocol amendment freezes these values in the protocol file.
D8993B2_RANGES = TrainingNoiseRanges(
    hiss_snr_db=(5.0, 30.0),
    hiss_colors=("pink", "brown"),
    rolloff_hz=(4000.0, 8000.0),
    crackle_threshold=0.5,
    crackle_density=(0.0002, 0.002),
    crackle_amplitude=(0.01, 0.1),
    hum_threshold=0.7,
    hum_amplitude=(0.005, 0.05),
    hum_freqs=(50.0, 60.0),
    saturation_threshold=0.6,
    saturation_drive=(0.1, 0.6),
    headroom=0.95,
)


@dataclass(frozen=True)
class DegradationParams:
    """One frame's drawn parameters, with the fields training recorded per pair."""

    hiss_snr_db: float
    hiss_color: str
    rolloff_hz: float
    enable_crackle: bool
    crackle_density: float
    crackle_amplitude: float
    enable_hum: bool
    hum_amplitude: float
    hum_freq: float
    enable_saturation: bool
    saturation_drive: float


@dataclass(frozen=True)
class DegradedFrame:
    """A degraded frame at training level, the gain that set that level, and its draws.

    `degraded` is float32 and read-only. The clean frame at the same level is
    `frame * gain`; dividing `degraded` by `gain` returns the frame's original level.
    """

    degraded: npt.NDArray[np.float32]
    gain: float
    params: DegradationParams


def frame_seed(seed: int, canonical_sha256: str, frame_index: int) -> np.random.SeedSequence:
    """The seed sequence for one frame: the seed, the digest's eight words, the index.

    Args:
        seed: The benchmark degradation seed, non-negative.
        canonical_sha256: The recording's canonical digest, 64 lowercase hex digits.
        frame_index: The frame's zero-based index in its recording.

    Returns:
        A seed sequence that differs for every recording and frame.

    Raises:
        ValueError: If any argument is outside its domain.
    """
    if seed < 0 or frame_index < 0:
        raise ValueError(f"seed and frame_index must be non-negative, got {seed}, {frame_index}")
    if len(canonical_sha256) != 64 or not all(c in "0123456789abcdef" for c in canonical_sha256):
        raise ValueError("canonical_sha256 must be 64 lowercase hex digits")
    words = struct.unpack(">8I", bytes.fromhex(canonical_sha256))
    return np.random.SeedSequence([seed, *words, frame_index])


def draw_params(rng: np.random.Generator, ranges: TrainingNoiseRanges) -> DegradationParams:
    """Draw one frame's parameters in the order training's `random_degradation` did.

    Keyword arguments are evaluated left to right, so this order is the draw order. Values
    are kept as the generator returns them, as training's were.
    """
    return DegradationParams(
        hiss_snr_db=rng.uniform(*ranges.hiss_snr_db),
        hiss_color=str(rng.choice(list(ranges.hiss_colors))),
        rolloff_hz=rng.uniform(*ranges.rolloff_hz),
        enable_crackle=bool(rng.random() > ranges.crackle_threshold),
        crackle_density=rng.uniform(*ranges.crackle_density),
        crackle_amplitude=rng.uniform(*ranges.crackle_amplitude),
        enable_hum=bool(rng.random() > ranges.hum_threshold),
        hum_amplitude=rng.uniform(*ranges.hum_amplitude),
        hum_freq=rng.choice(list(ranges.hum_freqs)),
        enable_saturation=bool(rng.random() > ranges.saturation_threshold),
        saturation_drive=rng.uniform(*ranges.saturation_drive),
    )


def _unit_peak(noise: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    peak = np.max(np.abs(noise))
    if peak > 0:
        noise /= peak
    return noise


def tape_hiss(
    length: int, sr: int, color: str, rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """Pink or brown noise normalised to unit peak, as at `d8993b2`."""
    if color not in _HISS_COLORS:
        raise ValueError(f"hiss color must be one of {_HISS_COLORS}, got {color!r}")
    white = rng.standard_normal(length).astype(np.float32)
    if color == "pink":
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(length, d=1.0 / sr)
        freqs[0] = 1.0
        spectrum *= 1.0 / np.sqrt(freqs)
        noise = np.fft.irfft(spectrum, n=length).astype(np.float32)
    else:
        noise = np.cumsum(white).astype(np.float32)
        noise -= np.mean(noise)
    return _unit_peak(noise)


def vinyl_crackle(
    length: int, sr: int, density: float, rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """Decaying impulses at the given density, normalised to unit peak, as at `d8993b2`."""
    crackle = np.zeros(length, dtype=np.float32)
    click_mask = rng.random(length) < density
    num_clicks = int(click_mask.sum())
    if num_clicks == 0:
        return crackle
    crackle[click_mask] = rng.standard_normal(num_clicks).astype(np.float32)
    decay_len = int(sr * _CRACKLE_DECAY_S)
    if decay_len > 1:
        decay = np.exp(-np.linspace(0, _CRACKLE_DECAY_SPAN, decay_len)).astype(np.float32)
        crackle = np.convolve(crackle, decay, mode="same")
    return _unit_peak(crackle)


def mains_hum(
    length: int, sr: int, freq: float, rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """A hum and three harmonics with random phases, normalised to unit peak."""
    t = np.arange(length, dtype=np.float32) / sr
    hum = np.zeros(length, dtype=np.float32)
    for h in range(1, _HUM_HARMONICS + 1):
        amplitude = 1.0 / (h**_HUM_HARMONIC_EXPONENT)
        phase = rng.uniform(0, 2 * np.pi)
        hum += amplitude * np.sin(2 * np.pi * freq * h * t + phase)
    return _unit_peak(hum)


def high_freq_rolloff(
    signal: npt.NDArray[np.float32], sr: int, cutoff_hz: float
) -> npt.NDArray[np.float32]:
    """A 4th-order Butterworth low-pass, applied causally, as at `d8993b2`."""
    nyquist = sr / 2.0
    if cutoff_hz >= nyquist:
        return signal
    b, a = butter(_ROLLOFF_ORDER, cutoff_hz / nyquist, btype="low")
    return np.asarray(lfilter(b, a, signal)).astype(np.float32)


def tape_saturation(signal: npt.NDArray[np.float32], drive: float) -> npt.NDArray[np.float32]:
    """tanh soft clipping, rescaled to the input's peak, as at `d8993b2`."""
    if drive <= 0:
        return signal
    gain = 1.0 + drive * _SATURATION_GAIN_PER_DRIVE
    saturated: npt.NDArray[np.float32] = np.tanh(signal * gain).astype(np.float32)
    peak_in = np.max(np.abs(signal))
    peak_out = np.max(np.abs(saturated))
    if peak_out > 0 and peak_in > 0:
        saturated *= peak_in / peak_out
    return saturated


def apply_degradation(
    signal: npt.NDArray[np.float32],
    sr: int,
    params: DegradationParams,
    rng: np.random.Generator,
) -> npt.NDArray[np.float32]:
    """Saturation, rolloff, hiss, crackle, then hum, clipped to [-1, 1], as at `d8993b2`.

    Hiss is always on, as `random_degradation` set it, and is scaled as though its unit
    peak were unit power, which is what made the training hiss quieter than recorded.
    """
    degraded = signal.copy()
    length = len(signal)
    if params.enable_saturation and params.saturation_drive > 0:
        degraded = tape_saturation(degraded, params.saturation_drive)
    if params.rolloff_hz < sr / 2:
        degraded = high_freq_rolloff(degraded, sr, params.rolloff_hz)
    hiss = tape_hiss(length, sr, params.hiss_color, rng)
    signal_power = np.mean(degraded**2)
    if signal_power > 0:
        noise_power = signal_power / 10 ** (params.hiss_snr_db / 10)
        hiss *= np.sqrt(noise_power)
        degraded += hiss
    if params.enable_crackle and params.crackle_amplitude > 0:
        crackle = vinyl_crackle(length, sr, params.crackle_density, rng)
        degraded += crackle * params.crackle_amplitude
    if params.enable_hum and params.hum_amplitude > 0:
        hum = mains_hum(length, sr, params.hum_freq, rng)
        degraded += hum * params.hum_amplitude
    return np.clip(degraded, -1.0, 1.0).astype(np.float32)


def degrade_frame(
    frame: npt.NDArray[np.float64],
    *,
    recording_peak: float,
    sample_rate: int,
    ranges: TrainingNoiseRanges,
    seed: np.random.SeedSequence,
) -> DegradedFrame:
    """Degrade one canonical frame at training level with its own generator.

    Args:
        frame: The frame's canonical samples.
        recording_peak: The largest absolute sample of the complete canonical waveform.
        sample_rate: The canonical rate.
        ranges: Training's parameter distributions.
        seed: The frame's seed sequence, from `frame_seed`.

    Returns:
        The degraded frame at training level, the gain, and the drawn parameters.

    Raises:
        ValueError: If the peak is not finite and positive.
    """
    if not (math.isfinite(recording_peak) and recording_peak > 0):
        raise ValueError(f"recording_peak must be finite and positive, got {recording_peak}")
    gain = ranges.headroom / recording_peak
    scaled = (frame * gain).astype(np.float32)
    rng = np.random.Generator(np.random.PCG64(seed))
    params = draw_params(rng, ranges)
    degraded = apply_degradation(scaled, sample_rate, params, rng)
    degraded.flags.writeable = False
    logger.debug("Degraded %d samples at gain %.6f with %s", frame.size, gain, params)
    return DegradedFrame(degraded=degraded, gain=gain, params=params)
