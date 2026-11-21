"""Sampling and reading the training pairs, and the quantities measured on them."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf

from benchmark.degradation import high_freq_rolloff, tape_saturation

__all__ = [
    "PAIRS_DIR",
    "TrainingPair",
    "expected_hiss_peak",
    "hiss_free_signal",
    "load_pair",
    "sampled_metadata",
    "spectral_slope",
]

PAIRS_DIR = Path("data/pairs")


@dataclass(frozen=True)
class TrainingPair:
    """One training pair as written: its recorded parameters and both float32 frames."""

    name: str
    params: dict[str, Any]
    clean: npt.NDArray[np.float32]
    noisy: npt.NDArray[np.float32]
    sample_rate: int


def sampled_metadata(
    seed: int, pairs_dir: Path = PAIRS_DIR
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Each pair's metadata, in sorted filename order permuted by `default_rng(seed)`."""
    names = sorted(path.name for path in (pairs_dir / "metadata").iterdir())
    for index in np.random.default_rng(seed).permutation(len(names)):
        name = names[index]
        yield name, json.loads((pairs_dir / "metadata" / name).read_text())


def load_pair(name: str, metadata: dict[str, Any], pairs_dir: Path = PAIRS_DIR) -> TrainingPair:
    """Read a pair's clean and noisy frames as float32, the format they were written in."""
    clean, sample_rate = sf.read(pairs_dir / "clean" / metadata["clean_frame"], dtype="float32")
    noisy, _ = sf.read(pairs_dir / "noisy" / name.replace(".json", ".wav"), dtype="float32")
    return TrainingPair(name, metadata["params"], clean, noisy, int(sample_rate))


def hiss_free_signal(pair: TrainingPair) -> npt.NDArray[np.float32]:
    """The clean frame through the port's saturation (when it was on) and rolloff.

    For a pair with crackle and hum off, the noisy frame minus this is the added hiss.
    """
    params = pair.params
    signal = pair.clean
    if params["enable_saturation"]:
        signal = tape_saturation(signal, params["saturation_drive"])
    return high_freq_rolloff(signal, pair.sample_rate, params["rolloff_hz"])


def expected_hiss_peak(signal: npt.NDArray[np.float32], snr_db: float) -> float:
    """The added hiss's peak under d8993b2's scaling: unit peak times sqrt(power / SNR)."""
    return float(np.sqrt(np.mean(signal**2) / 10 ** (snr_db / 10)))


def spectral_slope(
    samples: npt.NDArray[np.float32], sample_rate: int, low_hz: float, high_hz: float
) -> float:
    """Least-squares slope of log10 power against log10 frequency over a band."""
    freqs = np.fft.rfftfreq(samples.size, 1 / sample_rate)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    power = np.abs(np.fft.rfft(samples.astype(np.float64))) ** 2
    return float(np.polyfit(np.log10(freqs[band]), np.log10(power[band]), 1)[0])
