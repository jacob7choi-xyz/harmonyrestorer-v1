"""Spectral slope of the training pairs' added hiss against the port's `tape_hiss`.

Pairs: metadata in sorted order permuted by `default_rng(2)`, kept when crackle, hum, and
saturation are off and the noisy frame's peak is below 1.0, until 100 per hiss colour. The
slope is fitted over 100 Hz to 4 kHz. The port's hiss is drawn from `default_rng(10_000 + i)`
for 100 values of `i`. The criterion, fixed before the first run, is that the median slopes
agree within 0.1 per colour. Run from the repository root.
"""

from __future__ import annotations

import numpy as np

from benchmark.degradation import tape_hiss
from benchmark.measurements.training_pairs import (
    hiss_free_signal,
    load_pair,
    sampled_metadata,
    spectral_slope,
)

_SEED = 2
_PER_COLOUR = 100
_LOW_HZ, _HIGH_HZ = 100.0, 4000.0
_FRAME, _RATE = 32000, 16000
_COLOURS = ("pink", "brown")
_TOLERANCE = 0.1


def training_slopes() -> dict[str, list[float]]:
    """Slopes of the added hiss in the sampled training pairs, by colour."""
    slopes: dict[str, list[float]] = {colour: [] for colour in _COLOURS}
    for name, metadata in sampled_metadata(_SEED):
        params = metadata["params"]
        if params["enable_crackle"] or params["enable_hum"] or params["enable_saturation"]:
            continue
        colour = params["hiss_color"]
        if len(slopes[colour]) >= _PER_COLOUR:
            if all(len(found) >= _PER_COLOUR for found in slopes.values()):
                break
            continue
        pair = load_pair(name, metadata)
        if np.max(np.abs(pair.noisy)) >= 1.0:
            continue
        hiss = pair.noisy - hiss_free_signal(pair)
        slopes[colour].append(spectral_slope(hiss, pair.sample_rate, _LOW_HZ, _HIGH_HZ))
    return slopes


def main() -> None:
    """Print each colour's median slope for the training pairs and for the port."""
    training = training_slopes()
    for colour in _COLOURS:
        ported = [
            spectral_slope(
                tape_hiss(_FRAME, _RATE, colour, np.random.default_rng(10_000 + i)),
                _RATE,
                _LOW_HZ,
                _HIGH_HZ,
            )
            for i in range(_PER_COLOUR)
        ]
        a, b = float(np.median(training[colour])), float(np.median(ported))
        print(
            f"{colour}: training median slope {a:.3f} (n={len(training[colour])}), port {b:.3f} "
            f"(n={len(ported)}), difference {abs(a - b):.3f}, within {_TOLERANCE}: "
            f"{abs(a - b) <= _TOLERANCE}"
        )


if __name__ == "__main__":
    main()
