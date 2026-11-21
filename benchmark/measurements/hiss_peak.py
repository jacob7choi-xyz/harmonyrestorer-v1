"""The training pairs' hiss peak against d8993b2's scaling, using the port's functions.

Pairs: metadata in sorted order permuted by `default_rng(1)`, kept when crackle and hum are
off and the noisy frame's peak is below 1.0, until 200 with saturation on and 200 with it
off. For each, the added hiss is the noisy frame minus `hiss_free_signal`, and its peak is
compared with `expected_hiss_peak`. Run from the repository root.
"""

from __future__ import annotations

import numpy as np

from benchmark.measurements.training_pairs import (
    expected_hiss_peak,
    hiss_free_signal,
    load_pair,
    sampled_metadata,
)

_SEED = 1
_PER_GROUP = 200


def measure() -> tuple[list[float], dict[str, int], int]:
    """Relative errors of the hiss peak, the pairs kept per group, and the pairs skipped."""
    errors: list[float] = []
    kept = {"saturation": 0, "no_saturation": 0}
    clipped = 0
    for name, metadata in sampled_metadata(_SEED):
        params = metadata["params"]
        if params["enable_crackle"] or params["enable_hum"]:
            continue
        group = "saturation" if params["enable_saturation"] else "no_saturation"
        if kept[group] >= _PER_GROUP:
            if all(count >= _PER_GROUP for count in kept.values()):
                break
            continue
        pair = load_pair(name, metadata)
        if np.max(np.abs(pair.noisy)) >= 1.0:
            clipped += 1
            continue
        kept[group] += 1
        signal = hiss_free_signal(pair)
        peak = float(np.max(np.abs(pair.noisy - signal)))
        errors.append(abs(peak / expected_hiss_peak(signal, params["hiss_snr_db"]) - 1))
    return errors, kept, clipped


def main() -> None:
    """Print the median and maximum relative error over the sampled pairs."""
    errors, kept, clipped = measure()
    print(
        f"hiss peak on training pairs: n={len(errors)} {kept}, skipped clipped {clipped}, "
        f"relative error median {np.median(errors):.2e} max {max(errors):.2e}"
    )


if __name__ == "__main__":
    main()
