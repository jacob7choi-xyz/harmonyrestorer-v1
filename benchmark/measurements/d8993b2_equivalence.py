"""Bitwise comparison of `benchmark/degradation.py` with `dataset/noise.py` at `d8993b2`.

`d8993b2` draws its noise from the legacy global `np.random`, and the port draws from the
generator it is given. Seeding the global state and passing the port a `RandomState` with
the same seed makes both issue the same calls on the same stream, so a faithful port is
bit-identical. `random_degradation` at `d8993b2` makes its own `default_rng()`, which is
replaced by a seeded one while its parameters are compared with `draw_params`. Run from the
repository root; it reads `d8993b2` with `git show`.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

from benchmark import degradation as port
from benchmark.protocol import load_protocol

_COMMIT = "d8993b2"
_LENGTH, _RATE = 32000, 16000
_SEEDS = 25
_DRAW_SEEDS = 200


def load_noise_at(commit: str) -> ModuleType:
    """`dataset/noise.py` as it was at `commit`, imported under a private name."""
    source = subprocess.run(
        ["git", "show", f"{commit}:dataset/noise.py"], check=True, capture_output=True, text=True
    ).stdout
    name = f"noise_at_{commit}"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{name}.py"
        path.write_text(source)
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # dataclass resolves its module through sys.modules while the class is built.
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def bit_identical(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> bool:
    """Same dtype, same shape, and the same bytes."""
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compare_primitives(old: ModuleType, seed: int) -> list[tuple[str, bool]]:
    """Each primitive under one seed, across colours, densities, cutoffs, and drives."""
    results: list[tuple[str, bool]] = []
    for color in ("pink", "brown"):
        np.random.seed(seed)
        a = old.tape_hiss(_LENGTH, _RATE, color=color)
        b = port.tape_hiss(_LENGTH, _RATE, color, np.random.RandomState(seed))  # type: ignore[arg-type]
        results.append((f"tape_hiss {color}", bit_identical(a, b)))
    for density in (0.0002, 0.001, 0.002):
        np.random.seed(seed)
        a = old.vinyl_crackle(_LENGTH, _RATE, density=density)
        b = port.vinyl_crackle(_LENGTH, _RATE, density, np.random.RandomState(seed))  # type: ignore[arg-type]
        results.append(("vinyl_crackle", bit_identical(a, b)))
    for freq in (50.0, 60.0, np.float64(50.0)):
        np.random.seed(seed)
        a = old.mains_hum(_LENGTH, _RATE, freq=freq)
        b = port.mains_hum(_LENGTH, _RATE, freq, np.random.RandomState(seed))  # type: ignore[arg-type]
        results.append(("mains_hum", bit_identical(a, b)))
    signal = np.random.RandomState(1000 + seed).uniform(-0.9, 0.9, _LENGTH).astype(np.float32)
    for cutoff in (4000.0, 6123.4, 7999.9, 8000.0):
        same = bit_identical(
            old.high_freq_rolloff(signal, _RATE, cutoff_hz=cutoff),
            port.high_freq_rolloff(signal, _RATE, cutoff),
        )
        results.append(("high_freq_rolloff", same))
    for drive in (0.0, 0.1, 0.37, 0.6):
        same = bit_identical(
            old.tape_saturation(signal, drive=drive), port.tape_saturation(signal, drive)
        )
        results.append(("tape_saturation", same))
    return results


def compare_composite(old: ModuleType, seed: int) -> list[tuple[str, bool]]:
    """`apply_degradation` with every effect on, all off, and two mixes."""
    signal = np.random.RandomState(1000 + seed).uniform(-0.9, 0.9, _LENGTH).astype(np.float32)
    draws = np.random.RandomState(2000 + seed)
    ranges = load_protocol().degradation.ranges
    results: list[tuple[str, bool]] = []
    for crackle, hum, sat in (
        (True, True, True),
        (False, False, False),
        (True, False, True),
        (False, True, False),
    ):
        values: dict[str, Any] = {
            "hiss_snr_db": draws.uniform(*ranges.hiss_snr_db),
            "hiss_color": str(draws.choice(list(ranges.hiss_colors))),
            "rolloff_hz": draws.uniform(*ranges.rolloff_hz),
            "crackle_density": draws.uniform(*ranges.crackle_density),
            "crackle_amplitude": draws.uniform(*ranges.crackle_amplitude),
            "hum_amplitude": draws.uniform(*ranges.hum_amplitude),
            "hum_freq": draws.choice(list(ranges.hum_freqs)),
            "saturation_drive": draws.uniform(*ranges.saturation_drive),
        }
        flags: dict[str, Any] = {
            "enable_crackle": crackle,
            "enable_hum": hum,
            "enable_saturation": sat,
        }
        old_params = old.DegradationParams(**values, **flags, enable_hiss=True, enable_rolloff=True)
        new_params = port.DegradationParams(**values, **flags)
        np.random.seed(seed)
        a = old.apply_degradation(signal, _RATE, old_params)
        b = port.apply_degradation(signal, _RATE, new_params, np.random.RandomState(seed))  # type: ignore[arg-type]
        results.append(("apply_degradation", bit_identical(a, b)))
    return results


def compare_draws(old: ModuleType) -> int:
    """Seeds for which `draw_params` equals `random_degradation`'s parameters."""
    real = np.random.default_rng
    matched = 0
    try:
        for seed in range(_DRAW_SEEDS):
            setattr(np.random, "default_rng", lambda *_, s=seed: real(s))  # noqa: B010
            _, theirs = old.random_degradation(np.zeros(_LENGTH, dtype=np.float32), _RATE)
            ours = port.draw_params(real(seed), load_protocol().degradation.ranges)
            fields = port.DegradationParams.__dataclass_fields__
            matched += all(getattr(theirs, field) == getattr(ours, field) for field in fields)
    finally:
        setattr(np.random, "default_rng", real)  # noqa: B010
    return matched


def main() -> None:
    """Print, per function, how many comparisons were bit-identical."""
    old = load_noise_at(_COMMIT)
    total: Counter[str] = Counter()
    same: Counter[str] = Counter()
    for seed in range(_SEEDS):
        for name, identical in compare_primitives(old, seed) + compare_composite(old, seed):
            total[name] += 1
            same[name] += identical
    for name in total:
        print(f"{name}: {same[name]}/{total[name]} bit-identical")
    print(
        f"draw_params equals random_degradation's parameters: {compare_draws(old)}/{_DRAW_SEEDS} seeds"
    )


if __name__ == "__main__":
    main()
