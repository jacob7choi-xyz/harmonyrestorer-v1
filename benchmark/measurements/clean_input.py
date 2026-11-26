r"""How much do v1 and UVR change clean, full-band input?

The rule is fixed in `docs/v1-clean-input.md` before this runs. Clean audio is the benchmark's
demo partition (6 CC0 solo piano recordings), on the frames Phase 0 measured
(`real_noise_gap.frame_indices`), at training level as the systems stage sets it, with nothing
added and no low-pass. Each system restores each frame through the benchmark's in-process
engine, and its output is scored against that same frame with the benchmark's `score_output`.
This part is descriptive: the listening check in `clean_input_listen.py` decides.

Run from the repository root:

    PYTHONPATH=backend uv run --frozen python -m benchmark.measurements.clean_input \\
        --walk-dir DIR --uvr-model-dir DIR --out-dir NEW_DIR
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from benchmark.atomic_write import write_directory_atomically
from benchmark.frame_scoring import FrameScore, degrade_with_reference, score_output
from benchmark.measurements.real_noise_gap import (
    demo_admissions,
    frame_indices,
    recording_interval,
)
from benchmark.protocol import Protocol, load_protocol
from benchmark.run_systems import OPGAN_CHECKPOINT, build_production_systems
from benchmark.systems_pipeline import Admission
from benchmark.systems_stage import (
    FrameIdentity,
    System,
    SystemOutput,
    call_system,
    write_frame_input,
)

__all__ = ["KEYS", "SYSTEMS", "score_clean", "summary"]

logger = logging.getLogger(__name__)

SYSTEMS = ("opgan", "uvr")
KEYS = ("si_snr_db", "shift_tolerant_si_snr_db", "lsd_db")
_RESULTS_NAME = "clean_input.json"


class CleanInputError(RuntimeError):
    """Raised when the measurement cannot run as the rule fixes it; nothing is written."""


@dataclass(frozen=True)
class CleanRun:
    """What every frame needs: the protocol, the systems by name, and a scratch folder."""

    protocol: Protocol
    systems: Mapping[str, System]
    scratch: Path


def score_clean(
    run: CleanRun, name: str, identity: FrameIdentity, reference: npt.NDArray[np.float64]
) -> dict[str, float]:
    """One system's output on a clean frame, scored against that frame.

    Raises:
        CleanInputError: If the system fails, or conditioning refuses its output.
    """
    rate = run.protocol.canonicalisation.resample.target_sample_rate
    path = run.scratch / "input.wav"
    write_frame_input(reference.astype(np.float32), path, rate)
    marker = run.scratch / "marker.json"
    result = call_system(
        run.systems[name], path, identity=identity, marker_path=marker, sample_rate=rate
    )
    if not isinstance(result, SystemOutput):
        raise CleanInputError(f"{name} failed on {identity}: {result.reason}")
    score = score_output(reference, result.samples, run.protocol)
    if not isinstance(score, FrameScore):
        raise CleanInputError(f"conditioning refused {name}'s output on {identity}: {score}")
    return {
        "si_snr_db": float(score.results["si_snr"].value),
        "shift_tolerant_si_snr_db": float(score.shift_tolerant["si_snr"].value),
        "lsd_db": float(score.results["log_spectral_distance"].value),
    }


def measure_recording(run: CleanRun, admission: Admission) -> dict[str, dict[str, float]]:
    """Per system, each value's mean over the recording's measured frames."""
    waveform, framed, indices = frame_indices(admission, run.protocol)
    peak = float(np.max(np.abs(waveform.samples)))
    collected: dict[str, list[dict[str, float]]] = {name: [] for name in run.systems}
    for index in indices:
        _, reference = degrade_with_reference(
            framed.frames[index], peak, admission.canonical_sha256, index, run.protocol
        )
        identity = FrameIdentity(admission.canonical_sha256, index)
        for name in run.systems:
            collected[name].append(score_clean(run, name, identity, reference))
    logger.info("Measured %s on %d frames", admission.identifier, len(indices))
    return {
        name: {key: float(np.mean([entry[key] for entry in entries])) for key in KEYS}
        for name, entries in collected.items()
    }


def summary(
    per_recording: Mapping[str, Mapping[str, Mapping[str, float]]], protocol: Protocol
) -> dict[str, dict[str, dict[str, object]]]:
    """Per system and value, the mean over recordings and its recording-bootstrap interval.

    `opgan_minus_uvr` holds the same for each recording's OpGAN value minus its UVR value.
    """
    series: dict[str, dict[str, list[float]]] = {
        name: {key: [values[name][key] for values in per_recording.values()] for key in KEYS}
        for name in SYSTEMS
    }
    series["opgan_minus_uvr"] = {
        key: [a - b for a, b in zip(series["opgan"][key], series["uvr"][key], strict=True)]
        for key in KEYS
    }
    result: dict[str, dict[str, dict[str, object]]] = {}
    for name, values_by_key in series.items():
        result[name] = {}
        for key, values in values_by_key.items():
            low, high = recording_interval(values, protocol)
            result[name][key] = {"mean": float(np.mean(values)), "interval": [low, high]}
    return result


def run_measurement(args: argparse.Namespace) -> bytes:
    """Load everything, measure every demo recording, and return the results file's bytes."""
    protocol = load_protocol()
    admissions = demo_admissions(args.walk_dir, protocol)
    with tempfile.TemporaryDirectory(prefix="clean-input-") as scratch:
        opgan, uvr = build_production_systems(
            protocol, Path(scratch), OPGAN_CHECKPOINT, args.uvr_model_dir
        )
        run = CleanRun(
            protocol=protocol, systems={"opgan": opgan, "uvr": uvr}, scratch=Path(scratch)
        )
        per_recording = {a.identifier: measure_recording(run, a) for a in admissions}
    document = {
        "rule": "docs/v1-clean-input.md",
        "per_recording": per_recording,
        "summary": summary(per_recording, protocol),
    }
    return (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement and write its results directory, or report why it could not."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    for name in ("--walk-dir", "--uvr-model-dir", "--out-dir"):
        parser.add_argument(name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        print(f"clean_input: {args.out_dir} exists; the output directory must be new")
        return 1
    try:
        raw = run_measurement(args)
    except CleanInputError as error:
        logger.error("Measurement refused: %s", error)
        print(f"clean_input: {error}")
        return 1
    write_directory_atomically(args.out_dir, {_RESULTS_NAME: raw})
    print(f"clean_input: wrote {args.out_dir / _RESULTS_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
