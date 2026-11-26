r"""Model v2: which parts of the benchmark's degradation does v1's SI-SNR gain come from?

The rule is fixed in `docs/model-v2-gain-steps.md` before this runs. Clean audio is the
benchmark's demo partition (6 CC0 solo piano recordings), on the same frames Phase 0 measured
(`real_noise_gap.frame_indices`). Each frame is degraded four ways from one shared draw, its
benchmark seed: crackle and hum keep their samples, and hiss keeps its waveform, with its
level set per condition from the signal's power after the low-pass and saturation:

- `benchmark`: the benchmark's degraded frame exactly, low-pass and saturation as drawn;
- `no_lowpass`: the same, with the low-pass moved to the Nyquist rate;
- `added_noise_only`: the same, with saturation also switched off;
- `snr_grid`: the noise of `added_noise_only` rescaled to Phase 0's SNR grid.

v1 restores every input through the benchmark's in-process engine, and the input and the
output are scored against the clean reference with the benchmark's `score_output`. Three steps
are reported, each the mean over recordings of a difference in SI-SNR improvement between two
adjacent conditions, with the recording bootstrap's 95% interval.

Run from the repository root:

    PYTHONPATH=backend uv run --frozen python -m benchmark.measurements.gain_steps \\
        --walk-dir DIR --uvr-model-dir DIR --out-dir NEW_DIR
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

from benchmark.atomic_write import write_directory_atomically
from benchmark.degradation import apply_degradation, draw_params, frame_seed
from benchmark.frame_scoring import FrameScore, degrade_with_reference, score_output
from benchmark.measurements.real_noise_gap import (
    SNR_GRID_DB,
    demo_admissions,
    frame_indices,
    recording_interval,
    scale_to_snr,
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

__all__ = ["CONDITIONS", "STEPS", "conditions_for_frame", "degrade_with", "steps"]

logger = logging.getLogger(__name__)

CONDITIONS = ("benchmark", "no_lowpass", "added_noise_only", "snr_grid")
# Each step is (name, the condition with the part, the condition without it).
STEPS = (
    ("lowpass", "benchmark", "no_lowpass"),
    ("saturation", "no_lowpass", "added_noise_only"),
    ("noise_level", "added_noise_only", "snr_grid"),
)
STEP_METRICS = ("si_snr_improvement_db", "shift_tolerant_si_snr_improvement_db")
CONTRIBUTOR_MARGIN_DB = 1.0
_METRIC = "si_snr"
_KEYS = (
    "input_si_snr_db",
    "output_si_snr_db",
    "si_snr_improvement_db",
    "shift_tolerant_si_snr_improvement_db",
    "lsd_output",
)
_RESULTS_NAME = "gain_steps.json"


class StepsError(RuntimeError):
    """Raised when the measurement cannot run as the rule fixes it; nothing is written."""


@dataclass(frozen=True)
class FrameSource:
    """One frame's canonical samples and what seeds and levels it in the benchmark."""

    clean: npt.NDArray[np.float64]
    recording_peak: float
    canonical_sha256: str
    index: int


def degrade_with(
    source: FrameSource, protocol: Protocol, *, lowpass: bool, saturation: bool
) -> npt.NDArray[np.float32]:
    """The frame degraded as `degrade_frame` does, with the low-pass or saturation removable.

    The gain, seed and parameter draws are `degrade_frame`'s, so with both parts kept the
    result is the benchmark's degraded frame. Removing a part changes no draw: the rolloff is
    drawn either way and neither part reads the generator, so crackle and hum keep their
    samples and hiss keeps its waveform. Hiss is scaled from the signal's power after the
    low-pass and saturation, so its level can differ slightly between conditions.
    """
    ranges = protocol.degradation.ranges
    rate = protocol.canonicalisation.resample.target_sample_rate
    scaled = (source.clean * (ranges.headroom / source.recording_peak)).astype(np.float32)
    seed = frame_seed(protocol.degradation.seed, source.canonical_sha256, source.index)
    rng = np.random.Generator(np.random.PCG64(seed))
    params = draw_params(rng, ranges)
    if not lowpass:
        params = replace(params, rolloff_hz=float(rate) / 2)
    if not saturation:
        params = replace(params, enable_saturation=False)
    return apply_degradation(scaled, rate, params, rng)


def conditions_for_frame(
    source: FrameSource, protocol: Protocol
) -> tuple[npt.NDArray[np.float64], dict[str, list[npt.NDArray[np.float64]]]]:
    """The frame's clean reference and its inputs per condition (`snr_grid` has one per SNR).

    Raises:
        StepsError: If `degrade_with` does not reproduce the benchmark's degraded frame.
    """
    degraded, reference = degrade_with_reference(
        source.clean, source.recording_peak, source.canonical_sha256, source.index, protocol
    )
    full = degrade_with(source, protocol, lowpass=True, saturation=True)
    if not np.array_equal(full, degraded.degraded):
        raise StepsError(f"frame {source.index} does not reproduce the benchmark's degradation")
    no_lowpass = degrade_with(source, protocol, lowpass=False, saturation=True)
    noise_only = degrade_with(source, protocol, lowpass=False, saturation=False)
    scaled = (source.clean * degraded.gain).astype(np.float32).astype(np.float64)
    noise = noise_only.astype(np.float64) - scaled
    grid = [reference + scale_to_snr(reference, noise, snr) for snr in SNR_GRID_DB]
    inputs = {
        "benchmark": [full.astype(np.float64)],
        "no_lowpass": [no_lowpass.astype(np.float64)],
        "added_noise_only": [noise_only.astype(np.float64)],
        "snr_grid": grid,
    }
    return reference, inputs


@dataclass(frozen=True)
class StepsRun:
    """What every frame needs: the protocol, v1 and a scratch folder."""

    protocol: Protocol
    system: System
    scratch: Path


def _score(score: object) -> FrameScore:
    """The score, or a refusal if conditioning rejected the pair.

    Raises:
        StepsError: If conditioning refused the pair.
    """
    if not isinstance(score, FrameScore):
        raise StepsError(f"conditioning refused a pair: {score}")
    return score


def score_input(
    run: StepsRun,
    identity: FrameIdentity,
    reference: npt.NDArray[np.float64],
    mixture: npt.NDArray[np.float64],
) -> dict[str, float]:
    """v1's output on one input, scored with the input against the clean reference.

    Raises:
        StepsError: If v1 fails on the input.
    """
    rate = run.protocol.canonicalisation.resample.target_sample_rate
    path = run.scratch / "input.wav"
    write_frame_input(mixture.astype(np.float32), path, rate)
    marker = run.scratch / "marker.json"
    result = call_system(run.system, path, identity=identity, marker_path=marker, sample_rate=rate)
    if not isinstance(result, SystemOutput):
        raise StepsError(f"v1 failed on {identity}: {result.reason}")
    before = _score(score_output(reference, mixture, run.protocol))
    after = _score(score_output(reference, result.samples, run.protocol))
    shifted_before = float(before.shift_tolerant[_METRIC].value)
    shifted_after = float(after.shift_tolerant[_METRIC].value)
    return {
        "input_si_snr_db": float(before.results[_METRIC].value),
        "output_si_snr_db": float(after.results[_METRIC].value),
        "si_snr_improvement_db": float(
            after.results[_METRIC].value - before.results[_METRIC].value
        ),
        "shift_tolerant_si_snr_improvement_db": shifted_after - shifted_before,
        "lsd_output": float(after.results["log_spectral_distance"].value),
    }


def measure_recording(run: StepsRun, admission: Admission) -> dict[str, dict[str, float]]:
    """Per condition, each reported value's mean over the recording's frames (and SNRs)."""
    waveform, framed, indices = frame_indices(admission, run.protocol)
    peak = float(np.max(np.abs(waveform.samples)))
    collected: dict[str, list[dict[str, float]]] = {name: [] for name in CONDITIONS}
    for index in indices:
        source = FrameSource(framed.frames[index], peak, admission.canonical_sha256, index)
        reference, inputs = conditions_for_frame(source, run.protocol)
        identity = FrameIdentity(admission.canonical_sha256, index)
        for condition, mixtures in inputs.items():
            collected[condition].extend(
                score_input(run, identity, reference, mixture) for mixture in mixtures
            )
    logger.info("Measured %s on %d frames", admission.identifier, len(indices))
    return {
        name: {key: float(np.mean([entry[key] for entry in entries])) for key in _KEYS}
        for name, entries in collected.items()
    }


def steps(
    per_recording: Mapping[str, Mapping[str, Mapping[str, float]]], protocol: Protocol
) -> dict[str, dict[str, dict[str, object]]]:
    """Each step's mean difference over recordings, its interval, and whether it contributes.

    A step contributes on a metric when its mean is at least `CONTRIBUTOR_MARGIN_DB` and its
    interval lies above zero.
    """
    result: dict[str, dict[str, dict[str, object]]] = {}
    for name, with_part, without_part in STEPS:
        result[name] = {}
        for metric in STEP_METRICS:
            differences = [
                values[with_part][metric] - values[without_part][metric]
                for values in per_recording.values()
            ]
            low, high = recording_interval(differences, protocol)
            mean = float(np.mean(differences))
            result[name][metric] = {
                "mean_difference_db": mean,
                "interval_db": [low, high],
                "contributor": mean >= CONTRIBUTOR_MARGIN_DB and low > 0.0,
            }
    return result


def _document(
    per_recording: Mapping[str, Mapping[str, Mapping[str, float]]], protocol: Protocol
) -> bytes:
    """The results file: the rule's constants, per-recording means, and the steps."""
    document = {
        "rule": "docs/model-v2-gain-steps.md",
        "snr_grid_db": list(SNR_GRID_DB),
        "contributor_margin_db": CONTRIBUTOR_MARGIN_DB,
        "per_recording": per_recording,
        "steps": steps(per_recording, protocol),
    }
    return (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()


def run_measurement(args: argparse.Namespace) -> bytes:
    """Load everything, measure every demo recording, and return the results file's bytes."""
    protocol = load_protocol()
    admissions = demo_admissions(args.walk_dir, protocol)
    with tempfile.TemporaryDirectory(prefix="gain-steps-") as scratch:
        opgan, _ = build_production_systems(
            protocol, Path(scratch), OPGAN_CHECKPOINT, args.uvr_model_dir
        )
        run = StepsRun(protocol=protocol, system=opgan, scratch=Path(scratch))
        per_recording = {a.identifier: measure_recording(run, a) for a in admissions}
    return _document(per_recording, protocol)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement and write its results directory, or report why it could not."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    for name in ("--walk-dir", "--uvr-model-dir", "--out-dir"):
        parser.add_argument(name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        print(f"gain_steps: {args.out_dir} exists; the output directory must be new")
        return 1
    try:
        raw = run_measurement(args)
    except StepsError as error:
        logger.error("Measurement refused: %s", error)
        print(f"gain_steps: {error}")
        return 1
    write_directory_atomically(args.out_dir, {_RESULTS_NAME: raw})
    print(f"gain_steps: wrote {args.out_dir / _RESULTS_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
