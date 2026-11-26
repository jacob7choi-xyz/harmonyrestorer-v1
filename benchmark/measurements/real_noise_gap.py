r"""Model v2 Phase 0: is v1 measurably worse on real 78 noise than on its own synthetic noise?

The rule is fixed in `docs/model-v2-phase0.md` before this runs. Clean audio is the benchmark's
demo partition (6 CC0 solo piano recordings), at training level as the systems stage sets it.
From each recording, up to `FRAMES_PER_RECORDING` eligible frames are taken at evenly spaced
positions. Each frame gets three noises, each added and scaled to every SNR in `SNR_GRID_DB`:

- `synthetic`: v1's own training noise (hiss, crackle, hum) as `benchmark.degradation` makes
  it, with the parameters drawn from the protocol's ranges and the low-pass and saturation
  switched off, so that all three conditions are added noise only;
- `great78`: noise-only segments from the held-out evaluation discs of the frozen Aalto split;
- `freesound`: three CC0 78 rpm surface-noise recordings from other transfer chains.

v1 restores every mixture through the same in-process engine the benchmark uses, and the input
and the output are scored against the clean frame with the benchmark's `score_output`. The
gate passes when the mean over recordings of (SI-SNR improvement on `synthetic` minus that on
`great78`) is at least `GATE_MARGIN_DB`, and the recording bootstrap's 95% interval of that
difference lies above zero. `freesound` is reported and does not decide the gate.

Run from the repository root:

    PYTHONPATH=backend uv run --frozen python -m benchmark.measurements.real_noise_gap \\
        --walk-dir DIR --aalto-dir DIR --freesound-dir DIR --uvr-model-dir DIR --out-dir NEW_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import struct
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf

from benchmark.atomic_write import write_directory_atomically
from benchmark.canonicalisation import CanonicalWaveform
from benchmark.degradation import apply_degradation, draw_params
from benchmark.eligibility import frame_floors
from benchmark.eligibility_outcomes import frame_verdicts
from benchmark.frame_scoring import FrameScore, degrade_with_reference, score_output
from benchmark.framing import FramedWaveform, frame
from benchmark.measurements.real_noise_split import disc_number, load_split, recording_dirs
from benchmark.protocol import Protocol, load_protocol
from benchmark.run_systems import OPGAN_CHECKPOINT, build_production_systems
from benchmark.runner import CanonicalFile
from benchmark.systems_pipeline import Admission, load_walk_manifest
from benchmark.systems_stage import (
    FrameIdentity,
    System,
    SystemOutput,
    call_system,
    write_frame_input,
)

__all__ = [
    "GateResult",
    "demo_admissions",
    "gate",
    "noise_excerpt",
    "recording_interval",
    "scale_to_snr",
    "synthetic_noise",
]

logger = logging.getLogger(__name__)

MEASUREMENT_SEED = 20261126
SNR_GRID_DB = (5.0, 10.0, 15.0, 20.0)
FRAMES_PER_RECORDING = 40
CROSSFADE_S = 0.05
GATE_MARGIN_DB = 3.0
GATE_METRIC = "si_snr"
CONDITIONS = ("synthetic", "great78", "freesound")
_CONDITION_CODES = {name: index for index, name in enumerate(CONDITIONS)}
_RESULTS_NAME = "real_noise_gap.json"


class GapError(RuntimeError):
    """Raised when the measurement cannot run as the rule fixes it; nothing is written."""


@dataclass(frozen=True)
class GateResult:
    """The gate statistic: mean difference over recordings, its interval, and the verdict."""

    mean_difference_db: float
    interval_db: tuple[float, float]
    passed: bool


def scale_to_snr(
    clean: npt.NDArray[np.float64], noise: npt.NDArray[np.float64], snr_db: float
) -> npt.NDArray[np.float64]:
    """The noise scaled so that clean power over noise power is `snr_db` decibels.

    Raises:
        GapError: If either signal has zero power.
    """
    clean_power = float(np.mean(clean**2))
    noise_power = float(np.mean(noise**2))
    if clean_power <= 0 or noise_power <= 0:
        raise GapError("cannot set an SNR with a silent clean frame or a silent noise")
    scaled: npt.NDArray[np.float64] = noise * np.sqrt(
        clean_power / (noise_power * 10 ** (snr_db / 10))
    )
    return scaled


def noise_excerpt(
    pieces: Sequence[npt.NDArray[np.float64]],
    length: int,
    crossfade: int,
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """`length` samples of noise, from pieces joined in a random order with linear crossfades.

    Pieces are drawn with replacement and joined until the run is long enough, then a random
    window of `length` samples is cut from it.

    Raises:
        GapError: If there are no pieces, or a piece is not longer than the crossfade.
    """
    if not pieces or any(piece.size <= crossfade for piece in pieces):
        raise GapError("noise pieces must exist and each be longer than the crossfade")
    run = pieces[int(rng.integers(len(pieces)))].copy()
    ramp = np.linspace(0.0, 1.0, crossfade, endpoint=False)
    while run.size < length:
        piece = pieces[int(rng.integers(len(pieces)))]
        blended = run[-crossfade:] * (1.0 - ramp) + piece[:crossfade] * ramp
        run = np.concatenate([run[:-crossfade], blended, piece[crossfade:]])
    start = int(rng.integers(run.size - length + 1))
    excerpt: npt.NDArray[np.float64] = run[start : start + length]
    return excerpt


def synthetic_noise(
    clean: npt.NDArray[np.float64], protocol: Protocol, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """v1's training noise for one frame at training level, added noise only.

    The parameters are drawn from the protocol's ranges as training drew them; the low-pass is
    moved to the Nyquist rate and saturation is switched off, and the noise is the degraded
    frame minus the frame.
    """
    rate = protocol.canonicalisation.resample.target_sample_rate
    params = replace(
        draw_params(rng, protocol.degradation.ranges),
        rolloff_hz=float(rate) / 2,
        enable_saturation=False,
    )
    degraded = apply_degradation(clean.astype(np.float32), rate, params, rng)
    noise: npt.NDArray[np.float64] = degraded.astype(np.float64) - clean.astype(np.float32)
    return noise


def recording_interval(values: Sequence[float], protocol: Protocol) -> tuple[float, float]:
    """The 95% bootstrap interval of the mean of per-recording values.

    The bootstrap uses the protocol's seed and iteration count and resamples recordings with
    replacement; the interval is the 2.5th to 97.5th percentile of the resampled means.
    """
    array = np.asarray(values, dtype=np.float64)
    config = protocol.aggregation.bootstrap
    generator = np.random.Generator(np.random.PCG64(config.seed))
    draws = generator.integers(0, array.size, size=(config.iterations, array.size))
    means = array[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5], method="linear")
    return float(low), float(high)


def gate(differences: Sequence[float], protocol: Protocol) -> GateResult:
    """The gate over per-recording differences, with `recording_interval` over recordings."""
    low, high = recording_interval(differences, protocol)
    mean = float(np.asarray(differences, dtype=np.float64).mean())
    return GateResult(
        mean_difference_db=mean,
        interval_db=(low, high),
        passed=mean >= GATE_MARGIN_DB and low > 0.0,
    )


def _mono_at(path: Path, rate: int) -> npt.NDArray[np.float64]:
    """A noise file as float64 mono at `rate`, with its mean removed."""
    samples, source_rate = sf.read(str(path), dtype="float64", always_2d=True)
    mono = samples.mean(axis=1)
    if source_rate != rate:
        mono = librosa.resample(mono, orig_sr=source_rate, target_sr=rate)
    centred: npt.NDArray[np.float64] = mono - mono.mean()
    return centred


Pieces = list[npt.NDArray[np.float64]]


@dataclass(frozen=True)
class NoiseBanks:
    """The real noise: Great 78 evaluation segments grouped by disc, and the CC0 clips."""

    great78: list[Pieces]
    freesound: Pieces


def great78_pieces(aalto_dir: Path, rate: int) -> list[Pieces]:
    """The evaluation discs' segments, grouped by disc, each disc's segments in name order.

    Raises:
        GapError: If an evaluation disc has no segment file.
    """
    evaluation = set(load_split().evaluation)
    by_disc: dict[str, list[Path]] = {}
    for name, folder in recording_dirs(aalto_dir).items():
        disc = disc_number(name)
        if disc in evaluation:
            files = sorted(path for path in folder.rglob("*") if path.is_file())
            by_disc.setdefault(disc, []).extend(files)
    if set(by_disc) != evaluation:
        raise GapError("some evaluation discs have no segment files")
    return [[_mono_at(path, rate) for path in by_disc[disc]] for disc in sorted(by_disc)]


def freesound_pieces(freesound_dir: Path, rate: int) -> Pieces:
    """The CC0 clips listed in the folder's manifest, each checked against its digest.

    Raises:
        GapError: If a clip's bytes do not match its manifest digest.
    """
    manifest = json.loads((freesound_dir / "manifest.json").read_text())
    pieces = []
    for clip in manifest["clips"]:
        path = freesound_dir / clip["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != clip["sha256"]:
            raise GapError(f"{path} does not match its manifest digest")
        pieces.append(_mono_at(path, rate))
    return pieces


def _seed(canonical_sha256: str, index: int, condition: str) -> np.random.Generator:
    """One generator per recording, frame and condition.

    A frame's noise is then the same at every SNR, and only its scale changes.
    """
    words = struct.unpack(">8I", bytes.fromhex(canonical_sha256))
    entropy = [MEASUREMENT_SEED, *words, index, _CONDITION_CODES[condition]]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def frame_indices(
    admission: Admission, protocol: Protocol
) -> tuple[CanonicalWaveform, FramedWaveform, list[int]]:
    """A recording's waveform, its frames, and its measured frame indices.

    The indices are up to `FRAMES_PER_RECORDING` eligible frames at evenly spaced positions.
    """
    canonical = CanonicalFile(
        path=admission.canonical_path,
        sha256=admission.canonical_sha256,
        samples=admission.samples,
        raw_sha256=admission.raw_sha256,
        decoded_pcm_sha256=admission.decoded_pcm_sha256,
    )
    waveform = CanonicalWaveform(samples=canonical.load(), sha256=admission.canonical_sha256)
    framed = frame(waveform, protocol.framing)
    verdicts = frame_verdicts(framed, frame_floors(framed, protocol.eligibility), protocol.metrics)
    eligible = [index for index, verdict in enumerate(verdicts) if verdict.eligible]
    if len(eligible) <= FRAMES_PER_RECORDING:
        return waveform, framed, eligible
    positions = np.linspace(0, len(eligible) - 1, FRAMES_PER_RECORDING).round().astype(int)
    return waveform, framed, [eligible[int(position)] for position in positions]


@dataclass(frozen=True)
class MeasurementRun:
    """What every frame needs: the protocol, v1, the noise banks and a scratch folder."""

    protocol: Protocol
    system: System
    banks: NoiseBanks
    scratch: Path


def condition_noise(
    run: MeasurementRun,
    condition: str,
    reference: npt.NDArray[np.float64],
    rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """One condition's noise for one frame, before scaling."""
    if condition == "synthetic":
        return synthetic_noise(reference, run.protocol, rng)
    crossfade = int(CROSSFADE_S * run.protocol.canonicalisation.resample.target_sample_rate)
    if condition == "great78":
        disc = run.banks.great78[int(rng.integers(len(run.banks.great78)))]
        return noise_excerpt(disc, reference.size, crossfade, rng)
    return noise_excerpt(run.banks.freesound, reference.size, crossfade, rng)


def _value(score: object, metric: str) -> float:
    """A metric's value from a `FrameScore`.

    Raises:
        GapError: If conditioning refused the pair.
    """
    if not isinstance(score, FrameScore):
        raise GapError(f"conditioning refused a pair: {score}")
    return float(score.results[metric].value)


def score_mixture(
    run: MeasurementRun,
    identity: FrameIdentity,
    reference: npt.NDArray[np.float64],
    mixture: npt.NDArray[np.float64],
) -> dict[str, float]:
    """v1's SI-SNR improvement and its output's log-spectral distance for one mixture.

    Raises:
        GapError: If v1 fails on the mixture.
    """
    rate = run.protocol.canonicalisation.resample.target_sample_rate
    path = run.scratch / "input.wav"
    write_frame_input(mixture.astype(np.float32), path, rate)
    marker = run.scratch / "marker.json"
    result = call_system(run.system, path, identity=identity, marker_path=marker, sample_rate=rate)
    if not isinstance(result, SystemOutput):
        raise GapError(f"v1 failed on {identity}: {result.reason}")
    before = score_output(reference, mixture, run.protocol)
    after = score_output(reference, result.samples, run.protocol)
    return {
        "si_snr_improvement_db": _value(after, GATE_METRIC) - _value(before, GATE_METRIC),
        "lsd_output": _value(after, "log_spectral_distance"),
    }


def _frame_values(
    run: MeasurementRun, admission: Admission, index: int, reference: npt.NDArray[np.float64]
) -> dict[str, list[dict[str, float]]]:
    """Every condition's scores on one frame, one entry per SNR."""
    identity = FrameIdentity(admission.canonical_sha256, index)
    values: dict[str, list[dict[str, float]]] = {}
    for condition in CONDITIONS:
        rng = _seed(admission.canonical_sha256, index, condition)
        noise = condition_noise(run, condition, reference, rng)
        values[condition] = [
            score_mixture(run, identity, reference, reference + scale_to_snr(reference, noise, snr))
            for snr in SNR_GRID_DB
        ]
    return values


def measure_recording(run: MeasurementRun, admission: Admission) -> dict[str, dict[str, float]]:
    """Per condition, the mean SI-SNR improvement and output LSD over frames and SNRs."""
    waveform, framed, indices = frame_indices(admission, run.protocol)
    peak = float(np.max(np.abs(waveform.samples)))
    collected: dict[str, list[dict[str, float]]] = {name: [] for name in CONDITIONS}
    for index in indices:
        _, reference = degrade_with_reference(
            framed.frames[index], peak, admission.canonical_sha256, index, run.protocol
        )
        for condition, entries in _frame_values(run, admission, index, reference).items():
            collected[condition].extend(entries)
    logger.info("Measured %s on %d frames", admission.identifier, len(indices))
    return {
        name: {
            key: float(np.mean([entry[key] for entry in entries]))
            for key in ("si_snr_improvement_db", "lsd_output")
        }
        for name, entries in collected.items()
    }


def demo_admissions(walk_dir: Path, protocol: Protocol) -> list[Admission]:
    """The walk's demo admissions, in walk order.

    Raises:
        GapError: If the walk has no demo admission.
    """
    manifest = load_walk_manifest(walk_dir / "walk_manifest.json", protocol.source_sha256)
    demo = [admission for admission in manifest.admissions if admission.partition.value == "demo"]
    if not demo:
        raise GapError("the walk manifest has no demo admission")
    return demo


def _document(
    per_recording: Mapping[str, Mapping[str, Mapping[str, float]]], result: GateResult
) -> bytes:
    """The results file: the rule's constants, per-recording means, and the gate."""
    document = {
        "rule": "docs/model-v2-phase0.md",
        "seed": MEASUREMENT_SEED,
        "snr_grid_db": list(SNR_GRID_DB),
        "frames_per_recording": FRAMES_PER_RECORDING,
        "gate_margin_db": GATE_MARGIN_DB,
        "per_recording": per_recording,
        "gate": {
            "statistic": "mean over recordings of synthetic minus great78 SI-SNR improvement",
            "mean_difference_db": result.mean_difference_db,
            "interval_db": list(result.interval_db),
            "passed": result.passed,
        },
    }
    return (json.dumps(document, indent=1, sort_keys=True) + "\n").encode()


def run_measurement(args: argparse.Namespace) -> bytes:
    """Load everything, measure every demo recording, and return the results file's bytes."""
    protocol = load_protocol()
    rate = protocol.canonicalisation.resample.target_sample_rate
    banks = NoiseBanks(
        great78=great78_pieces(args.aalto_dir, rate),
        freesound=freesound_pieces(args.freesound_dir, rate),
    )
    admissions = demo_admissions(args.walk_dir, protocol)
    with tempfile.TemporaryDirectory(prefix="real-noise-gap-") as scratch:
        opgan, _ = build_production_systems(
            protocol, Path(scratch), OPGAN_CHECKPOINT, args.uvr_model_dir
        )
        run = MeasurementRun(protocol=protocol, system=opgan, banks=banks, scratch=Path(scratch))
        per_recording = {a.identifier: measure_recording(run, a) for a in admissions}
    differences = [
        values["synthetic"]["si_snr_improvement_db"] - values["great78"]["si_snr_improvement_db"]
        for values in per_recording.values()
    ]
    return _document(per_recording, gate(differences, protocol))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement and write its results directory, or report why it could not."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    for name in ("--walk-dir", "--aalto-dir", "--freesound-dir", "--uvr-model-dir", "--out-dir"):
        parser.add_argument(name, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out_dir.exists():
        print(f"real_noise_gap: {args.out_dir} exists; the output directory must be new")
        return 1
    try:
        raw = run_measurement(args)
    except GapError as error:
        logger.error("Measurement refused: %s", error)
        print(f"real_noise_gap: {error}")
        return 1
    write_directory_atomically(args.out_dir, {_RESULTS_NAME: raw})
    print(f"real_noise_gap: wrote {args.out_dir / _RESULTS_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
