"""Run the systems stage over a walk: preconditions, engines, scoring, and written results.

Usage, from the repository root:

    PYTHONPATH=backend uv run --frozen python -m benchmark.run_systems --walk-dir DIR \
        --run-dir NEW_DIR --uvr-model-dir DIR [--opgan-checkpoint PATH]

The run directory must not exist, because runs are never overwritten. The run writes
`systems_results.jsonl` (one line per recording, frame, and system, the baseline included) and
then `systems_summary.json`. A run that aborts writes neither, so a summary is present only for
a run that finished.

Engines are built once, after the run preconditions pass. Each engine loads its model lazily,
inside its first `denoise` call, so before the frame loop the run warms each engine up on a fixed
tone (see `benchmark.systems_pipeline`). A load fault therefore aborts the run as a harness
failure, and only calls on admitted frames are inside the engine boundary.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol as Interface

from app.config import OPGAN_CHECKPOINT_SHA256
from app.services.denoiser import DenoiserService
from app.services.opgan_denoiser import OpGANDenoiserService

from benchmark.atomic_write import write_atomically
from benchmark.golden_fixture import platform_token
from benchmark.protocol import Protocol, load_protocol
from benchmark.run_preconditions import file_sha256, verify_run_preconditions
from benchmark.scoring import OPGAN, UVR
from benchmark.source_outcome import RunAbortOutcome
from benchmark.systems_pipeline import (
    MANIFEST_NAME,
    AdmissionResult,
    StageContext,
    WalkManifest,
    build_summary,
    load_walk_manifest,
    population_to_json,
    record_to_json,
    score_manifest,
    warm_up_all,
)
from benchmark.systems_stage import HarnessFailure, System

__all__ = ["NamedSystem", "build_production_systems", "main", "run_systems"]

logger = logging.getLogger(__name__)

RESULTS_NAME = "systems_results.jsonl"
POPULATION_NAME = "systems_population.jsonl"
SUMMARY_NAME = "systems_summary.json"
PROGRESS_NAME = "progress.json"
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
OPGAN_CHECKPOINT = Path("checkpoints/final.pt")
_SEPARATOR_DIR_ENV = "AUDIO_SEPARATOR_MODEL_DIR"


class Engine(Interface):
    """What a backend engine offers the stage: one `denoise` call per input file."""

    def denoise(self, input_path: Path) -> Path:
        """Denoise the WAV at `input_path` and return the output WAV's path."""
        ...


class NamedSystem:
    """A backend engine under the protocol's system name, as the stage's `System` contract."""

    def __init__(self, name: str, engine: Engine) -> None:
        """Pair the engine with the name its results are recorded under."""
        self.name = name
        self._engine = engine

    def denoise(self, input_path: Path) -> Path:
        """Delegate the call to the engine."""
        return self._engine.denoise(input_path)


SystemsFactory = Callable[[Protocol, Path, Path, Path], Sequence[System]]
PreconditionCheck = Callable[..., RunAbortOutcome | None]


def build_production_systems(
    protocol: Protocol, run_dir: Path, opgan_checkpoint: Path, uvr_model_dir: Path
) -> tuple[System, System]:
    """Build the two backend engines for this run, OpGAN first and UVR second.

    audio-separator reads `AUDIO_SEPARATOR_MODEL_DIR` when its separator is created, which is on
    the first UVR call, so the variable is set here to the directory the preconditions checked.

    Args:
        protocol: The loaded protocol, for the UVR model name.
        run_dir: The run directory; each engine writes its outputs under it.
        opgan_checkpoint: The OpGAN checkpoint the preconditions verified.
        uvr_model_dir: The UVR model directory the preconditions verified.

    Returns:
        The OpGAN and UVR systems.
    """
    os.environ[_SEPARATOR_DIR_ENV] = str(uvr_model_dir)
    outputs = run_dir / OUTPUTS_DIR
    opgan = OpGANDenoiserService(
        output_dir=outputs / OPGAN,
        checkpoint_path=opgan_checkpoint,
        expected_sha256=OPGAN_CHECKPOINT_SHA256,
    )
    uvr = DenoiserService(output_dir=outputs / UVR, model_name=protocol.system_execution.uvr_model)
    logger.info("Built engines %s and %s for run %s", OPGAN, UVR, run_dir)
    return NamedSystem(OPGAN, opgan), NamedSystem(UVR, uvr)


def check_preflight(protocol: Protocol) -> None:
    """Refuse a protocol whose rates or engine pins would charge a setup fault to a system.

    The canonical rate sets the input's rate and the conditioning rate sets the output check, so
    they must agree. The backend checks its own OpGAN pin when it loads, so the protocol's pin
    must equal the backend's. A mismatch could make OpGAN's calls fail, and those failures would
    be charged to OpGAN although the fault is in the setup.

    Raises:
        HarnessFailure: If either pair disagrees.
    """
    canonical_rate = protocol.canonicalisation.resample.target_sample_rate
    if canonical_rate != protocol.conditioning.sample_rate:
        logger.error("Canonical rate %d differs from conditioning rate", canonical_rate)
        raise HarnessFailure(
            f"canonical rate {canonical_rate} differs from the conditioning rate "
            f"{protocol.conditioning.sample_rate}"
        )
    if protocol.system_execution.opgan_checkpoint_sha256 != OPGAN_CHECKPOINT_SHA256:
        logger.error("Protocol and backend name different OpGAN checkpoint digests")
        raise HarnessFailure("protocol and backend name different OpGAN checkpoint digests")


def _require_preconditions(
    protocol: Protocol,
    *,
    runtime_platform: str,
    opgan_checkpoint: Path,
    uvr_model_dir: Path,
    preconditions: PreconditionCheck,
) -> None:
    """Run the run preconditions; any abort stops the run before a candidate is opened."""
    abort = preconditions(
        protocol,
        runtime_platform=runtime_platform,
        opgan_checkpoint=opgan_checkpoint,
        uvr_model_dir=uvr_model_dir,
    )
    if abort is not None:
        logger.error("Run preconditions failed: %s: %s", abort.reason, abort.detail)
        raise HarnessFailure(f"run preconditions failed: {abort.reason}: {abort.detail}")


def _create_run_dir(run_dir: Path) -> None:
    """Create the run directory and its input and output folders, refusing an existing run.

    Raises:
        HarnessFailure: If the directory exists, or cannot be created.
    """
    if run_dir.exists():
        logger.error("Run directory %s already exists; runs are never overwritten", run_dir)
        raise HarnessFailure(f"run directory {run_dir} already exists")
    try:
        (run_dir / INPUTS_DIR).mkdir(parents=True)
        (run_dir / OUTPUTS_DIR).mkdir()
    except OSError as error:
        logger.error("Cannot create run directory %s: %s", run_dir, error)
        raise HarnessFailure(f"run directory {run_dir} cannot be created: {error}") from error


def _write_run(
    run_dir: Path,
    results: Sequence[AdmissionResult],
    manifest: WalkManifest,
    manifest_sha256: str,
) -> dict[str, object]:
    """Write the results, the population, then the summary, each atomically; the summary is last."""
    lines = [
        json.dumps(record_to_json(record), sort_keys=True, allow_nan=False)
        for result in results
        for record in result.records
    ]
    write_atomically(run_dir / RESULTS_NAME, "".join(f"{line}\n" for line in lines).encode())
    population = [
        json.dumps(population_to_json(result), sort_keys=True, allow_nan=False)
        for result in results
    ]
    write_atomically(
        run_dir / POPULATION_NAME, "".join(f"{line}\n" for line in population).encode()
    )
    summary = build_summary(manifest, manifest_sha256, results)
    text = json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n"
    write_atomically(run_dir / SUMMARY_NAME, text.encode("utf-8"))
    logger.info("Wrote %d result lines to %s", len(lines), run_dir / RESULTS_NAME)
    return summary


def _load_manifest(walk_dir: Path, protocol: Protocol) -> tuple[WalkManifest, str]:
    """The validated walk manifest and the digest of its file, which the summary names.

    Raises:
        HarnessFailure: If the manifest is refused or is not a regular file.
    """
    manifest_path = walk_dir / MANIFEST_NAME
    manifest = load_walk_manifest(manifest_path, protocol.source_sha256)
    digest = file_sha256(manifest_path)
    if digest is None:
        raise HarnessFailure(f"walk manifest {manifest_path} is not a regular file")
    return manifest, digest


def _stage_context(protocol: Protocol, systems: Sequence[System], run_dir: Path) -> StageContext:
    """The paths and systems one run scores with."""
    return StageContext(
        protocol=protocol,
        systems=systems,
        input_dir=run_dir / INPUTS_DIR,
        marker_path=run_dir / PROGRESS_NAME,
    )


def run_systems(
    *,
    walk_dir: Path,
    run_dir: Path,
    protocol: Protocol,
    runtime_platform: str,
    opgan_checkpoint: Path,
    uvr_model_dir: Path,
    build_systems: SystemsFactory = build_production_systems,
    preconditions: PreconditionCheck = verify_run_preconditions,
) -> dict[str, object]:
    """Run the systems stage over the walk manifest in `walk_dir`, writing results to `run_dir`.

    Args:
        walk_dir: The directory holding `walk_manifest.json`.
        run_dir: A new directory for this run. It must not exist.
        protocol: The loaded protocol.
        runtime_platform: This runtime's platform token.
        opgan_checkpoint: The OpGAN checkpoint file.
        uvr_model_dir: The directory holding the UVR model and its parameter files.
        build_systems: Builds the two systems after the preconditions pass.
        preconditions: The run-level precondition check.

    Returns:
        The summary that was written.

    Raises:
        HarnessFailure: On any failure outside a system's call. The run stops and no result
            file is left in `run_dir`.
    """
    check_preflight(protocol)
    manifest, manifest_sha256 = _load_manifest(walk_dir, protocol)
    _require_preconditions(
        protocol,
        runtime_platform=runtime_platform,
        opgan_checkpoint=opgan_checkpoint,
        uvr_model_dir=uvr_model_dir,
        preconditions=preconditions,
    )
    _create_run_dir(run_dir)
    systems = build_systems(protocol, run_dir, opgan_checkpoint, uvr_model_dir)
    warm_up_all(systems, run_dir, protocol, run_dir / PROGRESS_NAME)
    results = score_manifest(manifest, _stage_context(protocol, systems, run_dir))
    return _write_run(run_dir, results, manifest, manifest_sha256)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The command line: the walk, the new run directory, and the two model locations."""
    parser = argparse.ArgumentParser(description="Run the benchmark's systems stage.")
    parser.add_argument("--walk-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--uvr-model-dir", type=Path, required=True)
    parser.add_argument("--opgan-checkpoint", type=Path, default=OPGAN_CHECKPOINT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage from the command line. Returns 0 on success and 1 on a harness abort."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    try:
        run_systems(
            walk_dir=args.walk_dir,
            run_dir=args.run_dir,
            protocol=load_protocol(),
            runtime_platform=platform_token(),
            opgan_checkpoint=args.opgan_checkpoint,
            uvr_model_dir=args.uvr_model_dir,
        )
    except HarnessFailure as error:
        print(f"Systems stage aborted: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {args.run_dir / RESULTS_NAME} and {args.run_dir / SUMMARY_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
