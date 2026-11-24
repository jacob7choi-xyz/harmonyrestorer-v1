"""Fixtures for the post-hoc command: a solo comparison stratum scored by the systems stage itself.

`report_fixtures.write_base` gives a walk directory with a solo piano stratum that the report
publishes as a comparison, and a short chamber stratum it withholds. Its systems results are
synthetic. This module replaces the solo results with the rows `score_admission` produces, run
with two fake engines that write real 16-bit WAV files through the backend's writer. Every recorded
value and capped flag in the bundle therefore comes from the stage, so the post-hoc consistency
check compares the command against what the stage actually recorded.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from app.services.wav_io import write_wav_atomic
from report_fixtures import (
    SOLO,
    Bundle,
    bind,
    copy_walk,
    population_of,
    read_json,
    read_lines,
    write_base,
    write_lines,
)

from benchmark.protocol import Protocol
from benchmark.report import MANIFEST_NAME, RESULTS_NAME
from benchmark.scoring import OPGAN, UVR
from benchmark.systems_pipeline import (
    Admission,
    Partition,
    StageContext,
    record_to_json,
    score_admission,
)

__all__ = ["build_bundle", "copy_bundle", "update_results"]

# Each fake engine scales its input by a fixed gain, so the two outputs differ from each other.
OPGAN_GAIN = 0.9
UVR_GAIN = 0.7
# One least significant bit of 16-bit PCM. A file holding only this sample is not silent.
ONE_LSB = 1.0 / 32768.0
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
PROGRESS_NAME = "progress.json"


@dataclass(frozen=True)
class ScalingEngine:
    """A fake engine: scales its input, except for the stems it is told to silence or quieten."""

    name: str
    output_dir: Path
    suffix: str
    gain: float
    sample_rate: int
    zero_stems: frozenset[str] = frozenset()
    lsb_stems: frozenset[str] = frozenset()

    def denoise(self, input_path: Path) -> Path:
        """Write the scaled input as a 16-bit WAV beside the other outputs, and return its path."""
        samples, _ = sf.read(str(input_path), dtype="float64", always_2d=False)
        stem = input_path.stem
        if stem in self.zero_stems:
            output = np.zeros_like(samples)
        elif stem in self.lsb_stems:
            output = np.zeros_like(samples)
            output[0] = ONE_LSB
        else:
            output = samples * self.gain
        path = self.output_dir / f"{stem}{self.suffix}.wav"
        write_wav_atomic(path, output, self.sample_rate)
        return path


def _solo_test_admissions(bundle: Bundle) -> list[Admission]:
    """The solo test recordings, as the walk manifest lists them, in admission order."""
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    return [
        Admission(
            stratum=row["stratum"],
            partition=Partition(row["partition"]),
            position=row["position"],
            identifier=row["identifier"],
            cluster=row["cluster"],
            canonical_path=Path(row["canonical_path"]),
            canonical_sha256=row["canonical_sha256"],
            samples=row["samples"],
            raw_sha256=row["raw_sha256"],
            decoded_pcm_sha256=row["decoded_pcm_sha256"],
        )
        for row in manifest["admissions"]
        if row["stratum"] == SOLO and row["partition"] == Partition.TEST.value
    ]


def _eligible_stems(bundle: Bundle, protocol: Protocol) -> list[str]:
    """Every eligible frame's stem, in the order the stage scores them."""
    return [
        f"{admission.canonical_sha256}-{index:06d}"
        for admission in _solo_test_admissions(bundle)
        for index in population_of(protocol, admission).eligible
    ]


def _replace_solo_rows(bundle: Bundle, records: Sequence[dict[str, object]]) -> None:
    """Keep every row that is not solo, add the stage's solo rows, and bind the summary."""
    kept = [row for row in read_lines(bundle.systems / RESULTS_NAME) if row["stratum"] != SOLO]
    write_lines(bundle.systems / RESULTS_NAME, [*kept, *records])
    bind(bundle)


def build_bundle(
    root: Path, protocol: Protocol, *, zero_opgan: int = 0, lsb_opgan: int = 0
) -> Bundle:
    """A walk and a systems directory whose solo results the systems stage itself wrote.

    The first `zero_opgan` OpGAN frames in stage order write an all-zero output, and the next
    `lsb_opgan` write an output holding one least significant bit. All other frames get scaled
    copies of their input.
    """
    bundle = write_base(protocol, root)
    stems = _eligible_stems(bundle, protocol)
    rate = protocol.conditioning.sample_rate
    engines = (
        ScalingEngine(
            OPGAN,
            bundle.systems / OUTPUTS_DIR / OPGAN,
            "_opgan",
            OPGAN_GAIN,
            rate,
            zero_stems=frozenset(stems[:zero_opgan]),
            lsb_stems=frozenset(stems[zero_opgan : zero_opgan + lsb_opgan]),
        ),
        ScalingEngine(UVR, bundle.systems / OUTPUTS_DIR / UVR, "_uvr", UVR_GAIN, rate),
    )
    for engine in engines:
        engine.output_dir.mkdir(parents=True)
    (bundle.systems / INPUTS_DIR).mkdir()
    context = StageContext(
        protocol=protocol,
        systems=engines,
        input_dir=bundle.systems / INPUTS_DIR,
        marker_path=bundle.systems / PROGRESS_NAME,
    )
    records: list[dict[str, object]] = []
    for admission in _solo_test_admissions(bundle):
        result = score_admission(admission, context)
        records.extend(record_to_json(record) for record in result.records)
    _replace_solo_rows(bundle, records)
    return bundle


def update_results(bundle: Bundle, change: Callable[[dict[str, Any]], None]) -> None:
    """Apply a change to every row of the results, then bind the summary to the new bytes."""
    rows = read_lines(bundle.systems / RESULTS_NAME)
    for row in rows:
        change(row)
    write_lines(bundle.systems / RESULTS_NAME, rows)
    bind(bundle)


def copy_bundle(bundle: Bundle, root: Path) -> Bundle:
    """A copy of the walk and systems directories, so a test can change its own copy."""
    walk = root / "walk"
    systems = root / "systems"
    copy_walk(bundle.walk, walk)
    _copy_tree(bundle.systems, systems)
    copied = Bundle(walk, systems)
    bind(copied)
    return copied


def _copy_tree(source: Path, target: Path) -> None:
    """Copy a directory tree, file by file, into a new location."""
    target.mkdir(parents=True)
    for item in source.rglob("*"):
        destination = target / item.relative_to(source)
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(item.read_bytes())
