r"""A blind listening check: untouched clean input against v1's output on it.

The rule is fixed in `docs/v1-clean-input.md` before any clip is built. From each of the 6 CC0
demo recordings, two 8-second excerpts are cut at fixed positions among the stretches where
four consecutive frames are eligible. Each excerpt is presented untouched (`original`) and as
v1 restored it (`opgan`), at 16 kHz mono, loudness matched with the listening protocol's settings. That gives 12
scored pairs; two catch pairs play one file on both sides. The sides, the catch picks and the
trial order are drawn from a seed taken from the operating system, which is written only to
the sealed key.

Build the clips, then score the listener's answers after they are recorded:

    PYTHONPATH=backend uv run --frozen python -m benchmark.measurements.clean_input_listen \\
        build --walk-dir DIR --uvr-model-dir DIR --out-dir NEW_DIR --sealed-dir NEW_DIR
    PYTHONPATH=backend uv run --frozen python -m benchmark.measurements.clean_input_listen \\
        score --sealed-dir DIR --answers FILE --out-dir NEW_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import logging
import secrets
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf

from benchmark.atomic_write import write_directory_atomically
from benchmark.eligibility import frame_floors
from benchmark.eligibility_outcomes import frame_verdicts
from benchmark.listening.build import to_pcm16
from benchmark.listening.loudness import match_clip
from benchmark.listening.protocol import Loudness, load_listening_protocol
from benchmark.measurements.real_noise_gap import demo_admissions, frame_indices
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

__all__ = ["decide", "excerpt_starts", "render_pair", "score_answers", "trial_plan"]

logger = logging.getLogger(__name__)

EXCERPT_FRAMES = 4
CONTEXT_FRAMES = 1
EXCERPT_POSITIONS = (1 / 3, 2 / 3)
CATCH_TRIALS = 2
PREFERENCE_THRESHOLD = 10
CONDITIONS = ("original", "opgan")
ANSWERS = ("A", "B", "N")
_KEY_NAME = "key.json"
_BUILD_NAME = "build.json"
_RESULT_NAME = "clean_input_listen.json"


class ListenError(RuntimeError):
    """Raised when the clips or the answers do not meet the rule; nothing is written."""


def excerpt_starts(eligible: Sequence[bool]) -> list[int]:
    """The first frame of each excerpt, at fixed positions among fully eligible stretches.

    A candidate is a frame index whose next `EXCERPT_FRAMES` frames are all eligible.

    Raises:
        ListenError: If two picked excerpts would overlap.
    """
    candidates = [
        index
        for index in range(len(eligible) - EXCERPT_FRAMES + 1)
        if all(eligible[index : index + EXCERPT_FRAMES])
    ]
    if not candidates:
        raise ListenError("no stretch of eligible frames is long enough for an excerpt")
    picks = [candidates[round((len(candidates) - 1) * q)] for q in EXCERPT_POSITIONS]
    if any(b - a < EXCERPT_FRAMES for a, b in itertools.pairwise(picks)):
        raise ListenError(f"excerpts starting at frames {picks} would overlap")
    return picks


@dataclass(frozen=True)
class Excerpt:
    """One excerpt at training level, with up to `CONTEXT_FRAMES` frames of context each side.

    The excerpt itself is `padded[offset : offset + length]`.
    """

    start_frame: int
    padded: npt.NDArray[np.float64]
    offset: int
    length: int


def excerpt_window(
    samples: npt.NDArray[np.float64], start: int, width: int, gain: float
) -> Excerpt:
    """The excerpt starting at frame `start`, with context cut at the waveform's ends."""
    first = max(0, (start - CONTEXT_FRAMES) * width)
    last = min(samples.size, (start + EXCERPT_FRAMES + CONTEXT_FRAMES) * width)
    return Excerpt(start, samples[first:last] * gain, start * width - first, EXCERPT_FRAMES * width)


def _excerpts(admission: Admission, protocol: Protocol) -> list[Excerpt]:
    """Each excerpt of the recording, at training level with its context."""
    waveform, framed, _ = frame_indices(admission, protocol)
    floors = frame_floors(framed, protocol.eligibility)
    eligible = [v.eligible for v in frame_verdicts(framed, floors, protocol.metrics)]
    gain = protocol.degradation.ranges.headroom / float(np.max(np.abs(waveform.samples)))
    samples = np.asarray(waveform.samples, dtype=np.float64)
    width = framed.frames.shape[1]
    return [excerpt_window(samples, start, width, gain) for start in excerpt_starts(eligible)]


def render_pair(
    original: npt.NDArray[np.float64],
    restored: npt.NDArray[np.float64],
    rate: int,
    rule: Loudness,
) -> tuple[dict[str, npt.NDArray[np.int16]], dict[str, float]]:
    """Both conditions trimmed to a common length, loudness matched by `rule`, as 16-bit samples.

    Returns:
        Each condition's 16-bit samples, and each condition's applied gain.
    """
    length = min(original.size, restored.size)
    windows = {"original": original[:length], "opgan": restored[:length]}
    match = match_clip(windows, rate, rule)
    pcm = {name: to_pcm16(window * match.gains[name], name) for name, window in windows.items()}
    return pcm, dict(match.gains)


def trial_plan(excerpts: Sequence[str], rng: np.random.Generator) -> list[dict[str, Any]]:
    """Every trial, numbered from 1 in a random order: a scored pair per excerpt, then catches.

    A scored trial puts each condition on a random side. A catch trial plays one random
    condition of one random excerpt on both sides; the catch excerpts differ.
    """
    trials: list[dict[str, Any]] = []
    for excerpt in excerpts:
        first = CONDITIONS[int(rng.integers(2))]
        second = CONDITIONS[1] if first == CONDITIONS[0] else CONDITIONS[0]
        trials.append({"kind": "scored", "excerpt": excerpt, "A": first, "B": second})
    for pick in rng.choice(len(excerpts), size=CATCH_TRIALS, replace=False):
        condition = CONDITIONS[int(rng.integers(2))]
        trials.append(
            {"kind": "catch", "excerpt": excerpts[int(pick)], "A": condition, "B": condition}
        )
    order = rng.permutation(len(trials))
    return [{"trial": slot + 1, **trials[int(source)]} for slot, source in enumerate(order)]


def score_answers(
    trials: Sequence[Mapping[str, Any]], answers: Mapping[str, str]
) -> dict[str, Any]:
    """Counts of the scored trials by preferred condition, and every catch trial's answer.

    Raises:
        ListenError: If an answer is missing, extra, or not one of `ANSWERS`.
    """
    numbers = {str(trial["trial"]) for trial in trials}
    if set(answers) != numbers or any(answer not in ANSWERS for answer in answers.values()):
        raise ListenError(f"answers must give one of {ANSWERS} for exactly the trials {numbers}")
    counts = {"original": 0, "opgan": 0, "no_preference": 0}
    catches = []
    for trial in trials:
        answer = answers[str(trial["trial"])]
        if trial["kind"] == "catch":
            catches.append({"trial": trial["trial"], "answer": answer})
        else:
            counts[trial[answer] if answer != "N" else "no_preference"] += 1
    return {"scored_counts": counts, "catch_answers": catches, "decision": decide(counts)}


def decide(counts: Mapping[str, int]) -> str:
    """The rule's reading: the untouched input is preferred in at least the threshold."""
    if counts["original"] >= PREFERENCE_THRESHOLD:
        return "input_preferred"
    return "input_not_preferred"


def _wav_bytes(pcm: npt.NDArray[np.int16], rate: int) -> bytes:
    """A 16-bit PCM WAV."""
    buffer = io.BytesIO()
    sf.write(buffer, pcm, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _restore(
    system: System,
    scratch: Path,
    identity: FrameIdentity,
    samples: npt.NDArray[np.float64],
    rate: int,
) -> npt.NDArray[np.float64]:
    """v1's output on one excerpt.

    Raises:
        ListenError: If v1 fails on the excerpt.
    """
    path = scratch / "input.wav"
    write_frame_input(samples.astype(np.float32), path, rate)
    result = call_system(
        system, path, identity=identity, marker_path=scratch / "marker.json", sample_rate=rate
    )
    if not isinstance(result, SystemOutput):
        raise ListenError(f"v1 failed on {identity}: {result.reason}")
    return np.asarray(result.samples, dtype=np.float64)


@dataclass(frozen=True)
class RenderedExcerpt:
    """One excerpt's two conditions as 16-bit samples, and where it came from."""

    name: str
    recording: str
    start_frame: int
    pcm: Mapping[str, npt.NDArray[np.int16]]
    gains: Mapping[str, float]


def render_excerpts(
    system: System,
    scratch: Path,
    admissions: Sequence[Admission],
    protocol: Protocol,
    rule: Loudness,
) -> list[RenderedExcerpt]:
    """Every excerpt of every recording, rendered in both conditions."""
    rate = protocol.canonicalisation.resample.target_sample_rate
    rendered = []
    for number, admission in enumerate(admissions):
        for excerpt in _excerpts(admission, protocol):
            identity = FrameIdentity(admission.canonical_sha256, excerpt.start_frame)
            restored = _restore(system, scratch, identity, excerpt.padded, rate)
            middle = slice(excerpt.offset, excerpt.offset + excerpt.length)
            pcm, gains = render_pair(excerpt.padded[middle], restored[middle], rate, rule)
            name = f"r{number + 1}_f{excerpt.start_frame}"
            rendered.append(
                RenderedExcerpt(name, admission.identifier, excerpt.start_frame, pcm, gains)
            )
    return rendered


def package(
    rendered: Sequence[RenderedExcerpt], seed: int, rate: int
) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """The listener's files (two WAVs per trial and a build record) and the sealed key."""
    by_name = {excerpt.name: excerpt for excerpt in rendered}
    trials = trial_plan(list(by_name), np.random.Generator(np.random.PCG64(seed)))
    files: dict[str, bytes] = {}
    for trial in trials:
        for side in ("A", "B"):
            pcm = by_name[trial["excerpt"]].pcm[trial[side]]
            files[f"trial_{trial['trial']:02d}_{side}.wav"] = _wav_bytes(pcm, rate)
    record = {
        "rule": "docs/v1-clean-input.md",
        "trials": len(trials),
        "answers": "A, B or N (no preference) for each trial number",
        "files": sorted(files),
    }
    files[_BUILD_NAME] = (json.dumps(record, indent=1, sort_keys=True) + "\n").encode()
    key = {
        "seed": seed,
        "trials": trials,
        "file_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
        "excerpts": {
            e.name: {"recording": e.recording, "start_frame": e.start_frame, "gains": e.gains}
            for e in rendered
        },
    }
    return files, {_KEY_NAME: (json.dumps(key, indent=1, sort_keys=True) + "\n").encode()}


def build(args: argparse.Namespace) -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Render every excerpt with v1 and package the trials under a fresh secret seed."""
    protocol = load_protocol()
    admissions = demo_admissions(args.walk_dir, protocol)
    with tempfile.TemporaryDirectory(prefix="clean-input-listen-") as scratch:
        opgan, _ = build_production_systems(
            protocol, Path(scratch), OPGAN_CHECKPOINT, args.uvr_model_dir
        )
        rule = load_listening_protocol().loudness
        rendered = render_excerpts(opgan, Path(scratch), admissions, protocol, rule)
    rate = protocol.canonicalisation.resample.target_sample_rate
    return package(rendered, secrets.randbits(63), rate)


def _parser() -> argparse.ArgumentParser:
    """The command line: `build` and `score`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)
    build_command = commands.add_parser("build")
    for name in ("--walk-dir", "--uvr-model-dir", "--out-dir", "--sealed-dir"):
        build_command.add_argument(name, type=Path, required=True)
    score_command = commands.add_parser("score")
    for name in ("--sealed-dir", "--answers", "--out-dir"):
        score_command.add_argument(name, type=Path, required=True)
    return parser


def _run(args: argparse.Namespace) -> dict[Path, dict[str, bytes]]:
    """The directories to write for the chosen command.

    Raises:
        ListenError: If the clips or the answers do not meet the rule.
    """
    if args.command == "build":
        files, sealed = build(args)
        return {args.out_dir: files, args.sealed_dir: sealed}
    key = json.loads((args.sealed_dir / _KEY_NAME).read_text())
    answers = json.loads(args.answers.read_text())
    result = score_answers(key["trials"], answers)
    raw = (json.dumps(result, indent=1, sort_keys=True) + "\n").encode()
    return {args.out_dir: {_RESULT_NAME: raw}}


def main(argv: Sequence[str] | None = None) -> int:
    """Build the clips or score the answers, writing only to new directories."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    targets = [args.out_dir] + ([args.sealed_dir] if args.command == "build" else [])
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        print(f"clean_input_listen: {', '.join(existing)} exists; output directories must be new")
        return 1
    try:
        outputs = _run(args)
    except ListenError as error:
        logger.error("Refused: %s", error)
        print(f"clean_input_listen: {error}")
        return 1
    for directory, files in outputs.items():
        write_directory_atomically(directory, files)
        print(f"clean_input_listen: wrote {len(files)} files to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
