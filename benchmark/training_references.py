"""The training recordings as the duplicate screen compares against them, built at walk time.

Protocol v11 does not say where the training side's features come from. They are computed
again on every walk run rather than cached, so no stored file can go stale or be altered
between runs: each training file goes through the candidates' worker
(`benchmark.runner.run_source`) with its manifest digest as the expected digest, its canonical
waveform must have the canonical digest, sample count and decoded-PCM digest committed in
`benchmark/manifests/training_decoded_pcm.json`, and it is read back with
`CanonicalFile.load` before its features are computed. The work directory of each file is
deleted once its features are kept.

Measured on the development Mac: canonicalising the 145 files through the worker took about
4 minutes in each of two runs; computing features took 0.03 s for the shortest training file
and 0.31 s for the longest, the only two timed. By the frame count, 1 + samples // 512, the 145
feature sets hold 300,173,600 bytes. A process that computed features for 16,818,119 samples of
synthetic noise, the longest training file's length, reached a lifetime peak footprint of
1,228,998,072 bytes, which includes the interpreter, its imports and the input array.

What this module relies on and does not check itself:

- The golden fixtures, which the walk's run preconditions verify before any worker starts.
- The decoder: `benchmark.training_corpus.load_training_digests` refuses committed digests
  made by another soundfile or libsndfile version.
- A reference's features are not checked for degenerate values, as v11's
  `degenerate_scope` states.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NoReturn

from benchmark.duplicate_screen import RecordingIdentity, Reference, unit_features
from benchmark.protocol import Protocol
from benchmark.runner import (
    Candidate,
    CanonicalFile,
    CanonicalFileError,
    run_source,
    worker_command,
)
from benchmark.training_corpus import DecodedRecord, TrainingRecording

__all__ = [
    "TrainingReferenceError",
    "staged_candidates",
    "training_references",
]

logger = logging.getLogger(__name__)

WorkerCommand = Callable[[Candidate, Path], list[str]]


class TrainingReferenceError(RuntimeError):
    """Raised when a training file cannot be staged or does not reproduce its digests."""


def _refuse(message: str) -> NoReturn:
    """Log and raise *message* as a TrainingReferenceError."""
    logger.error("Training references: %s", message)
    raise TrainingReferenceError(message)


def staged_candidates(training: Sequence[TrainingRecording], raw_dir: Path) -> list[Candidate]:
    """One candidate per training file, its size checked against the manifest first.

    Raises:
        TrainingReferenceError: If *raw_dir* is relative or a file's size differs.
    """
    if not raw_dir.is_absolute():
        _refuse(f"raw directory must be absolute, got {raw_dir}")
    candidates = []
    for recording in training:
        path = raw_dir / recording.filename
        size = path.lstat().st_size if path.exists() else None
        if size != recording.bytes:
            _refuse(f"{recording.filename}: {size} bytes, manifest says {recording.bytes}")
        candidates.append(Candidate(recording.filename, path, recording.sha256))
    return candidates


def _reference(
    candidate: Candidate,
    record: DecodedRecord,
    out_dir: Path,
    protocol: Protocol,
    *,
    timeout_s: float,
    command: WorkerCommand,
) -> Reference:
    """One training file's features, from a canonical waveform that matches its record."""
    outcome = run_source(candidate, out_dir, protocol, timeout_s=timeout_s, command=command).outcome
    if not isinstance(outcome, CanonicalFile):
        _refuse(f"{record.filename}: {outcome!r}")
    produced = (outcome.sha256, outcome.samples, outcome.decoded_pcm_sha256)
    if produced != (record.canonical_sha256, record.canonical_samples, record.decoded_pcm_sha256):
        _refuse(f"{record.filename}: canonical waveform differs from the committed digests")
    try:
        waveform = outcome.load()
    except CanonicalFileError as error:
        logger.error("Training references: %s could not be read back", record.filename)
        raise TrainingReferenceError(f"{record.filename}: {error}") from error
    features, _ = unit_features(waveform, protocol.duplicate_detection.mfcc)
    shutil.rmtree(out_dir)
    identity = RecordingIdentity(record.filename, record.sha256, record.decoded_pcm_sha256)
    return Reference(identity, features)


def training_references(
    protocol: Protocol,
    training: Sequence[TrainingRecording],
    digests: Mapping[str, DecodedRecord],
    raw_dir: Path,
    work_dir: Path,
    *,
    timeout_s: float,
    command: WorkerCommand = worker_command,
) -> list[Reference]:
    """Every training recording's identity and features, in manifest order.

    Args:
        protocol: The loaded protocol.
        training: The training manifest's recordings.
        digests: The committed digests, keyed by file name, for exactly those recordings.
        raw_dir: The absolute directory holding the training files.
        work_dir: An absolute directory for the workers' output, one subdirectory per file.
        timeout_s: The per-file worker time bound.
        command: The worker command; tests replace it.

    Raises:
        TrainingReferenceError: If the digests do not cover exactly the training files, a
            file's size differs, or a file does not reproduce its committed digests.
    """
    if sorted(digests) != sorted(r.filename for r in training):
        _refuse("the committed digests do not name exactly the training files")
    candidates = staged_candidates(training, raw_dir)
    references = []
    for index, candidate in enumerate(candidates):
        out_dir = work_dir / f"training-{index:06d}"
        record = digests[candidate.logical_source_id]
        references.append(
            _reference(candidate, record, out_dir, protocol, timeout_s=timeout_s, command=command)
        )
        logger.info("Training reference %d/%d %s", index + 1, len(candidates), record.filename)
    return references
