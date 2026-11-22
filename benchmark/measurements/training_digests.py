"""The training recordings' decoded-PCM and canonical digests, through the candidates' worker.

The third exclusion rule compares a candidate's decoded-PCM digest with every training
recording's. Those training digests are computed here by the code a candidate goes through,
`benchmark.runner.run_source` and `benchmark.canonical_worker`, so both sides of the rule come
from one code path. The result is written to `benchmark/manifests/training_decoded_pcm.json`
and read by `benchmark.training_corpus`.

Before any file is decoded:

- this runtime's libsndfile must be the version the protocol records, since a decoded-PCM
  digest is comparable only with one from the same decoder, and the protocol records that
  version without enforcing it;
- the golden fixtures must reproduce their canonical digests. The run preconditions' model
  checks are not run: decoding does not load either model;
- each file's size must equal the manifest's byte count, a cheap check for the wrong file.

Each file runs in its own worker with the manifest's digest as the expected digest, in a fresh
temporary directory outside the repository, and its canonical file is deleted once read. Any
outcome other than a canonical waveform stops the run and nothing is written. The per-file
timeout is generous and unmeasured; the wall times this run prints are evidence for the
walk's bound, together with the capacity fixture's, which is not measured here.

Run from the repository root:
`uv run --frozen python -m benchmark.measurements.training_digests`. It reads `data/raw/`,
which is not in the repository.
"""

from __future__ import annotations

import logging
import shutil
import statistics
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import soundfile as sf

from benchmark import training_corpus
from benchmark.atomic_write import write_atomically
from benchmark.golden_fixture import platform_token, verify_golden_fixtures
from benchmark.protocol import Protocol, load_protocol
from benchmark.runner import Candidate, CanonicalFile, run_source, worker_command
from benchmark.source_outcome import RunAbortOutcome
from benchmark.training_corpus import DecodedRecord, TrainingRecording, decoded_manifest_bytes

logger = logging.getLogger(__name__)

# Generous and not measured: no training file has been timed through the worker before this.
TIMEOUT_S = 3600.0

GoldenCheck = Callable[..., RunAbortOutcome | None]


class GenerationError(RuntimeError):
    """Raised when a precondition fails or a training file does not canonicalise."""


def check_decoder(protocol: Protocol) -> None:
    """Refuse a runtime whose libsndfile is not the version the protocol records."""
    recorded = protocol.canonicalisation.decode.libsndfile_version
    if sf.__libsndfile_version__ != recorded:
        raise GenerationError(
            f"libsndfile is {sf.__libsndfile_version__}, the protocol records {recorded}"
        )


def staged_candidates(training: Sequence[TrainingRecording], raw_dir: Path) -> list[Candidate]:
    """One candidate per training file, its size checked against the manifest first."""
    if not raw_dir.is_absolute():
        raise GenerationError(f"raw directory must be absolute, got {raw_dir}")
    candidates = []
    for recording in training:
        path = raw_dir / recording.filename
        size = path.lstat().st_size if path.exists() else None
        if size != recording.bytes:
            raise GenerationError(
                f"{recording.filename}: {size} bytes, manifest says {recording.bytes}"
            )
        candidates.append(Candidate(recording.filename, path, recording.sha256))
    return candidates


def digest_one(
    candidate: Candidate,
    out_dir: Path,
    protocol: Protocol,
    *,
    command: Callable[[Candidate, Path], list[str]] = worker_command,
) -> tuple[DecodedRecord, float]:
    """One file's digests and its worker's wall time; its canonical file is then deleted."""
    source_run = run_source(candidate, out_dir, protocol, timeout_s=TIMEOUT_S, command=command)
    outcome = source_run.outcome
    if not isinstance(outcome, CanonicalFile):
        raise GenerationError(f"{candidate.logical_source_id}: {outcome!r}")
    record = DecodedRecord(
        filename=candidate.logical_source_id,
        sha256=outcome.raw_sha256,
        decoded_pcm_sha256=outcome.decoded_pcm_sha256,
        canonical_sha256=outcome.sha256,
        canonical_samples=outcome.samples,
    )
    shutil.rmtree(out_dir)
    return record, source_run.wall_seconds


def generate(
    protocol: Protocol,
    training: Sequence[TrainingRecording],
    raw_dir: Path,
    work_dir: Path,
    *,
    golden: GoldenCheck | None = None,
    command: Callable[[Candidate, Path], list[str]] = worker_command,
) -> list[tuple[DecodedRecord, float]]:
    """Every training file's digests and wall time, after the decoder and golden checks."""
    check_decoder(protocol)
    check = verify_golden_fixtures if golden is None else golden
    abort = check(protocol, runtime_platform=platform_token())
    if abort is not None:
        raise GenerationError(f"golden fixtures failed: {abort.reason}: {abort.detail}")
    candidates = staged_candidates(training, raw_dir)
    results = []
    for index, candidate in enumerate(candidates):
        out_dir = work_dir / f"source-{index:06d}"
        results.append(digest_one(candidate, out_dir, protocol, command=command))
        logger.info("%d/%d %s", index + 1, len(candidates), candidate.logical_source_id)
    return results


def main() -> int:
    """Generate the decoded manifest and print the worker wall times; 0 on success."""
    protocol = load_protocol()
    training = training_corpus.load_training_manifest()
    try:
        with tempfile.TemporaryDirectory(prefix="training-digests-") as work:
            results = generate(protocol, training, training_corpus.RAW_DIR, Path(work))
    except GenerationError as error:
        logger.error("Nothing written: %s", error)
        return 1
    records = [record for record, _ in results]
    data = decoded_manifest_bytes(
        training, records, protocol_sha256=protocol.source_sha256, platform=platform_token()
    )
    write_atomically(training_corpus.DECODED_MANIFEST_PATH, data)
    walls = [wall for _, wall in results]
    print(
        f"wrote {len(records)} records; worker wall time max {max(walls):.1f} s, "
        f"median {statistics.median(walls):.1f} s, total {sum(walls):.0f} s"
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
