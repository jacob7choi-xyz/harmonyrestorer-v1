"""The training corpus's identity: the frozen manifest and the decoded-PCM digests derived from it.

`benchmark/manifests/training_corpus.json` names the 145 recordings that trained the OpGAN
checkpoint, with each file's byte count and SHA-256. Its `corpus_sha256` is the SHA-256 of the
per-file digests joined by newlines in sorted filename order (`docs/benchmark-protocol.md`,
Identity and disjointness). `benchmark/manifests/training_decoded_pcm.json` adds, for each of
those files, the decoded-PCM digest the third exclusion rule compares, and the canonical digest
and sample count, produced by `benchmark.measurements.training_digests` through the same worker
the candidates go through.

**Fixed paths, like `load_protocol`.** The public loaders take no path, so nothing in a run can
be pointed at another manifest. The path-taking readers are private and used by tests.

**The decoder is enforced, the protocol is recorded.** A decoded-PCM digest is only comparable
with one computed by the same decoder, and the protocol records libsndfile's version without
enforcing it, so the loader refuses a decoded manifest whose soundfile or libsndfile version
differs from this runtime's. The protocol digest at generation is recorded and not compared:
an amendment that does not change decoding should not force the digests to be regenerated.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from benchmark._source_validation import SourceOutcomeError, checked_digest
from benchmark.protocol import load_protocol
from benchmark.selection import TrainingIdentities, training_identities

__all__ = [
    "DECODED_MANIFEST_PATH",
    "RAW_DIR",
    "TRAINING_MANIFEST_PATH",
    "DecodedRecord",
    "TrainingCorpusError",
    "TrainingRecording",
    "corpus_sha256",
    "decoded_manifest_bytes",
    "load_training_digests",
    "load_training_identities",
    "load_training_manifest",
]

_REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINING_MANIFEST_PATH = _REPO_ROOT / "benchmark" / "manifests" / "training_corpus.json"
DECODED_MANIFEST_PATH = _REPO_ROOT / "benchmark" / "manifests" / "training_decoded_pcm.json"
RAW_DIR = _REPO_ROOT / "data" / "raw"

_TRAINING_KEYS = frozenset(
    {"description", "count", "corpus_sha256", "checkpoint_sha256", "recordings"}
)
_TRAINING_ENTRY_KEYS = frozenset({"filename", "bytes", "sha256"})
_DECODED_KEYS = frozenset(
    {
        "description",
        "decoded_pcm_hash",
        "corpus_sha256",
        "protocol_sha256",
        "soundfile_version",
        "libsndfile_version",
        "platform",
        "count",
        "recordings",
    }
)
_DECODED_ENTRY_KEYS = frozenset(
    {"filename", "sha256", "decoded_pcm_sha256", "canonical_sha256", "canonical_samples"}
)
_DESCRIPTION = (
    "Decoded-PCM and canonical digests of the training recordings, produced by "
    "benchmark.measurements.training_digests through the canonicalisation worker."
)


class TrainingCorpusError(ValueError):
    """Raised when a training manifest is malformed or does not match what it derives from."""


@dataclass(frozen=True)
class TrainingRecording:
    """One training file as the frozen manifest names it."""

    filename: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class DecodedRecord:
    """One training file's digests as the worker reported them."""

    filename: str
    sha256: str
    decoded_pcm_sha256: str
    canonical_sha256: str
    canonical_samples: int


def corpus_sha256(raw_sha256_by_filename: Mapping[str, str]) -> str:
    """The corpus digest: per-file digests joined by newlines in sorted filename order."""
    ordered = [raw_sha256_by_filename[name] for name in sorted(raw_sha256_by_filename)]
    return hashlib.sha256("\n".join(ordered).encode()).hexdigest()


def _digest(value: object, field: str) -> str:
    """*value* as a 64-character lowercase hex digest, or a TrainingCorpusError."""
    if not isinstance(value, str):
        raise TrainingCorpusError(f"{field} must be a string digest, got {value!r}")
    try:
        return checked_digest(value, field)
    except SourceOutcomeError as error:
        raise TrainingCorpusError(str(error)) from error


def _count(value: object, field: str) -> int:
    """*value* as a positive int, refusing bool, which Python counts as an int."""
    if type(value) is not int or value <= 0:
        raise TrainingCorpusError(f"{field} must be a positive integer, got {value!r}")
    return value


def _exact_keys(data: object, expected: frozenset[str], where: str) -> dict[str, Any]:
    """*data* as a dict whose keys are exactly *expected*."""
    if not isinstance(data, dict) or set(data) != expected:
        found = sorted(data) if isinstance(data, dict) else type(data).__name__
        raise TrainingCorpusError(f"{where} must have keys {sorted(expected)}, got {found}")
    return data


def _entries(data: dict[str, Any], keys: frozenset[str], where: str) -> list[dict[str, Any]]:
    """The `recordings` list, each entry with exactly *keys*, its count matching `count`."""
    recordings = data["recordings"]
    if not isinstance(recordings, list):
        raise TrainingCorpusError(f"{where}.recordings must be a list")
    entries = [
        _exact_keys(entry, keys, f"{where}.recordings[{i}]") for i, entry in enumerate(recordings)
    ]
    if _count(data["count"], f"{where}.count") != len(entries):
        raise TrainingCorpusError(
            f"{where}.count is {data['count']}, but {len(entries)} are listed"
        )
    names = [entry["filename"] for entry in entries]
    if not all(isinstance(name, str) and name for name in names) or len(set(names)) != len(names):
        raise TrainingCorpusError(f"{where} file names must be non-empty and unique")
    return entries


def _read_training_manifest(path: Path) -> tuple[TrainingRecording, ...]:
    """The training manifest at *path*, checked, in sorted filename order."""
    data = _exact_keys(json.loads(path.read_bytes()), _TRAINING_KEYS, path.name)
    recordings = [
        TrainingRecording(
            filename=entry["filename"],
            bytes=_count(entry["bytes"], f"{entry['filename']}.bytes"),
            sha256=_digest(entry["sha256"], f"{entry['filename']}.sha256"),
        )
        for entry in _entries(data, _TRAINING_ENTRY_KEYS, path.name)
    ]
    stated = _digest(data["corpus_sha256"], "corpus_sha256")
    if corpus_sha256({r.filename: r.sha256 for r in recordings}) != stated:
        raise TrainingCorpusError(f"{path.name}: corpus_sha256 does not match its recordings")
    return tuple(sorted(recordings, key=lambda r: r.filename))


def load_training_manifest() -> tuple[TrainingRecording, ...]:
    """The frozen training manifest, checked, in sorted filename order.

    Raises:
        TrainingCorpusError: If the manifest is malformed or its corpus digest does not match.
    """
    return _read_training_manifest(TRAINING_MANIFEST_PATH)


def _decoder_versions() -> dict[str, str]:
    """This runtime's soundfile and libsndfile versions."""
    return {"soundfile_version": sf.__version__, "libsndfile_version": sf.__libsndfile_version__}


def _check_decoded_header(data: dict[str, Any], training: Sequence[TrainingRecording]) -> None:
    """The decoded manifest names this training corpus, this layout, and this decoder."""
    expected_corpus = corpus_sha256({r.filename: r.sha256 for r in training})
    if _digest(data["corpus_sha256"], "corpus_sha256") != expected_corpus:
        raise TrainingCorpusError("decoded manifest was made for another training corpus")
    layout = load_protocol().duplicate_detection.decoded_pcm_hash
    if data["decoded_pcm_hash"] != layout:
        raise TrainingCorpusError(f"decoded manifest layout is {data['decoded_pcm_hash']!r}")
    for field, version in _decoder_versions().items():
        if data[field] != version:
            raise TrainingCorpusError(
                f"decoded manifest {field} is {data[field]!r}, this runtime has {version!r}; "
                "digests from another decoder are not comparable"
            )
    _digest(data["protocol_sha256"], "protocol_sha256")


def _decoded_record(entry: dict[str, Any], raw: TrainingRecording) -> DecodedRecord:
    """One checked decoded entry, whose raw digest must equal the training manifest's."""
    name = entry["filename"]
    if _digest(entry["sha256"], f"{name}.sha256") != raw.sha256:
        raise TrainingCorpusError(f"{name}: raw digest differs from the training manifest")
    return DecodedRecord(
        filename=name,
        sha256=raw.sha256,
        decoded_pcm_sha256=_digest(entry["decoded_pcm_sha256"], f"{name}.decoded_pcm_sha256"),
        canonical_sha256=_digest(entry["canonical_sha256"], f"{name}.canonical_sha256"),
        canonical_samples=_count(entry["canonical_samples"], f"{name}.canonical_samples"),
    )


def _read_decoded_manifest(
    path: Path, training: Sequence[TrainingRecording]
) -> dict[str, DecodedRecord]:
    """The decoded manifest at *path*, checked against *training*, keyed by file name."""
    data = _exact_keys(json.loads(path.read_bytes()), _DECODED_KEYS, path.name)
    _check_decoded_header(data, training)
    entries = _entries(data, _DECODED_ENTRY_KEYS, path.name)
    by_name = {r.filename: r for r in training}
    if sorted(entry["filename"] for entry in entries) != sorted(by_name):
        raise TrainingCorpusError(f"{path.name} does not list exactly the training files")
    return {
        entry["filename"]: _decoded_record(entry, by_name[entry["filename"]]) for entry in entries
    }


def load_training_digests() -> dict[str, DecodedRecord]:
    """Every training file's digests, keyed by file name, checked against the training manifest.

    Raises:
        TrainingCorpusError: If either manifest is malformed, the decoded one was made for
            another corpus or layout, or by a different soundfile or libsndfile version.
    """
    return _read_decoded_manifest(DECODED_MANIFEST_PATH, load_training_manifest())


def load_training_identities() -> TrainingIdentities:
    """The three identity sets the exclusion rules compare against, from both manifests."""
    digests = load_training_digests()
    manifest = [{"filename": r.filename, "sha256": r.sha256} for r in digests.values()]
    return training_identities(manifest, {n: r.decoded_pcm_sha256 for n, r in digests.items()})


def decoded_manifest_bytes(
    training: Sequence[TrainingRecording],
    records: Sequence[DecodedRecord],
    *,
    protocol_sha256: str,
    platform: str,
) -> bytes:
    """The decoded manifest for *records*, in sorted filename order, ready to write.

    Every digest sits under a key the secrets gate exempts, one per line, so the gate can
    tell a digest from a credential.
    """
    data = {
        "description": _DESCRIPTION,
        "decoded_pcm_hash": load_protocol().duplicate_detection.decoded_pcm_hash,
        "corpus_sha256": corpus_sha256({r.filename: r.sha256 for r in training}),
        "protocol_sha256": protocol_sha256,
        **_decoder_versions(),
        "platform": platform,
        "count": len(records),
        "recordings": [
            {
                "filename": r.filename,
                "sha256": r.sha256,
                "decoded_pcm_sha256": r.decoded_pcm_sha256,
                "canonical_sha256": r.canonical_sha256,
                "canonical_samples": r.canonical_samples,
            }
            for r in sorted(records, key=lambda record: record.filename)
        ],
    }
    return (json.dumps(data, indent=2) + "\n").encode()
