"""Tests for reading the training manifest and the decoded-PCM digests derived from it.

The decoded manifest does not exist until the generator has run on the training files, so
these tests build one from the real training manifest with made-up digests and read it
through the private reader.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import soundfile as sf

from benchmark import training_corpus
from benchmark.training_corpus import (
    DecodedRecord,
    TrainingCorpusError,
    corpus_sha256,
    decoded_manifest_bytes,
    load_training_identities,
    load_training_manifest,
)

TRAINING = load_training_manifest()
PROTOCOL_DIGEST = "d" * 64


def records() -> list[DecodedRecord]:
    """One made-up decoded record per training file, its raw digest taken from the manifest."""
    return [
        DecodedRecord(r.filename, r.sha256, f"{i:064x}", f"{i + 1000:064x}", 16000 * 60 + i)
        for i, r in enumerate(TRAINING)
    ]


def written(tmp_path: Path, change: Callable[[dict[str, Any]], None] | None = None) -> Path:
    """A decoded manifest for the real training files, optionally changed, on disk."""
    data = json.loads(
        decoded_manifest_bytes(
            TRAINING, records(), protocol_sha256=PROTOCOL_DIGEST, platform="darwin_arm64"
        )
    )
    if change is not None:
        change(data)
    path = tmp_path / "training_decoded_pcm.json"
    path.write_text(json.dumps(data, indent=2))
    return path


def read(path: Path) -> dict[str, DecodedRecord]:
    """The decoded manifest at *path*, read against the real training manifest."""
    return training_corpus._read_decoded_manifest(path, TRAINING)


class TestTrainingManifest:
    """The frozen manifest, read and checked once for everything that uses it."""

    def test_it_lists_145_recordings_in_sorted_order(self) -> None:
        assert len(TRAINING) == 145
        assert [r.filename for r in TRAINING] == sorted(r.filename for r in TRAINING)

    def test_the_corpus_digest_does_not_depend_on_input_order(self) -> None:
        forward = {r.filename: r.sha256 for r in TRAINING}
        backward = dict(reversed(list(forward.items())))
        assert corpus_sha256(forward) == corpus_sha256(backward)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (lambda d: d.update(corpus_sha256="0" * 64), "corpus_sha256 does not match"),
            (lambda d: d.update(count=144), "count is 144"),
            (lambda d: d["recordings"][0].update(extra=1), "must have keys"),
            (lambda d: d["recordings"][0].update(bytes=True), "positive integer"),
            (lambda d: d["recordings"][0].update(sha256="A" * 64), "lowercase hexadecimal"),
            (
                lambda d: d["recordings"][1].update(filename=d["recordings"][0]["filename"]),
                "unique",
            ),
        ],
        ids=["corpus_digest", "count", "extra_key", "bytes_bool", "uppercase_digest", "repeat"],
    )
    def test_a_damaged_manifest_is_refused(
        self, tmp_path: Path, change: Callable[[dict[str, Any]], None], message: str
    ) -> None:
        data = json.loads(training_corpus.TRAINING_MANIFEST_PATH.read_text())
        change(data)
        path = tmp_path / "training_corpus.json"
        path.write_text(json.dumps(data))
        with pytest.raises(TrainingCorpusError, match=message):
            training_corpus._read_training_manifest(path)


class TestDecodedManifest:
    """The digests the third exclusion rule compares, tied to their corpus and decoder."""

    def test_a_written_manifest_reads_back_as_written(self, tmp_path: Path) -> None:
        read_back = read(written(tmp_path))
        assert sorted(read_back.values(), key=lambda r: r.filename) == records()

    def test_every_digest_sits_alone_on_a_line_under_its_key(self) -> None:
        """The form the secrets gate's exemption matches: "key": "64 hex"."""
        text = decoded_manifest_bytes(
            TRAINING, records(), protocol_sha256=PROTOCOL_DIGEST, platform="darwin_arm64"
        ).decode()
        digest_lines = [line.strip() for line in text.splitlines() if '_sha256": "' in line]
        assert digest_lines
        assert all(line.startswith('"') and line.count('": "') == 1 for line in digest_lines)

    @pytest.mark.parametrize(
        ("change", "message"),
        [
            (lambda d: d.update(corpus_sha256="0" * 64), "another training corpus"),
            (lambda d: d.update(decoded_pcm_hash="float32"), "layout"),
            (
                lambda d: d.update(libsndfile_version=sf.__libsndfile_version__ + "-other"),
                "libsndfile_version",
            ),
            (
                lambda d: d.update(soundfile_version=sf.__version__ + "-other"),
                "soundfile_version",
            ),
            (lambda d: d["recordings"].pop(), "count is 145"),
            (lambda d: d["recordings"][0].update(sha256="0" * 64), "raw digest differs"),
            (lambda d: d["recordings"][0].update(decoded_pcm_sha256="xyz"), "lowercase hex"),
            (lambda d: d["recordings"][0].update(canonical_samples=True), "positive integer"),
            (lambda d: d.update(extra=1), "must have keys"),
        ],
        ids=[
            "other_corpus",
            "other_layout",
            "other_libsndfile",
            "other_soundfile",
            "file_missing",
            "raw_digest_differs",
            "malformed_decoded",
            "samples_bool",
            "extra_key",
        ],
    )
    def test_a_manifest_that_does_not_match_is_refused(
        self, tmp_path: Path, change: Callable[[dict[str, Any]], None], message: str
    ) -> None:
        with pytest.raises(TrainingCorpusError, match=message):
            read(written(tmp_path, change))

    def test_a_manifest_naming_other_files_is_refused(self, tmp_path: Path) -> None:
        def rename(data: dict[str, Any]) -> None:
            data["recordings"][0]["filename"] = "MusopenCollectionAsFlac__other.flac"

        with pytest.raises(TrainingCorpusError, match="exactly the training files"):
            read(written(tmp_path, rename))

    def test_the_recorded_decoder_is_this_runtime_s(self, tmp_path: Path) -> None:
        data = json.loads(written(tmp_path).read_text())
        assert data["libsndfile_version"] == sf.__libsndfile_version__
        assert data["soundfile_version"] == sf.__version__


class TestTrainingIdentities:
    """The sets the exclusion rules compare, built from both manifests."""

    def test_the_identities_cover_every_training_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(training_corpus, "DECODED_MANIFEST_PATH", written(tmp_path))
        identities = load_training_identities()
        assert len(identities.filenames) == len(identities.decoded_sha256) == 145
        assert identities.raw_sha256 == frozenset(r.sha256 for r in TRAINING)


def _committed_decoder() -> tuple[str, str]:
    """The soundfile and libsndfile versions the committed decoded manifest was made with."""
    data = json.loads(training_corpus.DECODED_MANIFEST_PATH.read_text())
    return data["soundfile_version"], data["libsndfile_version"]


@pytest.mark.skipif(
    _committed_decoder() != (sf.__version__, sf.__libsndfile_version__),
    reason="the committed digests are comparable only under the decoder that made them",
)
class TestTheCommittedDigests:
    """The decoded manifest the generator wrote for the 145 training files."""

    def test_every_training_file_has_its_own_digests(self) -> None:
        digests = training_corpus.load_training_digests()
        assert sorted(digests) == [r.filename for r in TRAINING]
        assert len({r.decoded_pcm_sha256 for r in digests.values()}) == 145
        assert len({r.canonical_sha256 for r in digests.values()}) == 145

    def test_the_identities_load_from_the_committed_files(self) -> None:
        identities = load_training_identities()
        assert len(identities.filenames) == len(identities.decoded_sha256) == 145
