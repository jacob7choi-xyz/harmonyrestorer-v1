"""Tests for building the training references at walk time, through the real worker.

Short tone files stand in for training recordings. Their committed digests are produced the
way the real ones were, by the generator in `benchmark.measurements.training_digests`.
"""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from benchmark import training_references
from benchmark.duplicate_screen import RecordingIdentity, Reference, unit_features
from benchmark.measurements import training_digests
from benchmark.protocol import Protocol, load_protocol
from benchmark.runner import CanonicalFile, CanonicalFileError
from benchmark.source_outcome import RunAbortOutcome
from benchmark.training_corpus import DecodedRecord, TrainingRecording
from benchmark.training_references import TrainingReferenceError

RATE = 44100
TIMEOUT_S = 120.0


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The loaded protocol, once per module."""
    return load_protocol()


def passing_golden(protocol: Protocol, *, runtime_platform: str) -> RunAbortOutcome | None:
    """Stands in for the golden fixtures, which hold only on the reference platform."""
    del protocol, runtime_platform
    return None


Corpus = tuple[Path, list[TrainingRecording], dict[str, DecodedRecord]]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory, protocol: Protocol) -> Corpus:
    """Two tone files, and their digests as the generator records them, once per module."""
    root = tmp_path_factory.mktemp("corpus")
    raw = root / "raw"
    raw.mkdir()
    training = []
    for index in range(2):
        path = raw / f"MusopenCollectionAsFlac__tone{index}.flac"
        t = np.arange(RATE * 4) / RATE
        tone = 0.2 * np.sin(2 * np.pi * (220 + 110 * index) * t * (1.0 + t))
        sf.write(path, np.stack([tone, tone], axis=1), RATE, format="FLAC", subtype="PCM_16")
        data = path.read_bytes()
        training.append(TrainingRecording(path.name, len(data), hashlib.sha256(data).hexdigest()))
    scratch = root / "generate"
    scratch.mkdir()
    with pytest.MonkeyPatch.context() as patch:
        recorded = protocol.canonicalisation.decode.libsndfile_version
        patch.setattr(training_digests.sf, "__libsndfile_version__", recorded)
        results = training_digests.generate(protocol, training, raw, scratch, golden=passing_golden)
    return raw, training, {record.filename: record for record, _ in results}


def build(
    tmp_path: Path,
    protocol: Protocol,
    corpus: Corpus,
    training: list[TrainingRecording] | None = None,
    digests: dict[str, DecodedRecord] | None = None,
) -> list[Reference]:
    """The references for the corpus, or for a changed *training* or *digests*."""
    raw, corpus_training, corpus_digests = corpus
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    return training_references.training_references(
        protocol,
        corpus_training if training is None else training,
        corpus_digests if digests is None else digests,
        raw,
        work,
        timeout_s=TIMEOUT_S,
    )


class TestTrainingReferences:
    """Each training file is canonicalised again and must reproduce its committed digests."""

    def test_each_reference_is_named_by_its_record_and_holds_its_features(
        self,
        tmp_path: Path,
        protocol: Protocol,
        corpus: Corpus,
    ) -> None:
        _, training, digests = corpus
        references = build(tmp_path, protocol, corpus)

        assert [r.identity for r in references] == [
            RecordingIdentity(r.filename, r.sha256, digests[r.filename].decoded_pcm_sha256)
            for r in training
        ]
        for reference, recording in zip(references, training, strict=True):
            assert reference.features.shape[1] == protocol.duplicate_detection.mfcc.n_mfcc
            assert (
                len(reference.features) == 1 + digests[recording.filename].canonical_samples // 512
            )
        assert list((tmp_path / "work").iterdir()) == []

    def test_the_features_are_those_of_the_canonical_waveform(
        self,
        tmp_path: Path,
        protocol: Protocol,
        corpus: Corpus,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The waveform the features are computed from is the one CanonicalFile.load read."""
        loaded: list[np.ndarray] = []
        real_load = CanonicalFile.load

        def recording_load(self: CanonicalFile) -> np.ndarray:
            waveform = real_load(self)
            loaded.append(waveform)
            return waveform

        monkeypatch.setattr(CanonicalFile, "load", recording_load)
        references = build(tmp_path, protocol, corpus)

        config = protocol.duplicate_detection.mfcc
        for reference, waveform in zip(references, loaded, strict=True):
            assert np.array_equal(reference.features, unit_features(waveform, config)[0])

    @pytest.mark.parametrize(
        "field", ["canonical_sha256", "canonical_samples", "decoded_pcm_sha256"]
    )
    def test_a_file_that_does_not_reproduce_a_committed_value_is_refused(
        self,
        tmp_path: Path,
        protocol: Protocol,
        corpus: Corpus,
        field: str,
    ) -> None:
        _, training, digests = corpus
        name = training[0].filename
        other = 1 if field == "canonical_samples" else "0" * 64
        changed = {**digests, name: dataclasses.replace(digests[name], **{field: other})}
        with pytest.raises(TrainingReferenceError, match="differs from the committed digests"):
            build(tmp_path, protocol, corpus, digests=changed)

    def test_a_file_that_cannot_be_read_back_is_refused(
        self,
        tmp_path: Path,
        protocol: Protocol,
        corpus: Corpus,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The canonical file changed after the worker ended, so load refuses it."""

        def swapped(self: CanonicalFile) -> np.ndarray:
            raise CanonicalFileError(f"canonical file {self.path} does not hash")

        monkeypatch.setattr(CanonicalFile, "load", swapped)
        with pytest.raises(TrainingReferenceError, match="does not hash"):
            build(tmp_path, protocol, corpus)

    def test_digests_that_do_not_name_exactly_the_training_files_are_refused(
        self,
        tmp_path: Path,
        protocol: Protocol,
        corpus: Corpus,
    ) -> None:
        _, training, digests = corpus
        fewer = {training[0].filename: digests[training[0].filename]}
        with pytest.raises(TrainingReferenceError, match="exactly the training files"):
            build(tmp_path, protocol, corpus, digests=fewer)
        assert list((tmp_path / "work").iterdir()) == []

    def test_a_file_of_the_wrong_size_is_refused_before_any_worker(
        self,
        tmp_path: Path,
        protocol: Protocol,
        corpus: Corpus,
    ) -> None:
        _, training, _ = corpus
        wrong = [training[0], dataclasses.replace(training[1], bytes=training[1].bytes + 1)]
        with pytest.raises(TrainingReferenceError, match="bytes, manifest says"):
            build(tmp_path, protocol, corpus, training=wrong)
        assert list((tmp_path / "work").iterdir()) == []

    def test_a_file_with_other_bytes_is_refused_by_the_worker(
        self,
        tmp_path: Path,
        protocol: Protocol,
        corpus: Corpus,
    ) -> None:
        """The manifest digest is the worker's expected digest, so other bytes abort there."""
        _, training, _ = corpus
        wrong = [dataclasses.replace(training[0], sha256="0" * 64), training[1]]
        with pytest.raises(TrainingReferenceError, match="SOURCE_HASH_MISMATCH"):
            build(tmp_path, protocol, corpus, training=wrong)

    def test_a_relative_raw_directory_is_refused(self) -> None:
        with pytest.raises(TrainingReferenceError, match="absolute"):
            training_references.staged_candidates([], Path("data/raw"))
