"""Tests for the generator of the training recordings' decoded-PCM and canonical digests.

The real worker runs on short tone files standing in for training recordings. The golden
fixture check is replaced, because it requires the reference platform.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from benchmark import training_corpus
from benchmark.measurements import training_digests
from benchmark.measurements.training_digests import GenerationError, generate
from benchmark.protocol import Protocol, load_protocol
from benchmark.source_outcome import AbortReason, RunAbortOutcome
from benchmark.training_corpus import TrainingRecording

RATE = 44100


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The loaded protocol, once per module."""
    return load_protocol()


@pytest.fixture(autouse=True)
def protocol_decoder(protocol: Protocol, monkeypatch: pytest.MonkeyPatch) -> None:
    """Report the protocol's libsndfile, so these tests do not depend on the runtime's.

    The one test of the version check sets its own value after this fixture runs.
    """
    recorded = protocol.canonicalisation.decode.libsndfile_version
    monkeypatch.setattr(training_digests.sf, "__libsndfile_version__", recorded)


def passing_golden(protocol: Protocol, *, runtime_platform: str) -> RunAbortOutcome | None:
    """Stands in for the golden fixtures, which hold only on the reference platform."""
    del protocol, runtime_platform
    return None


def raw_files(raw_dir: Path, count: int) -> list[TrainingRecording]:
    """Short stereo FLAC files, and the manifest entries that describe them."""
    raw_dir.mkdir()
    recordings = []
    for index in range(count):
        path = raw_dir / f"MusopenCollectionAsFlac__tone{index}.flac"
        t = np.arange(RATE) / RATE
        tone = 0.2 * np.sin(2 * np.pi * (220 + 110 * index) * t)
        sf.write(path, np.stack([tone, tone], axis=1), RATE, format="FLAC", subtype="PCM_16")
        data = path.read_bytes()
        recordings.append(TrainingRecording(path.name, len(data), hashlib.sha256(data).hexdigest()))
    return recordings


class TestGenerate:
    """Each training file goes through the candidates' worker, or nothing is produced."""

    def test_each_file_gets_the_digests_the_worker_reports(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        training = raw_files(tmp_path / "raw", 2)
        work = tmp_path / "work"
        work.mkdir()
        results = generate(protocol, training, tmp_path / "raw", work, golden=passing_golden)
        assert [record.filename for record, _ in results] == [r.filename for r in training]
        for (record, wall), recording in zip(results, training, strict=True):
            decoded, _ = sf.read(
                tmp_path / "raw" / recording.filename, dtype="float64", always_2d=True
            )
            assert record.sha256 == recording.sha256
            assert (
                record.decoded_pcm_sha256
                == hashlib.sha256(decoded.astype("<f8").tobytes()).hexdigest()
            )
            assert record.canonical_samples == 16000
            assert wall > 0
        assert list(work.iterdir()) == []

    def test_a_file_of_the_wrong_size_stops_the_run_before_any_worker(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        training = raw_files(tmp_path / "raw", 2)
        wrong = TrainingRecording(training[1].filename, training[1].bytes + 1, training[1].sha256)
        work = tmp_path / "work"
        work.mkdir()
        with pytest.raises(GenerationError, match="bytes, manifest says"):
            generate(protocol, [training[0], wrong], tmp_path / "raw", work, golden=passing_golden)
        assert list(work.iterdir()) == []

    def test_a_file_with_other_bytes_of_the_same_size_stops_the_run(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        """The manifest's digest is the worker's expected digest, so other bytes abort."""
        training = raw_files(tmp_path / "raw", 1)
        other = TrainingRecording(training[0].filename, training[0].bytes, "0" * 64)
        work = tmp_path / "work"
        work.mkdir()
        with pytest.raises(GenerationError, match="SOURCE_HASH_MISMATCH"):
            generate(protocol, [other], tmp_path / "raw", work, golden=passing_golden)

    def test_a_golden_failure_stops_the_run_before_any_worker(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        def failing(protocol: Protocol, *, runtime_platform: str) -> RunAbortOutcome:
            return RunAbortOutcome(reason=AbortReason.GOLDEN_FIXTURE_FAILURE, detail="digest")

        training = raw_files(tmp_path / "raw", 1)
        work = tmp_path / "work"
        work.mkdir()
        with pytest.raises(GenerationError, match="golden fixtures failed"):
            generate(protocol, training, tmp_path / "raw", work, golden=failing)
        assert list(work.iterdir()) == []

    def test_another_libsndfile_stops_the_run(
        self, protocol: Protocol, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(training_digests.sf, "__libsndfile_version__", "0.0.0")
        with pytest.raises(GenerationError, match=r"libsndfile is 0\.0\.0"):
            generate(protocol, [], tmp_path, tmp_path, golden=passing_golden)

    def test_a_file_that_does_not_canonicalise_stops_the_run(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        raw.mkdir()
        noise = raw / "MusopenCollectionAsFlac__noise.flac"
        noise.write_bytes(b"not audio at all" * 64)
        data = noise.read_bytes()
        training = [TrainingRecording(noise.name, len(data), hashlib.sha256(data).hexdigest())]
        work = tmp_path / "work"
        work.mkdir()
        with pytest.raises(GenerationError, match="SOURCE_INSPECTION_REJECTED"):
            generate(protocol, training, raw, work, golden=passing_golden)


class TestMain:
    """The manifest is written whole after a complete run, and not at all otherwise."""

    def setup_paths(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, training: list[TrainingRecording]
    ) -> Path:
        output = tmp_path / "training_decoded_pcm.json"
        monkeypatch.setattr(training_corpus, "RAW_DIR", tmp_path / "raw")
        monkeypatch.setattr(training_corpus, "DECODED_MANIFEST_PATH", output)
        monkeypatch.setattr(training_corpus, "load_training_manifest", lambda: tuple(training))
        monkeypatch.setattr(training_digests, "verify_golden_fixtures", passing_golden)
        return output

    def test_a_complete_run_writes_a_manifest_the_loader_accepts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        training = raw_files(tmp_path / "raw", 2)
        output = self.setup_paths(tmp_path, monkeypatch, training)
        assert training_digests.main() == 0
        read_back = training_corpus._read_decoded_manifest(output, training)
        assert sorted(read_back) == [r.filename for r in training]
        assert json.loads(output.read_text())["count"] == 2

    def test_a_failed_run_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        training = raw_files(tmp_path / "raw", 1)
        wrong = [TrainingRecording(training[0].filename, 1, training[0].sha256)]
        output = self.setup_paths(tmp_path, monkeypatch, wrong)
        assert training_digests.main() == 1
        assert not output.exists()
