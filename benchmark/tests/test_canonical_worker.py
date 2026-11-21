"""Tests for the canonicalisation worker: what it returns, writes, and exits with.

Most tests call the worker in this process, which lets the decode bounds be lowered so the
over-size path needs no 512 MiB file. One test runs it the way the runner does, as a child
process from the repository root.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from benchmark import canonical_worker
from benchmark.canonical_worker import (
    CANONICAL_FILE,
    EXIT_MEMORY_ERROR,
    RECORD_FILE,
    canonicalise_source,
    main,
)
from benchmark.canonicalisation import CanonicalWaveform
from benchmark.outcome_wire import outcome_from_wire
from benchmark.protocol import Protocol, load_protocol
from benchmark.run_preconditions import file_sha256
from benchmark.source_outcome import (
    AbortReason,
    IneligibleReason,
    RunAbortOutcome,
    SourceIneligibleOutcome,
)

RATE = 16000
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol."""
    return load_protocol()


@pytest.fixture
def tone_flac(tmp_path: Path) -> Path:
    """One second of two tones, mono 24-bit FLAC at 16 kHz: a source that canonicalises."""
    t = np.arange(RATE) / RATE
    samples = 0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 1234.5 * t)
    path = tmp_path / "tones.flac"
    sf.write(path, samples, RATE, format="FLAC", subtype="PCM_24")
    return path


@pytest.fixture
def out_dir(tmp_path: Path) -> Path:
    """An empty directory for the worker's files."""
    path = tmp_path / "out"
    path.mkdir()
    return path


def with_source_bound(protocol: Protocol, max_source_bytes: int) -> Protocol:
    """The protocol with the source byte bound lowered and the identity bound left above it."""
    canon = protocol.canonicalisation
    decode = dataclasses.replace(
        canon.decode, max_source_bytes=max_source_bytes, max_identity_stream_bytes=1 << 30
    )
    return dataclasses.replace(protocol, canonicalisation=dataclasses.replace(canon, decode=decode))


def read_record(out_dir: Path) -> dict[str, object]:
    """The record the worker wrote."""
    return dict(json.loads((out_dir / RECORD_FILE).read_text()))


class TestCanonicaliseSource:
    """Each stage's outcome is returned as the first outcome reached."""

    def test_a_source_in_the_domain_canonicalises(
        self, protocol: Protocol, tone_flac: Path
    ) -> None:
        result = canonicalise_source(
            tone_flac, logical_source_id="item/tones.flac", expected_sha256=None, protocol=protocol
        )
        assert isinstance(result, CanonicalWaveform)
        assert result.samples.size == RATE

    def test_an_oversize_source_is_a_record_with_its_streamed_identity(
        self, protocol: Protocol, tone_flac: Path
    ) -> None:
        """The record `ArtifactTooLarge` was waiting for, built from what acquisition knew."""
        size = tone_flac.stat().st_size
        result = canonicalise_source(
            tone_flac,
            logical_source_id="item/tones.flac",
            expected_sha256=None,
            protocol=with_source_bound(protocol, size - 1),
        )
        assert isinstance(result, SourceIneligibleOutcome)
        assert result.primary_reason_code is IneligibleReason.SOURCE_EXCEEDS_MAX_BYTES
        assert result.identity.source_bytes == size
        assert result.identity.sha256 == hashlib.sha256(tone_flac.read_bytes()).hexdigest()
        assert result.observed_metadata is None
        assert result.protocol_sha256 == protocol.source_sha256

    def test_a_source_at_the_bound_is_not_oversize(
        self, protocol: Protocol, tone_flac: Path
    ) -> None:
        result = canonicalise_source(
            tone_flac,
            logical_source_id="item/tones.flac",
            expected_sha256=None,
            protocol=with_source_bound(protocol, tone_flac.stat().st_size),
        )
        assert isinstance(result, CanonicalWaveform)

    def test_an_undecodable_source_is_rejected_at_inspection(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        path = tmp_path / "noise.flac"
        path.write_bytes(b"not audio at all" * 64)
        result = canonicalise_source(
            path, logical_source_id="item/noise.flac", expected_sha256=None, protocol=protocol
        )
        assert isinstance(result, SourceIneligibleOutcome)
        assert result.primary_reason_code is IneligibleReason.SOURCE_INSPECTION_REJECTED

    def test_a_digest_mismatch_aborts_naming_the_source(
        self, protocol: Protocol, tone_flac: Path
    ) -> None:
        result = canonicalise_source(
            tone_flac,
            logical_source_id="item/tones.flac",
            expected_sha256="0" * 64,
            protocol=protocol,
        )
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.SOURCE_HASH_MISMATCH
        assert result.logical_source_id == "item/tones.flac"


class TestMain:
    """What the worker leaves behind for its parent, and the status it exits with."""

    def test_a_canonical_source_leaves_its_samples_and_then_its_record(
        self, protocol: Protocol, tone_flac: Path, out_dir: Path
    ) -> None:
        assert main([str(tone_flac), "item/tones.flac", str(out_dir)]) == 0

        record = read_record(out_dir)
        result = record["result"]
        assert isinstance(result, dict)
        assert record["protocol_sha256"] == protocol.source_sha256
        assert result["kind"] == "canonical"
        assert result["samples"] == RATE
        # The file is exactly the hashed bytes, so its digest is the canonical digest.
        assert file_sha256(out_dir / CANONICAL_FILE) == result["sha256"]
        expected = canonicalise_source(
            tone_flac, logical_source_id="item/tones.flac", expected_sha256=None, protocol=protocol
        )
        assert isinstance(expected, CanonicalWaveform)
        np.testing.assert_array_equal(
            np.fromfile(out_dir / CANONICAL_FILE, dtype="<f8"), expected.samples
        )

    def test_an_outcome_leaves_a_record_that_rebuilds_and_no_samples(
        self, protocol: Protocol, tone_flac: Path, out_dir: Path
    ) -> None:
        argv = [str(tone_flac), "item/tones.flac", str(out_dir), "--expected-sha256", "0" * 64]

        assert main(argv) == 0

        rebuilt = outcome_from_wire(
            read_record(out_dir)["result"], protocol.canonicalisation.decode
        )
        assert isinstance(rebuilt, RunAbortOutcome)
        assert rebuilt.reason is AbortReason.SOURCE_HASH_MISMATCH
        assert sorted(os.listdir(out_dir)) == [RECORD_FILE]

    def test_a_memory_error_exits_with_its_status_and_writes_nothing(
        self, tone_flac: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def exhausted(*args: object, **kwargs: object) -> None:
            raise MemoryError

        monkeypatch.setattr(canonical_worker, "canonicalise_source", exhausted)

        assert main([str(tone_flac), "item/tones.flac", str(out_dir)]) == EXIT_MEMORY_ERROR
        assert os.listdir(out_dir) == []

    def test_any_other_exception_is_not_caught(
        self, tone_flac: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Left to exit with status 1, which the parent cannot attribute to the source."""

        def broken(*args: object, **kwargs: object) -> None:
            raise RuntimeError("defect")

        monkeypatch.setattr(canonical_worker, "canonicalise_source", broken)

        with pytest.raises(RuntimeError, match="defect"):
            main([str(tone_flac), "item/tones.flac", str(out_dir)])
        assert os.listdir(out_dir) == []

    def test_a_failed_write_leaves_no_file_behind(
        self, tone_flac: Path, out_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def full_disk(fd: int) -> None:
            raise OSError(28, "No space left on device")

        monkeypatch.setattr(canonical_worker.os, "fsync", full_disk)

        with pytest.raises(OSError, match="No space left"):
            main([str(tone_flac), "item/tones.flac", str(out_dir)])
        assert os.listdir(out_dir) == []


def test_the_worker_runs_as_a_child_from_the_repository_root(
    protocol: Protocol, tone_flac: Path, out_dir: Path
) -> None:
    """The way the runner starts it: `-m` without `-I`, with the repository root as cwd."""
    command = [sys.executable, "-m", "benchmark.canonical_worker", str(tone_flac), "item/t.flac"]
    ended = subprocess.run(
        [*command, str(out_dir)], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert ended.returncode == 0, ended.stderr
    result = read_record(out_dir)["result"]
    assert isinstance(result, dict)
    assert result["kind"] == "canonical"
    assert file_sha256(out_dir / CANONICAL_FILE) == result["sha256"]
