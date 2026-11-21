"""Tests for the run-level preconditions: golden fixtures, degradation golden, model files."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from benchmark import run_preconditions
from benchmark.golden_fixture import platform_token
from benchmark.protocol import Protocol, load_protocol
from benchmark.run_preconditions import (
    degradation_golden_failure,
    file_sha256,
    model_file_failures,
    verify_run_preconditions,
)
from benchmark.source_outcome import AbortReason, RunAbortOutcome

PROTOCOL = load_protocol()
REFERENCE = PROTOCOL.golden_fixtures.reference_platform
UVR_FILES = ("UVR-DeNoise.pth", "vr_model_data.json", "mdx_model_data.json")


def model_files(root: Path) -> tuple[Path, Path, Protocol]:
    """Small stand-in model files, and the protocol with their digests frozen in."""
    checkpoint = root / "final.pt"
    checkpoint.write_bytes(b"checkpoint")
    uvr_dir = root / "models"
    uvr_dir.mkdir()
    digests = []
    for name in UVR_FILES:
        (uvr_dir / name).write_bytes(name.encode())
        digests.append(hashlib.sha256(name.encode()).hexdigest())
    execution = dataclasses.replace(
        PROTOCOL.system_execution,
        opgan_checkpoint_sha256=hashlib.sha256(b"checkpoint").hexdigest(),
        uvr_model_sha256=digests[0],
        uvr_vr_model_data_sha256=digests[1],
        uvr_mdx_model_data_sha256=digests[2],
    )
    return checkpoint, uvr_dir, dataclasses.replace(PROTOCOL, system_execution=execution)


@pytest.fixture
def goldens_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both golden steps pass, so the model check can be reached on any platform."""
    monkeypatch.setattr(run_preconditions, "verify_golden_fixtures", lambda *_, **__: None)
    frozen = PROTOCOL.degradation.golden.degraded_sha256
    monkeypatch.setattr(run_preconditions, "golden_degraded_sha256", lambda _: frozen)


class TestOrder:
    """Golden fixtures, then the degradation golden, then the model files."""

    def test_another_platform_aborts_before_any_model_file(self, tmp_path: Path) -> None:
        outcome = verify_run_preconditions(
            PROTOCOL,
            runtime_platform="linux_x86_64",
            opgan_checkpoint=tmp_path / "absent.pt",
            uvr_model_dir=tmp_path,
        )

        assert isinstance(outcome, RunAbortOutcome)
        assert outcome.reason is AbortReason.ENVIRONMENT_MISMATCH
        assert REFERENCE in outcome.detail and "absent.pt" not in outcome.detail
        assert outcome.logical_source_id is None

    def test_a_golden_fixture_failure_is_returned_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        failure = RunAbortOutcome(reason=AbortReason.GOLDEN_FIXTURE_FAILURE, detail="fixture")
        monkeypatch.setattr(run_preconditions, "verify_golden_fixtures", lambda *_, **__: failure)

        outcome = verify_run_preconditions(
            PROTOCOL, runtime_platform=REFERENCE, opgan_checkpoint=tmp_path, uvr_model_dir=tmp_path
        )

        assert outcome is failure

    def test_a_degradation_mismatch_aborts_before_the_model_files(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(run_preconditions, "verify_golden_fixtures", lambda *_, **__: None)
        monkeypatch.setattr(run_preconditions, "golden_degraded_sha256", lambda _: "0" * 64)

        outcome = verify_run_preconditions(
            PROTOCOL,
            runtime_platform=REFERENCE,
            opgan_checkpoint=tmp_path / "absent.pt",
            uvr_model_dir=tmp_path,
        )

        assert isinstance(outcome, RunAbortOutcome)
        assert outcome.reason is AbortReason.GOLDEN_FIXTURE_FAILURE
        assert outcome.detail.startswith("degradation golden produced " + "0" * 64)
        assert "absent.pt" not in outcome.detail
        assert outcome.logical_source_id is None


@pytest.mark.usefixtures("goldens_pass")
class TestModelFiles:
    """The four model files are compared with their frozen digests."""

    def test_matching_files_pass(self, tmp_path: Path) -> None:
        checkpoint, uvr_dir, protocol = model_files(tmp_path)

        outcome = verify_run_preconditions(
            protocol, runtime_platform=REFERENCE, opgan_checkpoint=checkpoint, uvr_model_dir=uvr_dir
        )

        assert outcome is None

    def test_a_changed_file_aborts_with_the_protocol_code(self, tmp_path: Path) -> None:
        checkpoint, uvr_dir, protocol = model_files(tmp_path)
        (uvr_dir / "vr_model_data.json").write_bytes(b"edited upstream")

        outcome = verify_run_preconditions(
            protocol, runtime_platform=REFERENCE, opgan_checkpoint=checkpoint, uvr_model_dir=uvr_dir
        )

        assert isinstance(outcome, RunAbortOutcome)
        assert outcome.reason is AbortReason(protocol.runner.model_digest_mismatch_abort)
        assert "vr_model_data.json has digest" in outcome.detail
        assert outcome.logical_source_id is None

    def test_every_failing_file_is_reported(self, tmp_path: Path) -> None:
        checkpoint, uvr_dir, protocol = model_files(tmp_path)
        checkpoint.write_bytes(b"retrained")
        (uvr_dir / "UVR-DeNoise.pth").unlink()

        failures = model_file_failures(protocol, opgan_checkpoint=checkpoint, uvr_model_dir=uvr_dir)

        assert len(failures) == 2
        assert "final.pt has digest" in failures[0]
        assert "UVR-DeNoise.pth is missing" in failures[1]

    def test_the_abort_names_every_failing_file(self, tmp_path: Path) -> None:
        checkpoint, uvr_dir, protocol = model_files(tmp_path)
        checkpoint.write_bytes(b"retrained")
        (uvr_dir / "mdx_model_data.json").write_bytes(b"edited upstream")

        outcome = verify_run_preconditions(
            protocol, runtime_platform=REFERENCE, opgan_checkpoint=checkpoint, uvr_model_dir=uvr_dir
        )

        assert isinstance(outcome, RunAbortOutcome)
        assert "final.pt has digest" in outcome.detail
        assert "mdx_model_data.json has digest" in outcome.detail

    def test_the_real_digests_reject_stand_in_files(self, tmp_path: Path) -> None:
        """Against the frozen digests, files that are not the real models all fail."""
        checkpoint, uvr_dir, _ = model_files(tmp_path)

        failures = model_file_failures(PROTOCOL, opgan_checkpoint=checkpoint, uvr_model_dir=uvr_dir)

        assert len(failures) == 4


@pytest.mark.skipif(
    platform_token() != REFERENCE, reason="the golden digests bind on the reference platform only"
)
def test_the_real_chain_passes_on_the_reference_platform(tmp_path: Path) -> None:
    """Real golden fixtures and degradation golden, then stand-in model files, no stubs."""
    checkpoint, uvr_dir, protocol = model_files(tmp_path)

    outcome = verify_run_preconditions(
        protocol,
        runtime_platform=platform_token(),
        opgan_checkpoint=checkpoint,
        uvr_model_dir=uvr_dir,
    )

    assert outcome is None


class TestDegradationGolden:
    def test_the_frozen_digest_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        frozen = PROTOCOL.degradation.golden.degraded_sha256
        monkeypatch.setattr(run_preconditions, "golden_degraded_sha256", lambda _: frozen)

        assert degradation_golden_failure(PROTOCOL) is None


class TestFileSha256:
    def test_chunks_hash_the_same_as_one_read(self, tmp_path: Path) -> None:
        path = tmp_path / "model.bin"
        data = bytes(range(256)) * 41  # 10,496 bytes, not a multiple of the chunk size
        path.write_bytes(data)

        assert file_sha256(path, chunk_bytes=1000) == hashlib.sha256(data).hexdigest()

    def test_a_missing_path_or_directory_is_none(self, tmp_path: Path) -> None:
        assert file_sha256(tmp_path / "absent") is None
        assert file_sha256(tmp_path) is None
