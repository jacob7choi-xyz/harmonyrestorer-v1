"""Tests for the systems stage's run: admissions selected, results and summary written, aborts."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from benchmark.protocol import Protocol, load_protocol
from benchmark.run_systems import (
    POPULATION_NAME,
    PROGRESS_NAME,
    RESULTS_NAME,
    SUMMARY_NAME,
    NamedSystem,
    _parse_args,
    build_production_systems,
    main,
    run_systems,
)
from benchmark.source_outcome import AbortReason, RunAbortOutcome
from benchmark.systems_pipeline import MANIFEST_NAME
from benchmark.systems_stage import HarnessFailure

RATE = 16000
FRAME = 32000
REMAINDER = 500
PEAK = 0.9
SOLO = "solo_piano"
CHAMBER = "chamber"


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol."""
    return load_protocol()


def make_samples() -> npt.NDArray[np.float64]:
    """Frame 0 silent (ineligible), frames 1 and 2 loud, and the peak in the remainder."""
    samples = np.zeros(3 * FRAME + REMAINDER)
    samples[FRAME : 3 * FRAME] = np.random.default_rng(11).uniform(-0.25, 0.25, 2 * FRAME)
    samples[-1] = PEAK
    return samples


def write_canonical(directory: Path, name: str, samples: npt.NDArray[np.float64]) -> dict[str, Any]:
    """Write a canonical file and return its manifest entry, with the file's real digest."""
    data = samples.astype("<f8").tobytes()
    path = directory / f"{name}.f64"
    path.write_bytes(data)
    return {
        "canonical_path": str(path),
        "canonical_sha256": hashlib.sha256(data).hexdigest(),
        "samples": samples.size,
    }


def entry(stratum: str, partition: str, position: int, located: dict[str, Any]) -> dict[str, Any]:
    """One admission as the walk writes it, with the located canonical file."""
    return {
        "stratum": stratum,
        "partition": partition,
        "position": position,
        "identifier": f"{stratum}-{position}-{partition}",
        "cluster": "c",
        "raw_sha256": "6" * 64,
        "decoded_pcm_sha256": "7" * 64,
        **located,
    }


def missing_file(name: str) -> dict[str, Any]:
    """A canonical location that does not exist, so a run that touched it would fail loudly.

    The digest is derived from the name, so no two missing admissions share one.
    """
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return {"canonical_path": f"/nonexistent/{name}.f64", "canonical_sha256": digest, "samples": 1}


def manifest(protocol: Protocol, strata: dict[str, Any], admissions: list[dict[str, Any]]) -> dict:
    """A manifest naming the loaded protocol. Any stratum not given is short with no admissions."""
    every_stratum = {
        SOLO: {"status": "short", "admitted": 0},
        CHAMBER: {"status": "short", "admitted": 0},
    }
    every_stratum.update(strata)
    return {
        "protocol_sha256": protocol.source_sha256,
        "universe_sha256": {"solo_piano": "3" * 64, "chamber": "4" * 64},
        "archive_sha256": None,
        "strata": every_stratum,
        "admissions": admissions,
    }


def write_walk(directory: Path, data: dict[str, Any]) -> Path:
    """Write the walk directory's manifest and return the directory."""
    directory.mkdir(exist_ok=True)
    (directory / MANIFEST_NAME).write_text(json.dumps(data), encoding="utf-8")
    return directory


def standard_walk(tmp_path: Path, protocol: Protocol) -> Path:
    """A walk with a complete piano stratum (one test recording, one demo recording whose file
    does not exist) and a short chamber stratum (one test recording whose file does not exist)."""
    located = write_canonical(tmp_path, "test-piano", make_samples())
    data = manifest(
        protocol,
        {
            SOLO: {"status": "complete", "admitted": 2},
            CHAMBER: {"status": "short", "admitted": 1},
        },
        [
            entry(SOLO, "test", 1, located),
            entry(SOLO, "demo", 2, missing_file("demo")),
            entry(CHAMBER, "test", 1, missing_file("chamber")),
        ],
    )
    return write_walk(tmp_path / "walk", data)


class EchoSystem:
    """A system that echoes its input, or raises for one frame index if asked to."""

    def __init__(self, name: str, output_dir: Path, fail_frame: int | None = None) -> None:
        """Store the name, the output folder, and the frame index that fails, if any."""
        self.name = name
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._fail_frame = fail_frame

    def denoise(self, input_path: Path) -> Path:
        """Raise for the failing frame, otherwise write the input back as the output."""
        if self._fail_frame is not None and input_path.stem.endswith(f"-{self._fail_frame:06d}"):
            raise RuntimeError("engine fault")
        samples, _ = sf.read(input_path, dtype="float32", always_2d=False)
        output = self._output_dir / f"{input_path.stem}_out.wav"
        sf.write(output, samples, RATE, subtype="FLOAT")
        return output


def passing(protocol: Protocol, **_: object) -> None:
    """A precondition check that every run passes."""
    return None


def factory(fail_uvr_frame: int | None = 1) -> Any:
    """A systems factory: OpGAN echoes; UVR echoes except for one frame index."""

    def build(protocol: Protocol, run_dir: Path, checkpoint: Path, models: Path) -> list[Any]:
        outputs = run_dir / "outputs"
        return [
            EchoSystem("opgan", outputs / "opgan"),
            EchoSystem("uvr", outputs / "uvr", fail_frame=fail_uvr_frame),
        ]

    return build


def run(walk: Path, run_dir: Path, protocol: Protocol, **overrides: Any) -> dict[str, Any]:
    """Run the stage with passing preconditions and fake systems, overriding any argument."""
    arguments: dict[str, Any] = {
        "walk_dir": walk,
        "run_dir": run_dir,
        "protocol": protocol,
        "runtime_platform": protocol.golden_fixtures.reference_platform,
        "opgan_checkpoint": run_dir.parent / "final.pt",
        "uvr_model_dir": run_dir.parent / "models",
        "build_systems": factory(),
        "preconditions": passing,
    }
    arguments.update(overrides)
    return run_systems(**arguments)


class TestSelectionAndOutputs:
    """Only test admissions of complete strata are scored, and the outputs are written."""

    def test_only_the_complete_strata_test_recording_is_loaded_and_scored(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        walk = standard_walk(tmp_path, protocol)
        run_dir = tmp_path / "run"

        run(walk, run_dir, protocol)

        rows = [json.loads(line) for line in (run_dir / RESULTS_NAME).read_text().splitlines()]
        assert {row["stratum"] for row in rows} == {SOLO}
        assert {row["partition"] for row in rows} == {"test"}
        assert {row["frame_index"] for row in rows} == {1, 2}
        assert len(rows) == 2 * 3

    def test_the_summary_gives_coverage_per_system_and_marks_the_short_stratum(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        summary = run(standard_walk(tmp_path, protocol), tmp_path / "run", protocol)

        strata = summary["strata"]
        assert strata[SOLO]["coverage"] == {"opgan": 1.0, "uvr": 0.5}
        assert strata[SOLO]["eligible_frames"] == 2
        assert strata[SOLO]["test_recordings"] == 1
        assert strata[SOLO]["failure_kinds"] == {"opgan": {}, "uvr": {"exception": 1}}
        assert strata[CHAMBER] == {"status": "short", "scored": False}

    def test_the_baseline_is_never_counted_in_coverage(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        summary = run(standard_walk(tmp_path, protocol), tmp_path / "run", protocol)

        assert set(summary["strata"][SOLO]["coverage"]) == {"opgan", "uvr"}
        assert summary["baseline"] == "degraded_input"

    def test_results_summary_and_progress_marker_are_written_and_match_the_return(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        run_dir = tmp_path / "run"
        summary = run(standard_walk(tmp_path, protocol), run_dir, protocol)

        assert (run_dir / PROGRESS_NAME).is_file()
        written = json.loads((run_dir / SUMMARY_NAME).read_text(encoding="utf-8"))
        assert written == json.loads(json.dumps(summary))
        assert written["walk_manifest_sha256"]

    def test_the_population_file_lists_every_candidate_frame_of_each_scored_recording(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        run_dir = tmp_path / "run"
        run(standard_walk(tmp_path, protocol), run_dir, protocol)

        rows = [json.loads(line) for line in (run_dir / POPULATION_NAME).read_text().splitlines()]
        assert len(rows) == 1
        assert rows[0]["stratum"] == SOLO
        assert rows[0]["frame_count"] == 3
        assert rows[0]["eligible_frames"] == [1, 2]
        assert rows[0]["dropped_samples"] == REMAINDER
        assert [frame["index"] for frame in rows[0]["frames"]] == [0, 1, 2]


class TestRefusals:
    """Runs that cannot start, or cannot be trusted, stop before any result is written."""

    def test_an_existing_run_directory_is_refused_and_left_untouched(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        walk = standard_walk(tmp_path, protocol)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "keep.txt").write_text("previous run", encoding="utf-8")

        with pytest.raises(HarnessFailure, match="already exists"):
            run(walk, run_dir, protocol)

        assert sorted(p.name for p in run_dir.iterdir()) == ["keep.txt"]

    def test_a_precondition_abort_stops_before_engines_are_built_and_files_are_read(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        walk = standard_walk(tmp_path, protocol)
        built: list[object] = []

        def abort(protocol: Protocol, **_: object) -> RunAbortOutcome:
            return RunAbortOutcome(reason=AbortReason.GOLDEN_FIXTURE_FAILURE, detail="drifted")

        with pytest.raises(HarnessFailure, match="preconditions failed"):
            run(
                walk,
                tmp_path / "run",
                protocol,
                preconditions=abort,
                build_systems=lambda *a: built.append(a),
            )

        assert built == []
        assert not (tmp_path / "run").exists()

    def test_a_manifest_naming_another_protocol_stops_before_the_preconditions(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        data = manifest(protocol, {}, [])
        data["protocol_sha256"] = "8" * 64
        calls: list[object] = []

        def record(protocol: Protocol, **_: object) -> None:
            calls.append(protocol)
            return None

        with pytest.raises(HarnessFailure, match="another protocol"):
            run(
                write_walk(tmp_path / "walk", data),
                tmp_path / "run",
                protocol,
                preconditions=record,
            )

        assert calls == []

    def test_a_conditioning_rate_that_differs_from_the_canonical_rate_is_refused(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        odd = dataclasses.replace(
            protocol, conditioning=dataclasses.replace(protocol.conditioning, sample_rate=22050)
        )

        with pytest.raises(HarnessFailure, match="conditioning rate"):
            run(standard_walk(tmp_path, protocol), tmp_path / "run", odd)

        assert not (tmp_path / "run").exists()

    def test_an_opgan_pin_that_differs_from_the_backend_is_refused(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        other = dataclasses.replace(protocol.system_execution, opgan_checkpoint_sha256="0" * 64)
        odd = dataclasses.replace(protocol, system_execution=other)

        with pytest.raises(HarnessFailure, match="OpGAN checkpoint digests"):
            run(standard_walk(tmp_path, protocol), tmp_path / "run", odd)

    def test_a_complete_stratum_with_no_test_recording_is_refused(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        data = manifest(
            protocol,
            {SOLO: {"status": "complete", "admitted": 1}},
            [entry(SOLO, "demo", 1, missing_file("demo"))],
        )

        with pytest.raises(HarnessFailure, match="no test admission"):
            run(write_walk(tmp_path / "walk", data), tmp_path / "run", protocol)


class TestEnginesAndCommandLine:
    """The production engines are wired to the checked model directory, and the CLI parses."""

    def test_the_separator_directory_is_set_to_the_checked_directory_before_the_engines_exist(
        self, tmp_path: Path, protocol: Protocol, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", "before")
        models = tmp_path / "models"

        opgan, uvr = build_production_systems(protocol, tmp_path / "run", tmp_path / "f.pt", models)

        assert os.environ["AUDIO_SEPARATOR_MODEL_DIR"] == str(models)
        assert isinstance(opgan, NamedSystem) and opgan.name == "opgan"
        assert isinstance(uvr, NamedSystem) and uvr.name == "uvr"
        assert (tmp_path / "run" / "outputs" / "opgan").is_dir()
        assert (tmp_path / "run" / "outputs" / "uvr").is_dir()

    def test_the_default_checkpoint_is_the_repository_checkpoint_path(self, tmp_path: Path) -> None:
        args = _parse_args(["--walk-dir", "w", "--run-dir", "r", "--uvr-model-dir", "m"])

        assert args.opgan_checkpoint == Path("checkpoints/final.pt")

    def test_a_harness_abort_on_the_command_line_exits_with_status_one(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        walk = write_walk(tmp_path / "walk", manifest(protocol, {}, []))
        run_dir = tmp_path / "run"

        status = main(
            [
                "--walk-dir",
                str(walk),
                "--run-dir",
                str(run_dir),
                "--uvr-model-dir",
                str(tmp_path / "models"),
                "--opgan-checkpoint",
                str(tmp_path / "missing.pt"),
            ]
        )

        assert status == 1
        assert not run_dir.exists()


class LoadOnFirstCall:
    """A system whose model loads on its first call, and whose later calls succeed."""

    def __init__(self, name: str, output_dir: Path, load_error: Exception | None = None) -> None:
        """Store the name, the output folder, and any error the first call's load raises."""
        self.name = name
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._load_error = load_error
        self._loaded = False
        self.calls: list[str] = []

    def denoise(self, input_path: Path) -> Path:
        """Record the call; the first call loads the model, then every call echoes its input."""
        self.calls.append(input_path.name)
        if not self._loaded:
            self._loaded = True
            if self._load_error is not None:
                raise self._load_error
        samples, _ = sf.read(input_path, dtype="float32", always_2d=False)
        output = self._output_dir / f"{input_path.stem}_out.wav"
        sf.write(output, samples, RATE, subtype="FLOAT")
        return output


def system_factory(opgan: Any, uvr: Any) -> Any:
    """A systems factory returning the two given systems."""

    def build(protocol: Protocol, run_dir: Path, checkpoint: Path, models: Path) -> list[Any]:
        return [opgan, uvr]

    return build


class TestWarmUpInTheRun:
    """A load fault in the warm-up aborts the run before any result is written."""

    def test_a_load_fault_in_the_warm_up_aborts_before_any_result_is_written(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        walk = standard_walk(tmp_path, protocol)
        run_dir = tmp_path / "run"
        opgan = LoadOnFirstCall("opgan", tmp_path / "fake_opgan", MemoryError("load"))
        uvr = EchoSystem("uvr", tmp_path / "fake_uvr")

        with pytest.raises(MemoryError, match="load"):
            run(walk, run_dir, protocol, build_systems=system_factory(opgan, uvr))

        assert opgan.calls == ["warm_up.wav"]
        assert not (run_dir / RESULTS_NAME).exists()
        assert not (run_dir / POPULATION_NAME).exists()
        assert not (run_dir / SUMMARY_NAME).exists()

    def test_a_wrong_format_warm_up_aborts_the_run(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        walk = standard_walk(tmp_path, protocol)
        run_dir = tmp_path / "run"
        opgan = EchoSystem("opgan", tmp_path / "fake_opgan")
        uvr = LoadOnFirstCall("uvr", tmp_path / "fake_uvr")
        original = uvr.denoise

        def wrong_rate(input_path: Path) -> Path:
            output = original(input_path)
            data, _ = sf.read(output, dtype="float32", always_2d=False)
            sf.write(output, data, 22050, subtype="FLOAT")
            return output

        uvr.denoise = wrong_rate

        with pytest.raises(HarnessFailure, match="warm-up of system uvr produced the wrong format"):
            run(walk, run_dir, protocol, build_systems=system_factory(opgan, uvr))

        assert not (run_dir / RESULTS_NAME).exists()

    def test_a_system_that_loads_on_its_first_call_produces_normal_results(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        walk = standard_walk(tmp_path, protocol)
        run_dir = tmp_path / "run"
        opgan = LoadOnFirstCall("opgan", tmp_path / "fake_opgan")
        uvr = EchoSystem("uvr", tmp_path / "fake_uvr")

        summary = run(walk, run_dir, protocol, build_systems=system_factory(opgan, uvr))

        assert opgan.calls[0] == "warm_up.wav"
        assert len(opgan.calls) == 1 + 2
        assert summary["strata"][SOLO]["coverage"]["opgan"] == 1.0
        text = (run_dir / RESULTS_NAME).read_text(encoding="utf-8")
        assert "warm_up" not in text
        assert len(text.splitlines()) == 2 * 3
