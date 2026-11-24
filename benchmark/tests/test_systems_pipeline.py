"""Tests for the systems pipeline: the walk manifest, and a recording's frames scored."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf
from systems_fakes import AUDIT_DIGEST, write_audit_trail

from benchmark.degradation import degrade_frame, frame_seed
from benchmark.frame_scoring import FrameScore, SystemFailure
from benchmark.protocol import Protocol, load_protocol
from benchmark.systems_pipeline import (
    AUDIT_NAME,
    BASELINE,
    MANIFEST_NAME,
    Admission,
    FailureKind,
    Partition,
    StageContext,
    StratumEntry,
    StratumStatus,
    load_walk_manifest,
    population_to_json,
    record_to_json,
    score_admission,
    warm_up_system,
    write_warm_up_input,
)
from benchmark.systems_stage import HarnessFailure

RATE = 16000
FRAME = 32000
REMAINDER = 500
PEAK = 0.9
STRATUM = "solo_piano"


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol."""
    return load_protocol()


def make_samples() -> npt.NDArray[np.float64]:
    """Three frames and a remainder: frame 0 silent (ineligible), frames 1 and 2 loud.

    The remainder holds the waveform's largest sample, so a peak taken over the complete
    waveform differs from one taken over the frames alone.
    """
    samples = np.zeros(3 * FRAME + REMAINDER)
    samples[FRAME : 3 * FRAME] = np.random.default_rng(7).uniform(-0.25, 0.25, 2 * FRAME)
    samples[-1] = PEAK
    return samples


def admission_for(
    directory: Path,
    samples: npt.NDArray[np.float64],
    *,
    name: str = "a",
    partition: Partition = Partition.TEST,
    position: int = 1,
) -> Admission:
    """Write a canonical file for the samples and describe it as a walk admission."""
    data = samples.astype("<f8").tobytes()
    path = directory / f"{name}_canonical.f64"
    path.write_bytes(data)
    return Admission(
        stratum=STRATUM,
        partition=partition,
        position=position,
        identifier=f"id-{name}",
        cluster=f"cluster-{name}",
        canonical_path=path,
        canonical_sha256=hashlib.sha256(data).hexdigest(),
        samples=samples.size,
        raw_sha256="1" * 64,
        decoded_pcm_sha256="2" * 64,
    )


class FakeSystem:
    """A system that records the input it received and answers with a chosen behaviour."""

    def __init__(
        self,
        name: str,
        output_dir: Path,
        behaviour: Callable[[Path, Path], Path] | None = None,
    ) -> None:
        """Store the name, the output folder, and the behaviour; default is to echo the input."""
        self.name = name
        self.output_dir = output_dir
        self._behaviour = behaviour or echo
        self.received: list[npt.NDArray[np.float32]] = []

    def denoise(self, input_path: Path) -> Path:
        """Record what arrived, then answer as the behaviour says."""
        samples, _ = sf.read(input_path, dtype="float32", always_2d=False)
        self.received.append(samples)
        return self._behaviour(input_path, self.output_dir)


def write_output(directory: Path, input_path: Path, samples: Any, rate: int = RATE) -> Path:
    """Write a float output WAV named after the input, as the engines name theirs."""
    output = directory / f"{input_path.stem}_out.wav"
    sf.write(output, samples, rate, subtype="FLOAT")
    return output


def echo(input_path: Path, directory: Path) -> Path:
    """Return the input's samples unchanged."""
    samples, _ = sf.read(input_path, dtype="float32", always_2d=False)
    return write_output(directory, input_path, samples)


def raising_on_frame(index: int) -> Callable[[Path, Path], Path]:
    """Echo every frame except the one with this index, whose call raises a RuntimeError."""

    def run(input_path: Path, directory: Path) -> Path:
        if input_path.stem.endswith(f"-{index:06d}"):
            raise RuntimeError("engine fault")
        return echo(input_path, directory)

    return run


def always_raising(error: BaseException) -> Callable[[Path, Path], Path]:
    """A behaviour that raises the given error on every call."""

    def run(input_path: Path, directory: Path) -> Path:
        raise error

    return run


def at_rate(rate: int) -> Callable[[Path, Path], Path]:
    """Echo the input at a different sample rate."""

    def run(input_path: Path, directory: Path) -> Path:
        samples, _ = sf.read(input_path, dtype="float32", always_2d=False)
        return write_output(directory, input_path, samples, rate)

    return run


def truncated(missing: int) -> Callable[[Path, Path], Path]:
    """Echo the input with its last `missing` samples dropped, beyond conditioning's tolerance."""

    def run(input_path: Path, directory: Path) -> Path:
        samples, _ = sf.read(input_path, dtype="float32", always_2d=False)
        return write_output(directory, input_path, samples[:-missing])

    return run


def context(protocol: Protocol, systems: list[FakeSystem], root: Path) -> StageContext:
    """A stage context whose input folder and marker live under the test's root."""
    inputs = root / "inputs"
    inputs.mkdir(exist_ok=True)
    return StageContext(
        protocol=protocol,
        systems=systems,
        input_dir=inputs,
        marker_path=root / "progress.json",
    )


def outputs_for(root: Path) -> Path:
    """A folder for a fake system's outputs."""
    folder = root / "outputs"
    folder.mkdir(exist_ok=True)
    return folder


class TestScoreAdmission:
    """A recording's eligible frames are scored for each system and the degraded input."""

    def test_only_eligible_frames_are_scored_for_every_system_and_the_baseline(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        admission = admission_for(tmp_path, make_samples())
        systems = [
            FakeSystem("opgan", outputs_for(tmp_path)),
            FakeSystem("uvr", outputs_for(tmp_path)),
        ]

        result = score_admission(admission, context(protocol, systems, tmp_path))

        assert result.eligible == (1, 2)
        seen = {(record.frame_index, record.system) for record in result.records}
        assert seen == {(i, s) for i in (1, 2) for s in ("opgan", "uvr", BASELINE)}

    def test_each_input_is_the_degraded_frame_with_its_true_index_and_the_complete_peak(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        samples = make_samples()
        admission = admission_for(tmp_path, samples)
        system = FakeSystem("opgan", outputs_for(tmp_path))

        score_admission(admission, context(protocol, [system], tmp_path))

        assert len(system.received) == 2
        for position, index in enumerate((1, 2)):
            expected = degrade_frame(
                samples[index * FRAME : (index + 1) * FRAME],
                recording_peak=PEAK,
                sample_rate=RATE,
                ranges=protocol.degradation.ranges,
                seed=frame_seed(protocol.degradation.seed, admission.canonical_sha256, index),
            )
            np.testing.assert_array_equal(system.received[position], expected.degraded)

    def test_the_baseline_is_scored_as_a_result_and_never_as_a_system_failure(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        admission = admission_for(tmp_path, make_samples())
        systems = [FakeSystem("opgan", outputs_for(tmp_path))]

        result = score_admission(admission, context(protocol, systems, tmp_path))

        baseline = [r for r in result.records if r.system == BASELINE]
        assert len(baseline) == 2
        assert all(isinstance(r.outcome, FrameScore) for r in baseline)
        assert all(r.failure_kind is None for r in baseline)


class TestSystemFailuresAreRecorded:
    """A system's failure stays on its record with the kind of failure that produced it."""

    def test_an_exception_in_the_call_is_recorded_as_that_systems_exception(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        admission = admission_for(tmp_path, make_samples())
        systems = [
            FakeSystem("opgan", outputs_for(tmp_path)),
            FakeSystem("uvr", outputs_for(tmp_path), raising_on_frame(1)),
        ]

        result = score_admission(admission, context(protocol, systems, tmp_path))

        failed = [r for r in result.records if r.system == "uvr" and r.frame_index == 1]
        assert len(failed) == 1
        assert isinstance(failed[0].outcome, SystemFailure)
        assert failed[0].failure_kind is FailureKind.EXCEPTION
        assert failed[0].exception_type == "RuntimeError"
        scored = [r for r in result.records if r.system == "uvr" and r.frame_index == 2]
        assert isinstance(scored[0].outcome, FrameScore)

    def test_a_wrong_format_output_is_recorded_as_an_output_format_failure(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        admission = admission_for(tmp_path, make_samples())
        systems = [FakeSystem("opgan", outputs_for(tmp_path), at_rate(22050))]

        result = score_admission(admission, context(protocol, systems, tmp_path))

        failed = [r for r in result.records if r.system == "opgan"]
        assert all(r.failure_kind is FailureKind.OUTPUT_FORMAT for r in failed)
        assert all(r.exception_type is None for r in failed)
        assert all(
            "22050" in r.outcome.reason for r in failed if isinstance(r.outcome, SystemFailure)
        )

    def test_an_output_conditioning_rejects_is_recorded_as_a_conditioning_failure(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        admission = admission_for(tmp_path, make_samples())
        missing = protocol.conditioning.max_length_difference_samples + 1
        systems = [FakeSystem("uvr", outputs_for(tmp_path), truncated(missing))]

        result = score_admission(admission, context(protocol, systems, tmp_path))

        failed = [r for r in result.records if r.system == "uvr"]
        assert all(r.failure_kind is FailureKind.CONDITIONING for r in failed)
        assert all(isinstance(r.outcome, SystemFailure) for r in failed)

    def test_the_json_record_names_the_failure_kind_and_the_exception(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        admission = admission_for(tmp_path, make_samples())
        systems = [FakeSystem("uvr", outputs_for(tmp_path), raising_on_frame(1))]

        result = score_admission(admission, context(protocol, systems, tmp_path))
        row = record_to_json(
            next(r for r in result.records if r.system == "uvr" and r.frame_index == 1)
        )

        assert row["outcome"] == "system_failure"
        assert row["failure_kind"] == "exception"
        assert row["exception_type"] == "RuntimeError"
        assert "results" not in row
        scored = record_to_json(next(r for r in result.records if r.system == BASELINE))
        assert scored["outcome"] == "scored"
        assert set(scored["results"]) == set(protocol.metrics.primary)


class TestHarnessFailuresAbortTheRecording:
    """Failures outside a system's call raise, and no system is charged for them."""

    def test_a_recording_with_no_eligible_frame_aborts_before_any_call(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        admission = admission_for(tmp_path, np.zeros(3 * FRAME + REMAINDER))
        system = FakeSystem("opgan", outputs_for(tmp_path))

        with pytest.raises(HarnessFailure, match="no eligible frame"):
            score_admission(admission, context(protocol, [system], tmp_path))

        assert system.received == []

    def test_a_canonical_file_that_no_longer_matches_its_digest_aborts_before_any_call(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        admission = admission_for(tmp_path, make_samples())
        admission.canonical_path.write_bytes(make_samples()[::-1].astype("<f8").tobytes())
        system = FakeSystem("opgan", outputs_for(tmp_path))

        with pytest.raises(HarnessFailure, match="canonical file"):
            score_admission(admission, context(protocol, [system], tmp_path))

        assert system.received == []

    def test_an_input_that_cannot_be_written_aborts_before_any_call(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        admission = admission_for(tmp_path, make_samples())
        system = FakeSystem("opgan", outputs_for(tmp_path))
        broken = StageContext(
            protocol=protocol,
            systems=[system],
            input_dir=tmp_path / "missing",
            marker_path=tmp_path / "progress.json",
        )

        with pytest.raises(HarnessFailure, match="could not be written"):
            score_admission(admission, broken)

        assert system.received == []

    def test_a_marker_that_cannot_be_written_aborts_before_the_call(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        admission = admission_for(tmp_path, make_samples())
        system = FakeSystem("opgan", outputs_for(tmp_path))
        broken = StageContext(
            protocol=protocol,
            systems=[system],
            input_dir=tmp_path,
            marker_path=tmp_path / "missing" / "progress.json",
        )

        with pytest.raises(HarnessFailure, match="progress marker"):
            score_admission(admission, broken)

        assert system.received == []


def valid_manifest(admissions: list[dict[str, Any]]) -> dict[str, Any]:
    """A manifest the contract allows, built from the admissions given."""
    return {
        "protocol_sha256": PROTOCOL_DIGEST,
        "universe_sha256": {"solo_piano": "3" * 64, "chamber": "4" * 64},
        "archive_sha256": None,
        "strata": {
            STRATUM: {"status": "complete", "admitted": len(admissions)},
            "chamber": {"status": "short", "admitted": 0},
        },
        "audit_trail_sha256": AUDIT_DIGEST,
        "admissions": admissions,
    }


PROTOCOL_DIGEST = "5" * 64


def admission_entry(position: int, digest: str | None = None) -> dict[str, Any]:
    """One admission as the walk writes it."""
    return {
        "stratum": STRATUM,
        "partition": "test",
        "position": position,
        "identifier": f"id-{position}",
        "cluster": f"cluster-{position}",
        "canonical_path": f"/abs/canonical-{position}.f64",
        "canonical_sha256": digest or f"{position:064x}",
        "samples": 96500,
        "raw_sha256": "6" * 64,
        "decoded_pcm_sha256": "7" * 64,
    }


PLACEHOLDER_CANONICAL_PREFIX = "/abs/"


def _admissions_of(data: object) -> list[dict[str, Any]]:
    """The admission objects of a manifest mapping, or none if its shape is not a mapping."""
    if not isinstance(data, dict) or not isinstance(data.get("admissions"), list):
        return []
    return [entry for entry in data["admissions"] if isinstance(entry, dict)]


def write_manifest(directory: Path, data: object) -> Path:
    """Write the manifest JSON into the directory and return the manifest's path.

    The placeholder canonical paths in `admission_entry` are moved into the directory, where the
    walk writes its canonical files. A test that sets its own canonical path keeps that path.
    """
    for entry in _admissions_of(data):
        if str(entry.get("canonical_path", "")).startswith(PLACEHOLDER_CANONICAL_PREFIX):
            name = Path(entry["canonical_path"]).name
            entry["canonical_path"] = str(directory / "candidates" / name)
    path = directory / MANIFEST_NAME
    path.write_text(json.dumps(data), encoding="utf-8")
    write_audit_trail(directory)
    return path


class TestLoadWalkManifest:
    """The manifest is read with exact keys and types, and refused otherwise."""

    def test_a_contract_manifest_loads_into_typed_admissions(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path, valid_manifest([admission_entry(1), admission_entry(2)]))

        manifest = load_walk_manifest(path, PROTOCOL_DIGEST)

        assert manifest.strata == {
            STRATUM: StratumEntry(StratumStatus.COMPLETE, 2),
            "chamber": StratumEntry(StratumStatus.SHORT, 0),
        }
        assert [a.position for a in manifest.admissions] == [1, 2]
        assert manifest.admissions[0].partition is Partition.TEST
        assert manifest.archive_sha256 is None
        assert manifest.audit_trail_sha256 == AUDIT_DIGEST

    @pytest.mark.parametrize(
        "mutate",
        [
            pytest.param(lambda d: d.update(extra=1), id="extra_top_level_key"),
            pytest.param(lambda d: d.pop("strata"), id="missing_top_level_key"),
            pytest.param(lambda d: d["strata"].pop("chamber"), id="stratum_missing_from_strata"),
            pytest.param(lambda d: d.update(protocol_sha256="8" * 64), id="other_protocol"),
            pytest.param(lambda d: d.pop("audit_trail_sha256"), id="audit_digest_missing"),
            pytest.param(lambda d: d["universe_sha256"].pop("chamber"), id="universe_key_missing"),
            pytest.param(
                lambda d: d["strata"][STRATUM].update(status="partial"), id="unknown_status"
            ),
            pytest.param(lambda d: d["strata"][STRATUM].update(admitted=3), id="admitted_count"),
            pytest.param(lambda d: d["admissions"][1].update(position=3), id="position_gap"),
            pytest.param(lambda d: d["admissions"][0].update(position=True), id="bool_position"),
            pytest.param(lambda d: d["admissions"][0].update(partition="train"), id="partition"),
            pytest.param(
                lambda d: d["admissions"][0].update(canonical_path="relative.f64"),
                id="relative_path",
            ),
            pytest.param(lambda d: d["admissions"][0].update(samples=0), id="empty_recording"),
            pytest.param(
                lambda d: d["admissions"][0].update(canonical_sha256="ABC"), id="bad_digest"
            ),
            pytest.param(
                lambda d: d["admissions"][1].update(
                    canonical_sha256=d["admissions"][0]["canonical_sha256"]
                ),
                id="shared_recording",
            ),
            pytest.param(
                lambda d: d["admissions"][0].update(stratum="orchestra"), id="unknown_stratum"
            ),
        ],
    )
    def test_each_departure_from_the_contract_is_refused(
        self, tmp_path: Path, mutate: Callable[[dict[str, Any]], object]
    ) -> None:
        data = valid_manifest([admission_entry(1), admission_entry(2)])
        mutate(data)
        path = write_manifest(tmp_path, data)

        with pytest.raises(HarnessFailure, match="walk manifest refused"):
            load_walk_manifest(path, PROTOCOL_DIGEST)

    def test_a_file_that_is_not_json_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / MANIFEST_NAME
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(HarnessFailure, match="not UTF-8 JSON"):
            load_walk_manifest(path, PROTOCOL_DIGEST)

    def test_a_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(HarnessFailure, match="cannot be read"):
            load_walk_manifest(tmp_path / MANIFEST_NAME, PROTOCOL_DIGEST)

    def test_a_missing_audit_trail_is_refused(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path, valid_manifest([admission_entry(1), admission_entry(2)]))
        (tmp_path / AUDIT_NAME).unlink()

        with pytest.raises(HarnessFailure, match=r"audit_trail\.jsonl is missing"):
            load_walk_manifest(path, PROTOCOL_DIGEST)

    def test_an_edited_audit_trail_is_refused(self, tmp_path: Path) -> None:
        path = write_manifest(tmp_path, valid_manifest([admission_entry(1), admission_entry(2)]))
        write_audit_trail(tmp_path, b'{"edited": true}\n')

        with pytest.raises(HarnessFailure, match="hashes to"):
            load_walk_manifest(path, PROTOCOL_DIGEST)

    def test_an_audit_digest_that_is_not_64_lowercase_hex_is_refused_before_the_trail_is_read(
        self, tmp_path: Path
    ) -> None:
        data = valid_manifest([admission_entry(1), admission_entry(2)])
        data["audit_trail_sha256"] = "ABC"
        path = write_manifest(tmp_path, data)

        with pytest.raises(HarnessFailure, match="audit_trail_sha256 must be 64 lowercase hex"):
            load_walk_manifest(path, PROTOCOL_DIGEST)


class TestPopulation:
    """Every candidate frame's verdict is recorded, eligible or not, with the dropped count."""

    def test_the_silent_frame_is_recorded_as_below_the_floor_with_null_rms(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        admission = admission_for(tmp_path, make_samples())
        systems = [FakeSystem("opgan", outputs_for(tmp_path))]

        result = score_admission(admission, context(protocol, systems, tmp_path))
        row = population_to_json(result)

        assert row["frame_count"] == 3
        assert row["dropped_samples"] == REMAINDER
        assert row["eligible_frames"] == [1, 2]
        frames = row["frames"]
        assert isinstance(frames, list)
        assert frames[0]["rms_dbfs"] is None
        assert "FRAME_BELOW_RMS_FLOOR" in frames[0]["reasons"]
        assert frames[1]["reasons"] == []
        assert isinstance(frames[1]["rms_dbfs"], float)
        json.dumps(row, allow_nan=False)


class TestWarmUp:
    """The warm-up call loads each system before any frame, and a fault in it aborts the run."""

    def test_the_warm_up_input_is_one_frame_of_mono_float_audio_at_the_rate(
        self, tmp_path: Path
    ) -> None:
        target = tmp_path / "warm_up.wav"

        write_warm_up_input(target, samples=FRAME, sample_rate=RATE)

        info = sf.info(target)
        assert (info.samplerate, info.channels, info.subtype) == (RATE, 1, "FLOAT")
        assert info.frames == FRAME

    def test_a_warm_up_call_that_raises_propagates_after_the_marker_names_it(
        self, tmp_path: Path
    ) -> None:
        warm_up = tmp_path / "warm_up.wav"
        write_warm_up_input(warm_up, samples=FRAME, sample_rate=RATE)
        system = FakeSystem(
            "opgan", outputs_for(tmp_path), always_raising(MemoryError("no memory"))
        )

        with pytest.raises(MemoryError, match="no memory"):
            warm_up_system(system, warm_up, tmp_path / "progress.json", RATE)
        assert "opgan" in (tmp_path / "progress.json").read_text()

    def test_a_warm_up_output_in_the_wrong_format_aborts_and_names_the_system(
        self, tmp_path: Path
    ) -> None:
        warm_up = tmp_path / "warm_up.wav"
        write_warm_up_input(warm_up, samples=FRAME, sample_rate=RATE)
        system = FakeSystem("uvr", outputs_for(tmp_path), at_rate(22050))

        with pytest.raises(HarnessFailure, match="warm-up of system uvr produced the wrong format"):
            warm_up_system(system, warm_up, tmp_path / "progress.json", RATE)

    def test_a_warm_up_that_succeeds_returns_nothing_and_is_not_scored(
        self, tmp_path: Path
    ) -> None:
        warm_up = tmp_path / "warm_up.wav"
        write_warm_up_input(warm_up, samples=FRAME, sample_rate=RATE)
        system = FakeSystem("opgan", outputs_for(tmp_path))

        assert warm_up_system(system, warm_up, tmp_path / "progress.json", RATE) is None
        assert system.received[0].size == FRAME

    def test_the_marker_names_the_warm_up_phase_before_the_call_is_made(
        self, tmp_path: Path
    ) -> None:
        warm_up = tmp_path / "warm_up.wav"
        write_warm_up_input(warm_up, samples=FRAME, sample_rate=RATE)
        marker = tmp_path / "progress.json"
        seen: list[dict[str, str]] = []

        class ReadsMarker:
            """Records the marker as it stands when the warm-up call begins."""

            name = "opgan"

            def denoise(self, input_path: Path) -> Path:
                seen.append(json.loads(marker.read_text(encoding="utf-8")))
                return echo(input_path, tmp_path)

        warm_up_system(ReadsMarker(), warm_up, marker, RATE)

        assert seen == [{"system": "opgan", "phase": "warm_up"}]

    def test_a_marker_that_cannot_be_written_aborts_before_the_call(self, tmp_path: Path) -> None:
        warm_up = tmp_path / "warm_up.wav"
        write_warm_up_input(warm_up, samples=FRAME, sample_rate=RATE)
        system = FakeSystem("opgan", outputs_for(tmp_path))

        with pytest.raises(HarnessFailure, match="warm-up marker"):
            warm_up_system(system, warm_up, tmp_path / "missing" / "progress.json", RATE)

        assert system.received == []


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("nan"), id="NaN"),
        pytest.param(float("inf"), id="Infinity"),
        pytest.param(float("-inf"), id="-Infinity"),
    ],
)
def test_a_non_finite_json_constant_in_the_walk_manifest_is_refused(
    tmp_path: Path, value: float
) -> None:
    """A walk manifest holding NaN, Infinity or -Infinity is refused before any field is read."""
    entry = admission_entry(1)
    entry["samples"] = value
    path = write_manifest(tmp_path, valid_manifest([entry]))
    with pytest.raises(HarnessFailure, match="non-finite JSON value"):
        load_walk_manifest(path, PROTOCOL_DIGEST)


def test_a_canonical_path_outside_the_walk_directory_is_refused(tmp_path: Path) -> None:
    """An admission whose canonical file resolves outside the walk directory is refused."""
    entry = admission_entry(1)
    entry["canonical_path"] = str(tmp_path.parent / "outside.f64")
    path = write_manifest(tmp_path, valid_manifest([entry]))
    with pytest.raises(HarnessFailure, match="resolves outside the walk directory"):
        load_walk_manifest(path, PROTOCOL_DIGEST)
