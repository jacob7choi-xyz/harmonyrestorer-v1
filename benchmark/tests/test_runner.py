"""Tests for the canonicalisation runner: preconditions, one worker per source, exit mapping.

Every way a worker can end is produced by a real child process, either the real worker or a
short script standing in for it, so the exit codes are the operating system's rather than
assumed ones. The footprint is read on macOS only, so only those assertions are skipped
elsewhere.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from benchmark import runner
from benchmark.canonical_worker import (
    CANONICAL_FILE,
    EXIT_MEMORY_ERROR,
    RECORD_FILE,
    canonicalise_source,
)
from benchmark.outcome_wire import outcome_to_wire
from benchmark.protocol import Protocol, load_protocol
from benchmark.runner import (
    WORKER_LOG,
    Candidate,
    CanonicalFile,
    run_canonicalisation,
    run_source,
)
from benchmark.source_outcome import (
    AbortReason,
    IneligibleReason,
    RunAbortOutcome,
    SourceIneligibleOutcome,
)

RATE = 16000
SOURCE_ID = "item/tones.flac"
RAW_SHA256 = "a" * 64


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


def script(code: str) -> runner.WorkerCommand:
    """A stand-in worker that runs *code* with its output directory as `out`."""

    def command(candidate: Candidate, out_dir: Path) -> list[str]:
        return [
            sys.executable,
            "-c",
            f"from pathlib import Path\nout = Path({str(out_dir)!r})\n{code}",
        ]

    return command


def writes_record(record: object, canonical: bytes | None = None) -> runner.WorkerCommand:
    """A stand-in worker that leaves *record*, and optionally canonical bytes, then exits 0."""
    lines = [f"(out / {RECORD_FILE!r}).write_text({json.dumps(record)!r})"]
    if canonical is not None:
        lines.insert(0, f"(out / {CANONICAL_FILE!r}).write_bytes({canonical!r})")
    return script("\n".join(lines))


def canonical_record(
    protocol: Protocol, values: np.ndarray, **changes: object
) -> dict[str, object]:
    """A canonical record for *values*, with fields replaced per test."""
    result: dict[str, object] = {
        "kind": "canonical",
        "sha256": hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
        "samples": int(values.size),
        "logical_source_id": SOURCE_ID,
        "raw_sha256": RAW_SHA256,
        "decoded_pcm_sha256": "b" * 64,
    }
    result.update(changes)
    return {"protocol_sha256": protocol.source_sha256, "result": result}


def run(
    protocol: Protocol,
    tmp_path: Path,
    command: runner.WorkerCommand,
    path: Path | None = None,
    expected_sha256: str | None = None,
) -> runner.SourceRun:
    """Run one candidate through `run_source` with *command* as its worker."""
    candidate = Candidate(
        SOURCE_ID, path if path is not None else tmp_path / "unused.flac", expected_sha256
    )
    return run_source(candidate, tmp_path / "source-000000", protocol, command=command)


def assert_aborts(source_run: runner.SourceRun, reason: AbortReason) -> None:
    """The source aborted the run under *reason*, naming the source and recording no verdict."""
    assert isinstance(source_run.outcome, RunAbortOutcome)
    assert source_run.outcome.reason is reason
    assert source_run.outcome.logical_source_id == SOURCE_ID


class TestAcceptedOutcomes:
    """A record is accepted from a worker that exited 0 and left one that holds together."""

    def test_the_real_worker_canonicalises_a_source(
        self, protocol: Protocol, tmp_path: Path, tone_flac: Path
    ) -> None:
        source_run = run(protocol, tmp_path, runner.worker_command, tone_flac)
        assert source_run.returncode == 0
        assert isinstance(source_run.outcome, CanonicalFile)
        assert source_run.outcome.samples == RATE
        assert source_run.outcome.path == tmp_path / "source-000000" / CANONICAL_FILE
        if sys.platform == "darwin":
            assert source_run.peak_footprint_bytes is not None
            assert source_run.peak_footprint_bytes > 0
        else:
            assert source_run.peak_footprint_bytes is None

    def test_an_ineligible_source_is_recorded_as_the_worker_reported_it(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        path = tmp_path / "noise.flac"
        path.write_bytes(b"not audio at all" * 64)
        source_run = run(protocol, tmp_path, runner.worker_command, path)
        assert isinstance(source_run.outcome, SourceIneligibleOutcome)
        assert source_run.outcome.primary_reason_code is IneligibleReason.SOURCE_INSPECTION_REJECTED

    def test_an_abort_the_worker_recorded_stands_as_recorded(
        self, protocol: Protocol, tmp_path: Path, tone_flac: Path
    ) -> None:
        candidate = Candidate(SOURCE_ID, tone_flac, expected_sha256="0" * 64)
        source_run = run_source(candidate, tmp_path / "source-000000", protocol)
        assert source_run.returncode == 0
        assert_aborts(source_run, AbortReason.SOURCE_HASH_MISMATCH)

    def test_the_worker_output_goes_to_its_log(self, protocol: Protocol, tmp_path: Path) -> None:
        run(protocol, tmp_path, script("import sys\nprint('to stderr', file=sys.stderr)"))
        assert "to stderr" in (tmp_path / "source-000000" / WORKER_LOG).read_text()


class TestCapacityExits:
    """A caught MemoryError reported by status, or death by SIGKILL, is a capacity abort."""

    @pytest.mark.parametrize(
        ("code", "returncode"),
        [
            (f"raise SystemExit({EXIT_MEMORY_ERROR})", EXIT_MEMORY_ERROR),
            ("import os, signal\nos.kill(os.getpid(), signal.SIGKILL)", -9),
        ],
        ids=["caught_memoryerror", "sigkill"],
    )
    def test_a_capacity_exit_aborts_with_the_capacity_code(
        self, protocol: Protocol, tmp_path: Path, code: str, returncode: int
    ) -> None:
        source_run = run(protocol, tmp_path, script(code))
        assert source_run.returncode == returncode
        assert_aborts(source_run, AbortReason.ENVIRONMENT_CAPACITY_FAILURE)

    def test_a_capacity_exit_is_not_overridden_by_a_record(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        """A record is read only after a 0 exit, so one left behind is never a verdict."""
        record = {"protocol_sha256": protocol.source_sha256, "result": {}}
        code = f"(out / {RECORD_FILE!r}).write_text({json.dumps(record)!r})\nraise SystemExit({EXIT_MEMORY_ERROR})"
        assert_aborts(
            run(protocol, tmp_path, script(code)), AbortReason.ENVIRONMENT_CAPACITY_FAILURE
        )

    @pytest.mark.skipif(sys.platform != "darwin", reason="footprint is macOS only")
    def test_a_killed_worker_still_has_a_footprint(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        code = "import numpy as np, os, signal\na = np.ones(100_000_000 // 8)\nos.kill(os.getpid(), signal.SIGKILL)"
        source_run = run(protocol, tmp_path, script(code))
        assert source_run.peak_footprint_bytes is not None
        assert source_run.peak_footprint_bytes >= 100_000_000


class TestUnattributedEnds:
    """Any other end aborts with the internal error naming the source, never a verdict."""

    @pytest.mark.parametrize(
        ("code", "returncode"),
        [
            ("import ctypes\nctypes.string_at(0)", -11),
            ("raise RuntimeError('defect')", 1),
            ("pass", 0),
        ],
        ids=["segfault", "uncaught_exception", "exit_0_without_a_record"],
    )
    def test_a_death_the_runner_cannot_attribute_aborts(
        self, protocol: Protocol, tmp_path: Path, code: str, returncode: int
    ) -> None:
        source_run = run(protocol, tmp_path, script(code))
        assert source_run.returncode == returncode
        assert_aborts(source_run, AbortReason.INTERNAL_CANONICALISATION_ERROR)

    def test_a_record_that_is_not_json_aborts(self, protocol: Protocol, tmp_path: Path) -> None:
        command = script(f"(out / {RECORD_FILE!r}).write_text('{{truncated')")
        assert_aborts(run(protocol, tmp_path, command), AbortReason.INTERNAL_CANONICALISATION_ERROR)

    def test_a_record_with_unexpected_keys_aborts(self, protocol: Protocol, tmp_path: Path) -> None:
        """Otherwise valid, with its file in place, so only the key check refuses it."""
        samples = np.zeros(4)
        record = {**canonical_record(protocol, samples), "verdict": "fine"}
        command = writes_record(record, samples.astype("<f8").tobytes())
        assert_aborts(run(protocol, tmp_path, command), AbortReason.INTERNAL_CANONICALISATION_ERROR)

    def test_a_valid_record_from_a_worker_that_then_failed_is_not_accepted(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        """A record is read only after a 0 exit, so a worker that wrote one and then died
        with status 1 leaves no verdict."""
        samples = np.zeros(4)
        record = json.dumps(canonical_record(protocol, samples))
        code = (
            f"(out / {CANONICAL_FILE!r}).write_bytes({samples.astype('<f8').tobytes()!r})\n"
            f"(out / {RECORD_FILE!r}).write_text({record!r})\n"
            "raise SystemExit(1)"
        )
        source_run = run(protocol, tmp_path, script(code))
        assert source_run.returncode == 1
        assert_aborts(source_run, AbortReason.INTERNAL_CANONICALISATION_ERROR)

    def test_a_record_under_another_protocol_aborts(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        samples = np.zeros(4)
        record = canonical_record(protocol, samples)
        record["protocol_sha256"] = "0" * 64
        command = writes_record(record, samples.astype("<f8").tobytes())
        assert_aborts(run(protocol, tmp_path, command), AbortReason.INTERNAL_CANONICALISATION_ERROR)

    @pytest.mark.parametrize(
        ("changes", "written"),
        [
            ({}, None),
            ({"sha256": "0" * 64}, np.zeros(4)),
            ({"samples": 5}, np.zeros(4)),
            ({"samples": True}, np.zeros(1)),
            ({"extra": 1}, np.zeros(4)),
            ({"logical_source_id": "item/other.flac"}, np.zeros(4)),
            ({"raw_sha256": "A" * 64}, np.zeros(4)),
            ({"decoded_pcm_sha256": None}, np.zeros(4)),
        ],
        ids=[
            "file_missing",
            "digest_differs",
            "count_differs",
            "count_is_a_bool",
            "extra_key",
            "names_another_source",
            "raw_digest_uppercase",
            "decoded_digest_missing",
        ],
    )
    def test_a_canonical_record_that_disagrees_with_its_file_aborts(
        self,
        protocol: Protocol,
        tmp_path: Path,
        changes: dict[str, object],
        written: np.ndarray | None,
    ) -> None:
        record = canonical_record(protocol, np.zeros(4) if written is None else written, **changes)
        canonical = None if written is None else written.astype("<f8").tobytes()
        command = writes_record(record, canonical)
        assert_aborts(run(protocol, tmp_path, command), AbortReason.INTERNAL_CANONICALISATION_ERROR)

    def test_a_canonical_record_that_holds_together_is_accepted(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        """The control for the refusals around it: the stand-in record is itself valid."""
        samples = np.zeros(4)
        command = writes_record(
            canonical_record(protocol, samples), samples.astype("<f8").tobytes()
        )
        source_run = run(protocol, tmp_path, command, expected_sha256=RAW_SHA256)
        assert isinstance(source_run.outcome, CanonicalFile)
        assert source_run.outcome.raw_sha256 == RAW_SHA256
        assert source_run.outcome.decoded_pcm_sha256 == "b" * 64

    def test_a_canonical_record_for_other_bytes_aborts(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        """Its raw digest is well formed but is not the one the candidate was staged with."""
        samples = np.zeros(4)
        command = writes_record(
            canonical_record(protocol, samples), samples.astype("<f8").tobytes()
        )
        source_run = run(protocol, tmp_path, command, expected_sha256="c" * 64)
        assert_aborts(source_run, AbortReason.INTERNAL_CANONICALISATION_ERROR)

    def test_an_ineligible_record_for_other_bytes_aborts(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        """A verdict is charged to a source only for the bytes it was staged with."""
        noise = tmp_path / "noise.flac"
        noise.write_bytes(b"not audio at all" * 64)
        outcome = canonicalise_source(
            noise, logical_source_id=SOURCE_ID, expected_sha256=None, protocol=protocol
        )
        assert isinstance(outcome, SourceIneligibleOutcome)
        record = {"protocol_sha256": protocol.source_sha256, "result": outcome_to_wire(outcome)}
        source_run = run(protocol, tmp_path, writes_record(record), expected_sha256="c" * 64)
        assert_aborts(source_run, AbortReason.INTERNAL_CANONICALISATION_ERROR)

    def test_an_ineligible_record_for_the_staged_bytes_is_accepted(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        noise = tmp_path / "noise.flac"
        noise.write_bytes(b"not audio at all" * 64)
        digest = hashlib.sha256(noise.read_bytes()).hexdigest()
        source_run = run(protocol, tmp_path, runner.worker_command, noise, expected_sha256=digest)
        assert isinstance(source_run.outcome, SourceIneligibleOutcome)

    def test_a_record_over_the_read_bound_aborts(self, protocol: Protocol, tmp_path: Path) -> None:
        """A valid canonical record padded with JSON whitespace, so only its size refuses it."""
        samples = np.zeros(4)
        record = json.dumps(canonical_record(protocol, samples))
        code = (
            f"(out / {CANONICAL_FILE!r}).write_bytes({samples.astype('<f8').tobytes()!r})\n"
            f"(out / {RECORD_FILE!r}).write_text({record!r} + ' ' * {runner._MAX_RECORD_BYTES})"
        )
        source_run = run(protocol, tmp_path, script(code), expected_sha256=RAW_SHA256)
        assert_aborts(source_run, AbortReason.INTERNAL_CANONICALISATION_ERROR)

    def test_reading_an_oversized_record_allocates_only_the_bound(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        """A 32 MiB record must not be read whole before its size refuses it."""
        out_dir = tmp_path / "source-000000"
        out_dir.mkdir()
        (out_dir / RECORD_FILE).write_bytes(b" " * (32 << 20))
        candidate = Candidate(SOURCE_ID, tmp_path / "unused.flac")
        tracemalloc.start()
        try:
            reported = runner._reported(candidate, out_dir, protocol)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert isinstance(reported, str)
        assert peak < 4 * runner._MAX_RECORD_BYTES

    def test_a_record_that_does_not_rebuild_aborts(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        """The case pickle would have let through: a golden-fixture failure naming a source."""
        result = {
            "kind": "abort",
            "reason": "GOLDEN_FIXTURE_FAILURE",
            "detail": "fixture",
            "logical_source_id": SOURCE_ID,
            "identity": None,
        }
        record = {"protocol_sha256": protocol.source_sha256, "result": result}
        assert_aborts(
            run(protocol, tmp_path, writes_record(record)),
            AbortReason.INTERNAL_CANONICALISATION_ERROR,
        )

    def test_a_record_naming_another_source_aborts(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        result = {
            "kind": "abort",
            "reason": "UNEXPECTED_IO_FAILURE",
            "detail": "read failed",
            "logical_source_id": "item/other.flac",
            "identity": None,
        }
        record = {"protocol_sha256": protocol.source_sha256, "result": result}
        assert_aborts(
            run(protocol, tmp_path, writes_record(record)),
            AbortReason.INTERNAL_CANONICALISATION_ERROR,
        )


class TestRunCanonicalisation:
    """Preconditions first, then each candidate in turn, stopping at the first abort."""

    @pytest.fixture
    def spawned(self) -> list[str]:
        """The logical identifiers of the workers started, in order."""
        return []

    def recording(self, spawned: list[str], command: runner.WorkerCommand) -> runner.WorkerCommand:
        """*command*, noting each candidate it is built for."""

        def build(candidate: Candidate, out_dir: Path) -> list[str]:
            spawned.append(candidate.logical_source_id)
            return command(candidate, out_dir)

        return build

    def start(
        self,
        protocol: Protocol,
        tmp_path: Path,
        candidates: list[Candidate],
        command: runner.WorkerCommand,
    ) -> runner.CanonicalisationRun:
        """Run the candidates with the preconditions' own inputs left to the patched check."""
        return run_canonicalisation(
            protocol,
            candidates,
            work_dir=tmp_path,
            runtime_platform="darwin_arm64",
            opgan_checkpoint=tmp_path / "final.pt",
            uvr_model_dir=tmp_path,
            command=command,
        )

    def test_a_failed_precondition_starts_no_worker(
        self,
        protocol: Protocol,
        tmp_path: Path,
        spawned: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        abort = RunAbortOutcome(reason=AbortReason.ENVIRONMENT_MISMATCH, detail="model differs")
        monkeypatch.setattr(runner, "verify_run_preconditions", lambda *a, **k: abort)
        candidates = [Candidate(SOURCE_ID, tmp_path / "a.flac")]

        result = self.start(protocol, tmp_path, candidates, self.recording(spawned, script("pass")))

        assert result == runner.CanonicalisationRun(abort, ())
        assert spawned == []
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    def test_the_run_continues_past_a_verdict_and_stops_at_the_first_abort(
        self,
        protocol: Protocol,
        tmp_path: Path,
        tone_flac: Path,
        spawned: list[str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(runner, "verify_run_preconditions", lambda *a, **k: None)
        noise = tmp_path / "noise.flac"
        noise.write_bytes(b"not audio at all" * 64)
        candidates = [
            Candidate("item/noise.flac", noise),
            Candidate(SOURCE_ID, tone_flac, expected_sha256="0" * 64),
            Candidate("item/never.flac", tone_flac),
        ]

        result = self.start(
            protocol, tmp_path, candidates, self.recording(spawned, runner.worker_command)
        )

        assert spawned == ["item/noise.flac", SOURCE_ID]
        assert [run.logical_source_id for run in result.sources] == spawned
        assert isinstance(result.sources[0].outcome, SourceIneligibleOutcome)
        assert result.abort is result.sources[1].outcome
        assert isinstance(result.abort, RunAbortOutcome)
        assert result.abort.reason is AbortReason.SOURCE_HASH_MISMATCH
        assert (tmp_path / "source-000001").is_dir()
        assert not (tmp_path / "source-000002").exists()

    def test_every_candidate_runs_when_none_aborts(
        self,
        protocol: Protocol,
        tmp_path: Path,
        tone_flac: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(runner, "verify_run_preconditions", lambda *a, **k: None)
        candidates = [Candidate("item/a.flac", tone_flac), Candidate("item/b.flac", tone_flac)]

        result = self.start(protocol, tmp_path, candidates, runner.worker_command)

        assert result.abort is None
        assert [type(run.outcome) for run in result.sources] == [CanonicalFile, CanonicalFile]

    def test_a_directory_left_by_an_earlier_run_is_never_reused(
        self, protocol: Protocol, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runner, "verify_run_preconditions", lambda *a, **k: None)
        (tmp_path / "source-000000").mkdir()
        candidates = [Candidate(SOURCE_ID, tmp_path / "a.flac")]

        with pytest.raises(FileExistsError):
            self.start(protocol, tmp_path, candidates, script("pass"))


class TestInputsCheckedBeforeStarting:
    """The worker runs from the repository root, so the parent refuses what it would misread."""

    @pytest.mark.parametrize(
        ("relative", "expected"),
        [("path", None), ("out_dir", None), (None, "A" * 64), (None, "ab"), (None, "\u00e9" * 64)],
        ids=[
            "relative_path",
            "relative_out_dir",
            "uppercase_digest",
            "short_digest",
            "non_ascii_digest_compare_digest_would_raise_on",
        ],
    )
    def test_a_bad_input_is_refused_before_anything_starts(
        self, protocol: Protocol, tmp_path: Path, relative: str | None, expected: str | None
    ) -> None:
        marker = tmp_path / "worker-started"
        path = Path("unused.flac") if relative == "path" else tmp_path / "unused.flac"
        out_dir = Path("source-000000") if relative == "out_dir" else tmp_path / "source-000000"
        candidate = Candidate(SOURCE_ID, path, expected)
        command = script(f"Path({str(marker)!r}).touch()")
        with pytest.raises(ValueError, match=r"must be absolute|64 lowercase hex"):
            run_source(candidate, out_dir, protocol, command=command)
        assert not marker.exists()
        assert not out_dir.exists()

    def test_a_relative_work_directory_is_refused_before_the_preconditions(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        work_dir = Path("relative-work-dir-that-must-not-be-created")
        with pytest.raises(ValueError, match="work directory must be absolute"):
            run_canonicalisation(
                protocol,
                [],
                work_dir=work_dir,
                runtime_platform="unused",
                opgan_checkpoint=tmp_path / "missing.pt",
                uvr_model_dir=tmp_path / "missing",
            )
        assert not work_dir.exists()
        assert not (tmp_path / "missing").exists()

    @pytest.mark.parametrize(
        ("field", "value"),
        [("path", "un\x00used.flac"), ("logical_source_id", "item/a\x00b.flac")],
        ids=["nul_in_path", "nul_in_id"],
    )
    def test_a_nul_character_is_refused_before_anything_starts(
        self, protocol: Protocol, tmp_path: Path, field: str, value: str
    ) -> None:
        """No process argument can carry a NUL, so Popen would raise after mkdir."""
        path = tmp_path / value if field == "path" else tmp_path / "unused.flac"
        source_id = value if field == "logical_source_id" else SOURCE_ID
        out_dir = tmp_path / "source-000000"
        with pytest.raises(ValueError, match="NUL character"):
            run_source(
                Candidate(source_id, path),
                out_dir,
                protocol,
                command=script("pass"),
            )
        assert not out_dir.exists()

    def test_an_identifier_that_begins_with_a_dash_reaches_the_worker(
        self, protocol: Protocol, tmp_path: Path, tone_flac: Path
    ) -> None:
        """Without `--` before the positionals, the worker's parser would exit 2 on it."""
        candidate = Candidate("-item/tones.flac", tone_flac)
        out_dir = tmp_path / "source-000000"
        source_run = run_source(candidate, out_dir, protocol)
        assert source_run.returncode == 0
        assert isinstance(source_run.outcome, CanonicalFile)
