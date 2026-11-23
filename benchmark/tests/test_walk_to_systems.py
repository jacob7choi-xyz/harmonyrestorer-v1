"""The walk's run directory, read by the systems stage: the manifest, the audit trail, and scores.

A fake walk over shrunk strata writes a real run directory, with its manifest and audit trail.
The systems stage then reads that directory with passing preconditions and echo systems, so the
walk-to-systems contract is exercised end to end without an engine or a model file.

The expected frames come from the walk manifest and the canonical files, not from the stage's own
output. They are derived with the framing and verdict functions the stage also calls, so the
check does not cover those functions, only the stage's selection and writing.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from systems_fakes import echo_factory, passing, passing_uvr
from walk_fakes import (
    CHAMBER_SOURCE,
    Harness,
    World,
    add_piano,
    long_tone_variant,
    shrunk,
    universe_of,
)

from benchmark.candidate_universe import ListedFile
from benchmark.canonicalisation import CanonicalWaveform
from benchmark.eligibility import frame_floors
from benchmark.eligibility_outcomes import frame_verdicts
from benchmark.framing import frame
from benchmark.protocol import Protocol, load_protocol
from benchmark.run_systems import POPULATION_NAME, RESULTS_NAME, run_systems
from benchmark.runner import CanonicalFile
from benchmark.systems_pipeline import AUDIT_NAME, BASELINE, MANIFEST_NAME
from benchmark.systems_stage import HarnessFailure
from benchmark.walk import RunLayout, WalkRun, run_walk

SOLO = "solo_piano"
CHAMBER = "chamber"
SYSTEMS = ("opgan", "uvr")


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol, which the systems stage checks the walk against."""
    return load_protocol()


def write_walk(root: Path, protocol: Protocol) -> Path:
    """Walk two complete strata into `root`, and return the directory the walk wrote."""
    root.mkdir()
    layout = RunLayout(root)
    layout.ensure_dirs()
    world = World(root, protocol)
    piano_files = add_piano(world, 3)
    proto = shrunk(protocol, piano={"test": 2, "demo": 1}, chamber={"test": 1})
    chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
    world.add(chamber_files[0].identifier, long_tone_variant(90), seed=400)
    universes = {
        SOLO: universe_of(proto, SOLO, piano_files),
        CHAMBER: universe_of(proto, CHAMBER, chamber_files),
    }
    outcome = run_walk(proto, layout, Harness(world, proto, layout).inputs(universes))
    assert isinstance(outcome, WalkRun)
    assert {walk.name: walk.status for walk in outcome.strata} == {
        SOLO: "complete",
        CHAMBER: "complete",
    }
    return root


def run_stage(walk: Path, run_dir: Path, protocol: Protocol) -> dict[str, Any]:
    """Run the systems stage over `walk` with passing preconditions and echo systems."""
    return run_systems(
        walk_dir=walk,
        run_dir=run_dir,
        protocol=protocol,
        runtime_platform=protocol.golden_fixtures.reference_platform,
        opgan_checkpoint=run_dir.parent / "final.pt",
        uvr_model_dir=run_dir.parent / "models",
        build_systems=echo_factory(fail_uvr_frame=None),
        preconditions=passing,
        uvr_check=passing_uvr,
    )


def read_rows(path: Path) -> list[dict[str, Any]]:
    """The JSON lines of a file the systems stage wrote."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def walk_test_admissions(walk: Path) -> list[dict[str, Any]]:
    """The walk manifest's test admissions of its complete strata, read straight from the file."""
    manifest = json.loads((walk / MANIFEST_NAME).read_text(encoding="utf-8"))
    complete = {name for name, entry in manifest["strata"].items() if entry["status"] == "complete"}
    return [
        entry
        for entry in manifest["admissions"]
        if entry["partition"] == "test" and entry["stratum"] in complete
    ]


def eligible_frames(walk: Path, protocol: Protocol) -> dict[str, set[int]]:
    """Each test admission's eligible frame indices, from its canonical file and the verdicts."""
    result: dict[str, set[int]] = {}
    for entry in walk_test_admissions(walk):
        canonical = CanonicalFile(
            Path(entry["canonical_path"]),
            entry["canonical_sha256"],
            entry["samples"],
            entry["raw_sha256"],
            entry["decoded_pcm_sha256"],
        )
        waveform = CanonicalWaveform(samples=canonical.load(), sha256=entry["canonical_sha256"])
        framed = frame(waveform, protocol.framing)
        verdicts = frame_verdicts(
            framed, frame_floors(framed, protocol.eligibility), protocol.metrics
        )
        result[entry["identifier"]] = {verdict.index for verdict in verdicts if verdict.eligible}
    return result


class TestWalkDirectoryScoredBySystems:
    """A walk's complete strata are scored frame by frame, and a tampered trail is refused."""

    def test_every_eligible_frame_of_every_test_admission_gets_one_line_per_system_and_baseline(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        walk = write_walk(tmp_path / "walk", protocol)
        run_dir = tmp_path / "systems"

        summary = run_stage(walk, run_dir, protocol)

        expected = eligible_frames(walk, protocol)
        admissions = walk_test_admissions(walk)
        assert sorted(entry["stratum"] for entry in admissions) == [CHAMBER, SOLO, SOLO]
        assert all(expected[entry["identifier"]] for entry in admissions)
        population = read_rows(run_dir / POPULATION_NAME)
        assert {row["identifier"]: set(row["eligible_frames"]) for row in population} == expected
        results = read_rows(run_dir / RESULTS_NAME)
        counts = Counter((row["identifier"], row["frame_index"], row["system"]) for row in results)
        assert set(counts.values()) == {1}
        scored = {
            (identifier, frame) for identifier, frames in expected.items() for frame in frames
        }
        assert {(row["identifier"], row["frame_index"]) for row in results} == scored
        assert len(results) == len(scored) * (len(SYSTEMS) + 1)
        for identifier, index in scored:
            for system in (*SYSTEMS, BASELINE):
                assert counts[(identifier, index, system)] == 1
        for stratum in (SOLO, CHAMBER):
            assert summary["strata"][stratum]["coverage"] == {"opgan": 1.0, "uvr": 1.0}

    def test_an_edited_audit_trail_makes_the_systems_stage_refuse_and_writes_nothing(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        walk = write_walk(tmp_path / "walk", protocol)
        trail = walk / AUDIT_NAME
        original = trail.read_bytes()
        assert original.count(b'"outcome": "admitted"') >= 1
        trail.write_bytes(original.replace(b'"outcome": "admitted"', b'"outcome": "rejected"', 1))
        run_dir = tmp_path / "systems"

        with pytest.raises(HarnessFailure, match=r"audit_trail\.jsonl hashes to"):
            run_stage(walk, run_dir, protocol)

        assert not run_dir.exists()
