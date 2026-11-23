"""Tests for the walk driver and the chamber archive path, on fakes and tiny tones.

The protocol is the real v13 with the partitions and clusters shrunk for the test, through
`dataclasses.replace`, so a walk fills in a few candidates. Candidates and features come from
`walk_fakes`, and the MusicNet archive is a small tar.gz served by a fake getter.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import tarfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from walk_fakes import (
    CHAMBER_SOURCE,
    Harness,
    World,
    add_piano,
    long_tone_variant,
    musicnet_csv,
    piano_id,
    read_lines,
    shrunk,
    stratum_named,
    tone,
    training_reference,
    unit_features,
    universe_of,
)

from benchmark.adjudication import AdjudicationRecord
from benchmark.candidate_universe import ListedFile, zenodo_file_url
from benchmark.duplicate_screen import Answer, RecordingIdentity, Reference
from benchmark.fetch import (
    Attempt,
    ListingMismatch,
    PermanentlyUnavailable,
    RetriesExhausted,
    Staged,
)
from benchmark.https_get import Response
from benchmark.protocol import (
    ArchiveItemStratum,
    Protocol,
    ZenodoArchiveStratum,
    load_protocol,
)
from benchmark.run_walk import main, new_run_layout, prepare_chamber
from benchmark.selection import walk_order
from benchmark.walk import (
    AwaitingAdjudication,
    ChamberPreparation,
    MetadataUnavailable,
    RunInvalid,
    RunLayout,
    WalkRun,
    archive_retries_invalid,
    run_walk,
)

AUDIT_KEYS = {
    "stratum",
    "position",
    "identifier",
    "outcome",
    "code",
    "partition",
    "admission_position",
    "details",
}
ADMISSION_KEYS = {
    "stratum",
    "partition",
    "position",
    "identifier",
    "cluster",
    "canonical_path",
    "canonical_sha256",
    "samples",
    "raw_sha256",
    "decoded_pcm_sha256",
}
MANIFEST_KEYS = {
    "protocol_sha256",
    "universe_sha256",
    "archive_sha256",
    "strata",
    "audit_trail_sha256",
    "admissions",
}
NO_MANIFEST = "walk_manifest.json"
NO_AUDIT = "audit_trail.jsonl"


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The real v13 protocol, once per module."""
    return load_protocol()


@pytest.fixture
def layout(tmp_path: Path) -> RunLayout:
    """A fresh, absolute run directory that the test has created."""
    root = tmp_path / "run"
    root.mkdir()
    layout = RunLayout(root)
    layout.ensure_dirs()
    return layout


class TestWalkCompletes:
    """A walk that fills its partitions stops there and writes the manifest."""

    def test_a_stratum_stops_when_its_partitions_fill(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """Two test admissions and one demo fill the piano stratum. The candidates after the third
        are not fetched, and the audit trail has one line per examined candidate."""
        world = World(layout.root, protocol)
        files = add_piano(world, 5)
        proto = shrunk(protocol, piano={"test": 2, "demo": 1}, chamber={"test": 1})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(90), seed=400)
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        assert isinstance(
            run_walk(proto, layout, Harness(world, proto, layout).inputs(universes)), WalkRun
        )
        order = walk_order([f.identifier for f in files], proto.selection.seed)
        piano_lines = [r for r in read_lines(layout.audit_path) if r["stratum"] == "solo_piano"]
        assert [r["identifier"] for r in piano_lines] == list(order[:3])
        assert [r["outcome"] for r in piano_lines] == ["admitted"] * 3
        assert all(identifier not in world.fetched for identifier in order[3:])

    def test_partitions_are_assigned_in_partition_order(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """Positions 1 and 2 fill test, 3 fills demo, and 4 fills test_reserve."""
        world = World(layout.root, protocol)
        files = add_piano(world, 6)
        proto = shrunk(
            protocol, piano={"test": 2, "demo": 1, "test_reserve": 1}, chamber={"test": 1}
        )
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(90), seed=400)
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        outcome = run_walk(proto, layout, Harness(world, proto, layout).inputs(universes))
        assert isinstance(outcome, WalkRun)
        piano = outcome.strata[0]
        assert [p.partition for p in piano.placed] == ["test", "test", "demo", "test_reserve"]
        assert [p.position for p in piano.placed] == [1, 2, 3, 4]
        assert piano.status == "complete"

    def test_the_audit_trail_has_one_line_per_candidate_with_exact_keys(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """Each examined candidate writes one record with the audit keys, in walk order."""
        world = World(layout.root, protocol)
        files = add_piano(world, 3)
        proto = shrunk(protocol, piano={"test": 2}, chamber={"test": 1})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(91), seed=401)
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        assert isinstance(
            run_walk(proto, layout, Harness(world, proto, layout).inputs(universes)), WalkRun
        )
        lines = read_lines(layout.audit_path)
        assert all(set(line) == AUDIT_KEYS for line in lines)
        piano_lines = [line for line in lines if line["stratum"] == "solo_piano"]
        assert [line["position"] for line in piano_lines] == list(range(1, len(piano_lines) + 1))
        assert [line["admission_position"] for line in piano_lines] == [1, 2]
        assert all(line["details"]["attempts"] for line in piano_lines)

    def test_a_full_cluster_rejects_the_candidate_and_the_walk_continues(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """With cap 1 and two candidates in one cluster, the walk admits one of them, rejects
        the other as capped, and admits the third from another cluster. Three test admissions
        are asked for, which the cap makes impossible, so every candidate is examined in any
        walk order and the stratum is short."""
        world = World(layout.root, protocol)
        first, second, third = piano_id(0), piano_id(1), piano_id(2)
        for index, identifier in enumerate((first, second, third)):
            world.add(identifier, long_tone_variant(index), seed=100 + index)
        files = [
            ListedFile(first, "Pianist A", None, None, None),
            ListedFile(second, "Pianist A", None, None, None),
            ListedFile(third, "Pianist B", None, None, None),
        ]
        proto = shrunk(protocol, piano={"test": 3}, chamber={"test": 1}, cap=1)
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(92), seed=402)
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        outcome = run_walk(proto, layout, Harness(world, proto, layout).inputs(universes))
        assert isinstance(outcome, WalkRun)
        assert {w.name: w.status for w in outcome.strata}["solo_piano"] == "short"
        lines = read_lines(layout.audit_path)
        capped = [
            line["identifier"]
            for line in lines
            if line["code"] == protocol.selection.cluster_cap_code
        ]
        admitted = [
            line["identifier"]
            for line in lines
            if line["outcome"] == "admitted" and line["stratum"] == "solo_piano"
        ]
        assert len(capped) == 1 and capped[0] in (first, second)
        assert sorted(admitted) == sorted({third, ({first, second} - set(capped)).pop()})
        rejected_position = next(line["position"] for line in lines if line["code"])
        assert not (layout.root / "candidates" / f"solo_piano-{rejected_position:06d}").exists()

    def test_a_candidate_rejected_after_canonicalisation_leaves_no_worker_output(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """A recording too short for the duration floor is rejected only after the worker has
        written its canonical file, so its output directory must be removed. Three test
        admissions are asked for with two eligible candidates, so the stratum is short and the
        short candidate is examined in any walk order."""
        world = World(layout.root, protocol)
        files = add_piano(world, 2)
        short = piano_id(9)
        world.add(short, tone(1.0), seed=500)
        files.append(ListedFile(short, "Pianist B", None, None, None))
        proto = shrunk(protocol, piano={"test": 3}, chamber={"test": 1})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(101), seed=411)
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        assert isinstance(
            run_walk(proto, layout, Harness(world, proto, layout).inputs(universes)), WalkRun
        )
        rejected = [line for line in read_lines(layout.audit_path) if line["identifier"] == short]
        assert [line["code"] for line in rejected] == ["RECORDING_BELOW_MIN_DURATION"]
        assert not (
            layout.root / "candidates" / f"solo_piano-{rejected[0]['position']:06d}"
        ).exists()

    def test_a_short_stratum_is_marked_and_the_other_still_stands(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """A chamber universe too small for its partitions is short; the piano stratum completes."""
        world = World(layout.root, protocol)
        files = add_piano(world, 2)
        proto = shrunk(protocol, piano={"test": 2}, chamber={"test": 3})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(93), seed=403)
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        outcome = run_walk(proto, layout, Harness(world, proto, layout).inputs(universes))
        assert isinstance(outcome, WalkRun)
        status = {walk.name: (walk.status, len(walk.placed)) for walk in outcome.strata}
        assert status == {"solo_piano": ("complete", 2), "chamber": ("short", 1)}
        manifest = json.loads((layout.root / NO_MANIFEST).read_text())
        assert manifest["strata"]["chamber"] == {"status": "short", "admitted": 1}


class TestManifest:
    """The manifest holds exactly the contract's keys, and every admission reloads."""

    @pytest.fixture
    def finished(self, layout: RunLayout, protocol: Protocol) -> tuple[Path, dict[str, Any]]:
        """A completed run with one admission per stratum, and its parsed manifest."""
        world = World(layout.root, protocol)
        files = add_piano(world, 2)
        proto = shrunk(protocol, piano={"test": 1}, chamber={"test": 1})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(94), seed=404)
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        assert isinstance(
            run_walk(proto, layout, Harness(world, proto, layout).inputs(universes)), WalkRun
        )
        return layout.root, json.loads((layout.root / NO_MANIFEST).read_text())

    def test_the_manifest_has_exactly_the_contract_keys(
        self, finished: tuple[Path, dict[str, Any]]
    ) -> None:
        """Top-level, stratum and admission entries carry exactly the keys the contract names."""
        _, manifest = finished
        assert set(manifest) == MANIFEST_KEYS
        assert set(manifest["universe_sha256"]) == {"solo_piano", "chamber"}
        assert set(manifest["strata"]["solo_piano"]) == {"status", "admitted"}
        assert all(set(entry) == ADMISSION_KEYS for entry in manifest["admissions"])
        assert manifest["archive_sha256"] == "a" * 64

    def test_admissions_are_piano_first_in_walk_order(
        self, finished: tuple[Path, dict[str, Any]]
    ) -> None:
        """Piano admissions come before chamber ones, each numbered from 1 within its stratum."""
        _, manifest = finished
        assert [(a["stratum"], a["position"]) for a in manifest["admissions"]] == [
            ("solo_piano", 1),
            ("chamber", 1),
        ]

    def test_every_admission_reloads_through_its_canonical_file(
        self, finished: tuple[Path, dict[str, Any]], protocol: Protocol
    ) -> None:
        """The systems stage's load rechecks each kept canonical file against its manifest digests."""
        from benchmark.runner import CanonicalFile

        _, manifest = finished
        for entry in manifest["admissions"]:
            path = Path(entry["canonical_path"])
            assert path.is_absolute()
            loaded = CanonicalFile(
                path,
                entry["canonical_sha256"],
                entry["samples"],
                entry["raw_sha256"],
                entry["decoded_pcm_sha256"],
            ).load()
            assert loaded.size == entry["samples"]

    def test_the_manifest_binds_the_audit_trail_as_it_sits_on_disk(
        self, finished: tuple[Path, dict[str, Any]]
    ) -> None:
        """The digest is the SHA-256 of the trail file the run left, not of the lines in memory."""
        root, manifest = finished
        trail = (root / NO_AUDIT).read_bytes()
        assert trail
        assert manifest["audit_trail_sha256"] == hashlib.sha256(trail).hexdigest()

    def test_a_walk_that_examines_no_candidate_writes_an_empty_trail_and_binds_it(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """With both universes empty, no candidate is examined. The trail is still written, empty,
        so the manifest has a trail to name, and it names the digest of zero bytes."""
        world = World(layout.root, protocol)
        proto = shrunk(protocol, piano={"test": 1}, chamber={"test": 1})
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", []),
            "chamber": universe_of(proto, "chamber", []),
        }
        outcome = run_walk(proto, layout, Harness(world, proto, layout).inputs(universes))
        assert isinstance(outcome, WalkRun)
        assert (layout.root / NO_AUDIT).read_bytes() == b""
        manifest = json.loads((layout.root / NO_MANIFEST).read_text())
        assert manifest["audit_trail_sha256"] == hashlib.sha256(b"").hexdigest()


class TestAcrossStrata:
    """An admission in one stratum is compared by the other stratum's walk, in both directions."""

    def test_a_chamber_candidate_matching_a_piano_admission_halts_the_run(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """The chamber candidate's features equal the admitted piano recording's."""
        world = World(layout.root, protocol)
        files = add_piano(world, 1)
        proto = shrunk(protocol, piano={"test": 1}, chamber={"test": 1})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(95), seed=405)
        world.features[chamber_files[0].identifier] = world.features[piano_id(0)]
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        harness = Harness(world, proto, layout)
        harness.metadata = harness.counted_metadata(musicnet_csv([1]))
        outcome = run_walk(proto, layout, harness.inputs(universes))
        assert isinstance(outcome, AwaitingAdjudication)
        halt = json.loads(layout.halt_path.read_text())
        assert halt["stratum"] == "chamber"
        assert [f["identifier"] for f in halt["flagged"]] == [piano_id(0)]
        assert halt["flagged"][0]["group"] == "admitted"
        assert not (layout.root / NO_MANIFEST).exists()


class TestHalt:
    """A halt writes the request and nothing that claims the walk completed."""

    def test_a_piano_halt_names_the_candidate_and_flagged_references_without_metadata(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """The request holds the candidate's and the flagged training reference's digests and score."""
        world = World(layout.root, protocol)
        files = add_piano(world, 1)
        training = Reference(
            RecordingIdentity("MusopenCollectionAsFlac__Same.flac", "4" * 64, "5" * 64),
            world.features[piano_id(0)],
        )
        proto = shrunk(protocol, piano={"test": 1}, chamber={"test": 1})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(96), seed=406)
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        harness = Harness(world, proto, layout)
        harness.training = [training]
        outcome = run_walk(proto, layout, harness.inputs(universes))
        assert isinstance(outcome, AwaitingAdjudication)
        halt = json.loads(layout.halt_path.read_text())
        assert halt["candidate"] == {
            "identifier": piano_id(0),
            "raw_sha256": world.raw(piano_id(0)),
            "decoded_pcm_sha256": world.decoded(piano_id(0)),
        }
        assert halt["flagged"] == [
            {
                "group": "training",
                "score": pytest.approx(1.0),
                "identifier": "MusopenCollectionAsFlac__Same.flac",
                "raw_sha256": "4" * 64,
                "decoded_pcm_sha256": "5" * 64,
            }
        ]
        assert halt["musicnet_metadata"] == {"status": "not_required"}
        assert harness.metadata_calls == 0

    def test_a_chamber_halt_includes_the_metadata_rows_of_the_named_ids(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """With a metadata file whose MD5 matches the frozen value, the candidate's row is written."""
        world = World(layout.root, protocol)
        files = add_piano(world, 1)
        data = musicnet_csv([1, 2])
        chamber = stratum_named(protocol, "chamber")
        assert isinstance(chamber, ZenodoArchiveStratum)
        frozen = dataclasses.replace(chamber, metadata_md5=hashlib.md5(data).hexdigest())
        proto = shrunk(protocol, piano={"test": 1}, chamber={"test": 1})
        proto = dataclasses.replace(
            proto,
            selection=dataclasses.replace(
                proto.selection,
                strata=tuple(frozen if s.name == "chamber" else s for s in proto.selection.strata),
            ),
        )
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(97), seed=407)
        world.features[chamber_files[0].identifier] = world.features[piano_id(0)]
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        harness = Harness(world, proto, layout)
        harness.metadata = harness.counted_metadata(data)
        assert isinstance(run_walk(proto, layout, harness.inputs(universes)), AwaitingAdjudication)
        section = json.loads(layout.halt_path.read_text())["musicnet_metadata"]
        assert section["status"] == "verified"
        assert section["ids"] == [1]
        assert [row["id"] for row in section["rows"]] == ["1"]
        assert harness.metadata_calls == 1

    def test_a_metadata_file_with_another_md5_is_reported_without_rows(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """A file that does not match the frozen MD5 is not trusted: the halt says so and gives no rows."""
        world = World(layout.root, protocol)
        files = add_piano(world, 1)
        proto = shrunk(protocol, piano={"test": 1}, chamber={"test": 1})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(98), seed=408)
        world.features[chamber_files[0].identifier] = world.features[piano_id(0)]
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        harness = Harness(world, proto, layout)
        harness.metadata = harness.counted_metadata(musicnet_csv([1]))
        assert isinstance(run_walk(proto, layout, harness.inputs(universes)), AwaitingAdjudication)
        section = json.loads(layout.halt_path.read_text())["musicnet_metadata"]
        assert section["status"] == "md5_mismatch"
        assert "rows" not in section

    def test_an_unavailable_metadata_file_still_writes_the_halt(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """A metadata fetch that fails does not lose the halt: the reason is recorded instead."""
        world = World(layout.root, protocol)
        files = add_piano(world, 1)
        proto = shrunk(protocol, piano={"test": 1}, chamber={"test": 1})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(99), seed=409)
        world.features[chamber_files[0].identifier] = world.features[piano_id(0)]
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        harness = Harness(world, proto, layout)
        assert isinstance(run_walk(proto, layout, harness.inputs(universes)), AwaitingAdjudication)
        section = json.loads(layout.halt_path.read_text())["musicnet_metadata"]
        assert section["status"] == "unavailable"
        assert not (layout.root / NO_MANIFEST).exists()

    def test_a_recorded_answer_lets_the_rerun_admit_the_flagged_candidate(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """The same halt, answered as a different performance, completes the walk on a rerun."""
        world = World(layout.root, protocol)
        files = add_piano(world, 1)
        training = Reference(
            RecordingIdentity("MusopenCollectionAsFlac__Same.flac", "4" * 64, "5" * 64),
            world.features[piano_id(0)],
        )
        proto = shrunk(protocol, piano={"test": 1}, chamber={"test": 1})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        world.add(chamber_files[0].identifier, long_tone_variant(100), seed=410)
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        harness = Harness(world, proto, layout)
        harness.training = [training]
        harness.adjudications = [
            AdjudicationRecord(
                candidate=world.identity(piano_id(0)),
                answers={training.identity: Answer.DIFFERENT_PERFORMANCE},
                evidence={training.identity: "A different pianist and session."},
            )
        ]
        assert isinstance(run_walk(proto, layout, harness.inputs(universes)), WalkRun)
        assert (layout.root / NO_MANIFEST).exists()
        assert not layout.halt_path.exists()


class TestInvalidRuns:
    """An invalid run publishes nothing: it writes its reason, and no manifest."""

    def test_exhausted_fetch_retries_stop_the_run_with_its_code(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """A candidate whose attempts all failed transiently invalidates the run."""
        world = World(layout.root, protocol)
        files = add_piano(world, 2)
        world.fetch_results[piano_id(0)] = RetriesExhausted(
            piano_id(0), (Attempt(500, None, (), ()),)
        )
        proto = shrunk(protocol, piano={"test": 2}, chamber={"test": 1})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        outcome = run_walk(proto, layout, Harness(world, proto, layout).inputs(universes))
        assert isinstance(outcome, RunInvalid)
        assert outcome.code == protocol.selection.retries_exhausted_code
        record = json.loads(layout.invalid_path.read_text())
        assert record["code"] == protocol.selection.retries_exhausted_code
        assert record["stratum"] == "solo_piano"
        assert not (layout.root / NO_MANIFEST).exists()
        assert not layout.halt_path.exists()

    def test_a_worker_abort_stops_the_run_under_its_reason(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """A harness abort on a candidate invalidates the run, and no later candidate is walked."""
        from benchmark.source_outcome import AbortReason, RunAbortOutcome

        world = World(layout.root, protocol)
        files = add_piano(world, 3)
        order = walk_order([f.identifier for f in files], protocol.selection.seed)
        world.aborts[order[0]] = RunAbortOutcome(
            reason=AbortReason.ENVIRONMENT_CAPACITY_FAILURE, detail="worker was killed"
        )
        proto = shrunk(protocol, piano={"test": 2}, chamber={"test": 1})
        chamber_files = [ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None)]
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        outcome = run_walk(proto, layout, Harness(world, proto, layout).inputs(universes))
        assert isinstance(outcome, RunInvalid)
        assert outcome.code == AbortReason.ENVIRONMENT_CAPACITY_FAILURE.value
        assert world.fetched == [order[0]]
        assert not (layout.root / NO_MANIFEST).exists()
        lines = read_lines(layout.audit_path)
        assert [line["outcome"] for line in lines] == ["run_invalid"]

    def test_archive_retries_are_recorded_as_an_invalid_run(
        self, layout: RunLayout, protocol: Protocol
    ) -> None:
        """A MusicNet archive whose attempts all failed is recorded, and nothing else is written."""
        error = RetriesExhausted(f"{CHAMBER_SOURCE}/musicnet.tar.gz", ())
        outcome = archive_retries_invalid(protocol, layout, error)
        assert outcome.code == protocol.selection.retries_exhausted_code
        record = json.loads(layout.invalid_path.read_text())
        assert record["identifier"] == f"{CHAMBER_SOURCE}/musicnet.tar.gz"
        assert not layout.audit_path.exists()
        assert not (layout.root / NO_MANIFEST).exists()


class TestChamberArchive:
    """The MusicNet archive is fetched whole, checked, and only its frozen members are extracted."""

    @staticmethod
    def archive(root: Path, ids: Sequence[int]) -> tuple[bytes, dict[str, bytes]]:
        """A tar.gz with a member per id and one member that is not frozen."""
        members = {f"musicnet/{i}.wav": f"audio {i}".encode() for i in ids}
        members["musicnet/extra.wav"] = b"not frozen"
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for name, data in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return buffer.getvalue(), members

    @staticmethod
    def getter(payload: bytes, status: int = 200) -> Callable[..., Response]:
        """A fake getter that writes *payload* for the archive, with the given status."""

        def get_(
            url: str, sink: io.BufferedIOBase, *, max_bytes: int, timeout_s: float
        ) -> Response:
            del url, max_bytes, timeout_s
            body = payload if status == 200 else b""
            sink.write(body)
            return Response(
                status,
                len(body),
                False,
                len(body),
                hashlib.sha256(body).hexdigest(),
                hashlib.md5(body, usedforsecurity=False).hexdigest(),
                hashlib.sha1(body, usedforsecurity=False).hexdigest(),
            )

        return get_

    def stratum_for(self, protocol: Protocol, payload: bytes) -> ZenodoArchiveStratum:
        """The protocol's chamber stratum with the fake archive's size and MD5 frozen in."""
        chamber = stratum_named(protocol, "chamber")
        assert isinstance(chamber, ZenodoArchiveStratum)
        return dataclasses.replace(
            chamber,
            candidate_ids=(1, 2),
            archive_bytes=len(payload),
            archive_md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        )

    def test_only_the_frozen_members_are_extracted_and_the_archive_is_digested(
        self, layout: RunLayout, protocol: Protocol, tmp_path: Path
    ) -> None:
        """Ids 1 and 2 are extracted, the extra member is not, and the archive's SHA-256 is kept."""
        payload, _ = self.archive(tmp_path, (1, 2))
        staging = tmp_path / "staging"
        staging.mkdir()
        stratum = self.stratum_for(protocol, payload)
        prep = prepare_chamber(
            stratum,
            protocol.selection,
            protocol.canonicalisation.decode.max_identity_stream_bytes,
            staging,
            layout.chamber_members_dir,
            get_=self.getter(payload),
            sleep=lambda seconds: None,
        )
        assert prep.archive_sha256 == hashlib.sha256(payload).hexdigest()
        assert sorted(p.name for p in layout.chamber_members_dir.iterdir()) == ["1.wav", "2.wav"]
        for index in (1, 2):
            fetched = prep.fetch(
                ListedFile(f"{CHAMBER_SOURCE}/{index}", "Ensemble", None, None, None)
            )
            assert isinstance(fetched, Staged)
            assert fetched.path == layout.chamber_members_dir / f"{index}.wav"

    def test_a_permanent_archive_status_fails_every_candidate_not_the_run(
        self, layout: RunLayout, protocol: Protocol, tmp_path: Path
    ) -> None:
        """A 404 on the archive leaves no digest and makes each chamber candidate permanently
        unavailable, so the stratum runs short rather than the run failing."""
        payload, _ = self.archive(tmp_path, (1, 2))
        staging = tmp_path / "staging"
        staging.mkdir()
        stratum = self.stratum_for(protocol, payload)
        prep = prepare_chamber(
            stratum,
            protocol.selection,
            protocol.canonicalisation.decode.max_identity_stream_bytes,
            staging,
            layout.chamber_members_dir,
            get_=self.getter(payload, status=404),
            sleep=lambda seconds: None,
        )
        assert prep.archive_sha256 is None
        fetched = prep.fetch(ListedFile(f"{CHAMBER_SOURCE}/1", "Ensemble", None, None, None))
        assert isinstance(fetched, PermanentlyUnavailable)
        assert fetched.status == 404

    def test_exhausted_archive_attempts_raise_for_the_run_to_record(
        self, layout: RunLayout, protocol: Protocol, tmp_path: Path
    ) -> None:
        """Transient failures on every attempt raise RetriesExhausted, which the CLI records."""
        payload, _ = self.archive(tmp_path, (1, 2))
        staging = tmp_path / "staging"
        staging.mkdir()
        stratum = self.stratum_for(protocol, payload)

        def failing(
            url: str, sink: io.BufferedIOBase, *, max_bytes: int, timeout_s: float
        ) -> Response:
            del url, sink, max_bytes, timeout_s
            raise OSError("connection reset")

        with pytest.raises(RetriesExhausted):
            prepare_chamber(
                stratum,
                protocol.selection,
                protocol.canonicalisation.decode.max_identity_stream_bytes,
                staging,
                layout.chamber_members_dir,
                get_=failing,
                sleep=lambda seconds: None,
            )


class TestChamberArchiveFailure:
    """A failed archive fetch fails each chamber candidate under its code, not the run."""

    @pytest.mark.parametrize(
        ("failure", "code_of"),
        [
            (
                lambda identifier: PermanentlyUnavailable(identifier, 404, ()),
                lambda protocol: protocol.selection.permanent_unavailability_code,
            ),
            (
                lambda identifier: ListingMismatch(identifier, ()),
                lambda protocol: protocol.selection.listing_mismatch_code,
            ),
        ],
        ids=["permanent status", "listing mismatch"],
    )
    def test_every_chamber_candidate_fails_under_the_archive_code_and_the_stratum_is_short(
        self,
        layout: RunLayout,
        protocol: Protocol,
        failure: Callable[[str], Any],
        code_of: Callable[[Protocol], str],
    ) -> None:
        """No chamber member exists, so each chamber candidate is rejected under the archive's
        code, the chamber stratum is short with nothing admitted, and the piano stratum stands."""
        world = World(layout.root, protocol)
        files = add_piano(world, 1)
        proto = shrunk(protocol, piano={"test": 1}, chamber={"test": 2})
        chamber_files = [
            ListedFile(f"{CHAMBER_SOURCE}/{i}", "Ensemble", None, None, None) for i in (1, 2)
        ]
        harness = Harness(world, proto, layout)
        harness.chamber = ChamberPreparation(None, lambda listed: failure(listed.identifier))
        universes = {
            "solo_piano": universe_of(proto, "solo_piano", files),
            "chamber": universe_of(proto, "chamber", chamber_files),
        }
        outcome = run_walk(proto, layout, harness.inputs(universes))
        assert isinstance(outcome, WalkRun)
        assert {w.name: (w.status, len(w.placed)) for w in outcome.strata} == {
            "solo_piano": ("complete", 1),
            "chamber": ("short", 0),
        }
        chamber_lines = [
            line for line in read_lines(layout.audit_path) if line["stratum"] == "chamber"
        ]
        assert [line["code"] for line in chamber_lines] == [code_of(proto)] * 2
        assert json.loads((layout.root / NO_MANIFEST).read_text())["archive_sha256"] is None


class TestMetadataProvider:
    """The MusicNet metadata file is fetched from its Zenodo URL and only a 200 is accepted."""

    @staticmethod
    def getter(status: int, body: bytes) -> Callable[..., Response]:
        """A fake getter that answers *status* with *body*, and records the URL it was asked for."""

        def get_(
            url: str, sink: io.BufferedIOBase, *, max_bytes: int, timeout_s: float
        ) -> Response:
            del max_bytes, timeout_s
            assert url == zenodo_file_url("5120004", "musicnet_metadata.csv")
            sink.write(body)
            digest = hashlib.sha256(body).hexdigest()
            return Response(status, len(body), False, len(body), digest, "", "")

        return get_

    def test_a_200_answer_yields_the_metadata_bytes(self, protocol: Protocol) -> None:
        """The provider returns the bytes the Zenodo record served."""
        from benchmark.run_walk import musicnet_metadata_provider

        chamber = stratum_named(protocol, "chamber")
        assert isinstance(chamber, ZenodoArchiveStratum)
        provide = musicnet_metadata_provider(chamber, get_=self.getter(200, b"id,source\n1,x\n"))
        assert provide() == b"id,source\n1,x\n"

    def test_a_non_200_answer_is_unavailable(self, protocol: Protocol) -> None:
        """Any other status is a MetadataUnavailable, which the halt file records as a reason."""
        from benchmark.run_walk import musicnet_metadata_provider

        chamber = stratum_named(protocol, "chamber")
        assert isinstance(chamber, ZenodoArchiveStratum)
        provide = musicnet_metadata_provider(chamber, get_=self.getter(404, b"missing"))
        with pytest.raises(MetadataUnavailable, match="answered 404"):
            provide()


class TestRunDirectory:
    """A run directory is absolute, and creating one that already exists is refused."""

    def test_a_new_run_directory_is_named_by_its_utc_start(self, tmp_path: Path) -> None:
        """The directory is named by the UTC time, and is created."""
        when = datetime(2026, 11, 22, 9, 30, 15, tzinfo=UTC)
        layout = new_run_layout(tmp_path / "runs", when)
        assert layout.root == (tmp_path / "runs" / "20261122T093015Z").resolve()
        assert layout.root.is_dir()

    def test_an_existing_run_directory_is_refused(self, tmp_path: Path) -> None:
        """A second run at the same instant fails rather than overwrite the first."""
        when = datetime(2026, 11, 22, 9, 30, 15, tzinfo=UTC)
        new_run_layout(tmp_path / "runs", when)
        with pytest.raises(FileExistsError):
            new_run_layout(tmp_path / "runs", when)

    def test_a_relative_run_directory_is_refused(self) -> None:
        """Every recorded path is absolute, so a relative run root is refused at construction."""
        with pytest.raises(ValueError, match="must be absolute"):
            RunLayout(Path("relative/run"))


class TestCommandLine:
    """The command refuses to start without a UVR model directory, before creating anything."""

    def test_a_missing_uvr_directory_refuses_before_any_run_directory_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a model directory the exit code is the refusal and no run directory appears."""
        from benchmark import run_walk as cli

        runs = tmp_path / "runs"
        monkeypatch.setattr(cli, "RUNS_ROOT", runs)

        def not_reached(*args: object, **kwargs: object) -> None:
            raise AssertionError("the preconditions were checked before the directory was refused")

        monkeypatch.setattr(cli, "verify_run_preconditions", not_reached)
        monkeypatch.delenv("AUDIO_SEPARATOR_MODEL_DIR", raising=False)
        code = main(["--uvr-model-dir", str(tmp_path / "missing")])
        assert code == cli.EXIT_REFUSED
        assert not runs.exists()


class TestPianoFetch:
    """The Internet Archive fetch uses the item's download URL and the protocol's oversize bound."""

    def test_the_download_url_and_the_oversize_cap_are_the_protocol_values(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        """The getter is asked for the percent-encoded URL, capped at the bound plus one byte."""
        from benchmark.fetch import download_url
        from benchmark.run_walk import piano_fetch

        item, _ = _strata_pair(protocol)
        bound = protocol.canonicalisation.decode.max_identity_stream_bytes
        seen: dict[str, int] = {}

        def get_(
            url: str, sink: io.BufferedIOBase, *, max_bytes: int, timeout_s: float
        ) -> Response:
            del timeout_s
            seen.update(url=url, max_bytes=max_bytes)
            return Response(404, 0, False, None, "", "", "")

        staging = tmp_path / "staging"
        staging.mkdir()
        fetch = piano_fetch(
            item, protocol.selection, bound, staging, get_=get_, sleep=lambda s: None
        )
        identifier = f"{item.source}/Chopin/Nocturne 1.flac"
        result = fetch(ListedFile(identifier, "Pianist A", None, None, None))
        assert isinstance(result, PermanentlyUnavailable)
        assert seen["url"] == download_url(item.source, "Chopin/Nocturne 1.flac")
        assert seen["max_bytes"] == bound + 1


def _strata_pair(protocol: Protocol) -> tuple[ArchiveItemStratum, ZenodoArchiveStratum]:
    """The protocol's item stratum and archive stratum."""
    item = next(s for s in protocol.selection.strata if isinstance(s, ArchiveItemStratum))
    archive = next(s for s in protocol.selection.strata if isinstance(s, ZenodoArchiveStratum))
    return item, archive


class TestRealWorker:
    """One candidate through the real canonicalisation worker, on a FLAC written to disk."""

    def test_the_real_worker_canonicalises_a_tone_flac_and_the_walk_admits_it(
        self, tmp_path: Path, protocol: Protocol
    ) -> None:
        """The worker's canonical file, its digests and its loaded samples satisfy the walk."""
        import soundfile as sf
        from walk_fakes import state_for

        from benchmark.duplicate_screen import Degenerate
        from benchmark.fetch import Staged
        from benchmark.runner import run_source
        from benchmark.walk import Admitted, WalkDeps, examine_candidate

        rate = 44100
        t = np.arange(61 * rate) / rate
        flac = tmp_path / "tone.flac"
        sf.write(flac, 0.3 * np.sin(2 * np.pi * 220.0 * t), rate, format="FLAC", subtype="PCM_16")
        data = flac.read_bytes()
        identifier = piano_id(0)
        staged = Staged(
            identifier, flac, len(data), hashlib.sha256(data).hexdigest(), False, (), (), False
        )
        deps = WalkDeps(
            fetch=lambda listed: staged,
            canonicalise=lambda candidate, out_dir: run_source(
                candidate, out_dir, protocol, timeout_s=120.0
            ),
            features=lambda samples: (unit_features(7), Degenerate(0, 0)),
        )
        stratum = next(s for s in protocol.selection.strata if s.name == "solo_piano")
        (tmp_path / "work").mkdir()
        out_dir = tmp_path / "work" / "candidate-000001"
        outcome = examine_candidate(
            protocol,
            stratum,
            ListedFile(identifier, "Pianist A", len(data), None, None),
            state_for([training_reference(2)]),
            out_dir,
            deps,
        )
        assert isinstance(outcome, Admitted), outcome
        canonical = outcome.admission.canonical
        assert outcome.admission.raw_sha256 == hashlib.sha256(data).hexdigest()
        assert canonical.path == out_dir / "canonical.f64"
        assert canonical.load().size == canonical.samples
