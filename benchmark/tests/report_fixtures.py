"""Shared fixtures for the report tests: a walk directory and a systems directory on disk.

The bundle is one complete solo piano stratum of six recordings, with one silent frame in the first
recording, and a short chamber stratum. The systems outputs come from the systems stage's own
writers (`record_to_json`, `population_to_json`), so the rows have the shape the stage writes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from benchmark import report as report_module
from benchmark.candidate_universe import UNIVERSE_SHA256, CandidateUniverse, ListedFile
from benchmark.canonicalisation import CanonicalWaveform
from benchmark.eligibility import frame_floors
from benchmark.eligibility_outcomes import frame_verdicts
from benchmark.frame_scoring import FrameScore
from benchmark.framing import frame
from benchmark.metrics import MetricResult
from benchmark.protocol import Protocol, Stratum
from benchmark.report import (
    AUDIT_NAME,
    MANIFEST_KEY,
    MANIFEST_NAME,
    POPULATION_KEY,
    POPULATION_NAME,
    RESULTS_KEY,
    RESULTS_NAME,
    SUMMARY_NAME,
)
from benchmark.runner import SAMPLE_FORMAT
from benchmark.scoring import BASELINE, OPGAN, UVR
from benchmark.systems_pipeline import (
    Admission,
    AdmissionResult,
    FrameRecord,
    Partition,
    population_to_json,
    record_to_json,
)
from benchmark.walk import partition_plan as real_partition_plan

SOLO = "solo_piano"
CHAMBER = "chamber"
# Six recordings by five performers. Five clusters are needed before the order of first
# admission moves a percentile interval (checked against the aggregation directly).
CLUSTERS = (
    "performer-f",
    "performer-a",
    "performer-d",
    "performer-b",
    "performer-f",
    "performer-c",
)
SILENT_RECORDING = 0
SILENT_FRAME = 5
PRIMARY = ("reconstruction_snr", "si_snr", "log_spectral_distance")
SHIFT = ("reconstruction_snr", "si_snr")


@dataclass(frozen=True)
class Bundle:
    """A walk directory and a systems directory, with the canonical files they name."""

    walk: Path
    systems: Path


def sha(data: bytes) -> str:
    """The SHA-256 of some bytes, as lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def universe_of(stratum: str) -> CandidateUniverse:
    """A stand-in pinned universe for the solo stratum: each fixture identifier with its cluster.

    The real universe files list the real corpus, which the fixtures do not contain. The loader is
    replaced by `pinned_universes`, and the stand-in lists the identifiers the admissions use.
    """
    files = tuple(
        ListedFile(f"musopen/track-{index}.flac", cluster, None, None, None)
        for index, cluster in enumerate(CLUSTERS)
    )
    return CandidateUniverse(stratum, "musopen", "", "", files, UNIVERSE_SHA256[stratum] or "")


def fixture_plan(stratum: Stratum) -> tuple[str, ...]:
    """The walk's plan, with the solo stratum cut to the fixture's admissions.

    The real solo plan is 42 positions (30 test, 6 demo, 6 test_reserve). The fixture writes six
    complete test admissions, and the first six real positions are test too, so the cut plan
    agrees with the real plan on every position the fixture uses. The real lengths are checked
    by test_the_walk_plans_have_the_v13_lengths.
    """
    if stratum.name == SOLO:
        return ("test",) * len(CLUSTERS)
    return real_partition_plan(stratum)


@contextmanager
def pinned_universes() -> Iterator[None]:
    """Replace the pinned universe loader and the solo plan with the fixture's stand-ins."""
    with (
        patch.object(
            report_module,
            "load_universe",
            side_effect=lambda stratum: universe_of(stratum.name),
        ),
        patch.object(report_module, "partition_plan", side_effect=fixture_plan),
    ):
        yield


def admission_specs(protocol: Protocol, canonical: Path) -> list[Admission]:
    """Six test recordings of solo piano, in admission order, each one minute long."""
    rate = protocol.canonicalisation.resample.target_sample_rate
    samples = protocol.eligibility.recording_min_duration_s * rate
    specs = []
    for index, cluster in enumerate(CLUSTERS):
        data = np.random.default_rng(1000 + index).normal(0.0, 0.1, samples)
        if index == SILENT_RECORDING:
            start = SILENT_FRAME * protocol.framing.frame_samples
            data[start : start + protocol.framing.frame_samples] = 0.0
        raw = data.astype(SAMPLE_FORMAT).tobytes()
        path = canonical / f"recording-{index}.f64"
        path.write_bytes(raw)
        specs.append(
            Admission(
                stratum=SOLO,
                partition=Partition.TEST,
                position=index + 1,
                identifier=f"musopen/track-{index}.flac",
                cluster=cluster,
                canonical_path=path,
                canonical_sha256=sha(raw),
                samples=samples,
                raw_sha256=sha(f"raw-{index}".encode()),
                decoded_pcm_sha256=sha(f"pcm-{index}".encode()),
            )
        )
    return specs


def frame_value(
    system: str, recording: int, index: int
) -> tuple[dict[str, float], dict[str, float]]:
    """A frame's primary and shift-tolerant values for one system, deterministic and continuous.

    The values are drawn from a seeded generator, so each recording has its own level. Values on a
    lattice would make several percentile bounds coincide and hide an ordering change.
    """
    base = {OPGAN: 12.0, UVR: 9.0, BASELINE: 5.0}[system]
    seed = (len(system), recording, index)
    snr = float(np.random.default_rng(seed).normal(base, 1.0))
    primary = {
        "reconstruction_snr": snr,
        "si_snr": snr - 1.0,
        "log_spectral_distance": 2.0 + abs(snr - base) / 10.0,
    }
    shift = {"reconstruction_snr": snr + 0.5, "si_snr": snr - 0.5}
    return primary, shift


def scored(system: str, recording: int, index: int) -> FrameScore:
    """A frame's score for one system, in the shape the systems stage records."""
    primary, shift = frame_value(system, recording, index)
    return FrameScore(
        {name: MetricResult(value, False) for name, value in primary.items()},
        0,
        0,
        {name: MetricResult(value, False) for name, value in shift.items()},
    )


def population_of(protocol: Protocol, admission: Admission) -> AdmissionResult:
    """The population of one recording, computed with the walk's own functions."""
    samples = np.frombuffer(admission.canonical_path.read_bytes(), dtype=SAMPLE_FORMAT)
    framed = frame(
        CanonicalWaveform(samples=samples.astype(np.float64), sha256=""), protocol.framing
    )
    verdicts = frame_verdicts(framed, frame_floors(framed, protocol.eligibility), protocol.metrics)
    eligible = tuple(v.index for v in verdicts if v.eligible)
    return AdmissionResult(admission, verdicts, framed.dropped_samples, eligible, records=())


def write_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    """A JSON-lines file, one object per line."""
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def read_lines(path: Path) -> list[dict[str, Any]]:
    """A JSON-lines file's objects."""
    return [json.loads(line) for line in path.read_text().splitlines()]


def read_json(path: Path) -> dict[str, Any]:
    """A JSON object from a file."""
    return json.loads(path.read_text())


def write_json(path: Path, value: dict[str, Any]) -> None:
    """A JSON object to a file."""
    path.write_text(json.dumps(value, sort_keys=True, indent=1))


def write_base(protocol: Protocol, root: Path) -> Bundle:
    """The walk and the systems outputs of a complete solo piano stratum and a short chamber one."""
    canonical = root / "canonical"
    walk = root / "walk"
    systems = root / "systems"
    for directory in (canonical, walk, systems):
        directory.mkdir(parents=True)
    admissions = admission_specs(protocol, canonical)
    audit = [
        {
            "stratum": SOLO,
            "position": a.position,
            "identifier": a.identifier,
            "outcome": "admitted",
            "code": None,
            "partition": a.partition.value,
            "admission_position": a.position,
            "details": {
                "screen": {"max_training": 0.1, "flagged": [], "uncertain": False},
                "raw_sha256": a.raw_sha256,
                "decoded_pcm_sha256": a.decoded_pcm_sha256,
                "samples": a.samples,
                "canonical_sha256": a.canonical_sha256,
            },
        }
        for a in admissions
    ]
    audit.append(
        {
            "stratum": SOLO,
            "position": len(admissions) + 1,
            "identifier": "musopen/rejected.flac",
            "outcome": "rejected",
            "code": "TRAINING_RAW_SHA256",
            "partition": None,
            "admission_position": None,
            "details": {"screen": None, "canonical_sha256": None},
        }
    )
    write_lines(walk / AUDIT_NAME, audit)
    write_walk_manifest(protocol, walk, admissions)
    write_systems(protocol, systems, admissions)
    return Bundle(walk, systems)


def write_walk_manifest(protocol: Protocol, walk: Path, admissions: list[Admission]) -> None:
    """The walk manifest with the audit trail's digest, so the screen gate can pass."""
    manifest = {
        "protocol_sha256": protocol.source_sha256,
        "universe_sha256": {
            SOLO: UNIVERSE_SHA256[SOLO],
            CHAMBER: UNIVERSE_SHA256[CHAMBER],
        },
        "archive_sha256": None,
        "strata": {
            SOLO: {"status": "complete", "admitted": len(admissions)},
            CHAMBER: {"status": "short", "admitted": 0},
        },
        "admissions": [
            {
                "stratum": a.stratum,
                "partition": a.partition.value,
                "position": a.position,
                "identifier": a.identifier,
                "cluster": a.cluster,
                "canonical_path": str(a.canonical_path),
                "canonical_sha256": a.canonical_sha256,
                "samples": a.samples,
                "raw_sha256": a.raw_sha256,
                "decoded_pcm_sha256": a.decoded_pcm_sha256,
            }
            for a in admissions
        ],
        "audit_trail_sha256": sha((walk / AUDIT_NAME).read_bytes()),
    }
    write_json(walk / MANIFEST_NAME, manifest)


def write_systems(protocol: Protocol, systems: Path, admissions: list[Admission]) -> None:
    """The systems results, population and summary, written with the systems stage's functions."""
    results: list[dict[str, object]] = []
    population: list[dict[str, object]] = []
    for recording, admission in enumerate(admissions):
        result = population_of(protocol, admission)
        population.append(population_to_json(result))
        for index in result.eligible:
            for system in (OPGAN, UVR, BASELINE):
                record = FrameRecord(
                    admission, index, system, scored(system, recording, index), None, None
                )
                results.append(record_to_json(record))
    write_lines(systems / RESULTS_NAME, results)
    write_lines(systems / POPULATION_NAME, population)
    summary = {
        "protocol_sha256": protocol.source_sha256,
        "systems": [OPGAN, UVR],
        "baseline": BASELINE,
        "strata": {
            SOLO: {"status": "complete", "scored": True},
            CHAMBER: {"status": "short", "scored": False},
        },
    }
    write_json(systems / SUMMARY_NAME, summary)
    bind(Bundle(systems.parent / "walk", systems))


def bind(bundle: Bundle) -> None:
    """Record the SHA-256 of each bound file's exact bytes in the summary, as the systems stage does.

    Call it after a test has changed the files it means to bind, so that only the test's own
    change is what the report sees. A test that wants the binding to fail does not call it.
    """
    summary = read_json(bundle.systems / SUMMARY_NAME)
    summary[RESULTS_KEY] = sha((bundle.systems / RESULTS_NAME).read_bytes())
    summary[POPULATION_KEY] = sha((bundle.systems / POPULATION_NAME).read_bytes())
    summary[MANIFEST_KEY] = sha((bundle.walk / MANIFEST_NAME).read_bytes())
    write_json(bundle.systems / SUMMARY_NAME, summary)
