"""Fake fetch, worker and features for the walk tests.

The worker is replaced by a function that writes the canonical bytes the candidate's samples
produce. Features are fixed per candidate and keyed by content: a candidate whose features equal
a reference's scores 1.0 against it, and independent random unit features score near 0.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from benchmark.adjudication import AdjudicationRecord
from benchmark.candidate_universe import ListedFile
from benchmark.duplicate_screen import (
    Degenerate,
    Features,
    RecordingIdentity,
    Reference,
)
from benchmark.fetch import (
    Attempt,
    FetchOutcome,
    Staged,
)
from benchmark.protocol import Protocol, Stratum
from benchmark.runner import Candidate, CanonicalFile, SourceRun
from benchmark.selection import TrainingIdentities
from benchmark.source_evidence import SourceIdentity
from benchmark.source_outcome import (
    IneligibleReason,
    RunAbortOutcome,
    SourceIneligibleOutcome,
)
from benchmark.walk import Outcome, Rejected, WalkDeps, WalkState, examine_candidate

PIANO_SOURCE = "musopen-chopin-complete-works-flac"
CHAMBER_SOURCE = "musicnet"
PIANO_ID = f"{PIANO_SOURCE}/Chopin/Nocturne.flac"
CHAMBER_ID = f"{CHAMBER_SOURCE}/1727"
RATE = 16000
FRAMES = 120
MFCC = 20


def sha256(data: bytes) -> str:
    """The SHA-256 of *data* as lowercase hex."""
    return hashlib.sha256(data).hexdigest()


def unit_features(seed: int) -> Features:
    """Random unit-length MFCC-shaped rows, reproducible from *seed*."""
    rng = np.random.default_rng(seed)
    rows = rng.standard_normal((FRAMES, MFCC))
    return np.ascontiguousarray(rows / np.linalg.norm(rows, axis=1, keepdims=True))


def tone(seconds: float, amplitude: float = 0.3) -> npt.NDArray[np.float64]:
    """A sine tone of *seconds* at the canonical rate."""
    t = np.arange(int(seconds * RATE)) / RATE
    return np.ascontiguousarray(amplitude * np.sin(2 * np.pi * 220.0 * t))


def long_tone() -> npt.NDArray[np.float64]:
    """A tone that clears the duration floor and has eligible frames."""
    return tone(61.0)


def quiet_remainder_tone(protocol: Protocol) -> npt.NDArray[np.float64]:
    """Silence in whole frames, then a loud tail shorter than one frame that framing drops.

    The recording clears both recording floors, but every complete frame is silent, so no frame
    is eligible.
    """
    frame = protocol.framing.frame_samples
    whole = (protocol.eligibility.recording_min_duration_s * RATE) // frame + 2
    tail = frame - 1
    head = np.zeros(whole * frame, dtype=np.float64)
    return np.concatenate([head, np.full(tail, 0.5, dtype=np.float64)])


class World:
    """Fake fetch, worker and features for candidates, keyed by identifier."""

    def __init__(self, root: Path, protocol: Protocol) -> None:
        self.root = root
        self.protocol = protocol
        self.samples: dict[str, npt.NDArray[np.float64]] = {}
        self.features: dict[str, Features] = {}
        self.degenerate: dict[str, Degenerate] = {}
        self.fetch_results: dict[str, FetchOutcome | BaseException] = {}
        self.aborts: dict[str, SourceIneligibleOutcome | RunAbortOutcome] = {}
        self.fetched: list[str] = []
        self.canonicalised: list[tuple[str, str | None]] = []
        self.by_decoded: dict[str, str] = {}

    def add(self, identifier: str, samples: npt.NDArray[np.float64], seed: int) -> None:
        """Register a candidate's samples and its seeded features."""
        self.samples[identifier] = samples
        self.features[identifier] = unit_features(seed)
        self.by_decoded[self.decoded(identifier)] = identifier

    def staged_bytes(self, identifier: str) -> bytes:
        """The bytes the fake fetch stages: the identifier and the samples."""
        return b"RAW:" + identifier.encode() + self.samples[identifier].astype("<f8").tobytes()

    def raw(self, identifier: str) -> str:
        """The candidate's raw SHA-256 as the fake fetch stages it."""
        return sha256(self.staged_bytes(identifier))

    def decoded(self, identifier: str) -> str:
        """The candidate's decoded-PCM SHA-256 as the fake worker reports it."""
        return sha256(self.samples[identifier].astype("<f8").tobytes())

    def identity(self, identifier: str) -> RecordingIdentity:
        """The candidate as an adjudication record names it."""
        return RecordingIdentity(identifier, self.raw(identifier), self.decoded(identifier))

    def fetch(self, listed: ListedFile) -> FetchOutcome:
        """The fake fetch: a configured result or exception, else the staged candidate."""
        self.fetched.append(listed.identifier)
        result = self.fetch_results.get(listed.identifier)
        if isinstance(result, BaseException):
            raise result
        if result is not None:
            return result
        path = self.root / "staged" / f"{sha256(listed.identifier.encode())}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.staged_bytes(listed.identifier)
        path.write_bytes(data)
        attempt = Attempt(200, None, ("size",), ())
        return Staged(listed.identifier, path, len(data), self.raw(listed.identifier), False,
                      ("size",), (attempt,), False)  # fmt: skip

    def canonicalise(self, candidate: Candidate, out_dir: Path) -> SourceRun:
        """The fake worker: an abort or verdict if configured, else the canonical file."""
        identifier = candidate.logical_source_id
        self.canonicalised.append((identifier, candidate.expected_sha256))
        if identifier in self.aborts:
            return SourceRun(identifier, self.aborts[identifier], 0, None, 0.0)
        out_dir.mkdir(parents=True)
        data = self.samples[identifier].astype("<f8").tobytes()
        path = out_dir / "canonical.f64"
        path.write_bytes(data)
        canonical = CanonicalFile(
            path=path,
            sha256=sha256(data),
            samples=self.samples[identifier].size,
            raw_sha256=self.raw(identifier),
            decoded_pcm_sha256=self.decoded(identifier),
        )
        return SourceRun(identifier, canonical, 0, None, 0.0)

    def features_of(self, samples: npt.NDArray[np.float64]) -> tuple[Features, Degenerate]:
        """The features keyed by the samples' decoded digest, with the configured degeneracy."""
        identifier = self.by_decoded[sha256(samples.astype("<f8").tobytes())]
        return self.features[identifier], self.degenerate.get(identifier, Degenerate(0, 0))

    def deps(self) -> WalkDeps:
        """The three injected functions, bound to this world."""
        return WalkDeps(self.fetch, self.canonicalise, self.features_of)


def training_reference(seed: int, name: str = "MusopenCollectionAsFlac__Bach.flac") -> Reference:
    """A training recording with random features that the candidates do not match."""
    raw, decoded = sha256(name.encode()), sha256(b"decoded:" + name.encode())
    return Reference(RecordingIdentity(name, raw, decoded), unit_features(seed))


def state_for(
    refs: list[Reference], adjudications: tuple[AdjudicationRecord, ...] = ()
) -> WalkState:
    """A walk state whose training identities are exactly the given references."""
    identities = TrainingIdentities(
        filenames=frozenset(r.identity.identifier for r in refs),
        raw_sha256=frozenset(r.identity.raw_sha256 for r in refs),
        decoded_sha256=frozenset(r.identity.decoded_pcm_sha256 for r in refs),
    )
    return WalkState(training=identities, training_references=refs, adjudications=adjudications)


def examine(
    world: World,
    protocol: Protocol,
    stratum: Stratum,
    identifier: str,
    state: WalkState,
    *,
    cluster: str = "Pianist A",
    out_root: Path | None = None,
) -> Outcome:
    """Examine one candidate with fresh evidence and a fresh output directory."""
    out_dir = (out_root or world.root / "work") / sha256(identifier.encode())
    listed = ListedFile(identifier, cluster, None, None, None)
    return examine_candidate(protocol, stratum, listed, state, out_dir, world.deps())


def snapshot(state: WalkState) -> tuple[Any, ...]:
    """Every mutable field of the walk state, copied, for a before and after comparison."""
    return (
        set(state.admitted_raw),
        set(state.admitted_decoded),
        list(state.admitted_references),
        dict(state.cluster_counts),
    )


def assert_rejected(outcome: Outcome, code: str) -> Rejected:
    """The outcome is a rejection under *code*; returns it for further checks."""
    assert isinstance(outcome, Rejected), outcome
    assert outcome.code == code
    return outcome


def assert_no_worker(world: World) -> None:
    """The walk stopped before the canonicalisation worker ran for any candidate."""
    assert world.canonicalised == []


def ineligible(
    protocol: Protocol, identifier: str, reason: IneligibleReason
) -> SourceIneligibleOutcome:
    """A source verdict on *identifier* for an oversize file, the one reason the walk test needs."""
    decode = protocol.canonicalisation.decode
    identity = SourceIdentity(decode.max_identity_stream_bytes + 1, None, decode)
    return SourceIneligibleOutcome(
        logical_source_id=identifier,
        primary_reason_code=reason,
        identity=identity,
        protocol_sha256=protocol.source_sha256,
    )
