"""Golden fixtures: known inputs whose canonical digests are frozen in the protocol.

A fixture is generated in code rather than stored: seeded integer samples from NumPy's
PCG64, written to FLAC in memory by the pinned soundfile, then run through the same
inspect, decode, and canonicalise steps as a candidate. The expected digest covers the
canonical samples, not the FLAC bytes, because the encoded bytes carry the libFLAC version
and can differ between libsndfile builds while the decoded integer PCM cannot.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import soundfile as sf

from benchmark.canonicalisation import CanonicalWaveform, canonicalise
from benchmark.protocol import Canonicalisation
from benchmark.source_acquisition import ArtifactBuffered
from benchmark.source_decode import decode_source
from benchmark.source_evidence import SourceIdentity
from benchmark.source_inspection import InspectedSource, inspect_source
from benchmark.source_outcome import RunAbortOutcome, SourceIneligibleOutcome

__all__ = ["FixtureSpec", "canonical_fixture", "fixture_flac", "fixture_integers"]

_BITS = {"PCM_16": 16, "PCM_24": 24}
_FIXTURE_ID = "golden-fixture"
_NO_PROTOCOL = "0" * 64


@dataclass(frozen=True)
class FixtureSpec:
    """Everything that determines a fixture's samples."""

    seed: int
    sample_rate: int
    channels: int
    frames: int
    subtype: str


def fixture_integers(spec: FixtureSpec) -> npt.NDArray[np.int64]:
    """The fixture's integer PCM samples, shaped (frames, channels).

    Uniform over the subtype's full range from PCG64 with the given seed, with the first
    frame set to the negative rail and the second to zero, so the inclusive amplitude
    bound and zero are both exercised.
    """
    bits = _BITS[spec.subtype]
    low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    generator = np.random.Generator(np.random.PCG64(spec.seed))
    samples = generator.integers(
        low, high, size=(spec.frames, spec.channels), endpoint=True, dtype=np.int64
    )
    samples[0, :] = low
    samples[1, :] = 0
    return samples


def fixture_flac(spec: FixtureSpec) -> bytes:
    """The fixture encoded as FLAC in memory, from its integers without any float step."""
    bits = _BITS[spec.subtype]
    # soundfile scales int32 by the full 32-bit range, so shifting left puts each value in
    # the top `bits` bits and the encoder stores it exactly.
    scaled = fixture_integers(spec).astype(np.int32) << (32 - bits)
    out = io.BytesIO()
    sf.write(out, scaled, spec.sample_rate, format="FLAC", subtype=spec.subtype)
    return out.getvalue()


def canonical_fixture(
    spec: FixtureSpec, config: Canonicalisation
) -> CanonicalWaveform | SourceIneligibleOutcome | RunAbortOutcome:
    """Run the fixture through inspection, decode, and canonicalisation."""
    data = fixture_flac(spec)
    identity = SourceIdentity(len(data), hashlib.sha256(data).hexdigest(), config.decode)
    artifact = ArtifactBuffered(identity=identity, data=data)
    inspected = inspect_source(
        artifact, logical_source_id=_FIXTURE_ID, protocol_sha256=_NO_PROTOCOL, config=config
    )
    if not isinstance(inspected, InspectedSource):
        return inspected
    decoded = decode_source(
        artifact,
        inspected,
        logical_source_id=_FIXTURE_ID,
        protocol_sha256=_NO_PROTOCOL,
        config=config,
    )
    if isinstance(decoded, SourceIneligibleOutcome):
        return decoded
    return canonicalise(decoded, inspected, logical_source_id=_FIXTURE_ID, config=config)
