"""Golden fixtures: known inputs a run must reproduce before it opens any candidate.

Protocol v5 freezes each fixture's recipe and the canonical digest it must produce. A
fixture is generated in code rather than stored: seeded samples from NumPy's PCG64,
written in memory by the pinned soundfile, then run through the same inspect, decode,
and canonicalise steps as a candidate. Integer subtypes are written to FLAC. The FLOAT
subtype is written to WAV, from float32 values that are exact 24-bit fractions. The digest
covers the canonical samples, not the file bytes. The FLAC bytes carry the libFLAC version,
which can differ between libsndfile builds while the decoded integer PCM cannot. The WAV
bytes carry a PEAK chunk with a write time: two writes of the same samples 1.2 s apart
produced different bytes on the reference platform, while the decoded samples are the same.

**The reference platform is part of the runtime conformance check.** Resampler output was
not bit-identical between macOS arm64 and Linux x86_64 with the same pinned packages, and
v5 generates canonical artifacts on the reference platform only. A run elsewhere aborts
with `ENVIRONMENT_MISMATCH` before any fixture is tried; on the reference platform, a
fixture that does not reproduce its digest aborts with `GOLDEN_FIXTURE_FAILURE`. Neither
names a candidate, because both are established before one is opened.
"""

from __future__ import annotations

import hashlib
import io
import logging
import platform
import sys

import numpy as np
import numpy.typing as npt
import soundfile as sf

from benchmark.canonicalisation import CanonicalWaveform, canonicalise
from benchmark.protocol import Canonicalisation, GoldenFixture, Protocol
from benchmark.source_acquisition import ArtifactBuffered
from benchmark.source_decode import decode_source
from benchmark.source_evidence import SourceIdentity
from benchmark.source_inspection import InspectedSource, inspect_source
from benchmark.source_outcome import AbortReason, RunAbortOutcome, SourceIneligibleOutcome

__all__ = [
    "FLOAT_GENERATOR",
    "canonical_fixture",
    "fixture_bytes",
    "fixture_failure",
    "fixture_flac",
    "fixture_float_samples",
    "fixture_integers",
    "fixture_wav_float",
    "platform_token",
    "verify_golden_fixtures",
]

logger = logging.getLogger(__name__)

_BITS = {"PCM_16": 16, "PCM_24": 24}
# The FLOAT generator's identity. It is pinned here rather than in the protocol file because
# the protocol file carries one generator field, for the integer subtypes. Protocol v14 kept
# that schema when it added the FLOAT subtype, so the float generator is not in the file. Every float sample is a signed
# 24-bit integer divided by 2**23, so each is exactly representable in float32.
FLOAT_GENERATOR = "pcg64_24bit_fraction_negative_rail_negative_zero_positive_rail"
_FLOAT_SCALE = 2**23
# Identifies the fixture in the records the pipeline builds along the way; those records
# are only inspected here and never published as source outcomes.
_FIXTURE_ID = "golden-fixture"
_NO_PROTOCOL_DIGEST = "0" * 64


def platform_token() -> str:
    """This runtime's platform, in the form v5 names the reference platform."""
    return f"{sys.platform}_{platform.machine()}"


def fixture_integers(fixture: GoldenFixture) -> npt.NDArray[np.int64]:
    """The fixture's integer PCM samples, shaped (frames, channels).

    Uniform over the subtype's full signed range from PCG64 with the fixture's seed, with
    the first frame set to the negative rail and the second to zero, per v5's generator.
    """
    bits = _BITS[fixture.subtype]
    low, high = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    generator = np.random.Generator(np.random.PCG64(fixture.seed))
    samples = generator.integers(
        low, high, size=(fixture.frames, fixture.channels), endpoint=True, dtype=np.int64
    )
    samples[0, :] = low
    samples[1, :] = 0
    return samples


def fixture_flac(fixture: GoldenFixture) -> bytes:
    """The fixture encoded as FLAC in memory, from its integers without any float step."""
    bits = _BITS[fixture.subtype]
    # soundfile scales int32 by the full 32-bit range, so shifting left puts each value in
    # the top `bits` bits and the encoder stores it exactly.
    scaled = fixture_integers(fixture).astype(np.int32) << (32 - bits)
    out = io.BytesIO()
    sf.write(out, scaled, fixture.sample_rate, format=fixture.container, subtype=fixture.subtype)
    return out.getvalue()


def fixture_float_samples(fixture: GoldenFixture) -> npt.NDArray[np.float32]:
    """The FLOAT fixture's samples, shaped (frames, channels), each exact in float32.

    A signed 24-bit integer from PCG64 with the fixture's seed, over the range the PCM_24
    fixture draws from, divided by 2**23. Three frames are then written explicitly: the first
    is the negative rail, -1.0, the second is negative zero, and the third is +1.0. The uniform
    draw never reaches +1.0, and the negative zero gives signed-zero normalisation a value to
    change. Frames 3 onward are the draw, so they match the PCM_24 stream under the same seed.
    """
    generator = np.random.Generator(np.random.PCG64(fixture.seed))
    integers = generator.integers(
        -(2**23),
        2**23 - 1,
        size=(fixture.frames, fixture.channels),
        endpoint=True,
        dtype=np.int64,
    )
    # Both operations are exact: the integers have at most 24 significant bits, and dividing
    # by a power of two only moves the exponent.
    samples = integers.astype(np.float32) / np.float32(_FLOAT_SCALE)
    samples[0, :] = -1.0
    samples[1, :] = -0.0
    samples[2, :] = 1.0
    return samples


def fixture_wav_float(fixture: GoldenFixture) -> bytes:
    """The FLOAT fixture written as WAV in memory, from its float32 samples."""
    out = io.BytesIO()
    sf.write(
        out,
        fixture_float_samples(fixture),
        fixture.sample_rate,
        format="WAV",
        subtype="FLOAT",
    )
    return out.getvalue()


def fixture_bytes(fixture: GoldenFixture) -> bytes:
    """The fixture's file bytes: WAV for the FLOAT subtype, FLAC for the integer subtypes."""
    if fixture.subtype == "FLOAT":
        return fixture_wav_float(fixture)
    return fixture_flac(fixture)


def canonical_fixture(
    fixture: GoldenFixture, config: Canonicalisation
) -> CanonicalWaveform | SourceIneligibleOutcome | RunAbortOutcome:
    """Run the fixture through inspection, decode, and canonicalisation."""
    data = fixture_bytes(fixture)
    identity = SourceIdentity(len(data), hashlib.sha256(data).hexdigest(), config.decode)
    artifact = ArtifactBuffered(identity=identity, data=data)
    inspected = inspect_source(
        artifact,
        logical_source_id=_FIXTURE_ID,
        protocol_sha256=_NO_PROTOCOL_DIGEST,
        config=config,
    )
    if not isinstance(inspected, InspectedSource):
        return inspected
    decoded = decode_source(
        artifact,
        inspected,
        logical_source_id=_FIXTURE_ID,
        protocol_sha256=_NO_PROTOCOL_DIGEST,
        config=config,
    )
    if isinstance(decoded, SourceIneligibleOutcome):
        return decoded
    return canonicalise(decoded, inspected, logical_source_id=_FIXTURE_ID, config=config)


def fixture_failure(fixture: GoldenFixture, config: Canonicalisation) -> str | None:
    """Why the fixture did not reproduce its frozen digest, or None when it did."""
    result = canonical_fixture(fixture, config)
    if isinstance(result, SourceIneligibleOutcome):
        return f"fixture {fixture.name} was ineligible: {result.primary_reason_code}"
    if isinstance(result, RunAbortOutcome):
        return f"fixture {fixture.name} aborted: {result.reason}: {result.detail}"
    if result.sha256 != fixture.canonical_sha256:
        return (
            f"fixture {fixture.name} produced canonical digest {result.sha256}, expected "
            f"{fixture.canonical_sha256}"
        )
    return None


def verify_golden_fixtures(protocol: Protocol, *, runtime_platform: str) -> RunAbortOutcome | None:
    """The run-level golden fixture precondition, on the given platform.

    Args:
        protocol: The loaded protocol.
        runtime_platform: This runtime's platform, normally `platform_token()`.

    Returns:
        None when every fixture reproduced its digest, otherwise the abort record.
    """
    golden = protocol.golden_fixtures
    if runtime_platform != golden.reference_platform:
        detail = (
            f"canonical artifacts are generated on {golden.reference_platform}, and this "
            f"runtime is {runtime_platform}"
        )
        logger.error("Golden fixtures not run: %s", detail)
        return RunAbortOutcome(reason=AbortReason.ENVIRONMENT_MISMATCH, detail=detail)
    for fixture in golden.fixtures:
        failure = fixture_failure(fixture, protocol.canonicalisation)
        if failure is not None:
            logger.error("Golden fixture failed: %s", failure)
            return RunAbortOutcome(reason=AbortReason.GOLDEN_FIXTURE_FAILURE, detail=failure)
    logger.info("Golden fixtures reproduced: %d", len(golden.fixtures))
    return None
