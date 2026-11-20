"""Golden fixtures: known inputs a run must reproduce before it opens any candidate.

Protocol v5 freezes each fixture's recipe and the canonical digest it must produce. A
fixture is generated in code rather than stored: seeded integer samples from NumPy's
PCG64, written to FLAC in memory by the pinned soundfile, then run through the same
inspect, decode, and canonicalise steps as a candidate. The digest covers the canonical
samples, not the FLAC bytes, because the encoded bytes carry the libFLAC version and can
differ between libsndfile builds while the decoded integer PCM cannot.

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
    "canonical_fixture",
    "fixture_failure",
    "fixture_flac",
    "fixture_integers",
    "platform_token",
    "verify_golden_fixtures",
]

logger = logging.getLogger(__name__)

_BITS = {"PCM_16": 16, "PCM_24": 24}
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


def canonical_fixture(
    fixture: GoldenFixture, config: Canonicalisation
) -> CanonicalWaveform | SourceIneligibleOutcome | RunAbortOutcome:
    """Run the fixture through inspection, decode, and canonicalisation."""
    data = fixture_flac(fixture)
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
