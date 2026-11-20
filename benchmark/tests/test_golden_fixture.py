"""Tests for the golden fixtures and the run-level precondition that checks them.

The analytic fixture's digest is asserted on every platform. The resampler fixture's is
asserted only on the reference platform, because resampler output differed between
macOS arm64 and Linux x86_64 with the same pinned packages; elsewhere that test is
skipped rather than passed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import platform
import sys

import numpy as np
import pytest

from benchmark import golden_fixture
from benchmark.golden_fixture import (
    fixture_failure,
    fixture_integers,
    platform_token,
    verify_golden_fixtures,
)
from benchmark.protocol import GoldenFixture, Protocol, load_protocol
from benchmark.source_outcome import AbortReason, RunAbortOutcome

PROTOCOL = load_protocol()
FIXTURES = {fixture.name: fixture for fixture in PROTOCOL.golden_fixtures.fixtures}
REFERENCE = PROTOCOL.golden_fixtures.reference_platform
ON_REFERENCE = platform_token() == REFERENCE
reference_only = pytest.mark.skipif(not ON_REFERENCE, reason=f"binding only on {REFERENCE}")


def with_fixture(replacement: GoldenFixture) -> Protocol:
    """The real protocol with one fixture replaced, in place."""
    fixtures = tuple(
        replacement if fixture.name == replacement.name else fixture
        for fixture in PROTOCOL.golden_fixtures.fixtures
    )
    golden = dataclasses.replace(PROTOCOL.golden_fixtures, fixtures=fixtures)
    return dataclasses.replace(PROTOCOL, golden_fixtures=golden)


class TestFixtureDigests:
    """Each fixture reproduces the digest v5 froze for it, where that digest binds."""

    def test_analytic_fixture_on_every_platform(self) -> None:
        """16 kHz mono, so neither the resampler nor the stereo downmix runs."""
        assert fixture_failure(FIXTURES["analytic"], PROTOCOL.canonicalisation) is None

    def test_analytic_digest_follows_from_its_integers(self) -> None:
        """Computed without the decoder: integers over full scale, as little-endian doubles."""
        oracle = fixture_integers(FIXTURES["analytic"])[:, 0] / 32768.0

        digest = hashlib.sha256(oracle.astype("<f8").tobytes(order="C")).hexdigest()

        assert digest == FIXTURES["analytic"].canonical_sha256

    @reference_only
    def test_resampler_fixture_on_the_reference_platform(self) -> None:
        """The digest measured on darwin_arm64."""
        assert fixture_failure(FIXTURES["resampler"], PROTOCOL.canonicalisation) is None


class TestFixtureFailure:
    """A fixture the pipeline refuses is reported as a failure, not raised."""

    def test_a_fixture_the_source_checks_refuse(self) -> None:
        """Below the minimum rate, so inspection refuses it on the real pipeline."""
        slow = dataclasses.replace(FIXTURES["analytic"], sample_rate=15999)

        failure = fixture_failure(slow, PROTOCOL.canonicalisation)

        assert failure == "fixture analytic was ineligible: SOURCE_BELOW_MIN_SAMPLE_RATE"

    def test_a_fixture_the_decoder_refuses(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Truncated bytes pass inspection and fail decode, as for a damaged candidate."""
        real = golden_fixture.fixture_flac

        def truncated(fixture: GoldenFixture) -> bytes:
            data = real(fixture)
            return data[: len(data) // 2]

        monkeypatch.setattr(golden_fixture, "fixture_flac", truncated)

        failure = fixture_failure(FIXTURES["analytic"], PROTOCOL.canonicalisation)

        assert failure == "fixture analytic was ineligible: SOURCE_DECODE_REJECTED"


class TestFixtureIntegers:
    """v5's generator: full range, the negative rail first, then zero."""

    @pytest.mark.parametrize("name", ["analytic", "resampler"])
    def test_rail_then_zero_within_range(self, name: str) -> None:
        fixture = FIXTURES[name]
        bits = 16 if fixture.subtype == "PCM_16" else 24

        samples = fixture_integers(fixture)

        assert samples.shape == (fixture.frames, fixture.channels)
        assert np.all(samples[0] == -(2 ** (bits - 1)))
        assert np.all(samples[1] == 0)
        assert samples.min() >= -(2 ** (bits - 1))
        assert samples.max() <= 2 ** (bits - 1) - 1


class TestPrecondition:
    """The run-level check: reference platform first, then every fixture."""

    def test_platform_token_form(self) -> None:
        assert platform_token() == f"{sys.platform}_{platform.machine()}"

    def test_another_platform_is_refused_before_any_fixture(self) -> None:
        """Canonical artifacts are generated on the reference platform only."""
        outcome = verify_golden_fixtures(PROTOCOL, runtime_platform="linux_x86_64")

        assert isinstance(outcome, RunAbortOutcome)
        assert outcome.reason is AbortReason.ENVIRONMENT_MISMATCH
        assert outcome.logical_source_id is None
        assert "darwin_arm64" in outcome.detail

    def test_a_wrong_digest_aborts_as_a_fixture_failure(self) -> None:
        """Tampering with the analytic digest fails before the resampler fixture is tried,
        so this holds on any platform."""
        tampered = dataclasses.replace(FIXTURES["analytic"], canonical_sha256="0" * 64)

        outcome = verify_golden_fixtures(with_fixture(tampered), runtime_platform=REFERENCE)

        assert isinstance(outcome, RunAbortOutcome)
        assert outcome.reason is AbortReason.GOLDEN_FIXTURE_FAILURE
        assert outcome.logical_source_id is None
        assert "fixture analytic produced canonical digest" in outcome.detail

    def test_a_pipeline_refusal_is_a_fixture_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A decoder that cannot handle the fixture aborts the run, not a candidate."""
        refusal = RunAbortOutcome(
            reason=AbortReason.INTERNAL_CANONICALISATION_ERROR, detail="refused"
        )
        monkeypatch.setattr(golden_fixture, "canonical_fixture", lambda fixture, config: refusal)

        outcome = verify_golden_fixtures(PROTOCOL, runtime_platform=REFERENCE)

        assert isinstance(outcome, RunAbortOutcome)
        assert outcome.reason is AbortReason.GOLDEN_FIXTURE_FAILURE
        assert (
            outcome.detail == "fixture analytic aborted: INTERNAL_CANONICALISATION_ERROR: refused"
        )

    @reference_only
    def test_every_fixture_passes_on_the_reference_platform(self) -> None:
        """The precondition as a run would execute it here."""
        assert verify_golden_fixtures(PROTOCOL, runtime_platform=platform_token()) is None
