"""Tests for the canonical waveform: downmix, resample, signed zero, validation, digest.

No resampler output values are pinned here. Native soxr may differ by platform, so these
tests assert invariants: the length rule, ownership and read-only state, exact passthrough
at the target rate, and a digest that matches an independent serialisation. Pinning a
resampler digest belongs to the golden fixture.

Several inputs are constructed rather than decoded, because no decoded source produced
them: negative zero (integer PCM does not decode to it), more than two channels (refused
at metadata validation), and a resampler that returns the wrong length or a non-finite
sample. Those tests replace the resampler or call the pure functions directly.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import pytest

from benchmark import canonicalisation
from benchmark.canonicalisation import (
    CanonicalWaveform,
    canonical_violation,
    canonicalise,
    downmix,
)
from benchmark.protocol import load_protocol
from benchmark.source_evidence import ObservedSourceMetadata
from benchmark.source_inspection import InspectedSource, canonical_length
from benchmark.source_outcome import AbortReason, RunAbortOutcome

CONFIG = load_protocol().canonicalisation
SOURCE_ID = "candidate-1"
TARGET = CONFIG.resample.target_sample_rate


def noise(frames: int, channels: int) -> npt.NDArray[np.float64]:
    """Seeded noise inside [-0.5, 0.5], as a decoded source could hold."""
    return np.random.default_rng(0).uniform(-0.5, 0.5, (frames, channels))


def inspected_for(decoded: npt.NDArray[np.float64], rate: int) -> InspectedSource:
    """What inspection would report for this array at this rate."""
    frames, channels = decoded.shape
    metadata = ObservedSourceMetadata(
        container="FLAC", subtype="PCM_16", channels=channels, sample_rate=rate, frames=frames
    )
    return InspectedSource(metadata, canonical_length(frames, rate, TARGET), 0)


def run(decoded: npt.NDArray[np.float64], rate: int) -> CanonicalWaveform | RunAbortOutcome:
    """Canonicalise under the production protocol."""
    return canonicalise(
        decoded, inspected_for(decoded, rate), logical_source_id=SOURCE_ID, config=CONFIG
    )


def canonical(result: CanonicalWaveform | RunAbortOutcome) -> CanonicalWaveform:
    """Narrow to a canonical waveform, failing the test otherwise."""
    assert isinstance(result, CanonicalWaveform), result
    return result


def aborted(result: CanonicalWaveform | RunAbortOutcome) -> RunAbortOutcome:
    """Narrow to an internal-error abort for this source, failing the test otherwise."""
    assert isinstance(result, RunAbortOutcome), result
    assert result.reason is AbortReason.INTERNAL_CANONICALISATION_ERROR
    assert result.logical_source_id == SOURCE_ID
    return result


def oracle_digest(samples: npt.NDArray[np.float64]) -> str:
    """SHA-256 of the samples packed one by one as little-endian doubles, without NumPy."""
    return hashlib.sha256(struct.pack(f"<{samples.size}d", *samples.tolist())).hexdigest()


class TestPassthrough:
    """At the target rate the resampler returns its input, so the result is exact."""

    def test_mono_is_the_source_and_leaves_it_unchanged(self) -> None:
        """The waveform equals the channel, and the decoded array is not written to."""
        decoded = noise(32000, 1)
        before = decoded.copy()

        result = canonical(run(decoded, TARGET))

        assert np.array_equal(result.samples, decoded[:, 0])
        assert np.array_equal(decoded, before)
        assert not np.shares_memory(result.samples, decoded)

    def test_stereo_is_the_explicit_mean(self) -> None:
        """(left + right) / 2.0, byte for byte."""
        decoded = noise(32000, 2)

        result = canonical(run(decoded, TARGET))

        expected = (decoded[:, 0] + decoded[:, 1]) / 2.0
        assert result.samples.tobytes() == expected.tobytes()


class TestResampled:
    """Invariants of a resampled waveform; its values are left to the golden fixture."""

    @pytest.mark.parametrize(
        ("frames", "rate"),
        [
            pytest.param(44100, 44100, id="44100"),
            pytest.param(48000, 48000, id="48000-exact"),
            pytest.param(48001, 48000, id="48000-remainder"),
        ],
    )
    def test_length_ownership_and_isolation(self, frames: int, rate: int) -> None:
        """v4's length, finite, owning its memory, read-only, and apart from the input.

        librosa returned a view for some of these lengths and not others on the platform
        checked, so which path each takes is not asserted.
        """
        decoded = noise(frames, 2)
        before = decoded.copy()

        samples = canonical(run(decoded, rate)).samples

        assert samples.shape == (canonical_length(frames, rate, TARGET),)
        assert np.all(np.isfinite(samples))
        assert samples.flags.owndata
        assert not samples.flags.writeable
        assert not np.shares_memory(samples, decoded)
        assert np.array_equal(decoded, before)

    def test_waveform_cannot_be_written(self) -> None:
        """The read-only flag refuses an assignment, so the digest keeps identifying it."""
        samples = canonical(run(noise(44100, 2), 44100)).samples

        with pytest.raises(ValueError, match="read-only"):
            samples[0] = 1.0


class TestNormalisedInPlace:
    """A resampled array that owns its memory is normalised where it is, not copied."""

    def test_an_owning_result_is_reused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Copying it would add a canonical-sized allocation, about 460 MB at the cap."""
        owned = np.full(16000, -0.0)
        monkeypatch.setattr(canonicalisation.librosa, "resample", lambda y, **kwargs: owned)

        result = canonical(run(noise(44100, 2), 44100))

        assert result.samples is owned
        assert not np.any(np.signbit(owned))


class TestSignedZero:
    """Negative zero becomes positive zero before validation and hashing."""

    def test_negative_zero_is_hashed_as_positive_zero(self) -> None:
        """Equal audio, equal bytes, equal digest."""
        decoded = np.array([[-0.0], [0.25], [-0.0]] * 1000)

        result = canonical(run(decoded, TARGET))

        assert not np.any(np.signbit(result.samples))
        assert result.sha256 == oracle_digest(np.abs(decoded[:, 0]))


class TestDigest:
    """The digest is SHA-256 over little-endian float64 bytes with no header."""

    @pytest.mark.parametrize("rate", [pytest.param(TARGET, id="passthrough"), 44100])
    def test_digest_matches_an_independent_serialisation(self, rate: int) -> None:
        """Packed by `struct`, so a NumPy byte-order mistake would not agree with it."""
        result = canonical(run(noise(44100, 2), rate))

        assert result.sha256 == oracle_digest(result.samples)


class TestResampleCall:
    """v4 freezes every resample argument, and the downmix comes first."""

    def test_call_arguments_are_the_frozen_ones(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real resampler runs; the test records what it received."""
        calls: list[tuple[tuple[int, ...], dict[str, object]]] = []
        real = librosa.resample

        def recording(y: npt.NDArray[np.float64], **kwargs: Any) -> npt.NDArray[np.float64]:
            calls.append((y.shape, kwargs))
            result: npt.NDArray[np.float64] = real(y, **kwargs)
            return result

        monkeypatch.setattr(canonicalisation.librosa, "resample", recording)

        canonical(run(noise(44100, 2), 44100))

        assert calls == [
            (
                (44100,),
                {
                    "orig_sr": 44100,
                    "target_sr": 16000,
                    "res_type": "soxr_hq",
                    "fix": True,
                    "scale": False,
                    "axis": -1,
                },
            )
        ]


class TestInternalErrors:
    """Harness failures abort the run rather than becoming a verdict on the source."""

    @pytest.mark.parametrize("channels", [0, 3])
    def test_channel_count_outside_the_domain(self, channels: int) -> None:
        """Refused at metadata validation, so reaching here is a harness defect.

        The metadata record cannot hold zero channels, so the inspection passed is a valid
        one-channel source and only the array is outside the domain.
        """
        inspected = inspected_for(np.zeros((100, 1)), TARGET)

        outcome = aborted(
            canonicalise(
                np.zeros((100, channels)), inspected, logical_source_id=SOURCE_ID, config=CONFIG
            )
        )

        assert f"shape (100, {channels})" in outcome.detail

    def test_resampled_length_off_by_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A length other than v4's integer rule is asserted, not trusted."""
        monkeypatch.setattr(
            canonicalisation.librosa, "resample", lambda y, **kwargs: np.zeros(14999)
        )

        outcome = aborted(run(noise(44100, 2), 44100))

        assert "expected (16000,)" in outcome.detail

    def test_resampler_refusal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """librosa's own input check failing on a validated source is the harness's fault."""

        def refuse(y: npt.NDArray[np.float64], **kwargs: object) -> None:
            raise librosa.util.exceptions.ParameterError("refused")

        monkeypatch.setattr(canonicalisation.librosa, "resample", refuse)

        outcome = aborted(run(noise(44100, 2), 44100))

        assert outcome.detail == "the resampler refused a validated source: refused"

    def test_non_finite_canonical_sample(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Canonical validation failing aborts instead of returning a waveform."""
        bad = np.zeros(16000)
        bad[7] = np.nan
        monkeypatch.setattr(canonicalisation.librosa, "resample", lambda y, **kwargs: bad)

        outcome = aborted(run(noise(44100, 2), 44100))

        assert outcome.detail == "canonical validation failed: non-finite samples"


class TestDownmix:
    """v4's downmix, stated per channel count."""

    def test_stereo_matches_the_explicit_expression(self) -> None:
        """The in-place division gives the same bytes as (left + right) / 2.0."""
        decoded = np.random.default_rng(1).uniform(-1.0, 1.0, (100003, 2))

        expected = (decoded[:, 0] + decoded[:, 1]) / 2.0
        assert downmix(decoded).tobytes() == expected.tobytes()

    def test_mono_is_a_copy(self) -> None:
        """A view would let an in-place step write into the decoded array."""
        decoded = noise(100, 1)

        mono = downmix(decoded)

        assert mono.flags.owndata
        assert not np.shares_memory(mono, decoded)


class TestCanonicalViolation:
    """v4's canonical validation, with no amplitude bound."""

    def test_overshoot_beyond_full_scale_is_valid(self) -> None:
        """Band-limited resampling overshoots, and v4 forbids forcing it back."""
        assert canonical_violation(np.array([1.2, -1.2, 0.0])) is None

    @pytest.mark.parametrize(
        ("waveform", "expected"),
        [
            pytest.param(np.zeros(4, dtype=np.float32), "dtype float32", id="float32"),
            pytest.param(np.zeros((2, 2)), "2 dimensions", id="2-d"),
            pytest.param(np.zeros(0), "no samples", id="empty"),
            pytest.param(np.array([0.0, np.nan]), "non-finite samples", id="nan"),
            pytest.param(np.array([0.0, np.inf]), "non-finite samples", id="+inf"),
            pytest.param(np.array([0.0, -np.inf]), "non-finite samples", id="-inf"),
        ],
    )
    def test_each_violation(self, waveform: npt.NDArray[np.float64], expected: str) -> None:
        """Each rule on its own is enough."""
        assert canonical_violation(waveform) == expected
