"""Tests for decoding an inspected source and validating what the decoder returned.

Buffers are written by libsndfile, then sometimes cut or given a different sample count in
their header, and every one passes through `inspect_source` before it is decoded. Four
reasons were not produced by any buffer tried: a decoded shape that disagrees with
inspection, an empty array, a non-finite sample, and a sample outside [-1.0, 1.0]. Their
checks are tested as pure functions on constructed arrays, and two tests replace the
decoder to show those results reach the record at the right stage.
"""

from __future__ import annotations

import hashlib
import io
import logging
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from benchmark import source_decode
from benchmark.protocol import load_protocol
from benchmark.source_acquisition import ArtifactBuffered
from benchmark.source_decode import decode_source, decoded_metadata_matches, waveform_violations
from benchmark.source_evidence import DiagnosticCategory, ObservedSourceMetadata, SourceIdentity
from benchmark.source_inspection import InspectedSource, inspect_source
from benchmark.source_outcome import EvaluationStage, IneligibleReason, SourceIneligibleOutcome

CONFIG = load_protocol().canonicalisation
PROTOCOL_DIGEST = "0" * 64
SOURCE_ID = "candidate-1"
FRAMES = 16384


def encoded(
    samples: npt.NDArray[np.float64], *, rate: int = 44100, subtype: str = "PCM_16"
) -> bytes:
    """A FLAC buffer libsndfile wrote from the given samples."""
    out = io.BytesIO()
    sf.write(out, samples, rate, format="FLAC", subtype=subtype)
    return out.getvalue()


def noise(frames: int = FRAMES, channels: int = 2) -> npt.NDArray[np.float64]:
    """Seeded noise, so the encoded stream is not trivially compressible."""
    return np.random.default_rng(0).uniform(-0.5, 0.5, (frames, channels))


def with_total_samples(data: bytes, total: int) -> bytes:
    """The buffer with STREAMINFO's 36-bit total-sample count replaced.

    The field is the low nibble of byte 21 followed by bytes 22 to 25.
    """
    edited = bytearray(data)
    edited[21] = (edited[21] & 0xF0) | ((total >> 32) & 0x0F)
    edited[22:26] = (total & 0xFFFFFFFF).to_bytes(4, "big")
    return bytes(edited)


def buffered(data: bytes) -> ArtifactBuffered:
    """An acquired artifact, as `acquire_local_artifact` would return it."""
    identity = SourceIdentity(len(data), hashlib.sha256(data).hexdigest(), CONFIG.decode)
    return ArtifactBuffered(identity=identity, data=data)


def decode(
    data: bytes,
) -> tuple[InspectedSource, npt.NDArray[np.float64] | SourceIneligibleOutcome]:
    """Inspect then decode a buffer under the production protocol, in that order."""
    artifact = buffered(data)
    inspected = inspect_source(
        artifact, logical_source_id=SOURCE_ID, protocol_sha256=PROTOCOL_DIGEST, config=CONFIG
    )
    assert isinstance(inspected, InspectedSource), inspected
    result = decode_source(
        artifact,
        inspected,
        logical_source_id=SOURCE_ID,
        protocol_sha256=PROTOCOL_DIGEST,
        config=CONFIG,
    )
    return inspected, result


def rejected(
    result: npt.NDArray[np.float64] | SourceIneligibleOutcome,
) -> SourceIneligibleOutcome:
    """Narrow to an ineligibility record, failing the test otherwise."""
    assert isinstance(result, SourceIneligibleOutcome), result
    return result


def decoded(result: npt.NDArray[np.float64] | SourceIneligibleOutcome) -> npt.NDArray[np.float64]:
    """Narrow to decoded samples, failing the test otherwise."""
    assert isinstance(result, np.ndarray), result
    return result


def metadata(frames: int = 1000, channels: int = 2, rate: int = 44100) -> ObservedSourceMetadata:
    """Admissible FLAC metadata with the given shape and rate."""
    return ObservedSourceMetadata(
        container="FLAC", subtype="PCM_16", channels=channels, sample_rate=rate, frames=frames
    )


class TestDecoded:
    """Admitted buffers decode to the samples that were written."""

    @pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24"])
    def test_admitted_subtype_decodes_the_written_samples(self, subtype: str) -> None:
        """0.25 is exact in both subtypes, so the decoded array equals what was written."""
        _, result = decode(encoded(np.full((1000, 2), 0.25), subtype=subtype))

        samples = decoded(result)
        assert samples.dtype == np.float64
        assert samples.shape == (1000, 2)
        assert np.all(samples == 0.25)

    def test_mono_keeps_two_dimensions(self) -> None:
        """One channel decodes as (frames, 1), the same rank as two channels."""
        _, result = decode(encoded(np.full((1000, 1), 0.25)))

        assert decoded(result).shape == (1000, 1)

    def test_negative_rail_is_admitted(self) -> None:
        """Integer PCM reaches exactly -1.0, which the inclusive bound admits."""
        _, result = decode(encoded(np.full((1000, 2), -1.0)))

        assert decoded(result).min() == -1.0

    def test_understated_length_decodes_only_the_stated_frames(self) -> None:
        """A header claiming fewer samples than the stream holds decodes that many.

        Pinned so a change is seen: the trailing audio is dropped without an error, and the
        record cannot show it. Protocol v5 states this behaviour as the rule.
        """
        inspected, result = decode(with_total_samples(encoded(noise()), FRAMES - 1000))

        assert inspected.metadata.frames == FRAMES - 1000
        assert decoded(result).shape == (FRAMES - 1000, 2)


class TestDecodeRejected:
    """A buffer inspection described and the decoder then refused."""

    @pytest.mark.parametrize(
        "damage",
        [
            pytest.param(lambda data: data[: len(data) // 2], id="truncated"),
            pytest.param(lambda data: with_total_samples(data, FRAMES + 1000), id="overstated"),
        ],
    )
    def test_decoder_error_is_decode_rejected(
        self, damage: Callable[[bytes], bytes], caplog: pytest.LogCaptureFixture
    ) -> None:
        """Classified by the stage that failed, with the inspected metadata and no raw text.

        An overstated header disagrees with its stream, yet the decoder raised, so the
        reason is the decode rejection and not an inconsistency.
        """
        data = damage(encoded(noise()))

        with caplog.at_level(logging.INFO, logger="benchmark.source_decode"):
            inspected, result = decode(data)

        outcome = rejected(result)
        assert outcome.primary_reason_code is IneligibleReason.SOURCE_DECODE_REJECTED
        assert outcome.terminating_evaluation_stage is EvaluationStage.DECODE
        assert outcome.unevaluated_evaluation_stages == (
            EvaluationStage.DECODED_METADATA_VALIDATION,
            EvaluationStage.WAVEFORM_VALIDATION,
        )
        assert outcome.observed_metadata == inspected.metadata
        assert outcome.decoder_diagnostic is not None
        assert outcome.decoder_diagnostic.category is DiagnosticCategory.DECODE_REJECTED
        assert outcome.identity.sha256 == hashlib.sha256(data).hexdigest()
        assert f"Decode rejected {SOURCE_ID}" in caplog.text


class TestDecodeCall:
    """v4 freezes the decode call's arguments rather than leaving any to a default."""

    def test_call_arguments_are_the_frozen_ones(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real decoder runs; the test records what it was asked for.

        The frame count is the inspected one, so the array soundfile allocates is capped at
        the number the metadata checks admitted.
        """
        calls: list[dict[str, object]] = []
        real_read = sf.read

        def recording_read(
            file: io.BytesIO, **kwargs: object
        ) -> tuple[npt.NDArray[np.float64], int]:
            calls.append(kwargs)
            decoded: tuple[npt.NDArray[np.float64], int] = real_read(file, **kwargs)
            return decoded

        monkeypatch.setattr(source_decode.sf, "read", recording_read)

        inspected, result = decode(encoded(np.full((1000, 2), 0.25)))

        decoded(result)
        assert calls == [
            {"frames": inspected.metadata.frames, "dtype": "float64", "always_2d": True}
        ]


class TestRecordsFromAReplacedDecoder:
    """Results no buffer tried produced, reaching the record through a replaced decoder."""

    def test_shape_disagreement_is_metadata_inconsistent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One frame short of what inspection reported stops at decoded-metadata validation."""
        monkeypatch.setattr(
            source_decode.sf, "read", lambda *args, **kwargs: (np.zeros((999, 2)), 44100)
        )

        _, result = decode(encoded(np.full((1000, 2), 0.25)))

        outcome = rejected(result)
        assert outcome.primary_reason_code is IneligibleReason.SOURCE_METADATA_INCONSISTENT
        assert outcome.terminating_evaluation_stage is EvaluationStage.DECODED_METADATA_VALIDATION
        assert outcome.observed_metadata is not None
        assert outcome.decoder_diagnostic is None

    def test_waveform_violations_are_recorded_together(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An infinity fails finiteness and amplitude; precedence makes finiteness primary."""
        samples = np.zeros((1000, 2))
        samples[10, 1] = np.inf
        monkeypatch.setattr(source_decode.sf, "read", lambda *args, **kwargs: (samples, 44100))

        _, result = decode(encoded(np.full((1000, 2), 0.25)))

        outcome = rejected(result)
        assert outcome.primary_reason_code is IneligibleReason.NONFINITE_SOURCE
        assert outcome.supplemental_reason_codes == (IneligibleReason.INVALID_SOURCE_AMPLITUDE,)
        assert outcome.terminating_evaluation_stage is EvaluationStage.WAVEFORM_VALIDATION
        assert outcome.decoder_diagnostic is None


class TestDecodedMetadataMatches:
    """Frames, channels, and rate must all be the ones inspection reported."""

    def test_matching_array_and_rate(self) -> None:
        """The same shape and rate agree."""
        assert decoded_metadata_matches(np.zeros((1000, 2)), 44100, metadata())

    @pytest.mark.parametrize(
        ("shape", "rate"),
        [
            pytest.param((999, 2), 44100, id="frames"),
            pytest.param((1000, 1), 44100, id="channels"),
            pytest.param((1000, 2), 48000, id="rate"),
        ],
    )
    def test_any_disagreement(self, shape: tuple[int, int], rate: int) -> None:
        """Each field on its own is enough to disagree."""
        assert not decoded_metadata_matches(np.zeros(shape), rate, metadata())


class TestWaveformViolations:
    """The waveform checks on constructed arrays, which no buffer tried produced."""

    def test_empty_array_is_empty_alone(self) -> None:
        """No samples cannot violate finiteness or amplitude."""
        assert waveform_violations(np.zeros((0, 2)), CONFIG) == (IneligibleReason.EMPTY_SOURCE,)

    def test_both_rails_are_admitted(self) -> None:
        """The bounds are inclusive at both ends."""
        assert waveform_violations(np.array([[-1.0, 1.0]]), CONFIG) == ()

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(np.nextafter(1.0, 2.0), id="above"),
            pytest.param(np.nextafter(-1.0, -2.0), id="below"),
        ],
    )
    def test_just_beyond_a_rail(self, value: float) -> None:
        """One ulp beyond either bound violates amplitude only."""
        samples = np.array([[0.0, value]])

        assert waveform_violations(samples, CONFIG) == (IneligibleReason.INVALID_SOURCE_AMPLITUDE,)

    @pytest.mark.parametrize(
        "value", [pytest.param(np.inf, id="+inf"), pytest.param(-np.inf, id="-inf")]
    )
    def test_infinity_violates_finiteness_and_amplitude(self, value: float) -> None:
        """An infinity is non-finite and outside the range, whichever its sign."""
        samples = np.array([[0.0, value]])

        assert waveform_violations(samples, CONFIG) == (
            IneligibleReason.NONFINITE_SOURCE,
            IneligibleReason.INVALID_SOURCE_AMPLITUDE,
        )

    def test_nan_violates_finiteness_only(self) -> None:
        """NaN has no position to compare against the range."""
        samples = np.array([[0.0, np.nan], [np.nan, np.nan]])

        assert waveform_violations(samples, CONFIG) == (IneligibleReason.NONFINITE_SOURCE,)

    @pytest.mark.parametrize(
        "value", [pytest.param(2.0, id="above"), pytest.param(-2.0, id="below")]
    )
    def test_nan_does_not_hide_an_amplitude_violation(self, value: float) -> None:
        """A finite sample out of range is still found beside a NaN, on either side."""
        samples = np.array([[np.nan, value]])

        assert waveform_violations(samples, CONFIG) == (
            IneligibleReason.NONFINITE_SOURCE,
            IneligibleReason.INVALID_SOURCE_AMPLITUDE,
        )
