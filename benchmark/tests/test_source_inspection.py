"""Tests for source inspection and the metadata checks that run before any decode.

Most buffers here are written by libsndfile, so those tests exercise metadata the pinned
decoder reports. The exceptions are a non-audio byte string and one libsndfile-written
FLAC with its sample count zeroed. Bounds whose boundary would need gigabytes of audio are
tested through `metadata_violations` with metadata the decoder could report, rather than
by allocating the audio.
"""

from __future__ import annotations

import hashlib
import io
import logging

import numpy as np
import pytest
import soundfile as sf

from benchmark.protocol import load_protocol
from benchmark.source_acquisition import ArtifactBuffered
from benchmark.source_evidence import (
    DiagnosticCategory,
    ObservedSourceMetadata,
    SourceIdentity,
)
from benchmark.source_inspection import (
    InspectedSource,
    canonical_length,
    inspect_source,
    metadata_violations,
)
from benchmark.source_outcome import (
    EvaluationStage,
    IneligibleReason,
    SourceIneligibleOutcome,
)

CONFIG = load_protocol().canonicalisation
PROTOCOL_DIGEST = "0" * 64
SOURCE_ID = "candidate-1"


def encoded(
    *,
    frames: int = 1000,
    channels: int = 2,
    rate: int = 44100,
    container: str = "FLAC",
    subtype: str = "PCM_16",
) -> bytes:
    """A buffer libsndfile wrote, holding a quiet constant signal."""
    out = io.BytesIO()
    sf.write(out, np.full((frames, channels), 0.25), rate, format=container, subtype=subtype)
    return out.getvalue()


def buffered(data: bytes) -> ArtifactBuffered:
    """An acquired artifact, as `acquire_local_artifact` would return it."""
    identity = SourceIdentity(len(data), hashlib.sha256(data).hexdigest(), CONFIG.decode)
    return ArtifactBuffered(identity=identity, data=data)


def inspect(data: bytes) -> InspectedSource | SourceIneligibleOutcome:
    """Inspect a buffer under the production protocol."""
    return inspect_source(
        buffered(data),
        logical_source_id=SOURCE_ID,
        protocol_sha256=PROTOCOL_DIGEST,
        config=CONFIG,
    )


def metadata(**overrides: int | str) -> ObservedSourceMetadata:
    """Admissible stereo 44.1 kHz FLAC metadata, with fields replaced as given."""
    fields: dict[str, int | str] = {
        "container": "FLAC",
        "subtype": "PCM_16",
        "channels": 2,
        "sample_rate": 44100,
        "frames": 44100,
    }
    fields.update(overrides)
    return ObservedSourceMetadata(**fields)  # type: ignore[arg-type]


def rejected(result: InspectedSource | SourceIneligibleOutcome) -> SourceIneligibleOutcome:
    """Narrow to an ineligibility record, failing the test otherwise."""
    assert isinstance(result, SourceIneligibleOutcome), result
    return result


class TestAdmitted:
    """Sources inside every bound reach the decoder with an exact projection."""

    @pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24"])
    def test_admitted_subtype_reports_decoder_metadata(self, subtype: str) -> None:
        """Both FLAC subtypes this libsndfile writes are admitted, as reported."""
        result = inspect(encoded(subtype=subtype))

        assert isinstance(result, InspectedSource)
        assert (result.metadata.container, result.metadata.subtype) == ("FLAC", subtype)
        assert (result.metadata.channels, result.metadata.sample_rate) == (2, 44100)
        assert result.metadata.frames == 1000

    def test_projection_is_the_live_bytes_rule(self) -> None:
        """Source bytes, decoded array, downmix, and canonical waveform, summed exactly."""
        data = encoded(frames=4801, channels=1, rate=48000)

        result = inspect(data)

        assert isinstance(result, InspectedSource)
        # ceil(4801 * 16000 / 48000) = ceil(1600.33) = 1601
        assert result.canonical_samples == 1601
        assert result.projected_live_bytes == len(data) + 4801 * 8 + 4801 * 8 + 1601 * 8

    def test_lowest_admitted_rate_and_channel_count(self) -> None:
        """Mono at exactly the rate floor sits on both lower bounds and is admitted."""
        result = inspect(encoded(channels=1, rate=16000))

        assert isinstance(result, InspectedSource)
        assert result.canonical_samples == 1000


class TestRejectedAtInspection:
    """Bytes the decoder cannot describe are a decoder outcome, not a metadata verdict."""

    def test_unrecognised_bytes_are_inspection_rejected(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The record carries the category, no metadata, and the raw text only in the log."""
        with caplog.at_level(logging.INFO, logger="benchmark.source_inspection"):
            outcome = rejected(inspect(b"not audio at all" * 8))

        assert outcome.primary_reason_code is IneligibleReason.SOURCE_INSPECTION_REJECTED
        assert outcome.terminating_evaluation_stage is EvaluationStage.INSPECTION
        assert outcome.observed_metadata is None
        assert outcome.decoder_diagnostic is not None
        assert outcome.decoder_diagnostic.category is DiagnosticCategory.INSPECTION_REJECTED
        # libsndfile's text is not on the publication allowlist, so it is withheld.
        assert outcome.decoder_diagnostic.detail is None
        assert "Format not recognised." in caplog.text

    def test_record_identifies_the_inspected_bytes(self) -> None:
        """The digest in the record is the digest of the buffer that was inspected."""
        data = b"not audio at all" * 8

        outcome = rejected(inspect(data))

        assert outcome.identity.sha256 == hashlib.sha256(data).hexdigest()
        assert outcome.protocol_sha256 == PROTOCOL_DIGEST
        assert outcome.logical_source_id == SOURCE_ID


class TestRejectedAtMetadataValidation:
    """Decoder-reported metadata outside the frozen domain, refused before decoding."""

    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"container": "WAV"}, IneligibleReason.UNSUPPORTED_CONTAINER),
            ({"subtype": "PCM_S8"}, IneligibleReason.UNSUPPORTED_SUBTYPE),
            ({"channels": 3}, IneligibleReason.UNSUPPORTED_CHANNEL_COUNT),
            ({"rate": 15999}, IneligibleReason.SOURCE_BELOW_MIN_SAMPLE_RATE),
        ],
    )
    def test_single_violation(self, kwargs: dict[str, int | str], reason: IneligibleReason) -> None:
        """Each violation through a real buffer, recorded alone with its metadata."""
        outcome = rejected(inspect(encoded(**kwargs)))  # type: ignore[arg-type]

        assert outcome.primary_reason_code is reason
        assert outcome.supplemental_reason_codes == ()
        assert outcome.terminating_evaluation_stage is EvaluationStage.METADATA_VALIDATION
        assert outcome.observed_metadata is not None
        assert outcome.decoder_diagnostic is None

    def test_unadmitted_container_does_not_also_judge_the_subtype(self) -> None:
        """WAV/FLOAT is no admitted pair, yet only the container is established."""
        outcome = rejected(inspect(encoded(container="WAV", subtype="FLOAT")))

        assert outcome.primary_reason_code is IneligibleReason.UNSUPPORTED_CONTAINER
        assert outcome.supplemental_reason_codes == ()

    def test_every_violation_is_recorded_and_one_is_primary(self) -> None:
        """Container, channels, and rate all fail; precedence picks the container."""
        outcome = rejected(inspect(encoded(container="WAV", channels=3, rate=8000)))

        assert outcome.primary_reason_code is IneligibleReason.UNSUPPORTED_CONTAINER
        assert outcome.supplemental_reason_codes == (
            IneligibleReason.UNSUPPORTED_CHANNEL_COUNT,
            IneligibleReason.SOURCE_BELOW_MIN_SAMPLE_RATE,
        )

    def test_unknown_sample_count_is_excluded_at_metadata_validation(self) -> None:
        """A FLAC header may leave its length unknown; libsndfile reports the maximum.

        Pinned so a change in that report, or in how this module treats it, is seen. The
        file holds 1000 frames, and the record says it exceeds the duration and decoded
        bounds, which is v4's decoder-reported authority applied as written.
        """
        data = bytearray(encoded())
        # STREAMINFO's 36-bit total-samples field: the low nibble of byte 21, then 22 to 25.
        data[21] &= 0xF0
        data[22:26] = bytes(4)

        outcome = rejected(inspect(bytes(data)))

        assert outcome.observed_metadata is not None
        assert outcome.observed_metadata.frames == 2**63 - 1
        assert outcome.terminating_evaluation_stage is EvaluationStage.METADATA_VALIDATION
        assert outcome.primary_reason_code is IneligibleReason.SOURCE_EXCEEDS_MAX_DURATION
        assert outcome.supplemental_reason_codes == (
            IneligibleReason.SOURCE_EXCEEDS_MAX_DECODED_BYTES,
        )


class TestMetadataBounds:
    """Exact boundaries of the duration and decoded-size bounds, without the audio."""

    def test_duration_bound_is_inclusive(self) -> None:
        """Exactly the cap is admitted and one frame more is not."""
        cap = CONFIG.decode.max_source_duration_s * 16000

        assert (
            metadata_violations(metadata(channels=1, sample_rate=16000, frames=cap), CONFIG) == ()
        )
        assert metadata_violations(
            metadata(channels=1, sample_rate=16000, frames=cap + 1), CONFIG
        ) == (IneligibleReason.SOURCE_EXCEEDS_MAX_DURATION,)

    def test_decoded_bound_is_inclusive(self) -> None:
        """A stereo source decoding to exactly the cap is admitted; one frame more is not."""
        frames = CONFIG.decode.max_decoded_bytes // (2 * 8)
        assert frames * 2 * 8 == CONFIG.decode.max_decoded_bytes

        assert metadata_violations(metadata(frames=frames), CONFIG) == ()
        assert metadata_violations(metadata(frames=frames + 1), CONFIG) == (
            IneligibleReason.SOURCE_EXCEEDS_MAX_DECODED_BYTES,
        )


class TestCanonicalLength:
    """The output-length rule, in exact integers."""

    @pytest.mark.parametrize(
        ("frames", "rate", "expected"),
        [
            (44100, 44100, 16000),
            (44101, 44100, 16001),
            (16000, 16000, 16000),
            (0, 48000, 0),
        ],
    )
    def test_ceiling(self, frames: int, rate: int, expected: int) -> None:
        """An exact quotient is not rounded up; any remainder is."""
        assert canonical_length(frames, rate, 16000) == expected
