"""Tests for the outcome wire format.

A record crosses from a worker process to its parent through this format, so a wire record
must rebuild only through the constructors: every combination a constructor refuses has to
be refused on the way in, and nothing a constructor derives may be supplied.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from benchmark.outcome_wire import OutcomeWireError, outcome_from_wire, outcome_to_wire
from benchmark.protocol import Decode, load_protocol
from benchmark.source_evidence import (
    DiagnosticCategory,
    ObservedSourceMetadata,
    SanitisedDiagnostic,
    SourceIdentity,
)
from benchmark.source_outcome import (
    AbortReason,
    IneligibleReason,
    RunAbortOutcome,
    SourceIneligibleOutcome,
)

DIGEST = "a" * 64
PROTOCOL_DIGEST = "b" * 64
METADATA = ObservedSourceMetadata(
    container="FLAC", subtype="PCM_24", channels=2, sample_rate=44100, frames=88200
)


@pytest.fixture(scope="module")
def decode() -> Decode:
    """The frozen decode section, which owns the identity bounds."""
    return load_protocol().canonicalisation.decode


def ineligible(decode: Decode, shape: str) -> SourceIneligibleOutcome:
    """One record per evidence shape a source can end in."""
    small = SourceIdentity(1024, DIGEST, decode)
    if shape == "too_large_streamed":
        return SourceIneligibleOutcome(
            logical_source_id="item/a.flac",
            primary_reason_code=IneligibleReason.SOURCE_EXCEEDS_MAX_BYTES,
            identity=SourceIdentity(decode.max_source_bytes + 1, DIGEST, decode),
            protocol_sha256=PROTOCOL_DIGEST,
        )
    if shape == "too_large_unhashed":
        return SourceIneligibleOutcome(
            logical_source_id="item/a.flac",
            primary_reason_code=IneligibleReason.SOURCE_EXCEEDS_MAX_BYTES,
            identity=SourceIdentity(decode.max_identity_stream_bytes + 1, None, decode),
            protocol_sha256=PROTOCOL_DIGEST,
        )
    if shape == "inspection_rejected":
        return SourceIneligibleOutcome(
            logical_source_id="item/a.flac",
            primary_reason_code=IneligibleReason.SOURCE_INSPECTION_REJECTED,
            identity=small,
            protocol_sha256=PROTOCOL_DIGEST,
            decoder_diagnostic=SanitisedDiagnostic(DiagnosticCategory.INSPECTION_REJECTED),
        )
    if shape == "metadata_with_supplemental":
        return SourceIneligibleOutcome(
            logical_source_id="item/a.flac",
            primary_reason_code=IneligibleReason.UNSUPPORTED_SUBTYPE,
            supplemental_reason_codes=[IneligibleReason.UNSUPPORTED_CONTAINER],
            identity=small,
            protocol_sha256=PROTOCOL_DIGEST,
            observed_metadata=METADATA,
        )
    if shape == "decode_rejected_with_detail":
        return SourceIneligibleOutcome(
            logical_source_id="item/a.flac",
            primary_reason_code=IneligibleReason.SOURCE_DECODE_REJECTED,
            identity=small,
            protocol_sha256=PROTOCOL_DIGEST,
            observed_metadata=METADATA,
            decoder_diagnostic=SanitisedDiagnostic(
                DiagnosticCategory.DECODE_REJECTED, "Internal psf_fseek() failed."
            ),
        )
    return SourceIneligibleOutcome(
        logical_source_id="item/a.flac",
        primary_reason_code=IneligibleReason.INVALID_SOURCE_AMPLITUDE,
        identity=small,
        protocol_sha256=PROTOCOL_DIGEST,
        observed_metadata=METADATA,
    )


SHAPES = (
    "too_large_streamed",
    "too_large_unhashed",
    "inspection_rejected",
    "metadata_with_supplemental",
    "decode_rejected_with_detail",
    "waveform_rejected",
)


def through_json(value: dict[str, Any]) -> Any:
    """What the receiving process parses: the wire form after a JSON round trip."""
    return json.loads(json.dumps(value))


class TestRoundTrip:
    """A record that crosses the wire comes back equal to the one that was sent."""

    @pytest.mark.parametrize("shape", SHAPES)
    def test_every_evidence_shape_round_trips(self, decode: Decode, shape: str) -> None:
        record = ineligible(decode, shape)
        assert outcome_from_wire(through_json(outcome_to_wire(record)), decode) == record

    def test_an_abort_naming_a_source_with_an_identity_round_trips(self, decode: Decode) -> None:
        record = RunAbortOutcome(
            reason=AbortReason.SOURCE_HASH_MISMATCH,
            detail="digest differs",
            logical_source_id="item/a.flac",
            identity=SourceIdentity(1024, DIGEST, decode),
        )
        assert outcome_from_wire(through_json(outcome_to_wire(record)), decode) == record

    def test_an_abort_naming_no_source_round_trips(self, decode: Decode) -> None:
        record = RunAbortOutcome(reason=AbortReason.GOLDEN_FIXTURE_FAILURE, detail="fixture")
        assert outcome_from_wire(through_json(outcome_to_wire(record)), decode) == record


class TestNothingDerivedIsSent:
    """A wire record cannot supply a value a constructor derives."""

    def test_an_ineligible_record_carries_constructor_inputs_only(self, decode: Decode) -> None:
        wire = outcome_to_wire(ineligible(decode, "decode_rejected_with_detail"))
        assert set(wire) == {
            "kind",
            "logical_source_id",
            "primary_reason_code",
            "supplemental_reason_codes",
            "identity",
            "protocol_sha256",
            "observed_metadata",
            "decoder_diagnostic",
        }
        assert set(wire["identity"]) == {"source_bytes", "sha256"}
        assert set(wire["observed_metadata"]) == {
            "container",
            "subtype",
            "channels",
            "sample_rate",
            "frames",
        }

    def test_an_abort_carries_constructor_inputs_only(self, decode: Decode) -> None:
        record = RunAbortOutcome(reason=AbortReason.GOLDEN_FIXTURE_FAILURE, detail="fixture")
        assert set(outcome_to_wire(record)) == {
            "kind",
            "reason",
            "detail",
            "logical_source_id",
            "identity",
        }


def _set(path: tuple[str, ...], value: object) -> Callable[[dict[str, Any]], None]:
    """A tamper that sets one nested key."""

    def apply(wire: dict[str, Any]) -> None:
        target = wire
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return apply


def _drop(key: str) -> Callable[[dict[str, Any]], None]:
    """A tamper that removes one top-level key."""
    return lambda wire: wire.pop(key)


# Each row is a record a worker cannot have built through the constructors, or a value of
# the wrong JSON type, applied to a valid decode-rejection record.
INELIGIBLE_TAMPERS = {
    "derived_stage_supplied": _set(("terminating_evaluation_stage",), "decode"),
    "key_missing": _drop("decoder_diagnostic"),
    "metadata_on_a_record_that_never_inspected": _set(
        ("primary_reason_code",), "SOURCE_EXCEEDS_MAX_BYTES"
    ),
    "diagnostic_missing_on_a_decoder_rejection": _set(("decoder_diagnostic",), None),
    "detail_outside_the_allowlist": _set(("decoder_diagnostic", "detail"), "/home/x/a.flac"),
    "digest_missing_within_the_identity_bound": _set(("identity", "sha256"), None),
    "channel_count_is_a_bool": _set(("observed_metadata", "channels"), True),
    "source_id_is_a_number": _set(("logical_source_id",), 7),
    "reason_outside_the_vocabulary": _set(("primary_reason_code",), "SOURCE_TOO_QUIET"),
    "reason_is_a_number": _set(("primary_reason_code",), 3),
    "supplemental_repeats_the_primary": _set(
        ("supplemental_reason_codes",), ["SOURCE_DECODE_REJECTED"]
    ),
    "category_outside_the_vocabulary": _set(("decoder_diagnostic", "category"), "other"),
    "identity_is_not_an_object": _set(("identity",), [1024, DIGEST]),
}


class TestRefusals:
    """Everything a constructor refuses is refused on the way in, as `OutcomeWireError`."""

    @pytest.mark.parametrize("tamper", INELIGIBLE_TAMPERS.values(), ids=INELIGIBLE_TAMPERS)
    def test_a_tampered_ineligible_record_is_refused(
        self, decode: Decode, tamper: Callable[[dict[str, Any]], None]
    ) -> None:
        wire = through_json(outcome_to_wire(ineligible(decode, "decode_rejected_with_detail")))
        tampered = copy.deepcopy(wire)
        tamper(tampered)
        assert outcome_from_wire(wire, decode) is not None
        with pytest.raises(OutcomeWireError):
            outcome_from_wire(tampered, decode)

    def test_supplemental_reasons_sent_as_an_object_are_refused(self, decode: Decode) -> None:
        """On a record where the code is a valid supplemental, so only the list check refuses
        it: an object iterates as its keys and would otherwise rebuild as that code."""
        wire = through_json(outcome_to_wire(ineligible(decode, "metadata_with_supplemental")))
        wire["supplemental_reason_codes"] = {"UNSUPPORTED_CONTAINER": True}
        with pytest.raises(OutcomeWireError, match="must be a list"):
            outcome_from_wire(wire, decode)

    def test_an_abort_that_must_not_name_a_source_is_refused(self, decode: Decode) -> None:
        """The case pickle let through: a golden-fixture failure naming a recording."""
        wire = through_json(
            outcome_to_wire(
                RunAbortOutcome(
                    reason=AbortReason.UNEXPECTED_IO_FAILURE,
                    detail="read failed",
                    logical_source_id="item/a.flac",
                )
            )
        )
        wire["reason"] = "GOLDEN_FIXTURE_FAILURE"
        with pytest.raises(OutcomeWireError, match="run-global failure"):
            outcome_from_wire(wire, decode)

    def test_a_value_that_is_not_an_object_is_refused(self, decode: Decode) -> None:
        with pytest.raises(OutcomeWireError, match="must be an object"):
            outcome_from_wire("abort", decode)

    def test_an_abort_shaped_record_of_another_kind_is_refused(self, decode: Decode) -> None:
        """Abort-shaped on purpose, so the kind check alone stands between it and a record."""
        wire = through_json(
            outcome_to_wire(RunAbortOutcome(reason=AbortReason.GOLDEN_FIXTURE_FAILURE, detail="x"))
        )
        wire["kind"] = "canonical"
        with pytest.raises(OutcomeWireError, match="record kind 'canonical'"):
            outcome_from_wire(wire, decode)
