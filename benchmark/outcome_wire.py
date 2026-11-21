"""A JSON wire format for the source outcome records, rebuilt only through their constructors.

A canonicalisation worker runs in its own process (protocol v8), so its record has to cross
a process boundary. Pickle cannot carry one safely: unpickling restores a frozen dataclass's
fields without calling its `__init__`. Measured before this module was written, a pickled
`UNEXPECTED_IO_FAILURE` abort naming a source, with its reason bytes swapped, unpickled as a
`GOLDEN_FIXTURE_FAILURE` naming a source, which the constructor refuses.

The wire form therefore carries constructor inputs only. Every field a constructor derives
(the terminating and unevaluated stages, the identity status and method, the duration, the
projected decoded bytes) is left out, so a wire record cannot supply a derived value at all,
and decoding goes through the same constructors as any other record. The receiving process
supplies its own frozen `canonicalisation.decode` section, from which `SourceIdentity`
derives its status and method.

Each object must have exactly its expected keys, as the protocol loader requires of the
protocol file. String and vocabulary values are type-checked here before construction,
because the shared validators assume strings and would fail on a number with an
`AttributeError` or `TypeError` rather than a `SourceOutcomeError`. Integer values are left
to the constructors, whose `checked_int` already refuses a `bool` or a non-integer.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, TypeVar

from benchmark.protocol import Decode
from benchmark.source_evidence import (
    DiagnosticCategory,
    ObservedSourceMetadata,
    SanitisedDiagnostic,
    SourceIdentity,
    SourceOutcomeError,
)
from benchmark.source_outcome import (
    AbortReason,
    IneligibleReason,
    RunAbortOutcome,
    SourceIneligibleOutcome,
)

__all__ = ["OutcomeWireError", "outcome_from_wire", "outcome_to_wire"]

_KIND = "kind"
_INELIGIBLE = "ineligible"
_ABORT = "abort"

_E = TypeVar("_E", bound=StrEnum)


class OutcomeWireError(Exception):
    """Raised when a wire record is malformed or describes a record that cannot be built."""


def outcome_to_wire(outcome: SourceIneligibleOutcome | RunAbortOutcome) -> dict[str, Any]:
    """The constructor inputs of a record, as JSON-ready values.

    Args:
        outcome: An ineligibility record or an abort record.

    Returns:
        A mapping holding a `kind` and the record's constructor inputs, and nothing derived.
    """
    if isinstance(outcome, SourceIneligibleOutcome):
        return {
            _KIND: _INELIGIBLE,
            "logical_source_id": outcome.logical_source_id,
            "primary_reason_code": outcome.primary_reason_code.value,
            "supplemental_reason_codes": [r.value for r in outcome.supplemental_reason_codes],
            "identity": _identity_to_wire(outcome.identity),
            "protocol_sha256": outcome.protocol_sha256,
            "observed_metadata": _metadata_to_wire(outcome.observed_metadata),
            "decoder_diagnostic": _diagnostic_to_wire(outcome.decoder_diagnostic),
        }
    return {
        _KIND: _ABORT,
        "reason": outcome.reason.value,
        "detail": outcome.detail,
        "logical_source_id": outcome.logical_source_id,
        "identity": None if outcome.identity is None else _identity_to_wire(outcome.identity),
    }


def outcome_from_wire(data: object, config: Decode) -> SourceIneligibleOutcome | RunAbortOutcome:
    """Rebuild a record from its wire form through its constructor.

    Args:
        data: A value parsed from JSON.
        config: The receiving process's frozen `canonicalisation.decode` section.

    Returns:
        The rebuilt record.

    Raises:
        OutcomeWireError: If a key is missing or unexpected, a value has the wrong type or
            is not in its vocabulary, or the constructor refuses the combination.
    """
    kind = _mapping(data, "record").get(_KIND)
    try:
        if kind == _INELIGIBLE:
            return _ineligible_from_wire(data, config)
        if kind == _ABORT:
            return _abort_from_wire(data, config)
    except SourceOutcomeError as error:
        raise OutcomeWireError(f"the record cannot be built: {error}") from error
    raise OutcomeWireError(f"record kind {kind!r} is neither {_INELIGIBLE!r} nor {_ABORT!r}")


def _ineligible_from_wire(data: object, config: Decode) -> SourceIneligibleOutcome:
    """Rebuild an ineligibility record."""
    fields = _exact(
        data,
        "ineligible record",
        (
            _KIND,
            "logical_source_id",
            "primary_reason_code",
            "supplemental_reason_codes",
            "identity",
            "protocol_sha256",
            "observed_metadata",
            "decoder_diagnostic",
        ),
    )
    supplemental = fields["supplemental_reason_codes"]
    if not isinstance(supplemental, list):
        raise OutcomeWireError(f"supplemental_reason_codes must be a list, got {supplemental!r}")
    metadata = fields["observed_metadata"]
    diagnostic = fields["decoder_diagnostic"]
    return SourceIneligibleOutcome(
        logical_source_id=_str(fields, "logical_source_id"),
        primary_reason_code=_member(IneligibleReason, fields["primary_reason_code"]),
        supplemental_reason_codes=[_member(IneligibleReason, code) for code in supplemental],
        identity=_identity_from_wire(fields["identity"], config),
        protocol_sha256=_str(fields, "protocol_sha256"),
        observed_metadata=None if metadata is None else _metadata_from_wire(metadata),
        decoder_diagnostic=None if diagnostic is None else _diagnostic_from_wire(diagnostic),
    )


def _abort_from_wire(data: object, config: Decode) -> RunAbortOutcome:
    """Rebuild an abort record."""
    fields = _exact(
        data, "abort record", (_KIND, "reason", "detail", "logical_source_id", "identity")
    )
    source = fields["logical_source_id"]
    identity = fields["identity"]
    return RunAbortOutcome(
        reason=_member(AbortReason, fields["reason"]),
        detail=_str(fields, "detail"),
        logical_source_id=None if source is None else _str(fields, "logical_source_id"),
        identity=None if identity is None else _identity_from_wire(identity, config),
    )


def _identity_to_wire(identity: SourceIdentity) -> dict[str, Any]:
    """An identity's constructor inputs; its status and method are derived on rebuild."""
    return {"source_bytes": identity.source_bytes, "sha256": identity.sha256}


def _identity_from_wire(data: object, config: Decode) -> SourceIdentity:
    """Rebuild an identity, deriving its status and method from the receiver's bounds."""
    fields = _exact(data, "identity", ("source_bytes", "sha256"))
    digest = fields["sha256"]
    return SourceIdentity(
        fields["source_bytes"], None if digest is None else _str(fields, "sha256"), config
    )


def _metadata_to_wire(metadata: ObservedSourceMetadata | None) -> dict[str, Any] | None:
    """Metadata's constructor inputs; its duration and projected size are derived."""
    if metadata is None:
        return None
    return {
        "container": metadata.container,
        "subtype": metadata.subtype,
        "channels": metadata.channels,
        "sample_rate": metadata.sample_rate,
        "frames": metadata.frames,
    }


def _metadata_from_wire(data: object) -> ObservedSourceMetadata:
    """Rebuild metadata; its constructor rejects a bool or a negative where an int belongs."""
    fields = _exact(
        data, "observed_metadata", ("container", "subtype", "channels", "sample_rate", "frames")
    )
    return ObservedSourceMetadata(
        container=_str(fields, "container"),
        subtype=_str(fields, "subtype"),
        channels=fields["channels"],
        sample_rate=fields["sample_rate"],
        frames=fields["frames"],
    )


def _diagnostic_to_wire(diagnostic: SanitisedDiagnostic | None) -> dict[str, Any] | None:
    """A diagnostic's category and allowlisted detail."""
    if diagnostic is None:
        return None
    return {"category": diagnostic.category.value, "detail": diagnostic.detail}


def _diagnostic_from_wire(data: object) -> SanitisedDiagnostic:
    """Rebuild a diagnostic; its constructor refuses a detail outside the allowlist."""
    fields = _exact(data, "decoder_diagnostic", ("category", "detail"))
    detail = fields["detail"]
    return SanitisedDiagnostic(
        _member(DiagnosticCategory, fields["category"]),
        None if detail is None else _str(fields, "detail"),
    )


def _mapping(data: object, name: str) -> Mapping[str, Any]:
    """Require a JSON object."""
    if not isinstance(data, Mapping):
        raise OutcomeWireError(f"{name} must be an object, got {type(data).__name__}")
    return data


def _exact(data: object, name: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    """Require a JSON object with exactly these keys, none missing and none unexpected."""
    fields = _mapping(data, name)
    missing = sorted(set(keys) - set(fields))
    unexpected = sorted(set(fields) - set(keys))
    if missing or unexpected:
        raise OutcomeWireError(f"{name} is missing {missing} and has unexpected {unexpected}")
    return fields


def _str(fields: Mapping[str, Any], key: str) -> str:
    """Require a string value."""
    value = fields[key]
    if not isinstance(value, str):
        raise OutcomeWireError(f"{key} must be a string, got {value!r}")
    return value


def _member(vocabulary: type[_E], value: object) -> _E:
    """Require a value from a closed vocabulary."""
    if not isinstance(value, str):
        raise OutcomeWireError(f"{value!r} is not a {vocabulary.__name__}")
    try:
        return vocabulary(value)
    except ValueError as error:
        raise OutcomeWireError(f"{value!r} is not a {vocabulary.__name__}") from error
