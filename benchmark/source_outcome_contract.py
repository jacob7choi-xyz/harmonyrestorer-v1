"""Conformance between the frozen protocol and its code-declared mirrors.

The top of the dependency graph, and the only module here that touches a loaded `Protocol`.
It imports evidence and outcomes; neither imports it. That direction is deliberate: a lower
layer importing the verifier would make the graph cyclic and quietly make import order part
of the contract.

`verify_source_outcome_contract` is called explicitly rather than at import, so importing
either lower module never depends on library versions, filesystem state, or the production
config path. Drift surfaces in the test that calls this, which is where it should.
"""

from __future__ import annotations

from dataclasses import fields

from benchmark.protocol import Decode, Protocol, SourceOutcomes
from benchmark.source_evidence import (
    IdentityMethod,
    IdentityStatus,
    SourceOutcomeError,
    identity_for_size,
)
from benchmark.source_outcome import (
    ABORT_SOURCE_CONTEXT,
    EVALUATION_STAGE_ORDER,
    REASON_PRECEDENCE,
    REASON_STAGE,
    AbortReason,
    IneligibleReason,
    SourceIneligibleOutcome,
)

__all__ = ["verify_source_outcome_contract"]


def verify_source_outcome_contract(protocol: Protocol) -> None:
    """Check every vocabulary, ordering, and mapping here against the protocol in force.

    Called explicitly rather than at import. Importing this module must not depend on
    library versions, filesystem state, or the production config path, or type checking
    and test collection would fail for reasons unrelated to what they are checking. The
    test suite calls this, which is where drift should surface.

    Args:
        protocol: A loaded, validated protocol.

    Raises:
        SourceOutcomeError: If anything this module owns disagrees with it.
    """
    outcomes = protocol.source_outcomes
    for name, mine, theirs in (
        ("evaluation_stage_order", EVALUATION_STAGE_ORDER, outcomes.evaluation_stage_order),
        ("reason_precedence", REASON_PRECEDENCE, outcomes.reason_precedence),
        ("abort_reasons", tuple(AbortReason), outcomes.abort_reasons),
        ("identity_status_values", tuple(IdentityStatus), outcomes.identity_status_values),
        ("identity_method_values", tuple(IdentityMethod), outcomes.identity_method_values),
    ):
        if tuple(str(value) for value in mine) != tuple(theirs):
            raise SourceOutcomeError(f"{name} disagrees with the protocol: {mine!r} vs {theirs!r}")

    if sorted(str(reason) for reason in IneligibleReason) != sorted(outcomes.ineligible_reasons):
        raise SourceOutcomeError("ineligible reason vocabulary disagrees with the protocol")
    if {str(k): str(v) for k, v in REASON_STAGE.items()} != dict(outcomes.reason_stage):
        raise SourceOutcomeError("reason-to-stage mapping disagrees with the protocol")
    # Completeness only. The policy itself is code-owned rather than loaded: protocol v4
    # scopes its record schema to source-ineligible outcomes and leaves abort-record
    # structure to the runner, so which aborts may name a candidate is a derivation from
    # v4's meanings rather than a value it states. Checking coverage here stops a new abort
    # reason arriving with no policy at all; it does not claim v4 authorises this map.
    if set(ABORT_SOURCE_CONTEXT) != set(AbortReason):
        raise SourceOutcomeError("the abort source-context policy does not cover every reason")

    _verify_record_shape(outcomes)
    _verify_identity_bounds(protocol.canonicalisation.decode)


def _verify_record_shape(outcomes: SourceOutcomes) -> None:
    """Check that this module carries every field and constraint the protocol names.

    Args:
        outcomes: The frozen `source_outcomes` section.

    Raises:
        SourceOutcomeError: If a required core field has no home here, a conditional field
            is not enforced, or a record constraint has no counterpart.
    """
    core = set(outcomes.record_required_core)
    present = {field.name for field in fields(SourceIneligibleOutcome)}
    # `identity_status`, `identity_method`, and `local_sha256` live on the identity object
    # rather than flattened onto the record, so they are satisfied by that field.
    present |= {"identity_status", "identity_method", "local_sha256"}
    missing = core - present
    if missing:
        raise SourceOutcomeError(f"record is missing required core fields: {sorted(missing)}")

    conditional = dict(outcomes.record_conditional)
    expected_conditions = {
        "local_sha256": "when_identity_complete_sha256",
        "observed_metadata": "when_inspection_succeeded",
        "decoder_diagnostic": "when_decoder_rejected",
        "supplemental_reason_codes": "when_additional_reasons_established",
    }
    if conditional != expected_conditions:
        raise SourceOutcomeError(f"record conditional fields disagree: {conditional!r}")

    unenforced = set(outcomes.record_constraints) - _ENFORCED_CONSTRAINTS
    if unenforced:
        raise SourceOutcomeError(
            f"record constraints with no enforcement here: {sorted(unenforced)}"
        )


def _verify_identity_bounds(decode: Decode) -> None:
    """Check that the derivation agrees with the frozen bounds at every boundary.

    Args:
        decode: The frozen `canonicalisation.decode` section.

    Raises:
        SourceOutcomeError: If a boundary size selects the wrong method.
    """
    expected = (
        (decode.max_source_bytes, IdentityMethod.BOUNDED_SINGLE_BUFFER_SHA256),
        (decode.max_source_bytes + 1, IdentityMethod.BOUNDED_STREAMING_SHA256),
        (decode.max_identity_stream_bytes, IdentityMethod.BOUNDED_STREAMING_SHA256),
        (decode.max_identity_stream_bytes + 1, IdentityMethod.NOT_COMPUTED_ABOVE_BOUND),
    )
    for size, method in expected:
        if identity_for_size(size, decode)[1] is not method:
            raise SourceOutcomeError(f"identity derivation disagrees with the bounds at {size}")


# Every constraint protocol v4 states about a record, and where it is enforced. A
# constraint here with no enforcement is documentation, so the verifier refuses one it does
# not recognise rather than letting it pass unnoticed.
_ENFORCED_CONSTRAINTS = frozenset(
    {
        "primary_reason_code_in_ineligible_reasons",
        "supplemental_reason_codes_subset_of_ineligible_reasons",
        "primary_reason_code_not_in_supplemental_reason_codes",
        "supplemental_reason_codes_unique",
        "supplemental_reason_codes_share_primary_reason_stage",
        "supplemental_reason_codes_ordered_by_reason_precedence",
        "unevaluated_evaluation_stages_ordered_by_evaluation_stage_order",
        "identity_status_consistent_with_identity_method",
        "local_sha256_present_iff_identity_complete",
        "identity_method_consistent_with_source_byte_bounds",
        "digest_discarded_when_instability_detected",
    }
)
