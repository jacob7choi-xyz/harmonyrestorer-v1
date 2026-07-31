"""Tests for the source-outcome record model.

Every constraint protocol v4 states about a record has a test here that constructs the
record it forbids and asserts the construction fails. A constraint stated in a config and
never exercised is documentation, which is the defect this benchmark has spent its whole
rebuild removing.

The bypass tests matter most. An earlier draft put the rules in a `create` classmethod and
left the generated constructor public, so a record with the wrong terminating stage, a
duplicated supplemental reason, and a fabricated digest constructed without complaint. A
validating factory beside a public constructor is a convention, not a boundary.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import pickle
import subprocess
import sys
import textwrap

import pytest

from benchmark.protocol import Decode, load_protocol
from benchmark.source_evidence import (
    _PUBLISHABLE_DETAILS,
    _TOKEN,
    DiagnosticCategory,
    IdentityMethod,
    IdentityStatus,
    ObservedSourceMetadata,
    SanitisedDiagnostic,
    SourceIdentity,
    SourceOutcomeError,
    identity_for_size,
    sanitise_decoder_diagnostic,
)
from benchmark.source_outcome import (
    ABORT_SOURCE_CONTEXT,
    EVALUATION_STAGE_ORDER,
    REASON_PRECEDENCE,
    REASON_STAGE,
    AbortReason,
    EvaluationStage,
    IneligibleReason,
    RunAbortOutcome,
    SourceContextPolicy,
    SourceIneligibleOutcome,
    terminating_stage_for,
    unevaluated_stages_after,
)
from benchmark.source_outcome_contract import verify_source_outcome_contract

DIGEST = "a" * 64
PROTOCOL_DIGEST = "b" * 64
METADATA = ObservedSourceMetadata(
    container="FLAC", subtype="PCM_24", channels=2, sample_rate=44100, frames=88200
)
DIAGNOSTIC = SanitisedDiagnostic(DiagnosticCategory.DECODE_REJECTED)
ALLOWLISTED = "Internal psf_fseek() failed."

_INSPECTION_POSITION = EVALUATION_STAGE_ORDER.index(EvaluationStage.INSPECTION)
_DECODER_REJECTIONS = (
    IneligibleReason.SOURCE_INSPECTION_REJECTED,
    IneligibleReason.SOURCE_DECODE_REJECTED,
)


@pytest.fixture
def decode() -> Decode:
    """The frozen decode section, which owns the identity bounds."""
    return load_protocol().canonicalisation.decode


@pytest.fixture
def small_identity(decode: Decode) -> SourceIdentity:
    """Identity for an artifact small enough for the single-buffer path."""
    return SourceIdentity(1024, DIGEST, decode)


def outcome(
    identity: SourceIdentity,
    reason: IneligibleReason = IneligibleReason.UNSUPPORTED_SUBTYPE,
    **overrides: object,
) -> SourceIneligibleOutcome:
    """Build a record valid for its reason, with one field replaced per test.

    Evidence defaults follow the stage, because it is required in both directions: metadata
    exactly when inspection succeeded, a diagnostic exactly on a decoder rejection. A fixed
    default would make most tests fail for the wrong reason.
    """
    stage = terminating_stage_for(reason)
    kwargs: dict[str, object] = {
        "logical_source_id": "candidate-0001",
        "primary_reason_code": reason,
        "identity": identity,
        "protocol_sha256": PROTOCOL_DIGEST,
        "observed_metadata": (
            METADATA if EVALUATION_STAGE_ORDER.index(stage) > _INSPECTION_POSITION else None
        ),
        "decoder_diagnostic": DIAGNOSTIC if reason in _DECODER_REJECTIONS else None,
    }
    kwargs.update(overrides)
    return SourceIneligibleOutcome(**kwargs)  # type: ignore[arg-type]


class TestThereIsNoUnvalidatedConstructionPath:
    """The finding that sent the first draft back: a factory is not a boundary."""

    def test_derived_fields_cannot_be_supplied_at_all(self) -> None:
        """Not merely checked. A caller that cannot pass a stage cannot pass a wrong one."""
        parameters = inspect.signature(SourceIneligibleOutcome.__init__).parameters
        for name in ("terminating_evaluation_stage", "unevaluated_evaluation_stages"):
            assert name not in parameters

    def test_identity_status_and_method_cannot_be_supplied(self) -> None:
        parameters = inspect.signature(SourceIdentity.__init__).parameters
        for name in ("status", "method"):
            assert name not in parameters

    @pytest.mark.parametrize(
        "cls",
        [
            SourceIneligibleOutcome,
            SourceIdentity,
            RunAbortOutcome,
            ObservedSourceMetadata,
            SanitisedDiagnostic,
        ],
    )
    def test_init_false_survives_the_deletion_of_the_written_constructor(self, cls: type) -> None:
        """`init=False` looks redundant while a hand-written `__init__` exists, because
        dataclasses skip generating one when the class already defines it. It is not: it is
        what makes a future deletion of that constructor fail loudly instead of silently
        restoring a generated one that assigns every field and validates nothing.

        Rebuilt here without the written `__init__`, which is the only way to observe the
        difference. With `init=False` the rebuilt class refuses construction; without it,
        the same class accepts a record with no checks at all.
        """
        annotations = dict.fromkeys((f.name for f in dataclasses.fields(cls)), object)
        guarded = dataclasses.dataclass(frozen=True, init=False)(
            type(cls.__name__, (), {"__annotations__": annotations})
        )
        unguarded = dataclasses.dataclass(frozen=True)(
            type(cls.__name__, (), {"__annotations__": annotations})
        )
        values = [None] * len(annotations)
        assert unguarded(*values) is not None
        with pytest.raises(TypeError):
            guarded(*values)
        assert cls.__dataclass_params__.init is False  # type: ignore[attr-defined]

    def test_the_previously_accepted_invalid_record_is_now_refused(self, small_identity) -> None:
        """The exact record the first draft constructed: wrong stage, duplicated
        supplemental, fabricated digest, diagnostic with no decoder rejection."""
        with pytest.raises((SourceOutcomeError, TypeError)):
            SourceIneligibleOutcome(  # type: ignore[call-arg]
                logical_source_id="x",
                primary_reason_code=IneligibleReason.UNSUPPORTED_SUBTYPE,
                terminating_evaluation_stage=EvaluationStage.WAVEFORM_VALIDATION,
                unevaluated_evaluation_stages=(),
                supplemental_reason_codes=(
                    IneligibleReason.NONFINITE_SOURCE,
                    IneligibleReason.NONFINITE_SOURCE,
                ),
                identity=small_identity,
                protocol_sha256="not-a-hash",
                observed_metadata=None,
                decoder_diagnostic="fabricated",
            )


class TestTheSplitPreservedWhatCallersSee:
    """A layered refactor can change observable identity while every test still passes."""

    def test_the_exception_keeps_its_name(self) -> None:
        """Defining the class as `SourceValidationError` and aliasing the historical name
        would preserve what a caller can catch while changing what a traceback, a log line,
        and pytest all print, because `type(error).__name__` follows the class rather than
        the alias."""
        assert SourceOutcomeError.__name__ == "SourceOutcomeError"
        with pytest.raises(SourceOutcomeError) as caught:
            SourceIdentity(1024, "not-a-digest", load_protocol().canonicalisation.decode)
        assert type(caught.value).__name__ == "SourceOutcomeError"

    def test_every_module_exposes_the_same_class_object(self) -> None:
        """Not merely a class of the same name. Two distinct classes would make an `except`
        in one module silently miss what another raises."""
        import benchmark._source_validation as validation
        import benchmark.source_evidence as evidence
        import benchmark.source_outcome as outcome

        assert evidence.SourceOutcomeError is validation.SourceOutcomeError
        assert outcome.SourceOutcomeError is validation.SourceOutcomeError


class TestContractAgreesWithTheProtocol:
    """These vocabularies are code-owned, so something has to notice when they drift."""

    def test_the_loaded_protocol_agrees(self) -> None:
        verify_source_outcome_contract(load_protocol())

    @pytest.mark.parametrize(
        "module",
        [
            "benchmark._source_validation",
            "benchmark.source_evidence",
            "benchmark.source_outcome",
        ],
    )
    def test_no_lower_module_loads_the_protocol_at_import(self, module: str) -> None:
        """The acceptance property of the module split, committed rather than checked once
        by hand. Every module below the contract layer must import with the production
        config path pointing at nothing: importing protocol dataclasses is fine, loading the
        protocol is not, because that would make type checking, test collection, and
        mutation runs fail over a library version."""
        script = textwrap.dedent(
            f"""
            import pathlib
            import benchmark.protocol as protocol

            protocol._PROTOCOL_PATH = pathlib.Path("/nonexistent/protocol.json")
            import {module} as loaded

            assert loaded is not None
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_importing_does_not_load_the_protocol(self) -> None:
        """Run in a fresh interpreter with the config path pointed at a file that does not
        exist. If the module loaded the protocol at import, the import would raise.

        A first draft did this by deleting the module from `sys.modules` in-process, which
        left a second copy behind and made a later test patch the wrong one.
        """
        script = textwrap.dedent(
            """
            import pathlib
            import benchmark.protocol as protocol

            protocol._PROTOCOL_PATH = pathlib.Path("/nonexistent/protocol.json")
            import benchmark.source_outcome as module

            assert module.IneligibleReason.EMPTY_SOURCE
            assert module.ABORT_SOURCE_CONTEXT
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_the_verifier_catches_a_drifting_vocabulary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The verifier is all that stands between a code-declared mirror and a silent
        divergence from the config, so it has to actually fail when they differ.

        Patched on the contract module rather than on `source_outcome`: the verifier binds
        these names in its own namespace, so patching the module they came from would not
        reach it and the test would pass without exercising anything.
        """
        import benchmark.source_outcome_contract as module

        monkeypatch.setattr(module, "REASON_PRECEDENCE", REASON_PRECEDENCE[::-1])
        with pytest.raises(SourceOutcomeError, match="reason_precedence disagrees"):
            verify_source_outcome_contract(load_protocol())

    def test_the_verifier_catches_an_unenforced_constraint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A constraint the protocol states and this module does not enforce would
        otherwise pass unnoticed, which is the documentation-theatre pattern again."""
        import benchmark.source_outcome_contract as module

        monkeypatch.setattr(
            module, "_ENFORCED_CONSTRAINTS", frozenset({"supplemental_reason_codes_unique"})
        )
        with pytest.raises(SourceOutcomeError, match="no enforcement here"):
            verify_source_outcome_contract(load_protocol())

    def test_the_verifier_checks_the_bounds_at_every_boundary(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import benchmark.source_outcome_contract as module

        monkeypatch.setattr(
            module, "identity_for_size", lambda *_: (IdentityStatus.COMPLETE_SHA256, None)
        )
        with pytest.raises(SourceOutcomeError, match="disagrees with the bounds"):
            verify_source_outcome_contract(load_protocol())


class TestDerivedFields:
    """A caller that cannot supply a value cannot supply a wrong one."""

    def test_the_terminating_stage_follows_from_the_reason(self, small_identity) -> None:
        record = outcome(small_identity, IneligibleReason.SOURCE_DECODE_REJECTED)
        assert record.terminating_evaluation_stage is EvaluationStage.DECODE

    def test_the_unevaluated_suffix_follows_from_the_stage(self, small_identity) -> None:
        record = outcome(small_identity, IneligibleReason.SOURCE_EXCEEDS_MAX_BYTES)
        assert record.unevaluated_evaluation_stages == EVALUATION_STAGE_ORDER[1:]

    def test_a_terminal_stage_leaves_nothing_unevaluated(self, small_identity) -> None:
        record = outcome(small_identity, IneligibleReason.NONFINITE_SOURCE)
        assert record.unevaluated_evaluation_stages == ()


class TestSupplementalReasons:
    """Membership, uniqueness, and ordering, each rejected rather than repaired."""

    def test_a_duplicate_is_rejected_not_collapsed(self, small_identity) -> None:
        """Converting to a set first would hide a caller defect and double-count one
        narrowing in the published reduction chain."""
        with pytest.raises(SourceOutcomeError, match="contain a duplicate"):
            outcome(
                small_identity,
                supplemental_reason_codes=[
                    IneligibleReason.UNSUPPORTED_CONTAINER,
                    IneligibleReason.UNSUPPORTED_CONTAINER,
                ],
            )

    def test_the_primary_cannot_repeat_as_supplemental(self, small_identity) -> None:
        with pytest.raises(SourceOutcomeError, match="cannot also be supplemental"):
            outcome(
                small_identity, supplemental_reason_codes=[IneligibleReason.UNSUPPORTED_SUBTYPE]
            )

    def test_a_reason_from_a_later_stage_is_rejected(self, small_identity) -> None:
        """The stage was never reached, so the violation cannot have been established."""
        with pytest.raises(SourceOutcomeError, match="never reached"):
            outcome(small_identity, supplemental_reason_codes=[IneligibleReason.NONFINITE_SOURCE])

    def test_a_reason_from_an_earlier_stage_is_rejected(self, small_identity) -> None:
        """An earlier stage cannot hold an unrecorded violation: it would already have
        terminated processing."""
        with pytest.raises(SourceOutcomeError, match="already have stopped"):
            outcome(
                small_identity,
                IneligibleReason.NONFINITE_SOURCE,
                supplemental_reason_codes=[IneligibleReason.UNSUPPORTED_SUBTYPE],
            )

    def test_order_is_normalised_not_trusted(self, small_identity) -> None:
        """Two callers holding the same set and serialising it differently would produce
        different manifest bytes, and therefore different digests."""
        pair = [
            IneligibleReason.SOURCE_EXCEEDS_MAX_DURATION,
            IneligibleReason.UNSUPPORTED_CONTAINER,
        ]
        forward = outcome(small_identity, supplemental_reason_codes=pair)
        reverse = outcome(small_identity, supplemental_reason_codes=list(reversed(pair)))
        assert forward.supplemental_reason_codes == reverse.supplemental_reason_codes
        assert forward.supplemental_reason_codes == (
            IneligibleReason.UNSUPPORTED_CONTAINER,
            IneligibleReason.SOURCE_EXCEEDS_MAX_DURATION,
        )

    def test_no_supplementals_is_the_empty_tuple(self, small_identity) -> None:
        assert outcome(small_identity).supplemental_reason_codes == ()


class TestIdentityCannotDisagreeWithItself:
    """Status, method, digest presence, and source size are four views of one fact."""

    def test_at_the_buffer_bound_the_single_buffer_method_is_used(self, decode) -> None:
        identity = SourceIdentity(decode.max_source_bytes, DIGEST, decode)
        assert identity.method is IdentityMethod.BOUNDED_SINGLE_BUFFER_SHA256
        assert identity.status is IdentityStatus.COMPLETE_SHA256

    def test_one_byte_over_switches_to_streaming(self, decode) -> None:
        identity = SourceIdentity(decode.max_source_bytes + 1, DIGEST, decode)
        assert identity.method is IdentityMethod.BOUNDED_STREAMING_SHA256

    def test_at_the_stream_bound_streaming_still_applies(self, decode) -> None:
        identity = SourceIdentity(decode.max_identity_stream_bytes, DIGEST, decode)
        assert identity.method is IdentityMethod.BOUNDED_STREAMING_SHA256

    def test_above_the_stream_bound_no_digest_is_claimed(self, decode) -> None:
        identity = SourceIdentity(decode.max_identity_stream_bytes + 1, None, decode)
        assert identity.method is IdentityMethod.NOT_COMPUTED_ABOVE_BOUND
        assert identity.status is IdentityStatus.UNAVAILABLE_ABOVE_IDENTITY_STREAM_BOUND
        assert identity.sha256 is None

    def test_a_digest_above_the_stream_bound_is_rejected(self, decode) -> None:
        with pytest.raises(SourceOutcomeError, match="no digest can have been computed"):
            SourceIdentity(decode.max_identity_stream_bytes + 1, DIGEST, decode)

    def test_a_missing_digest_below_the_bound_is_rejected(self, decode) -> None:
        with pytest.raises(SourceOutcomeError, match="a digest is required"):
            SourceIdentity(1024, None, decode)

    def test_a_negative_size_is_rejected(self, decode) -> None:
        with pytest.raises(SourceOutcomeError, match="must be at least 0"):
            SourceIdentity(-1, DIGEST, decode)


class TestDigestsAreValidated:
    """These fields carry identity, so a value that is not a digest is not a digest."""

    @pytest.mark.parametrize(
        "value",
        ["", "abc", "not a digest", "g" * 64, "A" * 64, " " + "a" * 63, "a" * 63, "a" * 65],
    )
    def test_a_non_canonical_source_digest_is_rejected(self, decode, value: str) -> None:
        with pytest.raises(SourceOutcomeError, match="64 lowercase hexadecimal"):
            SourceIdentity(1024, value, decode)

    @pytest.mark.parametrize("value", ["", "not-a-hash", "A" * 64, "a" * 63])
    def test_a_non_canonical_protocol_digest_is_rejected(self, small_identity, value: str) -> None:
        with pytest.raises(SourceOutcomeError, match="64 lowercase hexadecimal"):
            outcome(small_identity, protocol_sha256=value)

    def test_uppercase_is_rejected_rather_than_normalised(self, decode) -> None:
        """A record accepting two spellings of one digest breaks content addressing."""
        with pytest.raises(SourceOutcomeError):
            SourceIdentity(1024, DIGEST.upper(), decode)


class TestIdentifiersAreNotBlank:
    """Blank-as-present is how a record names nothing while appearing complete."""

    @pytest.mark.parametrize("value", ["", "   ", "\n", "\t", " candidate-1", "candidate-1 "])
    def test_a_blank_or_padded_source_id_is_rejected(self, small_identity, value: str) -> None:
        with pytest.raises(SourceOutcomeError, match="non-blank and unpadded"):
            outcome(small_identity, logical_source_id=value)

    @pytest.mark.parametrize("value", ["", "  ", "\n"])
    def test_a_blank_abort_detail_is_rejected(self, value: str) -> None:
        with pytest.raises(SourceOutcomeError, match="non-blank and unpadded"):
            RunAbortOutcome(reason=AbortReason.UNEXPECTED_IO_FAILURE, detail=value)


class TestConditionalEvidenceIsRequiredBothWays:
    """Rejecting impossible evidence is half the rule; the other half is requiring it."""

    def test_metadata_before_inspection_succeeded_is_rejected(self, small_identity) -> None:
        with pytest.raises(SourceOutcomeError, match="observed_metadata must be absent"):
            outcome(
                small_identity,
                IneligibleReason.SOURCE_EXCEEDS_MAX_BYTES,
                observed_metadata=METADATA,
            )

    def test_metadata_missing_after_inspection_succeeded_is_rejected(self, small_identity) -> None:
        """A record that knows less than its own stage implies. The first draft accepted
        this, because it only enforced the forbidding direction."""
        with pytest.raises(SourceOutcomeError, match="observed_metadata must be present"):
            outcome(small_identity, observed_metadata=None)

    def test_a_diagnostic_without_a_decoder_rejection_is_rejected(self, small_identity) -> None:
        with pytest.raises(SourceOutcomeError, match="decoder_diagnostic to be absent"):
            outcome(small_identity, decoder_diagnostic=DIAGNOSTIC)

    @pytest.mark.parametrize("reason", _DECODER_REJECTIONS)
    def test_a_decoder_rejection_without_a_diagnostic_is_rejected(
        self, small_identity, reason: IneligibleReason
    ) -> None:
        """Every observed libsndfile rejection carries a message, so requiring one asks for
        nothing that has to be invented."""
        with pytest.raises(SourceOutcomeError, match="decoder_diagnostic to be present"):
            outcome(small_identity, reason, decoder_diagnostic=None)


class TestDiagnosticsAreSanitisedNotInspected:
    """An earlier draft accepted arbitrary text and rejected it for containing a slash.

    That is a validator wearing a sanitiser's clothes, and it was wrong both ways: it
    refused `unsupported subtype PCM_24/32` while accepting a Windows path, a bare
    filename, a username, and an embedded newline. Safety is now constructed rather than
    inferred.
    """

    @pytest.mark.parametrize(
        ("label", "raw", "path"),
        [
            (
                "posix path",
                "Error opening '/Users/someone/data/raw/track.flac': malformed",
                "/Users/someone/data/raw/track.flac",
            ),
            (
                "linux path",
                "Error opening '/home/someone/corpus/track.flac': malformed",
                "/home/someone/corpus/track.flac",
            ),
            (
                "windows path",
                "Error opening 'C:\\Users\\someone\\track.flac': malformed",
                "C:\\Users\\someone\\track.flac",
            ),
            ("separatorless path", "C:UsersJacobtrack.flac unreadable", None),
            ("bare filename", "GoldbergVariations_JacobChoi.flac could not be opened", None),
            ("username", "permission denied for user jacobchoi", None),
            ("embedded newline", "malformed\nat offset 12", None),
        ],
    )
    def test_no_native_text_survives_unless_allowlisted(
        self, label: str, raw: str, path: str | None
    ) -> None:
        """Every one of these reaches a record as the category alone. The last four were
        accepted verbatim by the separator heuristic."""
        assert path is None or path  # the path is context for the reader, not an argument
        result = sanitise_decoder_diagnostic(raw, category=DiagnosticCategory.DECODE_REJECTED)
        assert result.detail is None, label
        assert result.category is DiagnosticCategory.DECODE_REJECTED

    def test_a_legitimate_message_is_no_longer_refused(self) -> None:
        """The heuristic rejected this outright because of one slash. It now simply
        contributes no detail, which loses nothing a record needed."""
        result = sanitise_decoder_diagnostic(
            "unsupported subtype PCM_24/32", category=DiagnosticCategory.INSPECTION_REJECTED
        )
        assert result.category is DiagnosticCategory.INSPECTION_REJECTED
        assert result.detail is None

    def test_an_allowlisted_fragment_is_published(self) -> None:
        result = sanitise_decoder_diagnostic(
            ALLOWLISTED, category=DiagnosticCategory.DECODE_REJECTED
        )
        assert result.detail == ALLOWLISTED

    def test_every_allowlist_entry_is_a_fixed_string(self) -> None:
        """The allowlist is the whole control, and that rests on it admitting no variable
        content. A path redaction and a byte bound were written alongside it and removed as
        unreachable; if an entry ever gains a wildcard, both must return with it."""
        for pattern in _PUBLISHABLE_DETAILS:
            body = pattern.pattern.removeprefix(r"\A").removesuffix(r"\Z")
            assert not any(token in body for token in (".*", ".+", "[", "(?", "\\d", "\\w"))

    def test_the_type_refuses_arbitrary_text_directly(self) -> None:
        """The sanitiser is not merely the recommended path. A caller holding a raw native
        string has no way to get it into a record by constructing the type."""
        with pytest.raises(SourceOutcomeError, match="not an allowlisted publishable fragment"):
            SanitisedDiagnostic(DiagnosticCategory.DECODE_REJECTED, "/Users/someone/track.flac")

    def test_a_wording_change_degrades_to_the_category_and_nothing_else(
        self, small_identity
    ) -> None:
        """The point of splitting category from detail. The allowlist is a code-owned
        privacy policy, so when a library rewords a message the detail disappears and the
        source's verdict, its terminating stage, and its place in the reduction chain are
        all untouched."""
        known = sanitise_decoder_diagnostic(
            ALLOWLISTED, category=DiagnosticCategory.DECODE_REJECTED
        )
        reworded = sanitise_decoder_diagnostic(
            "internal seek failure (reworded upstream)",
            category=DiagnosticCategory.DECODE_REJECTED,
        )
        before = outcome(
            small_identity, IneligibleReason.SOURCE_DECODE_REJECTED, decoder_diagnostic=known
        )
        after = outcome(
            small_identity, IneligibleReason.SOURCE_DECODE_REJECTED, decoder_diagnostic=reworded
        )
        assert known.detail == ALLOWLISTED
        assert reworded.detail is None
        assert reworded.category is known.category
        assert before.primary_reason_code == after.primary_reason_code
        assert before.terminating_evaluation_stage == after.terminating_evaluation_stage
        assert before.unevaluated_evaluation_stages == after.unevaluated_evaluation_stages

    def test_a_category_is_always_available(self) -> None:
        """Requiring a native message was justified by three fixtures on one libsndfile
        build. That shows availability there, not universality: another build, a
        wrapper-raised error, or a killed decode can carry nothing. The category is known
        from which call was made and never has to be invented."""
        result = sanitise_decoder_diagnostic("", category=DiagnosticCategory.INSPECTION_REJECTED)
        assert result.category is DiagnosticCategory.INSPECTION_REJECTED
        assert result.detail is None


class TestObservedMetadataIsTyped:
    """A mapping of scalars was too weak, and `bool` is an `int` in Python."""

    def test_a_bool_channel_count_is_rejected(self) -> None:
        """`isinstance(True, int)` is true, so a channel count of `True` was recorded as
        one channel. This project has been bitten by bool-as-int before."""
        with pytest.raises(SourceOutcomeError, match="must be a plain integer"):
            ObservedSourceMetadata(
                container="FLAC",
                subtype="PCM_24",
                channels=True,  # type: ignore[arg-type]
                sample_rate=44100,
                frames=10,
            )

    def test_duration_cannot_contradict_the_frames_and_rate(self) -> None:
        """The correlated-field defect this type exists to remove, one level in. A caller
        supplying all three could author 16,000 frames at 16 kHz lasting 900 seconds: every
        field valid alone, and together two different files."""
        assert (
            "duration_seconds" not in inspect.signature(ObservedSourceMetadata.__init__).parameters
        )
        derived = ObservedSourceMetadata(
            container="FLAC", subtype="PCM_24", channels=1, sample_rate=16000, frames=16000
        )
        assert derived.duration_seconds == 1.0

    def test_zero_frames_is_zero_seconds(self) -> None:
        empty = ObservedSourceMetadata(
            container="FLAC", subtype="PCM_24", channels=2, sample_rate=44100, frames=0
        )
        assert empty.duration_seconds == 0.0
        assert empty.projected_decoded_bytes == 0

    def test_an_overlong_token_is_rejected(self) -> None:
        """The longest token soundfile reports is MPEG_LAYER_III at 14 characters, so the
        bound has evidence behind it rather than being a round number."""
        with pytest.raises(SourceOutcomeError, match="over the 32 bound"):
            ObservedSourceMetadata(
                container="A" * 33,
                subtype="PCM_24",
                channels=2,
                sample_rate=44100,
                frames=10,
            )

    @pytest.mark.parametrize(
        "token", ["FLAC", "WAV", "PCM_16", "PCM_24", "PCM_32", "MPEG_LAYER_III", "VORBIS"]
    )
    def test_known_tokens_are_accepted(self, token: str) -> None:
        """Fixed examples beside the live inventory below. Without them, a library version
        that renamed or dropped a token would change what the inventory test covers without
        anything in this repository changing."""
        assert _TOKEN.fullmatch(token)
        assert len(token) <= 32

    def test_every_token_the_decoder_reports_is_accepted(self) -> None:
        """The grammar was measured against soundfile's own vocabulary, not assumed, so a
        legitimate container or subtype cannot be refused for its spelling."""
        import soundfile as sf

        for token in list(sf.available_formats()) + list(sf.available_subtypes()):
            assert _TOKEN.fullmatch(token), token
            assert len(token) <= 32, token

    @pytest.mark.parametrize(("field", "value"), [("channels", 0), ("sample_rate", 0)])
    def test_a_non_positive_count_is_rejected(self, field: str, value: int) -> None:
        kwargs = {
            "container": "FLAC",
            "subtype": "PCM_24",
            "channels": 2,
            "sample_rate": 44100,
            "frames": 10,
        }
        kwargs[field] = value
        with pytest.raises(SourceOutcomeError, match="must be at least 1"):
            ObservedSourceMetadata(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("value", ["", "   ", "flac", "PCM 24", " FLAC"])
    def test_a_malformed_container_token_is_rejected(self, value: str) -> None:
        with pytest.raises(SourceOutcomeError):
            ObservedSourceMetadata(
                container=value, subtype="PCM_24", channels=2, sample_rate=44100, frames=10
            )

    def test_the_projected_size_is_derived_not_supplied(self) -> None:
        """Two implementations cannot publish different projections for one file."""
        assert (
            "projected_decoded_bytes"
            not in inspect.signature(ObservedSourceMetadata.__init__).parameters
        )
        assert METADATA.projected_decoded_bytes == METADATA.frames * METADATA.channels * 8

    def test_the_record_cannot_take_a_bare_dictionary(self, small_identity) -> None:
        """A dict would reintroduce arbitrary keys and mutable values behind a frozen
        record, which is the defect this type replaced."""
        with pytest.raises(SourceOutcomeError, match="must be an ObservedSourceMetadata"):
            outcome(small_identity, observed_metadata={"channels": 2})  # type: ignore[arg-type]

    def test_the_record_cannot_take_a_raw_diagnostic_string(self, small_identity) -> None:
        """Native text reaches a record only through the sanitiser."""
        with pytest.raises(SourceOutcomeError, match="must be a SanitisedDiagnostic"):
            outcome(
                small_identity,
                IneligibleReason.SOURCE_DECODE_REJECTED,
                decoder_diagnostic="Error opening '/x/track.flac'",  # type: ignore[arg-type]
            )


class TestAbortsAreADifferentThing:
    """The two vocabularies are separate in v4; the types carry that separation."""

    @pytest.mark.parametrize(
        "reason",
        [r for r, p in ABORT_SOURCE_CONTEXT.items() if p is SourceContextPolicy.FORBIDDEN],
    )
    def test_a_run_global_abort_may_not_name_a_candidate(self, reason: AbortReason) -> None:
        """The fixture is decoded before any candidate is opened and an environment
        mismatch is established before the run touches audio."""
        with pytest.raises(SourceOutcomeError, match="before any candidate is opened"):
            RunAbortOutcome(reason=reason, detail="detail", logical_source_id="candidate-0001")

    @pytest.mark.parametrize(
        "reason", [r for r, p in ABORT_SOURCE_CONTEXT.items() if p is SourceContextPolicy.REQUIRED]
    )
    def test_an_artifact_specific_abort_must_name_the_candidate(self, reason: AbortReason) -> None:
        """Dropping the identifier would lose the only thing that says which artifact."""
        with pytest.raises(SourceOutcomeError, match="must name it"):
            RunAbortOutcome(reason=reason, detail="detail")

    @pytest.mark.parametrize(
        "reason", [r for r, p in ABORT_SOURCE_CONTEXT.items() if p is SourceContextPolicy.OPTIONAL]
    )
    def test_a_dual_scope_abort_may_go_either_way(self, reason: AbortReason) -> None:
        """These arise either while processing a candidate or outside one, so forcing
        either shape would encode a distinction the evidence does not support."""
        assert RunAbortOutcome(reason=reason, detail="d").logical_source_id is None
        assert RunAbortOutcome(reason=reason, detail="d", logical_source_id="c1") is not None

    def test_an_unstable_artifact_cannot_carry_a_digest(self, small_identity) -> None:
        """v4's eleventh record constraint, which lives here rather than on the ineligible
        record: a detected instability aborts, so no ineligible record for that source ever
        exists. A digest computed during that read identifies nothing."""
        with pytest.raises(SourceOutcomeError, match="must be discarded"):
            RunAbortOutcome(
                reason=AbortReason.LOCAL_ARTIFACT_UNSTABLE,
                detail="size changed during streaming",
                logical_source_id="candidate-0001",
                identity=small_identity,
            )

    def test_another_artifact_abort_may_carry_its_identity(self, small_identity) -> None:
        """A hash mismatch is precisely about a digest, so forbidding one everywhere would
        discard the evidence."""
        record = RunAbortOutcome(
            reason=AbortReason.SOURCE_HASH_MISMATCH,
            detail="digest differs from the manifest",
            logical_source_id="candidate-0001",
            identity=small_identity,
        )
        assert record.identity is small_identity

    def test_every_abort_reason_has_a_policy(self) -> None:
        assert set(ABORT_SOURCE_CONTEXT) == set(AbortReason)

    @pytest.mark.parametrize(
        ("reason", "policy"),
        [
            (AbortReason.GOLDEN_FIXTURE_FAILURE, SourceContextPolicy.FORBIDDEN),
            (AbortReason.ENVIRONMENT_MISMATCH, SourceContextPolicy.FORBIDDEN),
            (AbortReason.SOURCE_HASH_MISMATCH, SourceContextPolicy.REQUIRED),
            (AbortReason.LOCAL_ARTIFACT_UNSTABLE, SourceContextPolicy.REQUIRED),
            (AbortReason.LOCAL_ARTIFACT_NOT_REGULAR_FILE, SourceContextPolicy.REQUIRED),
            (AbortReason.ENVIRONMENT_CAPACITY_FAILURE, SourceContextPolicy.OPTIONAL),
            (AbortReason.UNEXPECTED_IO_FAILURE, SourceContextPolicy.OPTIONAL),
            (AbortReason.INTERNAL_CANONICALISATION_ERROR, SourceContextPolicy.OPTIONAL),
        ],
    )
    def test_each_policy_is_pinned_independently_of_the_map(
        self, reason: AbortReason, policy: SourceContextPolicy
    ) -> None:
        """The tests above parametrize over the map itself, so moving a reason between
        policies would silently move which test covers it and nothing would fail. That is
        the same self-referential weakness as an oracle written with the function it
        checks. These expectations are written out."""
        assert ABORT_SOURCE_CONTEXT[reason] is policy

    def test_the_two_vocabularies_do_not_overlap(self) -> None:
        assert not {str(r) for r in IneligibleReason} & {str(r) for r in AbortReason}


class TestConstructionCannotBeRoutedAround:
    """Paths that skip `__init__` entirely, which the first review named and I had not tested."""

    def test_dataclasses_replace_is_blocked(self, small_identity) -> None:
        """`replace` calls the constructor with every init field, and the derived ones are
        not init fields, so it cannot rebuild a record at all."""
        record = outcome(small_identity)
        with pytest.raises(TypeError):
            dataclasses.replace(record, logical_source_id="candidate-0002")

    def test_copy_can_only_reproduce_a_valid_record(self, small_identity) -> None:
        """`copy` skips validation, but it has nothing invalid to copy: the source record
        was validated at construction and is frozen."""
        record = outcome(small_identity)
        duplicate = copy.copy(record)
        assert duplicate == record

    def test_pickle_round_trips_without_revalidating(self, small_identity) -> None:
        """Stated rather than assumed. Unpickling restores fields without calling
        `__init__`, so pickle is not a validation boundary. That is acceptable here because
        nothing unpickles untrusted input and manifests are written as text; recording it
        stops a future caching or multiprocessing change inheriting the assumption."""
        record = outcome(small_identity)
        assert pickle.loads(pickle.dumps(record)) == record


class TestImmutability:
    """A record that can be edited after the fact is not evidence."""

    @pytest.mark.parametrize(
        ("target", "field"), [("record", "primary_reason_code"), ("identity", "sha256")]
    )
    def test_fields_cannot_be_reassigned(self, small_identity, target: str, field: str) -> None:
        record = outcome(small_identity)
        subject = record if target == "record" else record.identity
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(subject, field, None)

    def test_sequences_are_tuples(self, small_identity) -> None:
        record = outcome(
            small_identity, supplemental_reason_codes=[IneligibleReason.UNSUPPORTED_CONTAINER]
        )
        assert isinstance(record.supplemental_reason_codes, tuple)
        assert isinstance(record.unevaluated_evaluation_stages, tuple)


class TestHelpers:
    """The derivations the constructors depend on, checked directly."""

    def test_every_reason_has_a_stage(self) -> None:
        for reason in IneligibleReason:
            assert terminating_stage_for(reason) in EVALUATION_STAGE_ORDER

    def test_precedence_covers_the_vocabulary_exactly(self) -> None:
        assert sorted(REASON_PRECEDENCE) == sorted(IneligibleReason)
        assert len(REASON_PRECEDENCE) == len(set(REASON_PRECEDENCE))

    def test_precedence_never_contradicts_the_stage_order(self) -> None:
        positions = [
            EVALUATION_STAGE_ORDER.index(REASON_STAGE[reason]) for reason in REASON_PRECEDENCE
        ]
        assert positions == sorted(positions)

    def test_the_last_stage_has_no_suffix(self) -> None:
        assert unevaluated_stages_after(EVALUATION_STAGE_ORDER[-1]) == ()

    def test_the_size_derivation_is_shared_by_both_callers(self, decode) -> None:
        """One helper, so the status and method cannot be derived two ways and disagree."""
        for size in (0, decode.max_source_bytes, decode.max_identity_stream_bytes + 1):
            status, method = identity_for_size(size, decode)
            digest = DIGEST if status is IdentityStatus.COMPLETE_SHA256 else None
            identity = SourceIdentity(size, digest, decode)
            assert (identity.status, identity.method) == (status, method)
