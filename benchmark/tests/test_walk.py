"""Tests for one candidate through the walk's predicates, with fake fetch, worker and features.

The protocol is the real v13. The worker is replaced by a function that writes the canonical
bytes the candidate's samples produce, so every predicate's code can be reached by a minimal
fake. Features are fixed per candidate and keyed by content: a candidate whose features equal a
reference's scores 1.0 against it, and independent random unit features score near 0.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from walk_fakes import (
    CHAMBER_ID,
    PIANO_ID,
    PIANO_SOURCE,
    RATE,
    World,
    assert_no_worker,
    assert_rejected,
    examine,
    ineligible,
    long_tone,
    quiet_remainder_tone,
    snapshot,
    state_for,
    tone,
    training_reference,
)

from benchmark.adjudication import AdjudicationRecord
from benchmark.duplicate_screen import (
    Answer,
    Degenerate,
    RecordingIdentity,
    Reference,
    ScreenCode,
)
from benchmark.fetch import (
    ListingMismatch,
    PermanentlyUnavailable,
    RetriesExhausted,
    Staged,
)
from benchmark.protocol import Protocol, Stratum, load_protocol
from benchmark.selection import ExclusionRule, training_filename
from benchmark.source_evidence import SourceIdentity
from benchmark.source_outcome import (
    AbortReason,
    IneligibleReason,
    RunAbortOutcome,
    SourceIneligibleOutcome,
)
from benchmark.walk import (
    Admitted,
    AdmittedRule,
    AwaitingAdjudication,
    Rejected,
    RunInvalid,
    WalkState,
)


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The real v13 protocol, once per module."""
    return load_protocol()


@pytest.fixture(scope="module")
def strata(protocol: Protocol) -> dict[str, Stratum]:
    """The two v13 strata by name."""
    return {stratum.name: stratum for stratum in protocol.selection.strata}


@pytest.fixture
def world(tmp_path: Path, protocol: Protocol) -> World:
    """A fresh world with the piano candidate holding a long tone."""
    fake = World(tmp_path, protocol)
    fake.add(PIANO_ID, long_tone(), seed=1)
    return fake


class TestIdentityRulesBeforeAnyFetch:
    """Syntax, rule 1 and the cluster cap run before the fetch, in that order."""

    def test_a_malformed_identifier_is_rejected_before_any_fetch(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A dot-dot segment is malformed; the candidate is never fetched."""
        identifier = f"{PIANO_SOURCE}/../escape.flac"
        outcome = examine(
            world, protocol, strata["solo_piano"], identifier, state_for([training_reference(2)])
        )
        assert_rejected(outcome, protocol.selection.identifier_syntax_code)
        assert world.fetched == []

    def test_training_source_identity_is_rejected_before_any_fetch(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """Rule 1 needs an identifier in the training item's form, so the stratum's source is
        replaced with the training collection to reach it, as v13 matches nothing otherwise.
        """
        collection = protocol.selection.training_collection_identifier
        stratum = dataclasses.replace(strata["solo_piano"], source=collection)
        identifier = f"{collection}/Chopin/Nocturne.flac"
        flattened = training_filename(identifier, collection)
        refs = [training_reference(2, name=flattened)]
        outcome = examine(world, protocol, stratum, identifier, state_for(refs))
        assert_rejected(outcome, ExclusionRule.SOURCE_IDENTITY.value)
        assert world.fetched == []

    def test_a_full_cluster_is_rejected_before_any_fetch(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A cluster already holding its cap of admissions rejects the next candidate."""
        state = state_for([training_reference(2)])
        state.cluster_counts[(strata["solo_piano"].name, "Pianist A")] = strata[
            "solo_piano"
        ].cluster_cap
        outcome = examine(world, protocol, strata["solo_piano"], PIANO_ID, state)
        rejected = assert_rejected(outcome, protocol.selection.cluster_cap_code)
        assert rejected.evidence.cluster_admitted == strata["solo_piano"].cluster_cap
        assert world.fetched == []

    def test_a_cluster_below_its_cap_is_not_rejected(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """One admission below the cap leaves the candidate eligible for the cluster check."""
        state = state_for([training_reference(2)])
        state.cluster_counts[(strata["solo_piano"].name, "Pianist A")] = (
            strata["solo_piano"].cluster_cap - 1
        )
        assert isinstance(examine(world, protocol, strata["solo_piano"], PIANO_ID, state), Admitted)


class TestFetchPredicate:
    """The fetch's outcomes: permanent, listing mismatch, retries exhausted, or the staged file."""

    def test_a_permanent_status_excludes_the_candidate_with_its_status(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A 404 excludes under the permanent code and records the status."""
        world.fetch_results[PIANO_ID] = PermanentlyUnavailable(PIANO_ID, 404, ())
        rejected = assert_rejected(
            examine(
                world, protocol, strata["solo_piano"], PIANO_ID, state_for([training_reference(2)])
            ),
            protocol.selection.permanent_unavailability_code,
        )
        assert rejected.evidence.fetch_status == 404
        assert_no_worker(world)

    def test_a_listing_mismatch_excludes_the_candidate(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """Every attempt mismatching its listing excludes under the listing code."""
        world.fetch_results[PIANO_ID] = ListingMismatch(PIANO_ID, ())
        assert_rejected(
            examine(
                world, protocol, strata["solo_piano"], PIANO_ID, state_for([training_reference(2)])
            ),
            protocol.selection.listing_mismatch_code,
        )
        assert_no_worker(world)

    def test_exhausted_retries_make_the_run_invalid(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """Transient failures on every attempt invalidate the run, not the candidate."""
        world.fetch_results[PIANO_ID] = RetriesExhausted(PIANO_ID, ())
        state = state_for([training_reference(2)])
        before = snapshot(state)
        outcome = examine(world, protocol, strata["solo_piano"], PIANO_ID, state)
        assert isinstance(outcome, RunInvalid)
        assert outcome.code == protocol.selection.retries_exhausted_code
        assert snapshot(state) == before
        assert_no_worker(world)


class TestRawDigestRules:
    """Rules 2 on the raw digest: training first, then the earlier admissions."""

    def test_a_training_raw_digest_is_rejected_before_the_worker(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A training recording with the candidate's raw bytes excludes under the training code."""
        refs = [training_reference(2)]
        refs[0] = Reference(
            RecordingIdentity("MusopenCollectionAsFlac__Same.flac", world.raw(PIANO_ID), "1" * 64),
            refs[0].features,
        )
        assert_rejected(
            examine(world, protocol, strata["solo_piano"], PIANO_ID, state_for(refs)),
            ExclusionRule.RAW_SHA256.value,
        )
        assert_no_worker(world)

    def test_an_admitted_raw_digest_is_rejected_before_the_worker(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A candidate with the raw bytes of an earlier admission excludes under the admitted code."""
        state = state_for([training_reference(2)])
        state.admitted_raw.add(world.raw(PIANO_ID))
        assert_rejected(
            examine(world, protocol, strata["solo_piano"], PIANO_ID, state),
            AdmittedRule.RAW_SHA256.value,
        )
        assert_no_worker(world)

    def test_training_takes_precedence_over_an_admitted_raw_digest(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """When both sets hold the digest, the training code is recorded, as the protocol orders."""
        refs = [training_reference(2)]
        refs[0] = Reference(
            RecordingIdentity("MusopenCollectionAsFlac__Same.flac", world.raw(PIANO_ID), "2" * 64),
            refs[0].features,
        )
        state = state_for(refs)
        state.admitted_raw.add(world.raw(PIANO_ID))
        assert_rejected(
            examine(world, protocol, strata["solo_piano"], PIANO_ID, state),
            ExclusionRule.RAW_SHA256.value,
        )

    def test_an_oversize_prefix_is_worked_with_no_expected_digest(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """An oversize staged prefix has no digest, so the worker gets none and the size decides."""
        bound = protocol.canonicalisation.decode.max_identity_stream_bytes
        decode = protocol.canonicalisation.decode
        world.fetch_results[PIANO_ID] = Staged(
            PIANO_ID, world.root / "prefix.flac", bound + 1, None, True, (), (), False
        )
        (world.root / "prefix.flac").write_bytes(b"x" * 8)
        identity = SourceIdentity(bound + 1, None, decode)
        world.aborts[PIANO_ID] = SourceIneligibleOutcome(
            logical_source_id=PIANO_ID,
            primary_reason_code=IneligibleReason.SOURCE_EXCEEDS_MAX_BYTES,
            identity=identity,
            protocol_sha256=protocol.source_sha256,
        )
        assert_rejected(
            examine(
                world, protocol, strata["solo_piano"], PIANO_ID, state_for([training_reference(2)])
            ),
            IneligibleReason.SOURCE_EXCEEDS_MAX_BYTES.value,
        )
        assert world.canonicalised == [(PIANO_ID, None)]


class TestSourceOutcomeStage:
    """The worker's verdict or abort, and the canonical file it leaves."""

    def test_a_source_verdict_is_rejected_under_its_own_code(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A worker verdict excludes the candidate under the reason code the worker recorded."""
        reason = IneligibleReason.SOURCE_EXCEEDS_MAX_BYTES
        world.aborts[PIANO_ID] = ineligible(protocol, PIANO_ID, reason)
        rejected = assert_rejected(
            examine(
                world, protocol, strata["solo_piano"], PIANO_ID, state_for([training_reference(2)])
            ),
            reason.value,
        )
        assert rejected.evidence.screen is None

    def test_an_abort_makes_the_run_invalid_under_its_reason(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A harness abort invalidates the run and names the reason it recorded."""
        abort = RunAbortOutcome(
            reason=AbortReason.ENVIRONMENT_CAPACITY_FAILURE, detail="worker was killed"
        )
        world.aborts[PIANO_ID] = abort
        state = state_for([training_reference(2)])
        before = snapshot(state)
        outcome = examine(world, protocol, strata["solo_piano"], PIANO_ID, state)
        assert isinstance(outcome, RunInvalid)
        assert outcome.code == AbortReason.ENVIRONMENT_CAPACITY_FAILURE.value
        assert outcome.detail == "worker was killed"
        assert snapshot(state) == before


class TestDecodedDigestRules:
    """Rules 3 on the decoded-PCM digest: training first, then the earlier admissions."""

    def test_a_training_decoded_digest_is_rejected(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A training recording with the same decoded PCM excludes under the training code."""
        refs = [training_reference(2)]
        refs[0] = Reference(
            RecordingIdentity(
                "MusopenCollectionAsFlac__Lossless.flac", "3" * 64, world.decoded(PIANO_ID)
            ),
            refs[0].features,
        )
        assert_rejected(
            examine(world, protocol, strata["solo_piano"], PIANO_ID, state_for(refs)),
            ExclusionRule.DECODED_PCM_SHA256.value,
        )

    def test_an_admitted_decoded_digest_is_rejected(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A re-encode of an earlier admission, with the same decoded PCM, excludes under the admitted code."""
        state = state_for([training_reference(2)])
        state.admitted_decoded.add(world.decoded(PIANO_ID))
        assert_rejected(
            examine(world, protocol, strata["solo_piano"], PIANO_ID, state),
            AdmittedRule.DECODED_PCM_SHA256.value,
        )


class TestEligibilityStage:
    """The recording floors, then the frame verdicts, after the decoded digest rules."""

    def test_a_quiet_short_recording_is_rejected_by_the_duration_floor_first(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """Both floors fail, so the duration code is the primary reason, as v6 orders them."""
        world.add(PIANO_ID, np.zeros(RATE, dtype=np.float64), seed=1)
        assert_rejected(
            examine(
                world, protocol, strata["solo_piano"], PIANO_ID, state_for([training_reference(2)])
            ),
            "RECORDING_BELOW_MIN_DURATION",
        )

    def test_a_long_silent_recording_is_rejected_by_the_loudness_floor(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """Long enough but silent: the loudness floor is the reason."""
        world.add(PIANO_ID, np.zeros(61 * RATE, dtype=np.float64), seed=1)
        assert_rejected(
            examine(
                world, protocol, strata["solo_piano"], PIANO_ID, state_for([training_reference(2)])
            ),
            "RECORDING_BELOW_RMS_FLOOR",
        )

    def test_a_recording_with_no_eligible_frame_is_rejected_after_both_floors_pass(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """Both recording floors pass on the tail, but every complete frame is silent."""
        world.add(PIANO_ID, quiet_remainder_tone(protocol), seed=1)
        rejected = assert_rejected(
            examine(
                world, protocol, strata["solo_piano"], PIANO_ID, state_for([training_reference(2)])
            ),
            "RECORDING_WITHOUT_ELIGIBLE_FRAMES",
        )
        assert rejected.evidence.eligible_frames == 0
        assert rejected.evidence.recording_rms_dbfs is not None


class TestDuplicateScreen:
    """The screen against training and the earlier admissions, and the adjudication decision."""

    def test_an_unflagged_candidate_is_admitted_with_its_maxima(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """No reference reaches the threshold, so the candidate is admitted with its maxima."""
        outcome = examine(
            world, protocol, strata["solo_piano"], PIANO_ID, state_for([training_reference(2)])
        )
        assert isinstance(outcome, Admitted)
        screen = outcome.evidence.screen
        assert screen is not None
        assert screen.flagged == ()
        assert screen.max_training is not None and screen.max_training < 0.3
        assert screen.max_admitted is None
        assert outcome.admission.canonical.path.is_file()
        assert outcome.admission.raw_sha256 == world.raw(PIANO_ID)
        assert world.canonicalised == [(PIANO_ID, world.raw(PIANO_ID))]

    def test_a_flag_against_training_without_an_answer_awaits_adjudication(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A candidate that matches a training recording halts the walk when nothing answers it."""
        training = Reference(
            RecordingIdentity("MusopenCollectionAsFlac__Same.flac", "4" * 64, "5" * 64),
            world.features[PIANO_ID],
        )
        outcome = examine(world, protocol, strata["solo_piano"], PIANO_ID, state_for([training]))
        assert isinstance(outcome, AwaitingAdjudication)
        assert [flag.identity for flag in outcome.flagged] == [training.identity]
        assert outcome.evidence.adjudication is None

    def test_a_flag_answered_as_a_different_performance_admits_with_the_record(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A recorded different-performance answer for every flag admits, and the record is kept."""
        training = Reference(
            RecordingIdentity("MusopenCollectionAsFlac__Same.flac", "4" * 64, "5" * 64),
            world.features[PIANO_ID],
        )
        record = AdjudicationRecord(
            candidate=world.identity(PIANO_ID),
            answers={training.identity: Answer.DIFFERENT_PERFORMANCE},
            evidence={training.identity: "Another pianist, different movement."},
        )
        outcome = examine(
            world, protocol, strata["solo_piano"], PIANO_ID, state_for([training], (record,))
        )
        assert isinstance(outcome, Admitted)
        assert outcome.evidence.adjudication == record

    @pytest.mark.parametrize(
        ("answer", "code"),
        [
            (Answer.SAME_PERFORMANCE, ScreenCode.SAME_PERFORMANCE_TRAINING),
            (Answer.UNCERTAIN, ScreenCode.UNCERTAIN),
        ],
        ids=["same performance", "uncertain"],
    )
    def test_a_flag_against_training_answered_otherwise_excludes_under_its_code(
        self,
        world: World,
        protocol: Protocol,
        strata: dict[str, Stratum],
        answer: Answer,
        code: ScreenCode,
    ) -> None:
        """A same-performance answer gives the training code; an uncertain one, the uncertain code."""
        training = Reference(
            RecordingIdentity("MusopenCollectionAsFlac__Same.flac", "4" * 64, "5" * 64),
            world.features[PIANO_ID],
        )
        record = AdjudicationRecord(
            candidate=world.identity(PIANO_ID),
            answers={training.identity: answer},
            evidence={training.identity: "File names name one session."},
        )
        assert_rejected(
            examine(
                world, protocol, strata["solo_piano"], PIANO_ID, state_for([training], (record,))
            ),
            code.value,
        )

    def test_an_uncertain_screen_excludes_without_reading_any_record(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """Degenerate candidate features exclude as uncertain even with a record that would admit."""
        world.degenerate[PIANO_ID] = Degenerate(zero_variance_coefficients=1, zero_norm_frames=0)
        training = Reference(
            RecordingIdentity("MusopenCollectionAsFlac__Same.flac", "4" * 64, "5" * 64),
            world.features[PIANO_ID],
        )
        record = AdjudicationRecord(
            candidate=world.identity(PIANO_ID),
            answers={training.identity: Answer.DIFFERENT_PERFORMANCE},
            evidence={training.identity: "Another performer."},
        )
        rejected = assert_rejected(
            examine(
                world, protocol, strata["solo_piano"], PIANO_ID, state_for([training], (record,))
            ),
            ScreenCode.UNCERTAIN.value,
        )
        assert rejected.evidence.adjudication is None

    def test_a_record_for_other_bytes_does_not_apply(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A record naming the candidate by a different decoded digest leaves the flag awaiting."""
        training = Reference(
            RecordingIdentity("MusopenCollectionAsFlac__Same.flac", "4" * 64, "5" * 64),
            world.features[PIANO_ID],
        )
        wrong = RecordingIdentity(PIANO_ID, world.raw(PIANO_ID), "6" * 64)
        record = AdjudicationRecord(
            candidate=wrong,
            answers={training.identity: Answer.DIFFERENT_PERFORMANCE},
            evidence={training.identity: "Recorded for other bytes."},
        )
        outcome = examine(
            world, protocol, strata["solo_piano"], PIANO_ID, state_for([training], (record,))
        )
        assert isinstance(outcome, AwaitingAdjudication)
        assert outcome.evidence.adjudication is None

    def test_a_record_that_omits_a_flagged_reference_does_not_apply(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A record for the right bytes that answers a different reference leaves the flag awaiting."""
        flagged = Reference(
            RecordingIdentity("MusopenCollectionAsFlac__Same.flac", "4" * 64, "5" * 64),
            world.features[PIANO_ID],
        )
        other = RecordingIdentity("MusopenCollectionAsFlac__Other.flac", "7" * 64, "8" * 64)
        record = AdjudicationRecord(
            candidate=world.identity(PIANO_ID),
            answers={other: Answer.DIFFERENT_PERFORMANCE},
            evidence={other: "Answers a reference that was not flagged."},
        )
        outcome = examine(
            world, protocol, strata["solo_piano"], PIANO_ID, state_for([flagged], (record,))
        )
        assert isinstance(outcome, AwaitingAdjudication)
        assert outcome.evidence.adjudication is None


class TestAcrossStrata:
    """An admission in one stratum is a reference and a digest for the other stratum's walk."""

    @staticmethod
    def admit_piano(world: World, protocol: Protocol, strata: dict[str, Stratum]) -> WalkState:
        """Admit the piano candidate and record the admission, as the driver does."""
        state = state_for([training_reference(2)])
        world.add(CHAMBER_ID, tone(61.0, amplitude=0.2), seed=9)
        outcome = examine(world, protocol, strata["solo_piano"], PIANO_ID, state)
        assert isinstance(outcome, Admitted)
        state.record_admission(outcome.admission)
        return state

    def test_a_chamber_candidate_matching_a_piano_admission_awaits_adjudication(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """Its features equal the admitted piano recording's, so the admitted reference flags it."""
        state = self.admit_piano(world, protocol, strata)
        world.features[CHAMBER_ID] = world.features[PIANO_ID]
        outcome = examine(world, protocol, strata["chamber"], CHAMBER_ID, state, cluster="Ensemble")
        assert isinstance(outcome, AwaitingAdjudication)
        assert [flag.identity.identifier for flag in outcome.flagged] == [PIANO_ID]
        assert outcome.flagged[0].group.value == "admitted"

    def test_a_piano_admission_answered_as_the_same_performance_excludes_under_the_admitted_code(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A same-performance answer against an earlier admission gives the admitted code."""
        state = self.admit_piano(world, protocol, strata)
        world.features[CHAMBER_ID] = world.features[PIANO_ID]
        piano = world.identity(PIANO_ID)
        state.adjudications = (
            AdjudicationRecord(
                candidate=world.identity(CHAMBER_ID),
                answers={piano: Answer.SAME_PERFORMANCE},
                evidence={piano: "Same session, named in both files."},
            ),
        )
        assert_rejected(
            examine(world, protocol, strata["chamber"], CHAMBER_ID, state, cluster="Ensemble"),
            ScreenCode.SAME_PERFORMANCE_ADMITTED.value,
        )

    def test_a_different_performance_answer_admits_a_chamber_candidate(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """The same composition by another performer is admitted when the answer says so."""
        state = self.admit_piano(world, protocol, strata)
        world.features[CHAMBER_ID] = world.features[PIANO_ID]
        piano = world.identity(PIANO_ID)
        state.adjudications = (
            AdjudicationRecord(
                candidate=world.identity(CHAMBER_ID),
                answers={piano: Answer.DIFFERENT_PERFORMANCE},
                evidence={piano: "Different performer and session."},
            ),
        )
        outcome = examine(world, protocol, strata["chamber"], CHAMBER_ID, state, cluster="Ensemble")
        assert isinstance(outcome, Admitted)

    def test_a_piano_raw_digest_rejects_the_same_bytes_in_chamber(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """A chamber file whose raw bytes equal the admitted piano file's is rejected as admitted."""
        state = self.admit_piano(world, protocol, strata)
        world.fetch_results[CHAMBER_ID] = Staged(
            CHAMBER_ID, world.root / "x.flac", 1, world.raw(PIANO_ID), False, (), (), False
        )
        assert_rejected(
            examine(world, protocol, strata["chamber"], CHAMBER_ID, state, cluster="Ensemble"),
            AdmittedRule.RAW_SHA256.value,
        )


class TestStateAndEvidence:
    """Failure paths leave the walk state unchanged, and the admitted state is what later reads."""

    @pytest.mark.parametrize("scenario", ["rejected", "awaiting", "run_invalid"])
    def test_a_non_admitted_outcome_changes_no_state(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum], scenario: str
    ) -> None:
        """Rejection, a halt and a run invalid each leave digests, references and counts as they were."""
        state = state_for([training_reference(2)])
        if scenario == "rejected":
            state.admitted_raw.add(world.raw(PIANO_ID))
        elif scenario == "awaiting":
            training = Reference(
                RecordingIdentity("MusopenCollectionAsFlac__Same.flac", "4" * 64, "5" * 64),
                world.features[PIANO_ID],
            )
            state = state_for([training])
        else:
            world.fetch_results[PIANO_ID] = RetriesExhausted(PIANO_ID, ())
        before = snapshot(state)
        outcome = examine(world, protocol, strata["solo_piano"], PIANO_ID, state)
        assert not isinstance(outcome, Admitted)
        assert snapshot(state) == before

    def test_an_admission_updates_the_digests_references_and_cluster_count(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """After recording an admission, a later candidate sees its digests and cluster count."""
        state = state_for([training_reference(2)])
        outcome = examine(world, protocol, strata["solo_piano"], PIANO_ID, state)
        assert isinstance(outcome, Admitted)
        state.record_admission(outcome.admission)
        assert state.admitted_raw == {world.raw(PIANO_ID)}
        assert state.admitted_decoded == {world.decoded(PIANO_ID)}
        assert [r.identity for r in state.admitted_references] == [world.identity(PIANO_ID)]
        assert state.cluster_counts[(strata["solo_piano"].name, "Pianist A")] == 1

    def test_evidence_serialises_to_json_without_non_finite_numbers(
        self, world: World, protocol: Protocol, strata: dict[str, Stratum]
    ) -> None:
        """Silent recordings have an RMS of negative infinity, which the audit trail writes as null."""
        import json

        world.add(PIANO_ID, np.zeros(61 * RATE, dtype=np.float64), seed=1)
        outcome = examine(
            world, protocol, strata["solo_piano"], PIANO_ID, state_for([training_reference(2)])
        )
        assert isinstance(outcome, Rejected)
        encoded = json.dumps(outcome.evidence.to_json(), allow_nan=False)
        assert json.loads(encoded)["recording_rms_dbfs"] is None

    def test_every_code_the_walk_emits_is_a_protocol_code(self, protocol: Protocol) -> None:
        """Each rule and screen code the walk can emit appears in the protocol's rejection list."""
        emitted = {
            ExclusionRule.SOURCE_IDENTITY.value,
            ExclusionRule.RAW_SHA256.value,
            ExclusionRule.DECODED_PCM_SHA256.value,
            AdmittedRule.RAW_SHA256.value,
            AdmittedRule.DECODED_PCM_SHA256.value,
            ScreenCode.SAME_PERFORMANCE_TRAINING.value,
            ScreenCode.SAME_PERFORMANCE_ADMITTED.value,
            ScreenCode.UNCERTAIN.value,
        }
        assert emitted <= set(protocol.selection.rejection_codes)
