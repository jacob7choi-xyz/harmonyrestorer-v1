"""Tests for conditioning, whose whole job is one distinction.

The tests are organised around attribution rather than around code paths, because the
defect this layer prevents is a harness failure being counted as a system result. So the
central assertions are about which channel an outcome arrives through: a return value is
an observation about the item, an exception is the evaluator declining to make one.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark import conditioning as c
from benchmark.protocol import load_protocol

CONFIG = load_protocol().conditioning
TOLERANCE = CONFIG.max_length_difference_samples
UNIT = 32000


def signal(n: int, seed: int = 0) -> np.ndarray:
    """A deterministic non-constant waveform of the requested length."""
    return np.random.default_rng(seed).standard_normal(n) * 0.2


class TestLengthTolerance:
    """The boundary is exact and applies to either side being longer."""

    @pytest.mark.parametrize(
        ("reference_length", "estimate_length"),
        [
            (UNIT, UNIT),  # identical, the ordinary case
            (UNIT, UNIT - TOLERANCE),  # estimate short by exactly the tolerance
            (UNIT - TOLERANCE, UNIT),  # reference short by exactly the tolerance
            (UNIT, UNIT - 128),  # the observed UVR resampling deficit
            (UNIT - 128, UNIT),  # and its mirror, so a sign error cannot survive
        ],
    )
    def test_differences_within_tolerance_are_conditioned(
        self, reference_length: int, estimate_length: int
    ) -> None:
        outcome = c.condition_pair(signal(reference_length), signal(estimate_length), CONFIG)
        assert isinstance(outcome, c.ConditionedPair)
        assert (
            outcome.reference.size
            == outcome.estimate.size
            == min(reference_length, estimate_length)
        )

    @pytest.mark.parametrize(
        ("reference_length", "estimate_length"),
        [
            (UNIT, UNIT - TOLERANCE - 1),  # one sample past the bound
            (UNIT - TOLERANCE - 1, UNIT),  # and past it the other way
            (UNIT, UNIT - 500),
            (UNIT, UNIT + 500),
        ],
    )
    def test_differences_beyond_tolerance_are_invalid_for_the_system(
        self, reference_length: int, estimate_length: int
    ) -> None:
        """Truncating further would score the system on a tail it never produced, so the
        item becomes a system failure rather than a quietly shortened comparison."""
        outcome = c.condition_pair(signal(reference_length), signal(estimate_length), CONFIG)
        assert isinstance(outcome, c.InvalidSystemOutput)
        assert outcome.reference_minus_estimate_samples == reference_length - estimate_length
        assert str(TOLERANCE) in outcome.reason

    def test_the_boundary_is_inclusive(self) -> None:
        """160 conditions and 161 does not. Stated as its own test because an off-by-one
        here silently changes which items every system is scored on."""
        assert isinstance(
            c.condition_pair(signal(UNIT), signal(UNIT - TOLERANCE), CONFIG), c.ConditionedPair
        )
        assert isinstance(
            c.condition_pair(signal(UNIT), signal(UNIT - TOLERANCE - 1), CONFIG),
            c.InvalidSystemOutput,
        )


class TestDiscardAccounting:
    """Which side was trimmed, not just how much."""

    def test_a_short_estimate_discards_from_the_reference(self) -> None:
        outcome = c.condition_pair(signal(UNIT), signal(UNIT - 128), CONFIG)
        assert isinstance(outcome, c.ConditionedPair)
        assert outcome.reference_samples_discarded == 128
        assert outcome.estimate_samples_discarded == 0

    def test_a_long_estimate_discards_from_the_estimate(self) -> None:
        """The mirror case. A single combined count could not distinguish the two, which
        is why both are recorded even though one is always zero."""
        outcome = c.condition_pair(signal(UNIT), signal(UNIT + 128), CONFIG)
        assert isinstance(outcome, c.ConditionedPair)
        assert outcome.reference_samples_discarded == 0
        assert outcome.estimate_samples_discarded == 128

    def test_equal_lengths_discard_nothing(self) -> None:
        outcome = c.condition_pair(signal(UNIT), signal(UNIT), CONFIG)
        assert isinstance(outcome, c.ConditionedPair)
        assert outcome.reference_samples_discarded == 0
        assert outcome.estimate_samples_discarded == 0


class TestNoAlignmentSearch:
    """Front-aligned truncation is the entire policy."""

    def test_an_obvious_one_sample_offset_is_not_discovered(self) -> None:
        """The estimate is the reference delayed by one sample, which any correlation
        search would find immediately. Conditioning must not: it compares at the offset
        the output arrived with, so the pair stays misaligned and the metrics punish it.
        An alignment fix is indistinguishable from shifting outputs until scores improve.
        """
        reference = signal(UNIT)
        delayed = np.concatenate([[0.0], reference[:-1]])

        outcome = c.condition_pair(reference, delayed, CONFIG)
        assert isinstance(outcome, c.ConditionedPair)
        assert outcome.estimate[0] == 0.0
        assert not np.allclose(outcome.reference, outcome.estimate)

    def test_truncation_keeps_the_head_not_the_tail(self) -> None:
        """Front-aligned means the surviving samples are the first ones. Keeping the tail
        instead would be an undeclared alignment choice."""
        reference = np.arange(10, dtype=np.float64)
        outcome = c.condition_pair(reference, np.arange(8, dtype=np.float64), CONFIG)
        assert isinstance(outcome, c.ConditionedPair)
        assert np.array_equal(outcome.reference, np.arange(8, dtype=np.float64))

    def test_the_api_offers_no_way_to_request_a_search(self) -> None:
        """Absence of the behaviour is weaker than absence of the parameter: a caller
        cannot ask for something that has no argument."""
        import inspect

        parameters = set(inspect.signature(c.condition_pair).parameters)
        assert parameters == {"reference", "estimate", "config"}

    def test_a_config_requesting_a_search_is_refused(self) -> None:
        """Unreachable while the loader pins alignment_search false, and refused rather
        than ignored, because silently not performing a requested search would leave the
        run believing it had one."""
        import dataclasses

        searching = dataclasses.replace(CONFIG, alignment_search=True)
        with pytest.raises(c.ConditioningError, match="forbids one"):
            c.condition_pair(signal(UNIT), signal(UNIT), searching)


class TestAttribution:
    """A harness defect must never arrive as a system observation."""

    @pytest.mark.parametrize(
        ("reference", "expected"),
        [
            (np.zeros((2, UNIT)), "must be 1-D"),
            (np.array([]), "is empty"),
            (np.array([1.0, np.nan]), "non-finite"),
            ("not audio", "not numeric"),
        ],
    )
    def test_a_malformed_reference_raises_rather_than_returning(
        self, reference: np.ndarray, expected: str
    ) -> None:
        """The benchmark produced the reference, so no system is answerable for it. It
        raises so a caller looping over returned outcomes cannot fold it into coverage.
        """
        with pytest.raises(c.ConditioningError, match=expected):
            c.condition_pair(reference, signal(2), CONFIG)

    @pytest.mark.parametrize(
        ("estimate", "expected"),
        [
            (np.zeros((2, UNIT)), "must be 1-D"),
            (np.array([]), "is empty"),
            (np.full(UNIT, np.nan), "non-finite"),
            ("not audio", "not numeric"),
        ],
    )
    def test_a_malformed_estimate_returns_rather_than_raising(
        self, estimate: np.ndarray, expected: str
    ) -> None:
        """The system produced the artifact and the harness holds it, so the defect is
        attributable and is a result about that system."""
        outcome = c.condition_pair(signal(UNIT), estimate, CONFIG)
        assert isinstance(outcome, c.InvalidSystemOutput)
        assert expected in outcome.reason

    def test_the_two_outcomes_are_distinct_types_not_a_status_field(self) -> None:
        """Structural, so a caller cannot read the wrong branch. A status string would
        let `status == "invalid"` cover both a system failure and a harness failure."""
        conditioned = c.condition_pair(signal(UNIT), signal(UNIT), CONFIG)
        invalid = c.condition_pair(signal(UNIT), signal(UNIT - 500), CONFIG)
        assert type(conditioned) is not type(invalid)
        assert not isinstance(invalid, c.ConditionedPair)
        assert not isinstance(conditioned, c.InvalidSystemOutput)

    def test_a_harness_failure_is_not_in_the_outcome_union(self) -> None:
        """The union a caller aggregates over cannot contain a harness failure, which is
        what makes the separation more than a naming convention."""
        assert c.ConditioningOutcome == c.ConditionedPair | c.InvalidSystemOutput
        assert not issubclass(c.ConditioningError, c.ConditionedPair | c.InvalidSystemOutput)


class TestResultImmutability:
    """An outcome that can be edited after the fact is not a record.

    `frozen=True` alone does not achieve this. It stops a field being rebound and says
    nothing about the arrays a field points at, so the first version of this class passed
    while the result both aliased the caller's buffers and accepted direct writes.
    """

    def test_fields_cannot_be_reassigned(self) -> None:
        import dataclasses

        outcome = c.condition_pair(signal(UNIT), signal(UNIT), CONFIG)
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.reference_samples_discarded = 99  # type: ignore[misc]

    def test_an_invalid_outcome_cannot_be_reassigned(self) -> None:
        import dataclasses

        outcome = c.condition_pair(signal(UNIT), signal(UNIT - 500), CONFIG)
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.reason = "looks fine actually"  # type: ignore[misc]

    def test_the_result_does_not_alias_the_caller(self) -> None:
        """Slicing returns a view, so without copying, a later write by the caller would
        rewrite a result already recorded."""
        reference = signal(UNIT)
        estimate = signal(UNIT - 128, seed=1)
        outcome = c.condition_pair(reference, estimate, CONFIG)
        assert isinstance(outcome, c.ConditionedPair)
        recorded = float(outcome.reference[0])

        reference[0] = 999.0
        estimate[0] = -999.0
        assert float(outcome.reference[0]) == recorded
        assert float(outcome.estimate[0]) != -999.0

    def test_the_result_arrays_reject_direct_writes(self) -> None:
        outcome = c.condition_pair(signal(UNIT), signal(UNIT), CONFIG)
        assert isinstance(outcome, c.ConditionedPair)
        for array in (outcome.reference, outcome.estimate):
            assert not array.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                array[0] = 1.0


class TestTheHandoffToMetrics:
    """Structural comparability is the whole promise, and the tests say so."""

    def test_a_structurally_conditioned_pair_is_scoreable(self) -> None:
        """The ordinary case. Shapes and dtypes line up, so no metric rejects the pair for
        a structural reason, which is the part conditioning is responsible for."""
        from benchmark import metrics

        protocol = load_protocol()
        outcome = c.condition_pair(signal(UNIT), signal(UNIT - 128, seed=1), CONFIG)
        assert isinstance(outcome, c.ConditionedPair)
        result = metrics.reconstruction_snr(
            outcome.reference, outcome.estimate, protocol.metrics.reconstruction_snr
        )
        assert np.isfinite(result.value)

    def test_conditioning_does_not_promise_metric_domain_validity(self) -> None:
        """Deliberate, and worth pinning so nobody later reads ConditionedPair as a
        guarantee of scoreability.

        Scoreability is not one property. A constant reference is scored by reconstruction
        SNR and rejected by SI-SNR under protocol v3 onward, so conditioning could not promise it
        without knowing which metric is being computed. Those domains belong to the metrics
        and duplicating them here would put one rule in two places. The obligation this
        creates is upstream: when a metric rejects a pair that conditioned cleanly, the
        orchestration layer attributes it, and a reference-side domain failure is a dataset
        or harness failure rather than a failure of whichever system was paired with it.
        """
        from benchmark import metrics

        protocol = load_protocol()
        constant = np.full(UNIT, 0.1)
        outcome = c.condition_pair(constant, constant.copy(), CONFIG)
        assert isinstance(outcome, c.ConditionedPair)

        scored = metrics.reconstruction_snr(
            outcome.reference, outcome.estimate, protocol.metrics.reconstruction_snr
        )
        assert np.isfinite(scored.value)
        with pytest.raises(metrics.MetricInputError):
            metrics.si_snr(outcome.reference, outcome.estimate, protocol.metrics.si_snr)
