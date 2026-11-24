"""Tests for the trials one listener completes: counts, catch rule, sides and order."""

from __future__ import annotations

from collections import Counter

import pytest

from benchmark.listening.protocol import CONDITION_NAMES, ListeningProtocol, load_listening_protocol
from benchmark.listening.trials import Trial, listener_trials

_RECORDINGS = tuple(f"rec-{index:02d}" for index in range(10))
_LISTENER_INDICES = (0, 1, 5, 17)


@pytest.fixture(scope="module")
def protocol() -> ListeningProtocol:
    """The frozen listening protocol, with its digests checked."""
    return load_listening_protocol()


def test_each_listener_has_30_scored_trials_with_every_pair_once_per_recording(
    protocol: ListeningProtocol,
) -> None:
    """Each recording appears in each of the three pairs once, in protocol order, per listener."""
    expected = Counter(
        (recording, pair) for recording in _RECORDINGS for pair in protocol.trials.pairs
    )
    for index in _LISTENER_INDICES:
        scored = [t for t in listener_trials(protocol, index, _RECORDINGS) if t.kind == "scored"]
        observed = Counter((t.recording, (t.first, t.second)) for t in scored)

        assert len(scored) == protocol.trials.scored_trials_per_listener
        assert observed == expected


def test_each_listener_has_two_catch_trials_on_distinct_recordings(
    protocol: ListeningProtocol,
) -> None:
    """Two catch trials per listener, on different recordings, each with one condition per side."""
    for index in _LISTENER_INDICES:
        catches = [t for t in listener_trials(protocol, index, _RECORDINGS) if t.kind == "catch"]

        assert len(catches) == protocol.trials.catch_trials_per_listener
        assert len({t.recording for t in catches}) == 2
        for trial in catches:
            assert trial.first == trial.second and trial.first in CONDITION_NAMES
            assert trial.side_a == trial.side_b == trial.first


def test_trial_ids_run_from_t01_to_t32_in_presentation_order(protocol: ListeningProtocol) -> None:
    """The identifiers follow the presentation order and cover all 32 trials."""
    trials = listener_trials(protocol, 3, _RECORDINGS)

    assert [t.trial_id for t in trials] == [f"t{position:02d}" for position in range(1, 33)]


def test_the_pair_is_on_both_sides_of_each_scored_trial(protocol: ListeningProtocol) -> None:
    """The two sides of a scored trial are the pair's two conditions, in either order."""
    for trial in listener_trials(protocol, 2, _RECORDINGS):
        if trial.kind == "scored":
            assert sorted([trial.side_a, trial.side_b]) == sorted([trial.first, trial.second])


def test_the_first_condition_is_on_side_a_in_some_trials_and_side_b_in_others(
    protocol: ListeningProtocol,
) -> None:
    """Over twenty listeners, a pair's first condition is on side A in some trials."""
    flags = [
        trial.first_on_a
        for index in range(20)
        for trial in listener_trials(protocol, index, _RECORDINGS)
        if trial.kind == "scored"
    ]

    assert True in flags and False in flags


def test_the_same_index_gives_the_same_trials_and_another_index_does_not(
    protocol: ListeningProtocol,
) -> None:
    """The order is a function of the listener index alone, so a repeated call is reproducible."""
    first = listener_trials(protocol, 4, _RECORDINGS)

    assert listener_trials(protocol, 4, _RECORDINGS) == first
    assert listener_trials(protocol, 5, _RECORDINGS) != first


def test_a_negative_listener_index_is_refused(protocol: ListeningProtocol) -> None:
    """A negative index has no generator, so it is refused."""
    with pytest.raises(ValueError, match="non-negative"):
        listener_trials(protocol, -1, _RECORDINGS)


def test_too_few_recordings_for_the_catch_trials_is_refused(protocol: ListeningProtocol) -> None:
    """One recording cannot supply two distinct catch recordings, so it is refused."""
    with pytest.raises(ValueError, match="at least two recordings"):
        listener_trials(protocol, 0, ("rec-00",))


def test_the_trial_side_properties_follow_first_on_a() -> None:
    """A trial's sides are its first and second conditions, swapped when the first is on side B."""
    on_a = Trial("t01", "scored", "rec", "original", "uvr", True)
    on_b = Trial("t02", "scored", "rec", "original", "uvr", False)

    assert (on_a.side_a, on_a.side_b) == ("original", "uvr")
    assert (on_b.side_a, on_b.side_b) == ("uvr", "original")
