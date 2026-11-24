"""Tests for the listening selection walk and the clip-window rule. Pure: no network, no audio."""

from __future__ import annotations

import pytest

from benchmark.listening.manifest import UniverseItem
from benchmark.listening.metadata import split_performer_names
from benchmark.listening.protocol import load_listening_protocol
from benchmark.listening.selection import (
    SelectionError,
    SkipReason,
    Walk,
    WindowRule,
    fail_choice,
    run_walk,
    window_bounds,
    window_rule_for,
    window_start,
)
from benchmark.selection import walk_order

_SEED = 20261124
_RULE = WindowRule(seed=_SEED, margin_s=15, clip_s=8)


def _item(
    identifier: str, performers: tuple[str, ...], length_s: float | None = 120.0
) -> UniverseItem:
    """A synthetic universe item.

    Args:
        identifier: The identifier.
        performers: Performer names.
        length_s: The listed duration, or None.

    Returns:
        The item.
    """
    return UniverseItem(
        identifier=identifier,
        title=identifier,
        year=1910,
        performers=performers,
        instrumentation="Violin solo",
        flac_name=f"{identifier}.flac",
        flac_size=1,
        flac_md5="a" * 32,
        flac_sha1="b" * 40,
        length_s=length_s,
    )


def _names(state: Walk) -> list[str]:
    """The identifiers of the walk's current choices, in the order taken.

    Args:
        state: A walk state.

    Returns:
        The identifiers.
    """
    return [choice.identifier for choice in state.selected]


def test_window_bounds_are_inclusive_and_use_whole_seconds() -> None:
    """The range is from the margin to floor(duration) minus margin and clip, both inclusive."""
    assert window_bounds(45.9, 15, 8) == (15, 22)
    assert window_bounds(38.0, 15, 8) == (15, 15)


def test_a_recording_under_38_seconds_has_no_window() -> None:
    """Below 38 whole seconds no start fits, so the recording gets no window."""
    assert window_bounds(37.99, 15, 8) is None


def test_window_starts_fall_inside_their_bounds_for_many_identifiers() -> None:
    """Every start drawn lies within the bounds, and the draws are not all the same."""
    bounds = window_bounds(100.0, 15, 8)
    assert bounds is not None
    starts = {window_start(f"78_item-{i}", _RULE, bounds) for i in range(200)}
    assert starts <= set(range(bounds[0], bounds[1] + 1))
    assert len(starts) > 1


def test_a_degenerate_range_returns_its_only_start() -> None:
    """When low equals high there is one possible start, and the draw returns it."""
    assert window_start("78_any", _RULE, (15, 15)) == 15


def test_window_start_is_deterministic_and_depends_on_the_seed() -> None:
    """The same identifier and seed give the same start, and changing the seed changes draws."""
    bounds = (15, 60)
    assert window_start("78_fixed", _RULE, bounds) == window_start("78_fixed", _RULE, bounds)
    starts = {
        window_start("78_fixed", WindowRule(seed=seed, margin_s=15, clip_s=8), bounds)
        for seed in range(_SEED, _SEED + 20)
    }
    assert len(starts) > 1


def test_the_walk_takes_exactly_the_sample_size_and_is_deterministic() -> None:
    """The walk returns sample_size choices from the universe, the same way each time."""
    universe = [_item(f"78_{i:02d}", (f"Performer {i}",)) for i in range(12)]
    first = run_walk(universe, _RULE, sample_size=4)
    second = run_walk(universe, _RULE, sample_size=4)
    assert first.complete
    assert len(first.selected) == 4
    assert _names(first) == _names(second)
    assert set(_names(first)) <= {item.identifier for item in universe}


def test_a_repeated_performer_is_skipped_and_the_walk_stays_incomplete() -> None:
    """With one performer shared by two items, the second is skipped and the sample is short."""
    universe = [
        _item("78_a", ("Fritz Kreisler",)),
        _item("78_b", ("  fritz   KREISLER ",)),
        _item("78_c", ("Jan Kubelik",)),
    ]
    result = run_walk(universe, _RULE, sample_size=3)
    assert [skip.reason for skip in result.skipped] == [SkipReason.PERFORMER_REPEAT]
    assert len(result.selected) == 2
    assert not result.complete


def test_a_recording_too_short_for_a_window_is_passed_over_as_too_short() -> None:
    """An item with no valid window is skipped with the too_short reason, not taken."""
    universe = [_item("78_short", ("Ada",), length_s=30.0), _item("78_long", ("Ben",))]
    result = run_walk(universe, _RULE, sample_size=2)
    assert _names(result) == ["78_long"]
    assert [skip.identifier for skip in result.skipped] == ["78_short"]
    assert [skip.reason for skip in result.skipped] == [SkipReason.TOO_SHORT]
    assert not result.complete


def test_an_item_with_no_listed_duration_is_passed_over() -> None:
    """Without a duration no window can be drawn, so the item is skipped with no_duration."""
    universe = [_item("78_nodur", ("Ada",), length_s=None), _item("78_ok", ("Ben",))]
    result = run_walk(universe, _RULE, sample_size=1)
    assert _names(result) == ["78_ok"]
    assert result.skipped[0].reason is SkipReason.NO_DURATION


def test_a_short_universe_leaves_the_walk_incomplete_and_says_so() -> None:
    """When the universe runs out the walk is not complete, and it does not pad the sample."""
    universe = [_item("78_only", ("Ada",))]
    result = run_walk(universe, _RULE, sample_size=3)
    assert not result.complete
    assert _names(result) == ["78_only"]


def test_a_failed_choice_is_replaced_and_the_failure_is_recorded() -> None:
    """After a failure the failed item is gone, the walk refills, and the reason is kept."""
    universe = [
        _item("78_a", ("Ada",)),
        _item("78_b", ("Ben",)),
        _item("78_c", ("Cy",)),
        _item("78_d", ("Dee",)),
    ]
    initial = run_walk(universe, _RULE, sample_size=2)
    failed = initial.selected[0].identifier
    replaced = fail_choice(initial, failed, "md5_mismatch", universe, _RULE)
    assert failed not in _names(replaced)
    assert replaced.complete
    assert [(r.identifier, r.reason) for r in replaced.replaced] == [(failed, "md5_mismatch")]


def test_a_replacement_continues_after_the_last_item_considered() -> None:
    """A removed choice frees its performer, but the walk does not return to an earlier skip.

    The order is taken from the real walk order, so the case is built on the seeded permutation:
    order[0] and order[1] share a performer, order[2] and order[3] do not. The walk takes order[0]
    and skips order[1], then takes order[2]. When order[0] fails, the walk continues at order[3].
    order[1] stays out, although its performer is free again.
    """
    ids = [f"78_{i}" for i in range(4)]
    order = walk_order(ids, _SEED)
    performers = {
        order[0]: ("Ada",),
        order[1]: ("Ada",),
        order[2]: ("Cy",),
        order[3]: ("Dee",),
    }
    universe = [_item(identifier, performers[identifier]) for identifier in ids]
    initial = run_walk(universe, _RULE, sample_size=2)
    assert _names(initial) == [order[0], order[2]]
    assert [skip.identifier for skip in initial.skipped] == [order[1]]
    replaced = fail_choice(initial, order[0], "failed", universe, _RULE)
    assert _names(replaced) == [order[2], order[3]]
    assert order[1] not in _names(replaced)


def test_replacing_an_identifier_that_is_not_selected_is_refused() -> None:
    """Replacing an identifier that was never selected raises rather than changing the walk."""
    universe = [_item("78_a", ("Ada",)), _item("78_b", ("Ben",))]
    result = run_walk(universe, _RULE, sample_size=1)
    not_taken = next(item.identifier for item in universe if item.identifier not in _names(result))
    with pytest.raises(SelectionError, match="not currently selected"):
        fail_choice(result, not_taken, "x", universe, _RULE)


def test_a_refused_replacement_leaves_the_earlier_state_unchanged() -> None:
    """Replacement returns a new state, so the state it was given keeps its choices."""
    universe = [_item("78_a", ("Ada",)), _item("78_b", ("Ben",)), _item("78_c", ("Cy",))]
    before = run_walk(universe, _RULE, sample_size=2)
    snapshot = _names(before)
    replaced = fail_choice(before, before.selected[0].identifier, "x", universe, _RULE)
    assert _names(before) == snapshot
    assert _names(replaced) != snapshot


def _split_rule() -> WindowRule:
    """The window rule with the protocol's performer split patterns.

    Returns:
        The rule, built from the shipped protocol.
    """
    return window_rule_for(load_listening_protocol())


def test_a_joined_performer_name_splits_into_two_performers() -> None:
    """'Fritz Kreisler and Efrem Zimbalist' gives two performers under the shipped patterns."""
    parts = split_performer_names(
        ("Fritz Kreisler and Efrem Zimbalist",), _split_rule().performer_split
    )
    assert parts == ("Fritz Kreisler", "Efrem Zimbalist")


def test_an_ampersand_and_a_semicolon_also_split() -> None:
    """'&' and ';' are separators as well, and parts are stripped."""
    parts = split_performer_names(
        ("Mischa Elman & Percy B. Kahn; Ada Lovelace",), _split_rule().performer_split
    )
    assert parts == ("Mischa Elman", "Percy B. Kahn", "Ada Lovelace")


def test_a_name_containing_and_inside_a_word_is_not_split() -> None:
    """Only the whole word 'and' separates performers, so 'Sandy' and 'Anderson' stay whole."""
    parts = split_performer_names(("Sandy Anderson",), _split_rule().performer_split)
    assert parts == ("Sandy Anderson",)


def test_a_quartet_name_stays_one_performer() -> None:
    """'Flonzaley Quartet' is one performer under the shipped patterns."""
    parts = split_performer_names(("Flonzaley Quartet",), _split_rule().performer_split)
    assert parts == ("Flonzaley Quartet",)


def test_the_split_changes_which_items_the_walk_can_take() -> None:
    """With the split, an item naming Efrem Zimbalist after 'and' shares a performer and is skipped.

    Without the split the same universe is taken in full. The two items are the only two, and the
    sample is two, so the difference is exact rather than dependent on the seeded order.
    """
    universe = [
        _item("78_joint", ("Fritz Kreisler and Efrem Zimbalist",)),
        _item("78_solo", ("Efrem Zimbalist",)),
    ]
    split = run_walk(universe, _split_rule(), sample_size=2)
    unsplit = run_walk(universe, _RULE, sample_size=2)
    assert not split.complete
    assert [skip.reason for skip in split.skipped] == [SkipReason.PERFORMER_REPEAT]
    assert unsplit.complete


def test_the_window_rule_is_read_from_the_protocol() -> None:
    """The rule takes its seed, margin, clip length and split from the protocol file."""
    protocol = load_listening_protocol()
    rule = window_rule_for(protocol)
    assert rule.seed == protocol.selection.seed
    assert rule.margin_s == protocol.clips.margin_s
    assert rule.clip_s == 8
    assert rule.performer_split == protocol.selection.performer_split
