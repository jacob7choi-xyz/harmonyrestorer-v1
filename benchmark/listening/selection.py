"""Choosing the study's recordings and the clip window cut from each one. Pure: no I/O.

The walk takes the universe in the order `benchmark.selection.walk_order` gives (sorted by
identifier, then permuted by a PCG64 generator seeded with the study seed). It takes an item
when none of its performers has already been taken, and when a valid clip window exists.

When a taken item later fails, its choice is removed and the walk resumes after the last item
it considered. It does not return to the start, so the performer names of the removed item do
not reopen an item the walk already skipped. The protocol states this rule, and a test checks
the case where it matters.

The duration used for a window is the `length_s` value in the universe manifest, which is the
archive's listed length. The frozen rule uses the canonical sample count, which is only known
after decoding. A window reported from the listed length is therefore provisional.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from benchmark.listening.manifest import UniverseItem
from benchmark.listening.metadata import performer_key, split_performer_names
from benchmark.listening.protocol import ListeningProtocol
from benchmark.selection import walk_order

__all__ = [
    "Choice",
    "Replacement",
    "SelectionError",
    "Skip",
    "SkipReason",
    "Walk",
    "WindowRule",
    "fail_choice",
    "run_walk",
    "window_bounds",
    "window_rule_for",
    "window_start",
]


class SelectionError(ValueError):
    """Raised when a replacement names an item that is not currently selected."""


class SkipReason(StrEnum):
    """Why the walk passed over an item without taking it."""

    PERFORMER_REPEAT = "performer_repeat"
    NO_DURATION = "no_duration"
    TOO_SHORT = "too_short"


@dataclass(frozen=True)
class WindowRule:
    """The inputs of the clip window rule and of the performer comparison.

    Attributes:
        seed: The study seed, which seeds each window's generator with the identifier.
        margin_s: The whole seconds kept clear at each end of a recording.
        clip_s: The clip length in seconds, a whole number.
        performer_split: The patterns that split a listed performer name, in order.
    """

    seed: int
    margin_s: int
    clip_s: int
    performer_split: tuple[re.Pattern[str], ...] = ()


@dataclass(frozen=True)
class Choice:
    """One taken item and its clip window.

    Attributes:
        identifier: The item's identifier.
        start_s: The window's start, in whole seconds from the start of the recording.
        duration_s: The duration the window was drawn from (provisional, see the module).
        performer_keys: The comparison keys of the item's performers.
    """

    identifier: str
    start_s: int
    duration_s: float
    performer_keys: frozenset[str]


@dataclass(frozen=True)
class Skip:
    """An item the walk passed over, and why."""

    identifier: str
    reason: SkipReason


@dataclass(frozen=True)
class Replacement:
    """A taken item that was removed after it failed, with the failure's reason."""

    identifier: str
    reason: str


@dataclass(frozen=True)
class Walk:
    """The state of the walk. Immutable: each step returns a new state.

    Attributes:
        order: The universe identifiers in walk order.
        position: The index in `order` of the next item the walk considers.
        selected: The current choices, in the order they were taken.
        skipped: Every item passed over, in order, with its reason.
        replaced: Every removed choice, in order, with its reason.
        sample_size: The number of choices the walk aims for.
    """

    order: tuple[str, ...]
    position: int
    selected: tuple[Choice, ...]
    skipped: tuple[Skip, ...]
    replaced: tuple[Replacement, ...]
    sample_size: int

    @property
    def complete(self) -> bool:
        """Whether the walk holds `sample_size` choices."""
        return len(self.selected) == self.sample_size


def window_rule_for(protocol: ListeningProtocol) -> WindowRule:
    """The window rule and performer split the protocol names.

    Args:
        protocol: The loaded protocol.

    Returns:
        The rule, built from the protocol's seed, clip margin, clip length and split patterns.
    """
    return WindowRule(
        seed=protocol.selection.seed,
        margin_s=protocol.clips.margin_s,
        clip_s=int(protocol.clips.duration_s),
        performer_split=protocol.selection.performer_split,
    )


def window_bounds(duration_s: float, margin_s: int, clip_s: int) -> tuple[int, int] | None:
    """The inclusive range of whole-second window starts for a recording, or None.

    Args:
        duration_s: The recording's duration in seconds.
        margin_s: The whole seconds kept clear at each end.
        clip_s: The clip length in whole seconds.

    Returns:
        (low, high) with low = margin_s and high = floor(duration_s) - margin_s - clip_s, or None
        when high is below low, so that no valid window exists.
    """
    low = margin_s
    high = math.floor(duration_s) - margin_s - clip_s
    return (low, high) if high >= low else None


def window_start(identifier: str, rule: WindowRule, bounds: tuple[int, int]) -> int:
    """The window start for one recording: a deterministic draw from its own generator.

    The generator is numpy's PCG64 seeded with SeedSequence([seed, *words]), where words are
    the eight big-endian unsigned 32-bit integers of the SHA-256 of the identifier's UTF-8
    bytes, in digest order. The draw is inclusive at both ends of *bounds*.

    Args:
        identifier: The item's identifier.
        rule: The window rule, which supplies the study seed.
        bounds: The inclusive (low, high) range from `window_bounds`.

    Returns:
        The start, in whole seconds.
    """
    digest = hashlib.sha256(identifier.encode("utf-8")).digest()
    words = [int.from_bytes(digest[offset : offset + 4], "big") for offset in range(0, 32, 4)]
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([rule.seed, *words])))
    low, high = bounds
    return int(generator.integers(low, high, endpoint=True))


def _take(walk: Walk, item: UniverseItem, rule: WindowRule) -> Walk:
    """Take *item* if it can be taken, otherwise record why it was passed over.

    Args:
        walk: The current state.
        item: The item the walk is considering.
        rule: The window rule.

    Returns:
        The state after considering *item*.
    """
    parts = split_performer_names(item.performers, rule.performer_split)
    keys = frozenset(performer_key(part) for part in parts)
    taken = frozenset().union(*(choice.performer_keys for choice in walk.selected))
    if keys & taken:
        return _pass_over(walk, item.identifier, SkipReason.PERFORMER_REPEAT)
    if item.length_s is None:
        return _pass_over(walk, item.identifier, SkipReason.NO_DURATION)
    bounds = window_bounds(item.length_s, rule.margin_s, rule.clip_s)
    if bounds is None:
        return _pass_over(walk, item.identifier, SkipReason.TOO_SHORT)
    choice = Choice(
        identifier=item.identifier,
        start_s=window_start(item.identifier, rule, bounds),
        duration_s=item.length_s,
        performer_keys=keys,
    )
    return replace(walk, selected=(*walk.selected, choice))


def _pass_over(walk: Walk, identifier: str, reason: SkipReason) -> Walk:
    """Record that the walk passed over *identifier*.

    Args:
        walk: The current state.
        identifier: The passed-over item.
        reason: Why it was passed over.

    Returns:
        The state with the skip appended.
    """
    return replace(walk, skipped=(*walk.skipped, Skip(identifier, reason)))


def _advance(walk: Walk, items: dict[str, UniverseItem], rule: WindowRule) -> Walk:
    """Consider items in order until the walk is complete or the order is used up.

    Args:
        walk: The current state.
        items: The universe items keyed by identifier.
        rule: The window rule.

    Returns:
        The state when it is complete, or when no item is left.
    """
    current = walk
    while not current.complete and current.position < len(current.order):
        identifier = current.order[current.position]
        current = replace(current, position=current.position + 1)
        current = _take(current, items[identifier], rule)
    return current


def run_walk(universe: Sequence[UniverseItem], rule: WindowRule, sample_size: int) -> Walk:
    """Run the walk over a universe and return the resulting state.

    Args:
        universe: The kept items. Identifiers must be unique.
        rule: The window rule.
        sample_size: The number of choices to take.

    Returns:
        The state. Its `complete` property is False when the universe runs out first.

    Raises:
        SelectionError: If the universe repeats an identifier (raised by `walk_order`).
    """
    order = walk_order([item.identifier for item in universe], rule.seed)
    items = {item.identifier: item for item in universe}
    start = Walk(order, 0, (), (), (), sample_size)
    return _advance(start, items, rule)


def fail_choice(
    walk: Walk,
    identifier: str,
    reason: str,
    universe: Sequence[UniverseItem],
    rule: WindowRule,
) -> Walk:
    """Remove a failed choice and continue the walk after the last item it considered.

    Args:
        walk: The state in which the choice was taken.
        identifier: The failed item.
        reason: Why it failed, such as the failed check's name.
        universe: The same universe the walk was run over.
        rule: The window rule.

    Returns:
        The new state.

    Raises:
        SelectionError: If *identifier* is not currently selected.
    """
    if identifier not in {choice.identifier for choice in walk.selected}:
        raise SelectionError(f"{identifier!r} is not currently selected")
    remaining = tuple(choice for choice in walk.selected if choice.identifier != identifier)
    removed = replace(
        walk,
        selected=remaining,
        replaced=(*walk.replaced, Replacement(identifier, reason)),
    )
    items = {item.identifier: item for item in universe}
    return _advance(removed, items, rule)
