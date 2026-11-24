"""The universe inclusion rule: which listed items are instrumental solo or string-ensemble recordings.

The patterns are not written here. The frozen protocol file holds them as regex
sources, `benchmark.listening.protocol` compiles them into a `UniverseRule`, and this module
applies that rule to an item's facts. A change to a pattern is therefore a change to the
frozen protocol, and the pinned universe digest records its effect.

The checks run in a fixed order, and the first one that fails names the exclusion. The order
is part of the rule, because an item that fails two checks gets one reason.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from benchmark.listening.metadata import ItemFacts

__all__ = ["Decision", "Reason", "UniverseRule", "decide"]


class Reason(StrEnum):
    """Why an item was kept or excluded. The value is the text the manifest records."""

    KEPT = "kept"
    NO_INSTRUMENTATION_LINE = "no_instrumentation_line"
    MAIN_FLAC_COUNT = "main_flac_count_not_one"
    VOCAL = "vocal"
    ACCORDION = "accordion"
    NOT_SOLO_OR_STRING_ENSEMBLE = "not_solo_or_string_ensemble"


@dataclass(frozen=True)
class UniverseRule:
    """The compiled inclusion rule.

    Attributes:
        keep_patterns: At least one must match the instrumentation line.
        vocal_patterns: Any match on the instrumentation line or the title excludes the item.
        accordion_patterns: Any match on the instrumentation line or the title excludes the item.
        required_flac_count: The number of main FLAC files an item must have.
    """

    keep_patterns: tuple[re.Pattern[str], ...]
    vocal_patterns: tuple[re.Pattern[str], ...]
    accordion_patterns: tuple[re.Pattern[str], ...]
    required_flac_count: int


@dataclass(frozen=True)
class Decision:
    """The outcome of applying the rule to one item.

    Attributes:
        reason: `Reason.KEPT` or the exclusion reason.
        matched: The matched text of the patterns that decided the outcome, in order.
    """

    reason: Reason
    matched: tuple[str, ...]

    @property
    def kept(self) -> bool:
        """Whether the item stays in the universe."""
        return self.reason is Reason.KEPT


def _matches(patterns: Sequence[re.Pattern[str]], texts: Sequence[str]) -> tuple[str, ...]:
    """The matched text of every pattern hit in any of *texts*, in pattern then text order.

    Args:
        patterns: The patterns to search for.
        texts: The strings to search in.

    Returns:
        Each hit's text, lowercased. Empty when nothing matches.
    """
    return tuple(
        match.group(0).lower()
        for pattern in patterns
        for text in texts
        for match in pattern.finditer(text)
    )


def _keep_match(rule: UniverseRule, instrumentation: str) -> tuple[str, ...]:
    """The keep patterns' hits on the instrumentation line, in pattern order.

    Args:
        rule: The compiled rule.
        instrumentation: The instrumentation line.

    Returns:
        The hits, lowercased. Empty when no keep pattern matches.
    """
    return tuple(
        match.group(0).lower()
        for pattern in rule.keep_patterns
        for match in pattern.finditer(instrumentation)
    )


def decide(facts: ItemFacts, rule: UniverseRule) -> Decision:
    """Apply the inclusion rule to one item.

    Args:
        facts: The item's facts from `benchmark.listening.metadata.item_facts`.
        rule: The compiled rule.

    Returns:
        The decision. An excluded item carries the first failing check's reason.
    """
    if not facts.instrumentation.strip():
        return Decision(Reason.NO_INSTRUMENTATION_LINE, ())
    if facts.flac_count != rule.required_flac_count:
        return Decision(Reason.MAIN_FLAC_COUNT, ())
    texts = (facts.instrumentation, facts.title)
    vocal = _matches(rule.vocal_patterns, texts)
    if vocal:
        return Decision(Reason.VOCAL, tuple(sorted(set(vocal))))
    accordion = _matches(rule.accordion_patterns, texts)
    if accordion:
        return Decision(Reason.ACCORDION, tuple(sorted(set(accordion))))
    keep = _keep_match(rule, facts.instrumentation)
    if not keep:
        return Decision(Reason.NOT_SOLO_OR_STRING_ENSEMBLE, ())
    return Decision(Reason.KEPT, tuple(sorted(set(keep))))
