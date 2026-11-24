"""The trials one listener completes: 30 scored pairs and 2 catch trials, in a seeded order.

The protocol fixes the counts, the three pairs, the catch-trial rule and the seed, but not every
draw. This module fixes the rest, in this order, from one generator per listener:

1. The two catch trials: two distinct recordings, chosen with `choice` without replacement, and
   one condition for each, drawn with `integers` over the protocol's three condition names.
2. The presentation order: a permutation of the 32 trials. The scored trials are listed by
   recording (in the order the caller gives, which is sorted identifier order), then by pair in
   protocol order. The two catch trials follow them.
3. The side of each scored trial: one draw per scored trial, in list order, deciding whether the
   pair's first condition is on side A. A catch trial plays one condition on both sides, so it
   has no side draw.

Trial identifiers are `t01` to `t32`, in presentation order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from benchmark.listening.protocol import CONDITION_NAMES, ListeningProtocol

__all__ = ["Trial", "listener_trials"]

Kind = Literal["scored", "catch"]
_Draft = tuple[Kind, str, str, str, bool]


@dataclass(frozen=True)
class Trial:
    """One trial, with its conditions named. Only the analysis reads the condition names.

    Attributes:
        trial_id: The identifier the listener's page shows, `t01` to `t32`.
        kind: `scored` for a pair, `catch` for one condition on both sides.
        recording: The recording's identifier.
        first: The pair's first condition, in protocol order. A catch trial's condition.
        second: The pair's second condition. Equal to `first` for a catch trial.
        first_on_a: Whether `first` is on side A. Always True for a catch trial.
    """

    trial_id: str
    kind: Kind
    recording: str
    first: str
    second: str
    first_on_a: bool

    @property
    def side_a(self) -> str:
        """The condition played on side A."""
        return self.first if self.first_on_a else self.second

    @property
    def side_b(self) -> str:
        """The condition played on side B."""
        return self.second if self.first_on_a else self.first


def _generator(seed: int, listener_index: int) -> np.random.Generator:
    """The listener's generator, seeded by the study seed and the listener index."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, listener_index])))


def _catch_drafts(rng: np.random.Generator, recordings: Sequence[str]) -> list[_Draft]:
    """The two catch trials, drawn first, as drafts without identifiers."""
    clips = rng.choice(len(recordings), size=2, replace=False)
    picks = rng.integers(0, len(CONDITION_NAMES), size=2)
    return [
        (
            "catch",
            recordings[int(clip)],
            CONDITION_NAMES[int(pick)],
            CONDITION_NAMES[int(pick)],
            True,
        )
        for clip, pick in zip(clips, picks, strict=True)
    ]


def listener_trials(
    protocol: ListeningProtocol, listener_index: int, recordings: Sequence[str]
) -> tuple[Trial, ...]:
    """The trials for one listener, in presentation order.

    Args:
        protocol: The loaded protocol, which gives the seed, the pairs and the trial counts.
        listener_index: The listener's index in order of consent, from 0.
        recordings: The recording identifiers, in the order the design uses (sorted).

    Returns:
        The trials, with identifiers `t01` to `t32` in presentation order.

    Raises:
        ValueError: If the index is negative, or there are fewer than two recordings.
    """
    if listener_index < 0:
        raise ValueError(f"listener index must be non-negative, got {listener_index}")
    if len(recordings) < 2:
        raise ValueError("the catch trials need at least two recordings")
    rng = _generator(protocol.selection.seed, listener_index)
    catches = _catch_drafts(rng, recordings)
    scored: list[_Draft] = [
        ("scored", recording, first, second, True)
        for recording in recordings
        for first, second in protocol.trials.pairs
    ]
    drafts: list[_Draft] = scored + catches
    order = rng.permutation(len(drafts))
    flips = rng.integers(0, 2, size=len(scored))
    final: list[_Draft] = []
    for index, (kind, recording, first, second, first_on_a) in enumerate(drafts):
        if kind == "catch":
            final.append((kind, recording, first, second, first_on_a))
        else:
            final.append((kind, recording, first, second, not bool(flips[index])))
    return tuple(
        Trial(f"t{position + 1:02d}", kind, recording, first, second, first_on_a)
        for position, (kind, recording, first, second, first_on_a) in enumerate(
            final[int(index)] for index in order
        )
    )
