"""The selection walk's order and the automatic exclusions against the training corpus.

Protocol v6 freezes both, and neither needs the network:

- **Walk order.** The frozen universe's source identifiers are sorted as Python strings, by
  Unicode code point, and walked in the order that sorted list takes when indexed by
  `numpy.random.Generator(PCG64(seed)).permutation(n)`, where `n` is the size of the frozen
  universe. Duplicated identifiers are refused, because the permutation is over positions
  and a repeated identifier would be walked twice.
- **Automatic exclusion**, no discretion, against the frozen training corpus:
  1. source identity, compared in training's flattened file name form;
  2. the raw file's SHA-256;
  3. the decoded PCM's SHA-256.

**Rule 1's flattened form.** A candidate's source identifier is the item identifier, a
slash, and the file's path in the item. `dataset/acquire.py` named each training file
`f"{identifier}__{filename}".replace("/", "_")`, and v6 compares a candidate in that form.
The flattening is not one-to-one: `a/b_c.flac` and `a_b/c.flac` flatten alike. Under rule 1
that can only exclude a candidate whose flattened name matches a training file's, never
admit one that should have been excluded.

Every rule is evaluated and every match is returned, so the audit trail can show all the
grounds for an exclusion rather than the first.

> **Note:** v8 freezes the rules but names no code for each in the selection audit trail.
> The names of `ExclusionRule` are this module's own and are expected to be replaced by
> the amendment frozen before the candidate universe is enumerated.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

__all__ = [
    "ExclusionRule",
    "SelectionError",
    "TrainingIdentities",
    "exclusions",
    "training_filename",
    "training_identities",
    "walk_order",
]


class SelectionError(ValueError):
    """Raised when an input to selection is malformed."""


class ExclusionRule(StrEnum):
    """The three automatic exclusions, in v6's order. Names are code-owned until v9."""

    SOURCE_IDENTITY = "training_source_identity"
    RAW_SHA256 = "training_raw_sha256"
    DECODED_PCM_SHA256 = "training_decoded_pcm_sha256"


@dataclass(frozen=True)
class TrainingIdentities:
    """The training corpus as the three rules compare against it."""

    filenames: frozenset[str]
    raw_sha256: frozenset[str]
    decoded_sha256: frozenset[str]


def walk_order(identifiers: Iterable[str], seed: int) -> tuple[str, ...]:
    """The frozen universe in walk order.

    Args:
        identifiers: Every source identifier in the frozen universe, in any order.
        seed: `selection.seed`.

    Returns:
        The identifiers, sorted by code point and then indexed by the seeded permutation.

    Raises:
        SelectionError: If an identifier repeats.
    """
    ordered = sorted(identifiers)
    if len(set(ordered)) != len(ordered):
        raise SelectionError("the candidate universe repeats an identifier")
    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(len(ordered))
    return tuple(ordered[int(index)] for index in permutation)


def training_filename(source_identifier: str, collection: str) -> str:
    """A candidate's source identifier in the flattened form training files were named in.

    Args:
        source_identifier: `<item identifier>/<path in the item>`.
        collection: `selection.collection_identifier`, the only item candidates come from.

    Returns:
        `<item>__<path>` with every slash replaced by an underscore.

    Raises:
        SelectionError: If the identifier is not a path inside the collection's item.
    """
    item, slash, path = source_identifier.partition("/")
    if item != collection or not slash or not path:
        raise SelectionError(
            f"{source_identifier!r} is not a file path inside the {collection!r} item"
        )
    return f"{item}__{path}".replace("/", "_")


def exclusions(
    source_identifier: str,
    *,
    raw_sha256: str,
    decoded_sha256: str,
    collection: str,
    training: TrainingIdentities,
) -> tuple[ExclusionRule, ...]:
    """Every automatic exclusion rule a candidate meets, in rule order; empty if none.

    Args:
        source_identifier: The candidate's source identifier.
        raw_sha256: The SHA-256 of the candidate's file.
        decoded_sha256: The SHA-256 of its decoded PCM, float64 little-endian interleaved.
        collection: `selection.collection_identifier`.
        training: The training corpus's identities.

    Returns:
        The rules the candidate meets.
    """
    met: list[ExclusionRule] = []
    if training_filename(source_identifier, collection) in training.filenames:
        met.append(ExclusionRule.SOURCE_IDENTITY)
    if raw_sha256 in training.raw_sha256:
        met.append(ExclusionRule.RAW_SHA256)
    if decoded_sha256 in training.decoded_sha256:
        met.append(ExclusionRule.DECODED_PCM_SHA256)
    return tuple(met)


def training_identities(
    manifest: Sequence[dict[str, object]], decoded_sha256: Iterable[str]
) -> TrainingIdentities:
    """The three identity sets from the training manifest's entries and the decoded digests.

    Args:
        manifest: The `recordings` entries of `benchmark/manifests/training_corpus.json`.
        decoded_sha256: Each training recording's decoded-PCM digest.

    Returns:
        The identities the rules compare against.
    """
    return TrainingIdentities(
        filenames=frozenset(str(entry["filename"]) for entry in manifest),
        raw_sha256=frozenset(str(entry["sha256"]) for entry in manifest),
        decoded_sha256=frozenset(decoded_sha256),
    )
