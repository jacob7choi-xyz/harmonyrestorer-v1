"""The selection walk's order and the automatic exclusions against the training corpus.

Protocol v6 freezes both, v9 adds the candidate filter and the rejection codes, and none
of it needs the network:

- **Candidates.** A listed file is a candidate when the suffix of the final component of its
  path, lowercased, is `.flac` (v9), the test `dataset/acquire.py` applied when the training
  recordings were downloaded. Training recordings pass it and stay in the universe.

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

`ExclusionRule`'s values are v9's codes for the three rules against training. v9's codes
for a match against an earlier admission belong to the walk, which is not in this module;
the duplicate screen's codes are in `benchmark.duplicate_screen`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

import numpy as np

from benchmark._source_validation import SourceOutcomeError, checked_digest

__all__ = [
    "ExclusionRule",
    "SelectionError",
    "TrainingIdentities",
    "exclusions",
    "is_candidate",
    "training_filename",
    "training_identities",
    "walk_order",
]


class SelectionError(ValueError):
    """Raised when an input to selection is malformed."""


class ExclusionRule(StrEnum):
    """The three automatic exclusions against training, in v6's order, with v9's codes."""

    SOURCE_IDENTITY = "TRAINING_SOURCE_IDENTITY"
    RAW_SHA256 = "TRAINING_RAW_SHA256"
    DECODED_PCM_SHA256 = "TRAINING_DECODED_PCM_SHA256"


@dataclass(frozen=True)
class TrainingIdentities:
    """The training corpus as the three rules compare against it."""

    filenames: frozenset[str]
    raw_sha256: frozenset[str]
    decoded_sha256: frozenset[str]


def is_candidate(path_in_item: str) -> bool:
    """Whether a listed file is in the candidate universe under v9's filter.

    Args:
        path_in_item: The file's path within the item, as the item's listing names it.

    Returns:
        True when the final path component's suffix, lowercased, is `.flac`.
    """
    return PurePosixPath(path_in_item).suffix.lower() == ".flac"


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
        collection: `selection.training_collection_identifier`, the item the training
            recordings were downloaded from.

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
        collection: `selection.training_collection_identifier`.
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
    manifest: Sequence[Mapping[str, object]], decoded_sha256: Mapping[str, str]
) -> TrainingIdentities:
    """The three identity sets from the training manifest's entries and the decoded digests.

    Args:
        manifest: The `recordings` entries of `benchmark/manifests/training_corpus.json`.
        decoded_sha256: Each training recording's decoded-PCM digest, keyed by its file name.

    Returns:
        The identities the rules compare against.

    Raises:
        SelectionError: If a file name repeats, if the decoded digests do not cover exactly
            the manifest's file names, or if a digest is not 64 lowercase hexadecimal
            characters. A missing decoded digest would leave rule 3 unable to exclude a
            copy of that recording, so an incomplete set is refused rather than used.
    """
    filenames = [str(entry["filename"]) for entry in manifest]
    if len(set(filenames)) != len(filenames):
        raise SelectionError("the training manifest repeats a file name")
    if set(decoded_sha256) != set(filenames):
        missing = sorted(set(filenames) - set(decoded_sha256))
        unknown = sorted(set(decoded_sha256) - set(filenames))
        raise SelectionError(f"decoded digests missing for {missing}, unknown for {unknown}")
    raw = [str(entry["sha256"]) for entry in manifest]
    for digest in (*raw, *decoded_sha256.values()):
        try:
            checked_digest(digest, "sha256")
        except SourceOutcomeError as error:
            raise SelectionError(str(error)) from error
    return TrainingIdentities(
        filenames=frozenset(filenames),
        raw_sha256=frozenset(raw),
        decoded_sha256=frozenset(decoded_sha256.values()),
    )
