"""A disc-level split of the Aalto gramophone noise set, frozen before any model v2 work uses it.

The Aalto set (Moliner and Valimaki, ICASSP 2022) holds noise-only segments cut from Great 78
Project transfers. Its own train, validation and test splits were made by recording side, and
21 of its 48 test discs have their other side in its training split, so this module makes its
own split by disc. A Great 78 identifier carries the disc as a seven-digit `gbia` number and
the side as a trailing letter; both sides of one disc share a number.

Every disc of the listening study's universe listing (all 297 items, kept or excluded) is
removed before the split, so no study disc can reach model v2's noise. The remaining discs are
permuted with a fixed seed: the first share is held-out evaluation noise, the next share is
validation noise, and the rest is training noise for the research-only arm. The split is
written once to a tracked manifest, and `load_split` refuses any manifest whose digest is not
the pinned one.

Build the manifest from the repository root, with the set unpacked under `data/`:

    uv run --frozen python -m benchmark.measurements.real_noise_split --aalto-dir DIR

> **Note:** The Aalto set's noise comes from the Great 78 Project, whose collection is offered
> "for use in research, teaching, and private study only". It is used here for evaluation and
> for a research-only training arm whose model is never shipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmark.atomic_write import write_atomically
from benchmark.listening.manifest import UniverseError, read_universe, universe_path
from benchmark.listening.protocol import ListeningProtocolError, load_listening_protocol

__all__ = [
    "DiscSplit",
    "NoiseSplitError",
    "disc_number",
    "load_split",
    "parse_split",
    "recording_dirs",
    "split_discs",
    "study_discs",
]

logger = logging.getLogger(__name__)

SPLIT_SEED = 20261125
EVALUATION_FRACTION = 0.15
VALIDATION_FRACTION = 0.05
SPLIT_PATH = Path(__file__).resolve().parents[1] / "manifests" / "aalto_disc_split.json"
# Set from the manifest written by `main`; a manifest with any other digest is refused.
PINNED_SPLIT_SHA256 = (
    "75849f6248044aae9c2dc4218bf9d0e5214a86fd14fa700c7be6592c79a07363"  # pragma: allowlist secret
)
INFO_NAME = "info.csv"
SEGMENTS_DIR = "segments"
_DISC = re.compile(r"_gbia(\d{7})")
_NOT_ALPHANUMERIC = re.compile(r"[^a-z0-9]")


class NoiseSplitError(ValueError):
    """Raised when the noise set, the study universe or the split manifest cannot be used."""


@dataclass(frozen=True)
class DiscSplit:
    """Disc numbers per part, each part sorted, the parts pairwise disjoint."""

    evaluation: tuple[str, ...]
    validation: tuple[str, ...]
    training: tuple[str, ...]


def disc_number(identifier: str) -> str:
    """The seven-digit disc number of a Great 78 identifier.

    Raises:
        NoiseSplitError: If the identifier carries no `_gbia` disc number.
    """
    match = _DISC.search(identifier)
    if match is None:
        raise NoiseSplitError(f"identifier has no gbia disc number: {identifier!r}")
    return match.group(1)


def study_discs(universe_raw: bytes) -> frozenset[str]:
    """Every disc of the listening study's listing, kept or excluded, from the pinned manifest.

    Raises:
        NoiseSplitError: If the manifest is not the pinned one.
    """
    try:
        read_universe(load_listening_protocol(), universe_raw)
    except (UniverseError, ListeningProtocolError) as error:
        raise NoiseSplitError(f"study universe refused: {error}") from error
    document = json.loads(universe_raw)
    identifiers = [_identifier(item) for item in document["items"]]
    identifiers += [_identifier(item) for item in document["excluded"]]
    return frozenset(disc_number(identifier) for identifier in identifiers)


def _identifier(item: object) -> str:
    """An entry's identifier, whether the entry is a string or a mapping."""
    if isinstance(item, Mapping):
        return str(item["identifier"])
    return str(item)


def _normalised(name: str) -> str:
    """A name with case and punctuation removed, for matching info rows to folder names."""
    return _NOT_ALPHANUMERIC.sub("", name.lower())


def recording_dirs(aalto_dir: Path) -> dict[str, Path]:
    """Each info.csv recording's segment folder, matched by exact or normalised name.

    Info rows with no folder, and folders with no info row, are left out.
    """
    folders = {path.name: path for path in (aalto_dir / SEGMENTS_DIR).iterdir() if path.is_dir()}
    by_normalised = {_normalised(name): path for name, path in folders.items()}
    matched: dict[str, Path] = {}
    with (aalto_dir / INFO_NAME).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = row["recording"]
            folder = folders.get(name) or by_normalised.get(_normalised(name))
            if folder is not None:
                matched[name] = folder
    logger.info("Matched %d info.csv recordings to segment folders", len(matched))
    return matched


def split_discs(discs: Iterable[str], excluded: frozenset[str], seed: int) -> DiscSplit:
    """Permute the discs that are not excluded and cut them into the three parts.

    Raises:
        NoiseSplitError: If too few discs remain for every part to be non-empty.
    """
    candidates = sorted(set(discs) - excluded)
    order = np.random.Generator(np.random.PCG64(seed)).permutation(len(candidates))
    shuffled = [candidates[int(index)] for index in order]
    evaluation_count = round(len(shuffled) * EVALUATION_FRACTION)
    validation_count = round(len(shuffled) * VALIDATION_FRACTION)
    if min(evaluation_count, validation_count, len(shuffled) - evaluation_count) < 1:
        raise NoiseSplitError(f"too few discs to split: {len(shuffled)}")
    cut = evaluation_count + validation_count
    return DiscSplit(
        evaluation=tuple(sorted(shuffled[:evaluation_count])),
        validation=tuple(sorted(shuffled[evaluation_count:cut])),
        training=tuple(sorted(shuffled[cut:])),
    )


def _split_document(
    split: DiscSplit, excluded: frozenset[str], digests: Mapping[str, str]
) -> bytes:
    """The manifest's bytes: the rule, the source digests, the excluded discs and the parts."""
    document = {
        "description": "Disc-level split of the Aalto gramophone noise set for model v2.",
        "seed": SPLIT_SEED,
        "evaluation_fraction": EVALUATION_FRACTION,
        "validation_fraction": VALIDATION_FRACTION,
        "source": [{"file": name, "sha256": digest} for name, digest in sorted(digests.items())],
        "excluded_study_discs": sorted(excluded),
        "evaluation": list(split.evaluation),
        "validation": list(split.validation),
        "training": list(split.training),
    }
    return (json.dumps(document, indent=1) + "\n").encode()


def _file_sha256(path: Path) -> str:
    """The SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_split(raw: bytes) -> DiscSplit:
    """A manifest's parts, checked to be disjoint and to hold seven-digit disc numbers.

    Raises:
        NoiseSplitError: If a part is malformed or two parts share a disc.
    """
    document = json.loads(raw)
    parts = [tuple(document[name]) for name in ("evaluation", "validation", "training")]
    for part in parts:
        if any(not re.fullmatch(r"\d{7}", disc) for disc in part) or list(part) != sorted(part):
            raise NoiseSplitError("a split part holds a malformed or unsorted disc number")
    if len(set().union(*parts)) != sum(len(part) for part in parts):
        raise NoiseSplitError("two split parts share a disc")
    if set().union(*parts) & set(document["excluded_study_discs"]):
        raise NoiseSplitError("a study disc appears in the split")
    return DiscSplit(evaluation=parts[0], validation=parts[1], training=parts[2])


def load_split(path: Path = SPLIT_PATH) -> DiscSplit:
    """The frozen split, refused unless its bytes have the pinned digest.

    Raises:
        NoiseSplitError: If the manifest's digest is not the pinned one, or it is malformed.
    """
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PINNED_SPLIT_SHA256:
        logger.error("Split manifest %s has digest %s, not the pinned one", path, digest)
        raise NoiseSplitError(f"split manifest digest {digest} is not the pinned one")
    return parse_split(raw)


def build_split(aalto_dir: Path, universe_path: Path) -> bytes:
    """The manifest bytes for the unpacked set at `aalto_dir`."""
    excluded = study_discs(universe_path.read_bytes())
    discs = {disc_number(name) for name in recording_dirs(aalto_dir)}
    split = split_discs(discs, excluded, SPLIT_SEED)
    digests = {INFO_NAME: _file_sha256(aalto_dir / INFO_NAME)}
    logger.info(
        "Split %d discs: %d evaluation, %d validation, %d training; %d study discs removed",
        len(discs - excluded),
        len(split.evaluation),
        len(split.validation),
        len(split.training),
        len(discs & excluded),
    )
    return _split_document(split, excluded & discs, digests)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the split manifest once; refuse when it already exists."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--aalto-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if SPLIT_PATH.exists():
        print(f"real_noise_split: {SPLIT_PATH} exists; it is written once")
        return 1
    protocol = load_listening_protocol()
    raw = build_split(args.aalto_dir, universe_path(protocol))
    write_atomically(SPLIT_PATH, raw)
    print(f"real_noise_split: wrote {SPLIT_PATH} sha256 {hashlib.sha256(raw).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
