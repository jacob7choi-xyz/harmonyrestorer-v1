"""The candidate universes: one per v12 stratum, listed once and frozen.

Protocol v12 walks two strata, each over its own frozen universe (`selection.strata`). This
module builds each universe from its source's metadata and reads the frozen file back:

- **solo_piano** is an Internet Archive item. Its universe is every listed file that passes
  the suffix filter, less the stratum's `excluded_paths`, each with the size, md5 and sha1 the
  listing gives and its cluster: the file's artist tag after the stratum's spelling table.
- **chamber** is MusicNet. Its universe is exactly the stratum's `candidate_ids`, read from
  the metadata file whose md5 the stratum freezes, each with its cluster: the row's source
  after the spelling table. MusicNet gives no per-file size or checksum, so none is recorded.

A candidate's identifier is the stratum's source, a slash, and its path in that source.
The choices v12 leaves open (the query URLs, the file layout, the pinned digests) are stated
in `docs/benchmark-protocol.md`, How the sources are read. Each file's SHA-256 is pinned in
`UNIVERSE_SHA256`, so a change to a universe is a change to this code, and the loader refuses
a file with another digest or one that is not byte-for-byte its own canonical serialisation.

A malformed identifier is kept: the walk excludes it under its own code. A listing that
names a candidate twice, lacks a value the stratum needs, or does not hold every path or id
the stratum names is refused, since the frozen protocol describes what the source holds.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from benchmark.protocol import ArchiveItemStratum, Stratum, ZenodoArchiveStratum
from benchmark.selection import is_candidate

__all__ = [
    "UNIVERSE_SHA256",
    "CandidateUniverse",
    "CandidateUniverseError",
    "ListedFile",
    "archive_item_candidates",
    "chamber_candidates",
    "load_universe",
    "metadata_url",
    "universe_bytes",
    "universe_path",
    "zenodo_file_url",
]

logger = logging.getLogger(__name__)

_MANIFESTS = Path(__file__).resolve().parents[1] / "benchmark" / "manifests"
# Set in the commit that adds each listed file; None means that stratum has not been listed.
UNIVERSE_SHA256: dict[str, str | None] = {
    "solo_piano": "55a38b8d7d6f3bff450df9583d5db236c1065db8066933a7a177684aecfad1eb",  # pragma: allowlist secret
    "chamber": "b4dccbf7c4330c8bcf20129e056d99260ebbc40aa9c378f3677ee43680e29a90",  # pragma: allowlist secret
}

_HEX = {"md5": re.compile(r"[0-9a-f]{32}"), "sha1": re.compile(r"[0-9a-f]{40}")}
# ASCII digits only: str.isdigit also accepts superscripts and other scripts, which int refuses.
_ASCII_DIGITS = re.compile(r"[0-9]+")
_FILE_KEYS = frozenset(
    {"description", "stratum", "source", "query", "enumerated_at", "count", "candidates"}
)
_ENTRY_KEYS = frozenset({"identifier", "cluster", "size", "md5", "sha1"})
_DESCRIPTION = (
    "One stratum's candidate universe, listed from its source's metadata once and frozen "
    "before the selection walk."
)


class CandidateUniverseError(ValueError):
    """Raised when a listing or a frozen universe file is malformed or altered."""


@dataclass(frozen=True)
class ListedFile:
    """One candidate, its cluster, and the listing values its source gave (None if not)."""

    identifier: str
    cluster: str
    size: int | None
    md5: str | None
    sha1: str | None


@dataclass(frozen=True)
class CandidateUniverse:
    """One stratum's frozen universe and the facts of its enumeration."""

    stratum: str
    source: str
    query: str
    enumerated_at: str
    files: tuple[ListedFile, ...]
    sha256: str


def _refuse(message: str) -> NoReturn:
    """Log and raise *message* as a CandidateUniverseError."""
    logger.error("Candidate universe: %s", message)
    raise CandidateUniverseError(message)


def universe_path(stratum: str) -> Path:
    """Where the stratum's frozen universe lives."""
    return _MANIFESTS / f"universe_{stratum}.json"


def metadata_url(item: str) -> str:
    """The URL an Internet Archive item's listing is read from."""
    return f"https://archive.org/metadata/{item}"


def zenodo_file_url(record: str, file: str) -> str:
    """The URL a file of a Zenodo record is read from."""
    return f"https://zenodo.org/api/records/{record}/files/{file}/content"


def _size(value: object, name: str) -> int | None:
    """A listed size as a non-negative int; the archive gives it as a decimal string."""
    if value is None:
        return None
    if not isinstance(value, str) or not _ASCII_DIGITS.fullmatch(value):
        _refuse(f"{name}: listed size {value!r} is not a decimal byte count")
    return int(value)


def _hex(value: object, field: str, name: str) -> str | None:
    """A listed md5 or sha1 as lowercase hex of the right length."""
    if value is None:
        return None
    if not isinstance(value, str) or not _HEX[field].fullmatch(value):
        _refuse(f"{name}: listed {field} {value!r} is not lowercase hex")
    return value


def _cluster(value: object, spelling: Mapping[str, str], name: str) -> str:
    """The cluster a performer name gives, after the stratum's spelling table."""
    if not isinstance(value, str) or not value.strip():
        _refuse(f"{name}: no performer is listed, so its cluster is unknown")
    return spelling.get(value, value)


def _distinct_sorted(listed: list[ListedFile]) -> tuple[ListedFile, ...]:
    """*listed* sorted by identifier code point, refusing a repeated identifier."""
    identifiers = [f.identifier for f in listed]
    if len(set(identifiers)) != len(identifiers):
        _refuse("the listing names a candidate more than once")
    return tuple(sorted(listed, key=lambda f: f.identifier))


def archive_item_candidates(
    metadata: object, stratum: ArchiveItemStratum
) -> tuple[ListedFile, ...]:
    """The stratum's candidates from its item's metadata, sorted by identifier.

    Raises:
        CandidateUniverseError: If the metadata has no named files, a listed value or
            artist tag is missing or malformed, a candidate repeats, or an excluded path
            is not in the listing.
    """
    files = metadata.get("files") if isinstance(metadata, dict) else None
    if not isinstance(files, list):
        _refuse("the metadata has no files list")
    excluded, seen, listed = set(stratum.excluded_paths), set(), []
    for entry in files:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not name:
            _refuse(f"a listed file has no name: {entry!r}")
        if not is_candidate(name):
            continue
        if name in excluded:
            seen.add(name)
            continue
        listed.append(
            ListedFile(
                identifier=f"{stratum.source}/{name}",
                cluster=_cluster(entry.get("artist"), stratum.cluster_spelling, name),
                size=_size(entry.get("size"), name),
                md5=_hex(entry.get("md5"), "md5", name),
                sha1=_hex(entry.get("sha1"), "sha1", name),
            )
        )
    if seen != excluded:
        _refuse(f"excluded paths not in the listing: {sorted(excluded - seen)}")
    return _distinct_sorted(listed)


def chamber_candidates(
    metadata_csv: bytes, stratum: ZenodoArchiveStratum
) -> tuple[ListedFile, ...]:
    """The stratum's candidate ids from the metadata file the stratum freezes.

    Raises:
        CandidateUniverseError: If the file's md5 is not the frozen one, or a candidate id
            is missing, repeated, or has no source.
    """
    actual = hashlib.md5(metadata_csv, usedforsecurity=False).hexdigest()
    if actual != stratum.metadata_md5:
        _refuse(f"{stratum.metadata} has md5 {actual}, the protocol freezes {stratum.metadata_md5}")
    rows: dict[int, list[dict[str, str]]] = {}
    for row in csv.DictReader(io.StringIO(metadata_csv.decode("utf-8"))):
        if _ASCII_DIGITS.fullmatch(row.get("id", "")):
            rows.setdefault(int(row["id"]), []).append(row)
    listed = []
    for identifier in stratum.candidate_ids:
        found = rows.get(identifier, [])
        if len(found) != 1:
            _refuse(f"MusicNet id {identifier} appears {len(found)} times in {stratum.metadata}")
        cluster = _cluster(found[0].get("source"), stratum.cluster_spelling, str(identifier))
        listed.append(ListedFile(f"{stratum.source}/{identifier}", cluster, None, None, None))
    return _distinct_sorted(listed)


def universe_bytes(
    files: Sequence[ListedFile], *, stratum: str, source: str, query: str, enumerated_at: str
) -> bytes:
    """The frozen file for *files*, in the one serialisation the loader accepts."""
    data = {
        "description": _DESCRIPTION,
        "stratum": stratum,
        "source": source,
        "query": query,
        "enumerated_at": enumerated_at,
        "count": len(files),
        "candidates": [
            {
                "identifier": f.identifier,
                "cluster": f.cluster,
                "size": f.size,
                "md5": f.md5,
                "sha1": f.sha1,
            }
            for f in sorted(files, key=lambda f: f.identifier)
        ],
    }
    return (json.dumps(data, indent=2, ensure_ascii=True) + "\n").encode()


def _entry(entry: Any) -> ListedFile:
    """One frozen entry, its keys exact and its values typed."""
    if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
        _refuse(f"a candidate entry must have keys {sorted(_ENTRY_KEYS)}: {entry!r}")
    identifier, cluster, size = entry["identifier"], entry["cluster"], entry["size"]
    if not isinstance(identifier, str) or not identifier:
        _refuse(f"a candidate identifier must be a non-empty string: {identifier!r}")
    if not isinstance(cluster, str) or not cluster:
        _refuse(f"{identifier}: cluster must be a non-empty string")
    if size is not None and (type(size) is not int or size < 0):
        _refuse(f"{identifier}: size must be a non-negative integer or null")
    md5, sha1 = _hex(entry["md5"], "md5", identifier), _hex(entry["sha1"], "sha1", identifier)
    return ListedFile(identifier, cluster, size, md5, sha1)


def _check_membership(files: tuple[ListedFile, ...], stratum: Stratum) -> None:
    """Every frozen candidate is one the stratum admits, named and clustered as it says."""
    prefix = f"{stratum.source}/"
    paths = [f.identifier.removeprefix(prefix) for f in files]
    if any(not f.identifier.startswith(prefix) for f in files):
        _refuse(f"a candidate is not from {stratum.source}")
    if any(f.cluster in stratum.cluster_spelling for f in files):
        _refuse("a cluster still holds a misspelling the stratum's table corrects")
    if isinstance(stratum, ArchiveItemStratum):
        excluded = set(stratum.excluded_paths)
        if any(not is_candidate(p) or p in excluded for p in paths):
            _refuse("the file lists a path the stratum's filter or exclusions remove")
    elif sorted(paths) != sorted(str(i) for i in stratum.candidate_ids):
        _refuse("the file does not list exactly the stratum's candidate ids")


def _read_universe(path: Path, expected_sha256: str, stratum: Stratum) -> CandidateUniverse:
    """The frozen universe at *path*, refused unless its digest, form and content hold."""
    if not path.is_file():
        _refuse(f"{path} does not exist")
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        _refuse(f"{path.name} has SHA-256 {actual}, the pinned digest is {expected_sha256}")
    data = json.loads(raw)
    if not isinstance(data, dict) or set(data) != _FILE_KEYS:
        _refuse(f"{path.name} must have keys {sorted(_FILE_KEYS)}")
    if (data["stratum"], data["source"]) != (stratum.name, stratum.source):
        _refuse(f"{path.name} is for {data['stratum']!r} from {data['source']!r}")
    if not isinstance(data["candidates"], list):
        _refuse(f"{path.name} candidates must be a list")
    files = tuple(_entry(entry) for entry in data["candidates"])
    query, enumerated_at = data["query"], data["enumerated_at"]
    if not isinstance(query, str) or not isinstance(enumerated_at, str):
        _refuse(f"{path.name} query and enumerated_at must be strings")
    canonical = universe_bytes(
        files, stratum=stratum.name, source=stratum.source, query=query, enumerated_at=enumerated_at
    )
    if canonical != raw:
        _refuse(f"{path.name} is not the canonical serialisation of its own content")
    _check_membership(files, stratum)
    return CandidateUniverse(stratum.name, stratum.source, query, enumerated_at, files, actual)


def load_universe(stratum: Stratum) -> CandidateUniverse:
    """The stratum's frozen universe, checked against its pinned digest.

    Raises:
        CandidateUniverseError: If the stratum has not been listed, or its file's digest,
            keys, values, serialisation or membership do not hold.
    """
    pinned = UNIVERSE_SHA256.get(stratum.name)
    if pinned is None:
        _refuse(f"the {stratum.name} universe has not been listed and pinned yet")
    return _read_universe(universe_path(stratum.name), pinned, stratum)
