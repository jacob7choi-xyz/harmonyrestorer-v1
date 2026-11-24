"""The pinned universe manifest: its item type, its serialisation, and its verified reader.

The pinned universe manifest lists every recording the rule kept, with the
facts the selection reads, and records each excluded item with its reason. Reading it is
pure: this module checks the file's digest against the pinned value, then parses it with
exact keys. It makes no network request.

The serialisation is indented, with sorted keys and the default separators. The `md5` and
`sha1` keys are written only as the exact names the secrets gate exempts, which is why a
listed file's digests sit under a `flac` object and carry those two names.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from benchmark.listening.protocol import PINNED_UNIVERSE_SHA256, ListeningProtocol

__all__ = [
    "UniverseError",
    "UniverseItem",
    "manifest_bytes",
    "read_universe",
    "universe_path",
]

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TOP_KEYS = frozenset(
    {
        "study",
        "protocol_version",
        "listing_date",
        "listed_at_utc",
        "query",
        "sort",
        "num_found",
        "listed",
        "kept",
        "excluded_by_reason",
        "items",
        "excluded",
    }
)
_ITEM_KEYS = frozenset(
    {
        "identifier",
        "title",
        "year",
        "performers",
        "performer_source",
        "instrumentation",
        "rule",
        "flac",
    }
)
_FLAC_KEYS = frozenset({"name", "size", "md5", "sha1", "length_s"})


class UniverseError(ValueError):
    """Raised when a universe manifest is malformed, or its digest is not the pinned one."""


@dataclass(frozen=True)
class UniverseItem:
    """One kept recording, as the selection needs it.

    Attributes:
        identifier: The Internet Archive identifier.
        title: The title.
        year: The year, or None.
        performers: Performer names, in listed order.
        instrumentation: The instrumentation line the rule read.
        flac_name: The name of the single main FLAC.
        flac_size: Its listed size in bytes.
        flac_md5: Its listed md5.
        flac_sha1: Its listed sha1.
        length_s: The duration the archive lists for the FLAC, or None.
    """

    identifier: str
    title: str
    year: int | None
    performers: tuple[str, ...]
    instrumentation: str
    flac_name: str
    flac_size: int
    flac_md5: str
    flac_sha1: str
    length_s: float | None


def universe_path(protocol: ListeningProtocol) -> Path:
    """The absolute path of the universe manifest the protocol names.

    Args:
        protocol: The protocol.

    Returns:
        The manifest path under the repository root.
    """
    return _REPO_ROOT / protocol.universe.path


def manifest_bytes(data: Mapping[str, Any]) -> bytes:
    """The one serialisation the manifest is written and verified in.

    Args:
        data: The manifest object.

    Returns:
        UTF-8 JSON with sorted keys, two-space indentation and a final newline.
    """
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _refuse(message: str) -> NoReturn:
    """Log *message* as an error and raise it as a UniverseError.

    Args:
        message: What is wrong with the manifest.

    Raises:
        UniverseError: Always.
    """
    logger.error("Listening universe: %s", message)
    raise UniverseError(message)


def _item(raw: Mapping[str, Any]) -> UniverseItem:
    """Build one item from its manifest object, with exact keys and types.

    Args:
        raw: One entry of the manifest's `items` array.

    Returns:
        The item.

    Raises:
        UniverseError: On a missing or unknown key, or a wrong type.
    """
    if set(raw) != _ITEM_KEYS:
        _refuse(f"item keys are {sorted(raw)}, expected {sorted(_ITEM_KEYS)}")
    flac = raw["flac"]
    if not isinstance(flac, dict) or set(flac) != _FLAC_KEYS:
        _refuse(f"item {raw['identifier']!r} has a malformed flac object")
    performers = raw["performers"]
    if not isinstance(performers, list) or not all(isinstance(p, str) for p in performers):
        _refuse(f"item {raw['identifier']!r} has malformed performers")
    length = flac["length_s"]
    return UniverseItem(
        identifier=str(raw["identifier"]),
        title=str(raw["title"]),
        year=raw["year"],
        performers=tuple(performers),
        instrumentation=str(raw["instrumentation"]),
        flac_name=str(flac["name"]),
        flac_size=int(flac["size"]),
        flac_md5=str(flac["md5"]),
        flac_sha1=str(flac["sha1"]),
        length_s=None if length is None else float(length),
    )


def read_universe(
    protocol: ListeningProtocol, raw: bytes | None = None
) -> tuple[UniverseItem, ...]:
    """The kept items of the pinned universe manifest, verified against its digest.

    Args:
        protocol: The protocol, whose `universe.sha256` must equal the code pin.
        raw: The manifest's bytes. None reads the file at the protocol's path.

    Returns:
        The kept items, in the manifest's order (sorted by identifier).

    Raises:
        UniverseError: If the digest is not the pinned one, or the object is malformed.
    """
    data_bytes = universe_path(protocol).read_bytes() if raw is None else raw
    digest = hashlib.sha256(data_bytes).hexdigest()
    if digest != PINNED_UNIVERSE_SHA256 or protocol.universe.sha256 != digest:
        _refuse("the universe manifest's digest does not match the pinned digest")
    data = json.loads(data_bytes)
    if not isinstance(data, dict) or set(data) != _TOP_KEYS:
        _refuse("the universe manifest has an unexpected top-level shape")
    items = data["items"]
    if not isinstance(items, list) or len(items) != data["kept"]:
        _refuse("the universe manifest's kept count does not match its items")
    return tuple(_item(entry) for entry in items)
