"""Reading an Internet Archive item's metadata into the facts the listening universe needs.

The parsing here follows the one used for the 2026-11-24 candidate listing, so the
instrumentation line and the performer names mean the same thing in the frozen universe as
they did in that listing. Only the parsing is carried over. The inclusion decision is made by
`benchmark.listening.rule`, which reads the facts this module returns.

Everything here is a pure function of the metadata it is given. No network access happens
in this module.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict

__all__ = [
    "ItemFacts",
    "ParsedDescription",
    "as_text",
    "item_facts",
    "main_flac",
    "parse_description",
    "performer_key",
    "split_performer_names",
    "strip_html",
    "year_of",
]

_FLAC_SOURCE = "original"
_FLAC_FORMAT = "24bit Flac"
_LABEL_RE = re.compile(r"^(Performer|Writer):\s*(.*)$")
# Description lines that name the digitisation or the matrix rather than the music.
_BOILERPLATE_PREFIXES = (
    "Digitized at",
    "Digitized from",
    "We were unable",
    "The preferred version",
    "Matrix number:",
    "Catalog number:",
    "Other IDs from the record include:",
)
_BREAK_RE = re.compile(r"<\s*br\s*/?\s*>|</\s*(p|div|li|tr)\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_YEAR_RE = re.compile(r"(\d{4})")


@dataclass(frozen=True)
class ParsedDescription:
    """The labelled names and the free instrumentation lines of one description.

    Attributes:
        performers: Names from the Performer lines, in the order given.
        writers: Names from the Writer lines, in the order given.
        instrumentation_lines: Every other line that is not boilerplate, in order.
    """

    performers: tuple[str, ...]
    writers: tuple[str, ...]
    instrumentation_lines: tuple[str, ...]


@dataclass(frozen=True)
class ItemFacts:
    """The facts about one item that the universe rule and the selection read.

    Attributes:
        identifier: The Internet Archive identifier.
        title: The title, or an empty string if the item has none.
        year: The first four-digit year in the date, or None.
        performers: Performer names, or the creator when no Performer line exists.
        performer_source: "description", "creator_fallback" or "none", where `performers` came from.
        instrumentation: The instrumentation lines joined with " | ", or "" if there are none.
        flac_count: How many files match the main-FLAC test.
        flac_name: The name of the single main FLAC, or None unless `flac_count` is 1.
        flac_size: Its listed size in bytes, or None.
        flac_md5: Its listed md5, or None.
        flac_sha1: Its listed sha1, or None.
        length_s: The duration the archive lists for the main FLAC, or None.
    """

    identifier: str
    title: str
    year: int | None
    performers: tuple[str, ...]
    performer_source: str
    instrumentation: str
    flac_count: int
    flac_name: str | None
    flac_size: int | None
    flac_md5: str | None
    flac_sha1: str | None
    length_s: float | None


def as_text(value: object) -> str:
    """Join a metadata value that may be a string, a list of strings, or absent.

    Args:
        value: The raw metadata value.

    Returns:
        The value as one string. A list is joined with " | ", and None becomes "".
    """
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(str(part) for part in value)
    return str(value)


def strip_html(raw: str) -> tuple[str, ...]:
    """Convert an HTML description to its non-blank plain-text lines.

    Line breaks and block-level closing tags become newlines before the tags are removed,
    so each line of the description keeps its own text.

    Args:
        raw: The description as the archive stores it.

    Returns:
        The stripped, non-blank lines, in order.
    """
    text = html.unescape(_TAG_RE.sub("", _BREAK_RE.sub("\n", raw)))
    return tuple(line.strip() for line in text.split("\n") if line.strip())


def _split_names(value: str) -> tuple[str, ...]:
    """Split a labelled value on semicolons into its non-blank names.

    Args:
        value: The text after a Performer or Writer label.

    Returns:
        The names, stripped, in order.
    """
    return tuple(part.strip() for part in value.split(";") if part.strip())


def parse_description(raw: str) -> ParsedDescription:
    """Split one description into its performers, writers and instrumentation lines.

    A line that starts with a boilerplate prefix is neither a name nor an instrumentation
    line, and is dropped.

    Args:
        raw: The description as the archive stores it.

    Returns:
        The parsed description.
    """
    performers: list[str] = []
    writers: list[str] = []
    instrumentation: list[str] = []
    for line in strip_html(raw):
        label = _LABEL_RE.match(line)
        if label is not None:
            target = performers if label.group(1) == "Performer" else writers
            target.extend(_split_names(label.group(2)))
        elif not line.startswith(_BOILERPLATE_PREFIXES):
            instrumentation.append(line)
    return ParsedDescription(tuple(performers), tuple(writers), tuple(instrumentation))


def split_performer_names(
    names: Sequence[str], patterns: Sequence[re.Pattern[str]]
) -> tuple[str, ...]:
    """Split listed performer names into single performers, pattern by pattern.

    Each pattern is applied to every part the previous patterns produced, so the patterns apply
    in list order. Parts are stripped, and empty parts are dropped.

    Args:
        names: The listed performer names, as the universe manifest stores them.
        patterns: The split patterns, in the order the protocol gives them.

    Returns:
        The performer parts, in order.
    """
    parts = list(names)
    for pattern in patterns:
        parts = [part for piece in parts for part in pattern.split(piece)]
    return tuple(stripped for stripped in (part.strip() for part in parts) if stripped)


def performer_key(name: str) -> str:
    """The comparison key for a performer name: casefolded, with whitespace collapsed.

    Accents and punctuation are kept, so two names differing only in them stay distinct.

    Args:
        name: A performer name as listed.

    Returns:
        The key. Two names are the same performer for selection when their keys are equal.
    """
    return " ".join(name.casefold().split())


def _is_main_flac_candidate(entry: Mapping[str, object]) -> bool:
    """Whether a listed file is an original 24-bit FLAC at the top level of the item.

    Args:
        entry: One file entry from the item's metadata.

    Returns:
        True when the file is an original 24-bit FLAC whose name has no directory part.
    """
    name = as_text(entry.get("name"))
    return (
        entry.get("source") == _FLAC_SOURCE
        and entry.get("format") == _FLAC_FORMAT
        and "/" not in name
        and name.lower().endswith(".flac")
    )


def main_flac(files: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object] | None, int]:
    """The single main FLAC entry, or None, with the number of candidates found.

    Args:
        files: The file entries from the item's metadata.

    Returns:
        The entry when exactly one file is a candidate, otherwise None, and the count.
    """
    candidates = [entry for entry in files if _is_main_flac_candidate(entry)]
    return (candidates[0] if len(candidates) == 1 else None), len(candidates)


def year_of(date_raw: str) -> int | None:
    """The first four-digit number in a date string, as a year.

    Args:
        date_raw: The archive's date field.

    Returns:
        The year, or None when the string has no four digits.
    """
    match = _YEAR_RE.search(date_raw)
    return int(match.group(1)) if match else None


def _optional_int(value: object) -> int | None:
    """A listed byte size as an int, or None when the archive gave none.

    Args:
        value: The raw size, which the archive gives as a decimal string.

    Returns:
        The size, or None when it is absent or empty.
    """
    return int(str(value)) if value not in (None, "") else None


def _optional_float(value: object) -> float | None:
    """A listed duration as a float, or None when the archive gave none.

    Args:
        value: The raw duration, which the archive gives as a decimal string.

    Returns:
        The duration, or None when it is absent or empty.
    """
    return float(str(value)) if value not in (None, "") else None


def _performers(parsed: ParsedDescription, metadata: Mapping[str, object]) -> tuple[str, ...]:
    """Performer names from the description, or the creator when there are none.

    Args:
        parsed: The parsed description.
        metadata: The item's metadata, which may carry a creator.

    Returns:
        The performer names, or an empty tuple when neither source has one.
    """
    if parsed.performers:
        return parsed.performers
    creator = as_text(metadata.get("creator"))
    return (creator,) if creator else ()


def item_facts(
    identifier: str,
    metadata: Mapping[str, object],
    files: Sequence[Mapping[str, object]],
) -> ItemFacts:
    """The universe facts for one item, from its metadata and its file list.

    Args:
        identifier: The item's identifier, as the listing gave it.
        metadata: The item's metadata object.
        files: The item's file entries.

    Returns:
        The facts. A missing field is recorded as None or an empty value, and no rule is
        applied here.
    """
    parsed = parse_description(as_text(metadata.get("description")))
    performers = _performers(parsed, metadata)
    flac, count = main_flac(files)
    return ItemFacts(
        identifier=identifier,
        title=as_text(metadata.get("title")),
        year=year_of(as_text(metadata.get("date"))),
        performers=performers,
        performer_source=(
            "description" if parsed.performers else ("creator_fallback" if performers else "none")
        ),
        instrumentation=" | ".join(parsed.instrumentation_lines),
        flac_count=count,
        **_flac_fields(flac),
    )


class _FlacFields(TypedDict):
    """The main-FLAC fields of ItemFacts, passed to its constructor."""

    flac_name: str | None
    flac_size: int | None
    flac_md5: str | None
    flac_sha1: str | None
    length_s: float | None


def _flac_fields(flac: Mapping[str, object] | None) -> _FlacFields:
    """The main-FLAC fields of ItemFacts; all None when there is no single main FLAC.

    Args:
        flac: The main FLAC entry, or None.

    Returns:
        Keyword arguments for the flac_* and length_s fields of ItemFacts.
    """
    if flac is None:
        return {
            "flac_name": None,
            "flac_size": None,
            "flac_md5": None,
            "flac_sha1": None,
            "length_s": None,
        }
    return {
        "flac_name": as_text(flac.get("name")),
        "flac_size": _optional_int(flac.get("size")),
        "flac_md5": as_text(flac.get("md5")) or None,
        "flac_sha1": as_text(flac.get("sha1")) or None,
        "length_s": _optional_float(flac.get("length")),
    }
