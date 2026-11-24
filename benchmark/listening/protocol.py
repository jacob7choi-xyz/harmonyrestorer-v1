"""Immutable, validated access to the frozen listening-study protocol.

`benchmark/listening/protocols/v2.json` is the study's frozen values. This module answers two
questions. Is the file well formed: every required key is present, no key is unknown, each
value has its type, and the values agree with one another. Is it the protocol this code was
written for: the file's own digest, the digest of the benchmark protocol it refers to, and the
digest of the universe manifest must each equal the value pinned in this module. The amendment
chain is checked as well: the predecessor's file and archived document must hash to the digests
the protocol records.

Explanatory text is carried in keys whose names end in `note` (and `mfcc_skip_reason`, which is
required). Those keys are checked to be strings and are not otherwise interpreted.

`load_listening_protocol` is the only supported way to read the protocol. It takes no path, so
a caller cannot point it at another file. `load_protocol_for_listing` is the one exception: it
does not check the universe digest, because the listing is the step that creates the universe
file that digest refers to. It is used by `benchmark.listening.universe` and by nothing else.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, NoReturn

from benchmark.listening.rule import UniverseRule

__all__ = [
    "PINNED_BENCHMARK_PROTOCOL_SHA256",
    "PINNED_PROTOCOL_SHA256",
    "PINNED_UNIVERSE_SHA256",
    "ListeningProtocol",
    "ListeningProtocolError",
    "load_listening_protocol",
    "load_protocol_for_listing",
]

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Private, so that the only supported route to the file is `load_listening_protocol`.
_PROTOCOL_PATH = _REPO_ROOT / "benchmark" / "listening" / "protocols" / "v2.json"
_STUDY = "historical_listening"
_VERSION = 2
# The predecessor that version 2 amends, named by path rather than read from the file, so the
# amendment chain cannot be redirected by editing the protocol file.
_PREDECESSOR_VERSION = 1
_PREDECESSOR_PROTOCOL_PATH = "benchmark/listening/protocols/v1.json"
_PREDECESSOR_DOCUMENT_PATH = "docs/listening-study-protocol-v1.md"
_BENCHMARK_PATH = "benchmark/protocols/v14.json"
_UNIVERSE_PATH = "benchmark/listening/manifests/universe_v1.json"
_SAMPLE_RATE_HZ = 16000
_CLIP_S = 8.0
_PRESENTATION_BITS = 16
_CONDITIONS = ("original", "opgan", "uvr")
_ANSWERS = ("A", "B", "No preference")
_MIN_ADULT_AGE = 18
_DIGEST = re.compile(r"[0-9a-f]{64}")
# The top-level keys of the protocol file, in file order.
_ROOT_KEYS = frozenset(
    {
        "study",
        "protocol_version",
        "amends",
        "benchmark_protocol",
        "source",
        "fetch",
        "universe",
        "selection",
        "clips",
        "conditions",
        "loudness",
        "trials",
        "listeners",
        "exclusion",
        "stages",
        "analysis",
        "publication",
    }
)

# Pinned digests. The protocol digest is the digest of v2.json's bytes, set last, once the file's
# content is final. The universe digest is set once the universe file is built, from its own
# bytes. The predecessor digests are not pinned here: v2.json records them under `amends`, and
# `_check_amends` recomputes them from the archived files.
PINNED_PROTOCOL_SHA256 = (
    "1b5c1d33aa4d64a9d35e2434f957338824ec19a63df958c5f61a259dbf55c89e"  # pragma: allowlist secret
)
PINNED_UNIVERSE_SHA256 = (
    "c21230d99ccc69c728da36913f428f3fe0647b2210a8b0d8b225c22eb2a83497"  # pragma: allowlist secret
)
PINNED_BENCHMARK_PROTOCOL_SHA256 = (
    "c381a0103c0ad8337c8abd3e5a3c0425757ddfa4d69325152345045983c3c2f2"  # pragma: allowlist secret
)


class ListeningProtocolError(ValueError):
    """Raised when the protocol file is malformed, inconsistent, or not the pinned protocol."""


@dataclass(frozen=True)
class BenchmarkReference:
    """The benchmark protocol this study refers to, and the digest of its bytes."""

    path: str
    sha256: str


@dataclass(frozen=True)
class Amends:
    """The predecessor this version amends, with the digests of its file and archived document.

    The loader recomputes both digests from the archived files, so the chain is checked rather
    than only recorded.
    """

    protocol_version: int
    protocol_path: str
    protocol_sha256: str
    document_path: str
    document_sha256: str
    reason: str


@dataclass(frozen=True)
class Source:
    """Where the candidate recordings are listed from."""

    collection: str
    search_query: str
    search_fields: tuple[str, ...]
    search_sort: str
    search_rows_per_page: int
    search_url: str
    metadata_url_template: str
    listing_date: date


@dataclass(frozen=True)
class Fetch:
    """How the listing and the metadata are requested."""

    user_agent: str
    timeout_s: float
    max_response_bytes: int
    pause_s_between_requests: float


@dataclass(frozen=True)
class Universe:
    """The universe manifest's location, pinned digest, and inclusion rule."""

    path: str
    sha256: str
    rule: UniverseRule
    rule_sources: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class Selection:
    """The seed, the number of recordings, and the rule that splits performer names.

    Attributes:
        seed: The study seed.
        sample_size: The number of recordings the walk takes.
        performer_split: The compiled patterns that split a listed performer name, in order.
    """

    seed: int
    sample_size: int
    performer_split: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class Clips:
    """The clip length, the margin at each end, and the sample rate of the presented audio."""

    duration_s: float
    margin_s: int
    sample_rate_hz: int


@dataclass(frozen=True)
class Loudness:
    """The loudness target and the sample-peak ceiling for the presented windows."""

    target_lkfs: float
    sample_peak_ceiling_dbfs: float
    target_step_lu: float
    presentation_bits: int


@dataclass(frozen=True)
class Trials:
    """The comparison design each listener completes."""

    pairs: tuple[tuple[str, str], ...]
    scored_trials_per_listener: int
    catch_trials_per_listener: int
    answers: tuple[str, ...]
    question: str


@dataclass(frozen=True)
class Listeners:
    """Who may take part and what they agree to."""

    minimum_age_years: int
    consent_text: str


@dataclass(frozen=True)
class Stages:
    """The pilot size and the threshold for the full study."""

    pilot_max_listeners: int
    full_study_minimum_listeners_total: int


@dataclass(frozen=True)
class Analysis:
    """The descriptive interval setting."""

    interval_level: float
    interval_iterations: int
    interval_seed: int
    interval_minimum_included_listeners: int


@dataclass(frozen=True)
class ListeningProtocol:
    """The whole frozen protocol, and the digest of the bytes it was read from.

    Attributes:
        sha256: The digest of the file's bytes, which is the file's identity.
    """

    sha256: str
    study: str
    protocol_version: int
    amends: Amends
    benchmark: BenchmarkReference
    source: Source
    fetch: Fetch
    universe: Universe
    selection: Selection
    clips: Clips
    loudness: Loudness
    trials: Trials
    listeners: Listeners
    stages: Stages
    analysis: Analysis


def _refuse(message: str) -> NoReturn:
    """Log *message* as an error and raise it as a ListeningProtocolError.

    Args:
        message: What is wrong with the protocol.

    Raises:
        ListeningProtocolError: Always.
    """
    logger.error("Listening protocol: %s", message)
    raise ListeningProtocolError(message)


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object from JSON pairs, refusing a repeated key rather than keeping the last.

    Args:
        pairs: The key and value pairs of one JSON object, in file order.

    Returns:
        The object.

    Raises:
        ListeningProtocolError: If a key appears more than once.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _refuse(f"duplicate key {key!r}")
        result[key] = value
    return result


def _exact(section: Mapping[str, Any], where: str, required: frozenset[str]) -> None:
    """Require exactly the *required* keys, plus any key whose name ends in `note`.

    Args:
        section: The JSON object to check.
        where: Its location in the file, for the message.
        required: Every key the section must have.

    Raises:
        ListeningProtocolError: On a missing key, an unknown key, or a non-string note.
    """
    notes = {key for key in section if key.endswith("note")}
    missing = sorted(required - set(section))
    unknown = sorted(set(section) - required - notes)
    if missing or unknown:
        _refuse(f"{where}: missing keys {missing}, unknown keys {unknown}")
    for key in notes:
        if not isinstance(section[key], str):
            _refuse(f"{where}.{key} must be a string")


def _get(section: Mapping[str, Any], where: str, key: str, kind: type) -> Any:
    """The value of *key*, checked to have type *kind* (bool is not accepted as an int).

    Args:
        section: The JSON object.
        where: Its location, for the message.
        key: The key to read.
        kind: The required Python type.

    Returns:
        The value.

    Raises:
        ListeningProtocolError: If the type is wrong.
    """
    value = section[key]
    is_bool_for_int = kind is int and isinstance(value, bool)
    if not isinstance(value, kind) or is_bool_for_int:
        _refuse(f"{where}.{key} must be of type {kind.__name__}")
    return value


def _float(section: Mapping[str, Any], where: str, key: str) -> float:
    """A JSON number as a float; an integer is accepted and converted.

    Args:
        section: The JSON object.
        where: Its location, for the message.
        key: The key to read.

    Returns:
        The value as a float.

    Raises:
        ListeningProtocolError: If the value is not a number.
    """
    value = section[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _refuse(f"{where}.{key} must be a number")
    return float(value)


def _strings(section: Mapping[str, Any], where: str, key: str) -> tuple[str, ...]:
    """A JSON array of strings.

    Args:
        section: The JSON object.
        where: Its location, for the message.
        key: The key to read.

    Returns:
        The strings, in order.

    Raises:
        ListeningProtocolError: If the value is not an array of strings.
    """
    value = section[key]
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _refuse(f"{where}.{key} must be an array of strings")
    return tuple(value)


def _date(section: Mapping[str, Any], where: str, key: str) -> date:
    """A YYYY-MM-DD date string.

    Args:
        section: The JSON object.
        where: Its location, for the message.
        key: The key to read.

    Returns:
        The date.

    Raises:
        ListeningProtocolError: If the string is not an ISO date.
    """
    text = _get(section, where, key, str)
    try:
        return date.fromisoformat(text)
    except ValueError:
        _refuse(f"{where}.{key} is not a YYYY-MM-DD date: {text!r}")


def _digest(section: Mapping[str, Any], where: str, key: str) -> str:
    """A lowercase SHA-256 hex digest.

    Args:
        section: The JSON object.
        where: Its location, for the message.
        key: The key to read.

    Returns:
        The digest.

    Raises:
        ListeningProtocolError: If the value is not 64 lowercase hexadecimal characters.
    """
    value: str = _get(section, where, key, str)
    if not _DIGEST.fullmatch(value):
        _refuse(f"{where}.{key} must be 64 lowercase hexadecimal characters")
    return value


def _compile(sources: tuple[str, ...], where: str) -> tuple[re.Pattern[str], ...]:
    """Compile regex sources with the protocol's flags.

    Args:
        sources: The pattern texts.
        where: Their location, for the message.

    Returns:
        The compiled patterns.

    Raises:
        ListeningProtocolError: If a pattern is empty or does not compile.
    """
    if not sources:
        _refuse(f"{where} must name at least one pattern")
    compiled: list[re.Pattern[str]] = []
    for source in sources:
        try:
            compiled.append(re.compile(source, re.IGNORECASE))
        except re.error as error:
            _refuse(f"{where} pattern does not compile: {error}")
    return tuple(compiled)


def _parse_benchmark(data: Mapping[str, Any]) -> BenchmarkReference:
    """The benchmark-protocol reference section.

    Args:
        data: The section.

    Returns:
        The reference, with its path and digest checked against the pins.
    """
    _exact(data, "benchmark_protocol", frozenset({"path", "sha256"}))
    path = _get(data, "benchmark_protocol", "path", str)
    if path != _BENCHMARK_PATH:
        _refuse(f"benchmark_protocol.path must be {_BENCHMARK_PATH}, got {path!r}")
    sha256 = _digest(data, "benchmark_protocol", "sha256")
    if sha256 != PINNED_BENCHMARK_PROTOCOL_SHA256:
        _refuse("benchmark_protocol.sha256 does not match the pinned benchmark protocol")
    return BenchmarkReference(path=path, sha256=sha256)


def _parse_amends(data: Mapping[str, Any]) -> Amends:
    """The amends section: the predecessor's version and paths, its digests, and the reason.

    Args:
        data: The section.

    Returns:
        The amendment record. Its paths are the frozen predecessor paths, and its digests are
        checked for form here and against the files on disk by `_check_amends`.

    Raises:
        ListeningProtocolError: If a key is missing or unknown, the version or a path is not the
            predecessor's, or the reason is empty.
    """
    keys = frozenset(
        {
            "protocol_version",
            "protocol_path",
            "protocol_sha256",
            "document_path",
            "document_sha256",
            "reason",
        }
    )
    _exact(data, "amends", keys)
    version = _get(data, "amends", "protocol_version", int)
    if version != _PREDECESSOR_VERSION:
        _refuse(f"amends.protocol_version must be {_PREDECESSOR_VERSION}, got {version}")
    for key, frozen in (
        ("protocol_path", _PREDECESSOR_PROTOCOL_PATH),
        ("document_path", _PREDECESSOR_DOCUMENT_PATH),
    ):
        if _get(data, "amends", key, str) != frozen:
            _refuse(f"amends.{key} must be {frozen!r}; the amendment chain is not redirectable")
    reason = _get(data, "amends", "reason", str)
    if not reason.strip():
        _refuse("amends.reason must not be empty")
    return Amends(
        protocol_version=version,
        protocol_path=_PREDECESSOR_PROTOCOL_PATH,
        protocol_sha256=_digest(data, "amends", "protocol_sha256"),
        document_path=_PREDECESSOR_DOCUMENT_PATH,
        document_sha256=_digest(data, "amends", "document_sha256"),
        reason=reason,
    )


def _parse_source(data: Mapping[str, Any]) -> Source:
    """The source section.

    Args:
        data: The section.

    Returns:
        The source description.
    """
    keys = frozenset(
        {
            "collection",
            "search_query",
            "search_fields",
            "search_sort",
            "search_rows_per_page",
            "search_url",
            "metadata_url_template",
            "listing_date",
        }
    )
    _exact(data, "source", keys)
    rows = _get(data, "source", "search_rows_per_page", int)
    if not 1 <= rows <= 1000:
        _refuse(f"source.search_rows_per_page must be from 1 to 1000, got {rows}")
    if _get(data, "source", "search_sort", str) != "identifier asc":
        _refuse("source.search_sort must be 'identifier asc'")
    return Source(
        collection=_get(data, "source", "collection", str),
        search_query=_get(data, "source", "search_query", str),
        search_fields=_strings(data, "source", "search_fields"),
        search_sort=_get(data, "source", "search_sort", str),
        search_rows_per_page=rows,
        search_url=_get(data, "source", "search_url", str),
        metadata_url_template=_get(data, "source", "metadata_url_template", str),
        listing_date=_date(data, "source", "listing_date"),
    )


def _parse_fetch(data: Mapping[str, Any]) -> Fetch:
    """The fetch section.

    Args:
        data: The section.

    Returns:
        The request settings.
    """
    keys = frozenset({"user_agent", "timeout_s", "max_response_bytes", "pause_s_between_requests"})
    _exact(data, "fetch", keys)
    timeout = _float(data, "fetch", "timeout_s")
    max_bytes = _get(data, "fetch", "max_response_bytes", int)
    pause = _float(data, "fetch", "pause_s_between_requests")
    if timeout <= 0 or max_bytes <= 0 or pause < 0:
        _refuse("fetch timeout and byte cap must be positive, and the pause non-negative")
    return Fetch(
        user_agent=_get(data, "fetch", "user_agent", str),
        timeout_s=timeout,
        max_response_bytes=max_bytes,
        pause_s_between_requests=pause,
    )


def _parse_rule(data: Mapping[str, Any]) -> tuple[UniverseRule, dict[str, tuple[str, ...]]]:
    """The universe rule section, compiled, with its raw sources kept for the manifest.

    Args:
        data: The `rule` object.

    Returns:
        The compiled rule and the sources it was compiled from, keyed by field name.
    """
    keys = frozenset(
        {
            "require_instrumentation_line",
            "require_main_flac_count",
            "keep_if_instrumentation_matches",
            "drop_if_instrumentation_or_title_matches_vocal",
            "drop_if_instrumentation_or_title_matches_accordion",
            "flags",
        }
    )
    _exact(data, "universe.rule", keys)
    if _get(data, "universe.rule", "require_instrumentation_line", bool) is not True:
        _refuse("universe.rule.require_instrumentation_line must be true")
    if _get(data, "universe.rule", "require_main_flac_count", int) != 1:
        _refuse("universe.rule.require_main_flac_count must be 1")
    if _strings(data, "universe.rule", "flags") != ("ignorecase",):
        _refuse("universe.rule.flags must be exactly ['ignorecase']")
    sources = {
        "keep_if_instrumentation_matches": _strings(
            data, "universe.rule", "keep_if_instrumentation_matches"
        ),
        "drop_if_instrumentation_or_title_matches_vocal": _strings(
            data, "universe.rule", "drop_if_instrumentation_or_title_matches_vocal"
        ),
        "drop_if_instrumentation_or_title_matches_accordion": _strings(
            data, "universe.rule", "drop_if_instrumentation_or_title_matches_accordion"
        ),
    }
    rule = UniverseRule(
        keep_patterns=_compile(
            sources["keep_if_instrumentation_matches"], "keep_if_instrumentation_matches"
        ),
        vocal_patterns=_compile(sources["drop_if_instrumentation_or_title_matches_vocal"], "vocal"),
        accordion_patterns=_compile(
            sources["drop_if_instrumentation_or_title_matches_accordion"], "accordion"
        ),
        required_flac_count=1,
    )
    return rule, sources


def _parse_universe(data: Mapping[str, Any]) -> Universe:
    """The universe section.

    Args:
        data: The section.

    Returns:
        The universe location, digest and rule.
    """
    _exact(data, "universe", frozenset({"path", "sha256", "rule"}))
    if _get(data, "universe", "path", str) != _UNIVERSE_PATH:
        _refuse(f"universe.path must be {_UNIVERSE_PATH}")
    sha256 = _digest(data, "universe", "sha256")
    rule, sources = _parse_rule(_get(data, "universe", "rule", dict))
    return Universe(path=_UNIVERSE_PATH, sha256=sha256, rule=rule, rule_sources=sources)


def _parse_selection(data: Mapping[str, Any]) -> Selection:
    """The selection section.

    Args:
        data: The section.

    Returns:
        The seed and sample size.
    """
    keys = frozenset(
        {
            "seed",
            "sample_size",
            "order",
            "performer_rule",
            "performer_split_patterns",
            "performer_split_note",
            "replacement_rule",
            "item_with_no_valid_window_rule",
        }
    )
    _exact(data, "selection", keys)
    seed = _get(data, "selection", "seed", int)
    size = _get(data, "selection", "sample_size", int)
    if seed < 0 or size < 1:
        _refuse("selection.seed must be non-negative and sample_size at least 1")
    split = _compile(
        _strings(data, "selection", "performer_split_patterns"), "performer_split_patterns"
    )
    return Selection(seed=seed, sample_size=size, performer_split=split)


def _parse_clips(data: Mapping[str, Any]) -> Clips:
    """The clips section.

    Args:
        data: The section.

    Returns:
        The clip geometry.
    """
    keys = frozenset(
        {
            "duration_s",
            "margin_s",
            "sample_rate_hz",
            "duration_source",
            "start_range_whole_seconds",
            "seed_rule",
            "one_window_per_recording",
        }
    )
    _exact(data, "clips", keys)
    duration = _float(data, "clips", "duration_s")
    margin = _get(data, "clips", "margin_s", int)
    rate = _get(data, "clips", "sample_rate_hz", int)
    if duration != _CLIP_S or margin < 0 or rate != _SAMPLE_RATE_HZ:
        _refuse("clips: duration_s must be 8.0, margin_s non-negative, sample_rate_hz 16000")
    if _get(data, "clips", "one_window_per_recording", bool) is not True:
        _refuse("clips.one_window_per_recording must be true")
    return Clips(duration_s=duration, margin_s=margin, sample_rate_hz=rate)


def _parse_conditions(data: Mapping[str, Any]) -> None:
    """The conditions section, checked for its names and its text keys only.

    Args:
        data: The section.
    """
    keys = frozenset({"names", "original", "opgan", "uvr", "presentation", "same_window_rule"})
    _exact(data, "conditions", keys)
    if _strings(data, "conditions", "names") != _CONDITIONS:
        _refuse(f"conditions.names must be {list(_CONDITIONS)}")


def _parse_loudness(data: Mapping[str, Any]) -> Loudness:
    """The loudness section.

    Args:
        data: The section.

    Returns:
        The loudness settings.
    """
    keys = frozenset(
        {
            "meter",
            "measured_on",
            "target_lkfs",
            "sample_peak_ceiling_dbfs",
            "target_step_lu",
            "target_lowering_rule",
            "presentation_format",
        }
    )
    _exact(data, "loudness", keys)
    target = _float(data, "loudness", "target_lkfs")
    ceiling = _float(data, "loudness", "sample_peak_ceiling_dbfs")
    step = _float(data, "loudness", "target_step_lu")
    if ceiling > 0 or step <= 0 or target >= 0:
        _refuse("loudness: target must be negative, ceiling at most 0 dBFS, step positive")
    return Loudness(
        target_lkfs=target,
        sample_peak_ceiling_dbfs=ceiling,
        target_step_lu=step,
        presentation_bits=_PRESENTATION_BITS,
    )


def _parse_trials(data: Mapping[str, Any], sample_size: int) -> Trials:
    """The trials section, with its pairs checked against the conditions and the clip count.

    Args:
        data: The section.
        sample_size: The number of clips, from the selection section.

    Returns:
        The trial design.

    Raises:
        ListeningProtocolError: If a pair is not two distinct conditions, or the scored trial
            count is not pairs times clips.
    """
    keys = frozenset(
        {
            "pairs",
            "scored_trials_per_listener",
            "catch_trials_per_listener",
            "catch_trial_rule",
            "question",
            "answers",
            "comment",
            "order_rule",
        }
    )
    _exact(data, "trials", keys)
    raw_pairs = data["pairs"]
    if not isinstance(raw_pairs, list) or not all(
        isinstance(pair, list) and len(pair) == 2 and all(isinstance(c, str) for c in pair)
        for pair in raw_pairs
    ):
        _refuse("trials.pairs must be a list of two-condition pairs")
    pairs = tuple((pair[0], pair[1]) for pair in raw_pairs)
    for left, right in pairs:
        if left == right or {left, right} - set(_CONDITIONS):
            _refuse(f"trials.pairs holds an invalid pair {left!r}, {right!r}")
    if _strings(data, "trials", "answers") != _ANSWERS:
        _refuse(f"trials.answers must be {list(_ANSWERS)}")
    scored = _get(data, "trials", "scored_trials_per_listener", int)
    catch = _get(data, "trials", "catch_trials_per_listener", int)
    if scored != len(pairs) * sample_size or catch != 2:
        _refuse("trials: scored trials must equal pairs times clips, with two catch trials")
    return Trials(
        pairs=pairs,
        scored_trials_per_listener=scored,
        catch_trials_per_listener=catch,
        answers=_ANSWERS,
        question=_get(data, "trials", "question", str),
    )


def _parse_listeners(data: Mapping[str, Any]) -> Listeners:
    """The listeners section.

    Args:
        data: The section.

    Returns:
        The eligibility age and the consent text.
    """
    keys = frozenset(
        {
            "minimum_age_years",
            "self_reported_hearing",
            "playback",
            "excluded_from_scored_panel",
            "identifier",
            "consent_text",
        }
    )
    _exact(data, "listeners", keys)
    age = _get(data, "listeners", "minimum_age_years", int)
    if age < _MIN_ADULT_AGE:
        _refuse(f"listeners.minimum_age_years must be at least {_MIN_ADULT_AGE}")
    consent = _get(data, "listeners", "consent_text", str)
    if not consent.strip() or consent.count(".") != 1:
        _refuse("listeners.consent_text must be a single sentence")
    return Listeners(minimum_age_years=age, consent_text=consent)


def _parse_stages(data: Mapping[str, Any]) -> Stages:
    """The stages section.

    Args:
        data: The section.

    Returns:
        The pilot size and the full-study threshold.
    """
    _exact(data, "stages", frozenset({"pilot", "full_study", "pooling_rule"}))
    pilot = _get(data, "stages", "pilot", dict)
    _exact(pilot, "stages.pilot", frozenset({"max_listeners", "definition"}))
    full = _get(data, "stages", "full_study", dict)
    _exact(
        full,
        "stages.full_study",
        frozenset({"minimum_listeners_total", "count_basis", "decision_rule"}),
    )
    pilot_max = _get(pilot, "stages.pilot", "max_listeners", int)
    full_min = _get(full, "stages.full_study", "minimum_listeners_total", int)
    if pilot_max < 1 or full_min < pilot_max:
        _refuse(
            "stages: the pilot must hold at least one listener, and the full-study threshold must not be below it"
        )
    return Stages(pilot_max_listeners=pilot_max, full_study_minimum_listeners_total=full_min)


def _parse_analysis(data: Mapping[str, Any]) -> Analysis:
    """The analysis section.

    Args:
        data: The section.

    Returns:
        The interval settings.
    """
    keys = frozenset(
        {
            "descriptive_only",
            "reported_per_pair",
            "reported_per_listener",
            "reported_per_clip",
            "interval",
            "reporting_rules",
        }
    )
    _exact(data, "analysis", keys)
    if _get(data, "analysis", "descriptive_only", bool) is not True:
        _refuse("analysis.descriptive_only must be true")
    interval = _get(data, "analysis", "interval", dict)
    _exact(
        interval,
        "analysis.interval",
        frozenset({"method", "level", "iterations", "seed", "minimum_included_listeners", "label"}),
    )
    level = _float(interval, "analysis.interval", "level")
    iterations = _get(interval, "analysis.interval", "iterations", int)
    minimum = _get(interval, "analysis.interval", "minimum_included_listeners", int)
    if not 0 < level < 1 or iterations < 1 or minimum < 1:
        _refuse("analysis.interval: level must be between 0 and 1, and counts at least 1")
    return Analysis(
        interval_level=level,
        interval_iterations=iterations,
        interval_seed=_get(interval, "analysis.interval", "seed", int),
        interval_minimum_included_listeners=minimum,
    )


def _parse_publication(data: Mapping[str, Any]) -> None:
    """The publication section, checked for its keys only.

    Args:
        data: The section.
    """
    _exact(data, "publication", frozenset({"published", "great78_audio", "training_overlap"}))
    overlap = _get(data, "publication", "training_overlap", dict)
    _exact(
        overlap,
        "publication.training_overlap",
        frozenset({"raw_digest_check", "decoded_digest_check", "mfcc_screen", "mfcc_skip_reason"}),
    )


def _parse_document(data: Mapping[str, Any], sha256: str) -> ListeningProtocol:
    """Validate the whole document and build the protocol. Digests are checked by the caller.

    Args:
        data: The parsed JSON object.
        sha256: The digest of the bytes *data* was parsed from.

    Returns:
        The protocol.

    Raises:
        ListeningProtocolError: On any malformed or inconsistent value.
    """
    _exact(data, "protocol", _ROOT_KEYS)
    if _get(data, "protocol", "study", str) != _STUDY:
        _refuse(f"study must be {_STUDY!r}")
    if _get(data, "protocol", "protocol_version", int) != _VERSION:
        _refuse(f"only protocol version {_VERSION} is supported")
    selection = _parse_selection(_get(data, "protocol", "selection", dict))
    _parse_conditions(_get(data, "protocol", "conditions", dict))
    _exact(
        _get(data, "protocol", "exclusion", dict),
        "exclusion",
        frozenset({"catch_trial_rule", "incomplete_rule", "reporting"}),
    )
    _parse_publication(_get(data, "protocol", "publication", dict))
    return ListeningProtocol(
        sha256=sha256,
        study=_STUDY,
        protocol_version=_VERSION,
        amends=_parse_amends(_get(data, "protocol", "amends", dict)),
        benchmark=_parse_benchmark(_get(data, "protocol", "benchmark_protocol", dict)),
        source=_parse_source(_get(data, "protocol", "source", dict)),
        fetch=_parse_fetch(_get(data, "protocol", "fetch", dict)),
        universe=_parse_universe(_get(data, "protocol", "universe", dict)),
        selection=selection,
        clips=_parse_clips(_get(data, "protocol", "clips", dict)),
        loudness=_parse_loudness(_get(data, "protocol", "loudness", dict)),
        trials=_parse_trials(_get(data, "protocol", "trials", dict), selection.sample_size),
        listeners=_parse_listeners(_get(data, "protocol", "listeners", dict)),
        stages=_parse_stages(_get(data, "protocol", "stages", dict)),
        analysis=_parse_analysis(_get(data, "protocol", "analysis", dict)),
    )


def _read_bytes() -> bytes:
    """The protocol file's bytes, read once.

    Returns:
        The bytes of the file.
    """
    return _PROTOCOL_PATH.read_bytes()


def _parse_bytes(raw: bytes) -> ListeningProtocol:
    """Parse and validate protocol bytes, without checking the digests against any pin.

    Args:
        raw: The file's bytes.

    Returns:
        The protocol.

    Raises:
        ListeningProtocolError: If the bytes are not one valid protocol.
    """
    try:
        data = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except json.JSONDecodeError as error:
        _refuse(f"the protocol file is not valid JSON: {error}")
    if not isinstance(data, dict):
        _refuse("the protocol file must hold one JSON object")
    return _parse_document(data, hashlib.sha256(raw).hexdigest())


def _check_pins(protocol: ListeningProtocol) -> None:
    """Refuse a protocol whose file or universe digest is not the pinned one.

    Args:
        protocol: The parsed protocol, which carries the digest of its file.

    Raises:
        ListeningProtocolError: On any mismatch.
    """
    if protocol.sha256 != PINNED_PROTOCOL_SHA256:
        _refuse("the protocol file's digest does not match the pinned protocol digest")
    if protocol.universe.sha256 != PINNED_UNIVERSE_SHA256:
        _refuse("universe.sha256 in the protocol file does not match the pinned universe digest")


def _check_benchmark_file(benchmark: BenchmarkReference) -> None:
    """Refuse when the benchmark protocol file on disk is not the one the study pins.

    Args:
        benchmark: The reference, with the digest the study pins.

    Raises:
        ListeningProtocolError: If the file is missing or its digest differs.
    """
    path = _REPO_ROOT / benchmark.path
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        logger.error("Listening protocol: cannot read %s: %s", benchmark.path, error)
        raise ListeningProtocolError(f"cannot read {benchmark.path}") from error
    if actual != benchmark.sha256:
        _refuse(f"{benchmark.path} on disk does not match the digest this study pins")


def _file_sha256(rel_path: str) -> str:
    """The SHA-256 of a repository file, read from the path relative to the repository root.

    Args:
        rel_path: The repository-relative path.

    Returns:
        The lowercase hexadecimal digest of the file's bytes.

    Raises:
        ListeningProtocolError: If the file cannot be read.
    """
    try:
        return hashlib.sha256((_REPO_ROOT / rel_path).read_bytes()).hexdigest()
    except OSError as error:
        logger.error("Listening protocol: cannot read %s: %s", rel_path, error)
        raise ListeningProtocolError(f"cannot read {rel_path}") from error


def _check_amends(amends: Amends) -> None:
    """Refuse an amendment chain whose predecessor files differ from the digests it records.

    Args:
        amends: The amendment record, whose digests are the ones the protocol file states.

    Raises:
        ListeningProtocolError: If a predecessor file is missing or its digest differs.
    """
    for label, rel_path, recorded in (
        ("protocol", amends.protocol_path, amends.protocol_sha256),
        ("document", amends.document_path, amends.document_sha256),
    ):
        actual = _file_sha256(rel_path)
        if actual != recorded:
            _refuse(
                f"amends.{label}_sha256 records {recorded[:12]} but {rel_path} hashes to "
                f"{actual[:12]}; the archived predecessor has been modified"
            )


def load_listening_protocol() -> ListeningProtocol:
    """The frozen listening protocol, with every digest checked.

    Returns:
        The protocol. Its `sha256` is the digest of the file's bytes.

    Raises:
        ListeningProtocolError: If the file is malformed, or any pinned digest does not match.
    """
    raw = _read_bytes()
    protocol = _parse_bytes(raw)
    _check_pins(protocol)
    _check_benchmark_file(protocol.benchmark)
    _check_amends(protocol.amends)
    return protocol


def load_protocol_for_listing() -> ListeningProtocol:
    """The protocol for building the universe, with every digest except the universe's checked.

    The universe digest cannot be checked before the universe exists, so this is the one
    caller that skips that check. It still checks the protocol file, the benchmark file and the
    amendment chain.

    Returns:
        The protocol.

    Raises:
        ListeningProtocolError: If the file is malformed or its own digest or the benchmark
            digest does not match.
    """
    raw = _read_bytes()
    protocol = _parse_bytes(raw)
    if protocol.sha256 != PINNED_PROTOCOL_SHA256:
        _refuse("the protocol file's digest does not match the pinned protocol digest")
    _check_benchmark_file(protocol.benchmark)
    _check_amends(protocol.amends)
    return protocol
