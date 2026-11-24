"""Building the listening universe from the Internet Archive listing, once, and writing it.

The build has three steps. The listing reads every page of the collection search, sorted by
identifier, and refuses a listing whose page count disagrees with its own total or that repeats
an identifier. Each listed item's metadata is then read and turned into facts. The rule in
`benchmark.listening.rule` decides each item, and the manifest records every kept item and every
exclusion with its reason.

The network is used for the metadata only. No audio is requested. The one call that writes the
file refuses to overwrite it, so the universe is created once and afterwards only verified.

Run as `python -m benchmark.listening.universe` from the repository root.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import time
import urllib.parse
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, NoReturn

from benchmark import https_get
from benchmark.atomic_write import write_atomically
from benchmark.listening.manifest import manifest_bytes, universe_path
from benchmark.listening.metadata import ItemFacts, as_text, item_facts
from benchmark.listening.protocol import ListeningProtocol, load_protocol_for_listing
from benchmark.listening.rule import Decision, UniverseRule, decide

__all__ = [
    "Listing",
    "UniverseBuildError",
    "build_manifest",
    "collect_listing",
    "facts_from_response",
    "main",
    "write_universe",
]

logger = logging.getLogger(__name__)

# A fetch takes a URL and returns the body, and raises UniverseBuildError on anything but a 200.
Fetch = Callable[[str], bytes]


class UniverseBuildError(RuntimeError):
    """Raised when the listing or a metadata response is not the one the protocol describes."""


@dataclass(frozen=True)
class Listing:
    """Every identifier the search returned, and the total the search reported.

    Attributes:
        identifiers: The identifiers, in the order the search gave them (sorted).
        num_found: The total the first page reported.
    """

    identifiers: tuple[str, ...]
    num_found: int


def _refuse(message: str) -> NoReturn:
    """Log *message* as an error and raise it as a UniverseBuildError.

    Args:
        message: What is wrong with the listing or the metadata.

    Raises:
        UniverseBuildError: Always.
    """
    logger.error("Listening universe build: %s", message)
    raise UniverseBuildError(message)


def search_url(protocol: ListeningProtocol, page: int) -> str:
    """The search URL for one page of the listing.

    Args:
        protocol: The protocol, which names the query, fields, sort and page size.
        page: The one-based page number.

    Returns:
        The URL, with every parameter percent-encoded.
    """
    source = protocol.source
    params: list[tuple[str, str]] = [
        ("q", source.search_query),
        ("rows", str(source.search_rows_per_page)),
        ("page", str(page)),
        ("output", "json"),
        ("sort[]", source.search_sort),
    ]
    params += [("fl[]", field) for field in source.search_fields]
    return source.search_url + "?" + urllib.parse.urlencode(params)


def metadata_url(protocol: ListeningProtocol, identifier: str) -> str:
    """The metadata URL for one item.

    Args:
        protocol: The protocol, which holds the URL template.
        identifier: The item's identifier.

    Returns:
        The URL, with the identifier percent-encoded.
    """
    quoted = urllib.parse.quote(identifier, safe="")
    return protocol.source.metadata_url_template.format(identifier=quoted)


def _page_docs(raw: bytes, expected_total: int | None) -> tuple[list[str], int]:
    """The identifiers on one search page and the total it reports, checked for consistency.

    Args:
        raw: The page body.
        expected_total: The total reported by earlier pages, or None on the first page.

    Returns:
        The identifiers on the page, in order, and the total the page reports.

    Raises:
        UniverseBuildError: If the page is malformed or reports a different total.
    """
    body = json.loads(raw)
    response = body.get("response") if isinstance(body, dict) else None
    if not isinstance(response, dict) or not isinstance(response.get("docs"), list):
        _refuse("a search page has no response.docs array")
    total = response["numFound"]
    if expected_total is not None and total != expected_total:
        _refuse(f"the search total changed between pages: {expected_total} then {total}")
    identifiers = [str(doc["identifier"]) for doc in response["docs"]]
    return identifiers, int(total)


def collect_listing(protocol: ListeningProtocol, fetch: Fetch) -> Listing:
    """Read every page of the search and check the result is the whole listing.

    Args:
        protocol: The protocol.
        fetch: The HTTPS fetch used for each page.

    Returns:
        The listing.

    Raises:
        UniverseBuildError: If the total changes between pages, a page is empty before the
            total is reached, an identifier repeats, or the count differs from the total.
    """
    collected: list[str] = []
    total: int | None = None
    page = 1
    while total is None or len(collected) < total:
        identifiers, total = _page_docs(fetch(search_url(protocol, page)), total)
        if not identifiers:
            _refuse(f"search page {page} is empty before the total {total} is reached")
        collected.extend(identifiers)
        page += 1
    if len(set(collected)) != len(collected):
        _refuse("the search repeats an identifier across pages")
    if len(collected) != total:
        _refuse(f"the listing holds {len(collected)} identifiers, not the total {total}")
    return Listing(identifiers=tuple(collected), num_found=total)


def facts_from_response(identifier: str, raw: bytes) -> ItemFacts:
    """The universe facts for one item, from its metadata response.

    Args:
        identifier: The identifier the listing gave.
        raw: The metadata response body.

    Returns:
        The facts.

    Raises:
        UniverseBuildError: If the body is malformed, or its identifier is not the requested one.
    """
    body = json.loads(raw)
    metadata = body.get("metadata") if isinstance(body, dict) else None
    if not isinstance(metadata, dict):
        _refuse(f"metadata for {identifier!r} has no metadata object")
    if as_text(metadata.get("identifier")) != identifier:
        _refuse(f"metadata for {identifier!r} names a different identifier")
    files = body.get("files", [])
    if not isinstance(files, list):
        _refuse(f"metadata for {identifier!r} has a files value that is not a list")
    return item_facts(identifier, metadata, files)


def _entry(facts: ItemFacts, decision: Decision) -> dict[str, Any]:
    """One kept item's manifest object.

    Args:
        facts: The item's facts.
        decision: The kept decision.

    Returns:
        The object written under `items`.
    """
    return {
        "identifier": facts.identifier,
        "title": facts.title,
        "year": facts.year,
        "performers": list(facts.performers),
        "performer_source": facts.performer_source,
        "instrumentation": facts.instrumentation,
        "rule": {"decision": decision.reason.value, "matched": list(decision.matched)},
        "flac": {
            "name": facts.flac_name,
            "size": facts.flac_size,
            "md5": facts.flac_md5,
            "sha1": facts.flac_sha1,
            "length_s": facts.length_s,
        },
    }


def _partition(
    listing: Listing, facts: Mapping[str, ItemFacts], rule: UniverseRule
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split the listing into kept entries and excluded entries, in identifier order.

    Args:
        listing: The listing whose identifiers are decided.
        facts: The facts for every listed identifier.
        rule: The compiled rule.

    Returns:
        The kept item entries and the excluded entries, each with its reason.

    Raises:
        UniverseBuildError: If a listed identifier has no facts.
    """
    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for identifier in sorted(listing.identifiers):
        if identifier not in facts:
            _refuse(f"no metadata was read for listed identifier {identifier!r}")
        decision = decide(facts[identifier], rule)
        if decision.kept:
            kept.append(_entry(facts[identifier], decision))
        else:
            excluded.append(
                {
                    "identifier": identifier,
                    "reason": decision.reason.value,
                    "matched": list(decision.matched),
                }
            )
    return kept, excluded


def build_manifest(
    protocol: ListeningProtocol,
    listing: Listing,
    facts: Mapping[str, ItemFacts],
    rule: UniverseRule,
    listed_at_utc: str,
) -> dict[str, Any]:
    """The manifest object for the listing, with every item decided by the rule.

    Args:
        protocol: The protocol, which supplies the study name, version, query and date.
        listing: The listing the facts were read for.
        facts: The facts for every identifier in the listing.
        rule: The compiled rule.
        listed_at_utc: The time the listing was read, as an ISO 8601 string.

    Returns:
        The manifest object, ready for `manifest_bytes`.

    Raises:
        UniverseBuildError: If an identifier in the listing has no facts.
    """
    kept, excluded = _partition(listing, facts, rule)
    reasons = Counter(entry["reason"] for entry in excluded)
    return {
        "study": protocol.study,
        "protocol_version": protocol.protocol_version,
        "listing_date": protocol.source.listing_date.isoformat(),
        "listed_at_utc": listed_at_utc,
        "query": protocol.source.search_query,
        "sort": protocol.source.search_sort,
        "num_found": listing.num_found,
        "listed": len(listing.identifiers),
        "kept": len(kept),
        "excluded_by_reason": dict(sorted(reasons.items())),
        "items": kept,
        "excluded": excluded,
    }


def write_universe(protocol: ListeningProtocol, data: Mapping[str, Any]) -> bytes:
    """Create the universe file once. Refuses to replace an existing one.

    Args:
        protocol: The protocol, which names the file.
        data: The manifest object.

    Returns:
        The bytes written.

    Raises:
        FileExistsError: If the file already exists, since the universe is frozen once written.
    """
    target = universe_path(protocol)
    if target.exists():
        logger.error("Listening universe: %s already exists and is frozen", target)
        raise FileExistsError(str(target))
    payload = manifest_bytes(data)
    write_atomically(target, payload)
    logger.info("Wrote the listening universe to %s (%d bytes)", target, len(payload))
    return payload


def live_fetch(protocol: ListeningProtocol) -> Fetch:
    """The network fetch for the listing: HTTPS only, byte-capped, one attempt per call.

    Args:
        protocol: The protocol, which supplies the user agent, timeout, byte cap and pause.

    Returns:
        A fetch that pauses between calls and refuses any response that is not a complete 200.
    """
    fetch_settings = protocol.fetch
    state: dict[str, int] = {"calls": 0}

    def fetch(url: str) -> bytes:
        if state["calls"]:
            time.sleep(fetch_settings.pause_s_between_requests)
        state["calls"] += 1
        sink = io.BytesIO()
        response = https_get.get(
            url,
            sink,
            max_bytes=fetch_settings.max_response_bytes,
            timeout_s=fetch_settings.timeout_s,
            user_agent=fetch_settings.user_agent,
        )
        if response.status != 200 or response.exceeded:
            _refuse(f"GET {url} returned status {response.status}, exceeded={response.exceeded}")
        return sink.getvalue()

    return fetch


def _read_all_facts(
    protocol: ListeningProtocol, identifiers: Sequence[str], fetch: Fetch
) -> dict[str, ItemFacts]:
    """The facts for every identifier, read one metadata response at a time.

    Args:
        protocol: The protocol, which holds the metadata URL template.
        identifiers: The identifiers to read.
        fetch: The fetch.

    Returns:
        The facts keyed by identifier.
    """
    result: dict[str, ItemFacts] = {}
    for identifier in identifiers:
        raw = fetch(metadata_url(protocol, identifier))
        result[identifier] = facts_from_response(identifier, raw)
    return result


def main() -> None:
    """Read the listing, decide every item, and write the universe file once.

    The protocol is loaded without the universe digest check, since this run creates the
    file that digest refers to. The written file's digest must then be pinned in the code and
    in the protocol file before the universe is used.
    """
    protocol = load_protocol_for_listing()
    fetch = live_fetch(protocol)
    listed_at = datetime.now(UTC).isoformat()
    listing = collect_listing(protocol, fetch)
    facts = _read_all_facts(protocol, listing.identifiers, fetch)
    data = build_manifest(protocol, listing, facts, protocol.universe.rule, listed_at)
    payload = write_universe(protocol, data)
    print(json.dumps({"listed": data["listed"], "kept": data["kept"]}))
    print(f"sha256 {hashlib.sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()
