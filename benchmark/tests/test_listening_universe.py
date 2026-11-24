"""Tests for building the listening universe: the listing, the facts, the manifest, the write."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from benchmark import https_get
from benchmark.listening import universe
from benchmark.listening.manifest import UniverseError, manifest_bytes, read_universe
from benchmark.listening.protocol import (
    PINNED_UNIVERSE_SHA256,
    ListeningProtocol,
    load_listening_protocol,
    load_protocol_for_listing,
)
from benchmark.listening.rule import Reason
from benchmark.listening.universe import Listing, UniverseBuildError


@pytest.fixture
def protocol() -> ListeningProtocol:
    """The frozen protocol, as the listing reads it."""
    return load_protocol_for_listing()


def _page(identifiers: list[str], total: int) -> bytes:
    """A search page body with the given identifiers and reported total.

    Args:
        identifiers: The identifiers on this page.
        total: The numFound the page reports.

    Returns:
        The JSON bytes.
    """
    docs = [{"identifier": identifier} for identifier in identifiers]
    return json.dumps({"response": {"numFound": total, "docs": docs}}).encode("utf-8")


def _page_fetch(pages: dict[int, bytes]) -> universe.Fetch:
    """A fake fetch that serves search pages by their `page` parameter.

    Args:
        pages: The body for each one-based page number.

    Returns:
        A fetch for `collect_listing`.
    """

    def fetch(url: str) -> bytes:
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        return pages[int(query["page"][0])]

    return fetch


def _metadata(identifier: str, description: str, title: str = "Item") -> bytes:
    """A metadata response with one main FLAC and the given description.

    Args:
        identifier: The identifier the response names.
        description: The HTML description.
        title: The title.

    Returns:
        The JSON bytes.
    """
    flac = {
        "name": "track.flac",
        "source": "original",
        "format": "24bit Flac",
        "size": "1000",
        "md5": "c" * 32,
        "sha1": "d" * 40,
        "length": "95.0",
    }
    body = {
        "metadata": {"identifier": identifier, "title": title, "description": description},
        "files": [flac],
    }
    return json.dumps(body).encode("utf-8")


def test_the_listing_reads_every_page_and_checks_the_total(protocol: ListeningProtocol) -> None:
    """Pages are read until the reported total is reached, and the listing keeps their order."""
    fetch = _page_fetch({1: _page(["78_a", "78_b"], 3), 2: _page(["78_c"], 3)})
    listing = universe.collect_listing(protocol, fetch)
    assert listing == Listing(identifiers=("78_a", "78_b", "78_c"), num_found=3)


def test_a_total_that_changes_between_pages_is_refused(protocol: ListeningProtocol) -> None:
    """If the search total moves while it is read, the listing is not the one the protocol names."""
    fetch = _page_fetch({1: _page(["78_a"], 2), 2: _page(["78_b"], 5)})
    with pytest.raises(UniverseBuildError, match="total changed"):
        universe.collect_listing(protocol, fetch)


def test_a_repeated_identifier_across_pages_is_refused(protocol: ListeningProtocol) -> None:
    """A repeated identifier means the paged search is unstable, and the listing is refused."""
    fetch = _page_fetch({1: _page(["78_a", "78_b"], 3), 2: _page(["78_b"], 3)})
    with pytest.raises(UniverseBuildError, match="repeats an identifier"):
        universe.collect_listing(protocol, fetch)


def test_an_empty_page_before_the_total_is_refused(protocol: ListeningProtocol) -> None:
    """An empty page while the total is not reached would otherwise loop or undercount."""
    fetch = _page_fetch({1: _page(["78_a"], 3), 2: _page([], 3)})
    with pytest.raises(UniverseBuildError, match="is empty before the total"):
        universe.collect_listing(protocol, fetch)


def test_metadata_naming_another_identifier_is_refused() -> None:
    """A metadata response for a different item is refused, not attributed to the one asked for."""
    with pytest.raises(UniverseBuildError, match="names a different identifier"):
        universe.facts_from_response("78_asked", _metadata("78_other", "Violin solo."))


def test_the_manifest_records_each_decision_and_counts_the_exclusions(
    protocol: ListeningProtocol,
) -> None:
    """Kept items are written with their rule decision, and each exclusion carries its reason."""
    ids = ("78_kept", "78_vocal", "78_cornet")
    descriptions = {
        "78_kept": "Performer: Jan Fiddle<br>Violin solo; Pianoforte.",
        "78_vocal": "Performer: Ada Singer<br>Soprano and Tenor; sung duet.",
        "78_cornet": "Performer: Sam Brass<br>Cornet Solo.",
    }
    facts = {
        identifier: universe.facts_from_response(
            identifier, _metadata(identifier, descriptions[identifier])
        )
        for identifier in ids
    }
    data = universe.build_manifest(
        protocol, Listing(ids, 3), facts, protocol.universe.rule, "2026-11-24T00:00:00+00:00"
    )
    assert data["study"] == "historical_listening"
    assert data["protocol_version"] == protocol.protocol_version
    assert data["kept"] == 1
    assert data["listed"] == 3
    assert [entry["identifier"] for entry in data["items"]] == ["78_kept"]
    assert data["items"][0]["rule"]["decision"] == Reason.KEPT.value
    assert data["excluded_by_reason"] == {
        Reason.NOT_SOLO_OR_STRING_ENSEMBLE.value: 1,
        Reason.VOCAL.value: 1,
    }


def test_a_listed_identifier_without_metadata_is_refused(protocol: ListeningProtocol) -> None:
    """A manifest is never written with an identifier that had no metadata read for it."""
    with pytest.raises(UniverseBuildError, match="no metadata was read"):
        universe.build_manifest(
            protocol, Listing(("78_missing",), 1), {}, protocol.universe.rule, "t"
        )


def test_the_universe_file_is_written_once_and_an_existing_one_is_left_alone(
    protocol: ListeningProtocol, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second write refuses, and the file the first write made is unchanged afterwards."""
    target = tmp_path / "universe_v1.json"
    monkeypatch.setattr(universe, "universe_path", lambda _protocol: target)
    data: dict[str, Any] = {"study": "x"}
    universe.write_universe(protocol, data)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        universe.write_universe(protocol, {"study": "replacement"})
    assert target.read_bytes() == original == manifest_bytes(data)


def test_a_universe_with_a_digest_other_than_the_pin_is_refused(
    protocol: ListeningProtocol,
) -> None:
    """The universe reader refuses bytes whose digest is not the pinned one."""
    with pytest.raises(UniverseError, match="digest does not match"):
        read_universe(protocol, raw=b'{"not": "the pinned universe"}')


def test_the_listing_sends_the_protocol_user_agent(
    protocol: ListeningProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each listing request names the protocol's user agent through the shared transport."""
    sent: list[object] = []

    def fake_get(url: str, sink: object, **kwargs: object) -> https_get.Response:
        sent.append(kwargs.get("user_agent"))
        return https_get.Response(200, 0, False, None, "", "", "")

    monkeypatch.setattr(universe.https_get, "get", fake_get)
    universe.live_fetch(protocol)("https://archive.org/advancedsearch.php")
    assert sent == ["research-client"]


def test_a_non_200_metadata_response_stops_the_fetch(
    protocol: ListeningProtocol, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-200 response is a build failure, so no partial listing is produced."""

    def fake_get(url: str, sink: object, **kwargs: object) -> https_get.Response:
        return https_get.Response(429, 0, False, None, "", "", "")

    monkeypatch.setattr(universe.https_get, "get", fake_get)
    monkeypatch.setattr(universe.time, "sleep", lambda _seconds: None)
    fetch = universe.live_fetch(protocol)
    with pytest.raises(UniverseBuildError, match="returned status 429"):
        fetch("https://archive.org/advancedsearch.php")


def test_the_full_loader_accepts_the_shipped_protocol_and_universe_pins() -> None:
    """With every pin set, the loader that checks the universe digest accepts the shipped files."""
    loaded = load_listening_protocol()
    items = read_universe(loaded)
    assert loaded.universe.sha256 == PINNED_UNIVERSE_SHA256
    assert len(items) == len({item.identifier for item in items})
    assert all(item.flac_md5 and item.flac_sha1 for item in items)
