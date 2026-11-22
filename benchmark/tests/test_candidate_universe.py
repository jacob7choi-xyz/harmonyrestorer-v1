"""Tests for listing each stratum's candidate universe and reading the frozen file back.

No test reads the network: listings are served by a stand-in opener. The strata are the
loaded protocol's, narrowed with `dataclasses.replace` where a small case is clearer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import re
import urllib.request
from email.message import Message
from pathlib import Path
from typing import IO, Any

import pytest

from benchmark import candidate_universe
from benchmark.candidate_universe import (
    CandidateUniverseError,
    ListedFile,
    archive_item_candidates,
    chamber_candidates,
    universe_bytes,
)
from benchmark.https_get import Opener
from benchmark.measurements import candidate_listing
from benchmark.measurements.candidate_listing import ListingError, list_stratum
from benchmark.protocol import ArchiveItemStratum, ZenodoArchiveStratum, load_protocol

MD5 = "a" * 32
SHA1 = "b" * 40
CSV_HEADER = '"id","composer","composition","movement","ensemble","source","transcriber","catalog_name","seconds"\n'


@pytest.fixture(scope="module")
def strata() -> tuple[ArchiveItemStratum, ZenodoArchiveStratum]:
    """The protocol's two strata, piano first."""
    piano, chamber = load_protocol().selection.strata
    assert isinstance(piano, ArchiveItemStratum)
    assert isinstance(chamber, ZenodoArchiveStratum)
    return piano, chamber


@pytest.fixture
def piano(strata: tuple[ArchiveItemStratum, ZenodoArchiveStratum]) -> ArchiveItemStratum:
    """The piano stratum with one excluded path and one spelling correction."""
    return dataclasses.replace(
        strata[0], excluded_paths=("Cello Sonata.flac",), cluster_spelling={"OIga X": "Olga X"}
    )


@pytest.fixture
def chamber(strata: tuple[ArchiveItemStratum, ZenodoArchiveStratum]) -> ZenodoArchiveStratum:
    """The chamber stratum narrowed to ids 11 and 12, with CSV matching its metadata md5."""
    return dataclasses.replace(
        strata[1],
        candidate_ids=(11, 12),
        metadata_md5=hashlib.md5(CSV, usedforsecurity=False).hexdigest(),
        cluster_spelling={"Soni V": "Soni V Quintet"},
    )


def flac(name: str, **values: Any) -> dict[str, Any]:
    """A listed file with a size, md5, sha1 and artist unless *values* says otherwise."""
    return {"name": name, "size": "1234", "md5": MD5, "sha1": SHA1, "artist": "Ann", **values}


def item(*files: dict[str, Any]) -> dict[str, Any]:
    """Item metadata whose files list holds *files* and the excluded cello file."""
    return {"metadata": {}, "files": [*files, flac("Cello Sonata.flac", artist="Duo")]}


def row(identifier: int, source: str, ensemble: str = "String Quartet") -> str:
    """One MusicNet metadata row."""
    return f'{identifier},"C","Work","1","{ensemble}","{source}","t","K1",300\n'


CSV = (CSV_HEADER + row(10, "Other") + row(11, "Soni V") + row(12, "Pascal")).encode()


class TestArchiveItemCandidates:
    """The piano universe from its item's listing."""

    def test_flac_files_less_exclusions_are_kept_clustered_and_sorted(
        self, piano: ArchiveItemStratum
    ) -> None:
        listed = archive_item_candidates(
            item(flac("b.FLAC", artist="OIga X"), flac("a.flac"), {"name": "a.mp3"}), piano
        )
        assert listed == (
            ListedFile(f"{piano.source}/a.flac", "Ann", 1234, MD5, SHA1),
            ListedFile(f"{piano.source}/b.FLAC", "Olga X", 1234, MD5, SHA1),
        )

    def test_an_excluded_path_missing_from_the_listing_refuses_it(
        self, piano: ArchiveItemStratum
    ) -> None:
        with pytest.raises(CandidateUniverseError, match="excluded paths not in the listing"):
            archive_item_candidates({"files": [flac("a.flac")]}, piano)

    @pytest.mark.parametrize("artist", [None, "", "  ", 3])
    def test_a_file_without_a_performer_refuses_the_listing(
        self, piano: ArchiveItemStratum, artist: object
    ) -> None:
        with pytest.raises(CandidateUniverseError, match="no performer"):
            archive_item_candidates(item(flac("a.flac", artist=artist)), piano)

    @pytest.mark.parametrize(
        "values",
        [{"size": "12kB"}, {"md5": MD5.upper()}, {"sha1": SHA1[:-1]}],
        ids=["size_text", "md5_uppercase", "sha1_short"],
    )
    def test_a_malformed_listing_value_refuses_the_listing(
        self, piano: ArchiveItemStratum, values: dict[str, Any]
    ) -> None:
        with pytest.raises(CandidateUniverseError):
            archive_item_candidates(item(flac("a.flac", **values)), piano)

    def test_a_candidate_listed_twice_refuses_the_listing(self, piano: ArchiveItemStratum) -> None:
        with pytest.raises(CandidateUniverseError, match="more than once"):
            archive_item_candidates(item(flac("a.flac"), flac("a.flac")), piano)

    @pytest.mark.parametrize("data", [{}, {"files": "x"}, [], {"files": [{"size": "1"}]}])
    def test_metadata_without_named_files_is_refused(
        self, piano: ArchiveItemStratum, data: object
    ) -> None:
        with pytest.raises(CandidateUniverseError):
            archive_item_candidates(data, piano)


class TestChamberCandidates:
    """The chamber universe from MusicNet's metadata file."""

    def test_exactly_the_candidate_ids_are_kept_with_their_clusters(
        self, chamber: ZenodoArchiveStratum
    ) -> None:
        assert chamber_candidates(CSV, chamber) == (
            ListedFile("musicnet/11", "Soni V Quintet", None, None, None),
            ListedFile("musicnet/12", "Pascal", None, None, None),
        )

    def test_a_metadata_file_with_another_md5_is_refused(
        self, chamber: ZenodoArchiveStratum
    ) -> None:
        with pytest.raises(CandidateUniverseError, match="the protocol freezes"):
            chamber_candidates(CSV + b" ", chamber)

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            (CSV_HEADER + row(11, "A"), "id 12 appears 0 times"),
            (CSV_HEADER + row(11, "A") + row(12, "B") + row(12, "C"), "id 12 appears 2 times"),
            (CSV_HEADER + row(11, "A") + row(12, ""), "no performer"),
        ],
        ids=["id_missing", "id_twice", "no_source"],
    )
    def test_a_file_that_does_not_hold_each_id_once_is_refused(
        self, chamber: ZenodoArchiveStratum, body: str, message: str
    ) -> None:
        data = body.encode()
        narrowed = dataclasses.replace(
            chamber, metadata_md5=hashlib.md5(data, usedforsecurity=False).hexdigest()
        )
        with pytest.raises(CandidateUniverseError, match=message):
            chamber_candidates(data, narrowed)


def frozen(tmp_path: Path, data: bytes) -> tuple[Path, str]:
    """*data* written to a universe file, and its digest."""
    path = tmp_path / "universe.json"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def canonical(stratum: Any, files: tuple[ListedFile, ...]) -> bytes:
    """The frozen file for *files* in *stratum*."""
    return universe_bytes(
        files,
        stratum=stratum.name,
        source=stratum.source,
        query="https://example.invalid/q",
        enumerated_at="2026-11-22T00:00:00Z",
    )


def piano_files(piano: ArchiveItemStratum) -> tuple[ListedFile, ...]:
    """Two well-formed piano candidates."""
    return (
        ListedFile(f"{piano.source}/a.flac", "Ann", 10, MD5, SHA1),
        ListedFile(f"{piano.source}/b.flac", "Olga X", None, None, None),
    )


def chamber_files() -> tuple[ListedFile, ...]:
    """The two chamber candidates of the narrowed stratum."""
    return (
        ListedFile("musicnet/11", "Soni V Quintet", None, None, None),
        ListedFile("musicnet/12", "Pascal", None, None, None),
    )


class TestFrozenFile:
    """A frozen file reads back only if its digest, form and membership hold."""

    def test_each_kind_reads_back_as_written(
        self, tmp_path: Path, piano: ArchiveItemStratum, chamber: ZenodoArchiveStratum
    ) -> None:
        for stratum, files in ((piano, piano_files(piano)), (chamber, chamber_files())):
            path, digest = frozen(tmp_path, canonical(stratum, files))
            universe = candidate_universe._read_universe(path, digest, stratum)
            assert (universe.files, universe.sha256, universe.stratum) == (
                files,
                digest,
                stratum.name,
            )

    def test_another_digest_is_refused(self, tmp_path: Path, piano: ArchiveItemStratum) -> None:
        path, _ = frozen(tmp_path, canonical(piano, piano_files(piano)))
        with pytest.raises(CandidateUniverseError, match="pinned digest"):
            candidate_universe._read_universe(path, "0" * 64, piano)

    def test_an_edited_file_repinned_out_of_canonical_form_is_refused(
        self, tmp_path: Path, piano: ArchiveItemStratum
    ) -> None:
        edited = json.loads(canonical(piano, piano_files(piano)))
        edited["count"] = 3
        path, digest = frozen(tmp_path, (json.dumps(edited, indent=2) + "\n").encode())
        with pytest.raises(CandidateUniverseError, match="canonical"):
            candidate_universe._read_universe(path, digest, piano)

    def test_a_file_for_the_other_stratum_is_refused(
        self, tmp_path: Path, piano: ArchiveItemStratum, chamber: ZenodoArchiveStratum
    ) -> None:
        path, digest = frozen(tmp_path, canonical(chamber, chamber_files()))
        with pytest.raises(CandidateUniverseError, match="is for 'chamber'"):
            candidate_universe._read_universe(path, digest, piano)

    @pytest.mark.parametrize(
        ("identifier", "cluster", "message"),
        [
            ("x.flac", "Ann", "not from"),
            ("{source}/Cello Sonata.flac", "Ann", "filter or exclusions"),
            ("{source}/a.mp3", "Ann", "filter or exclusions"),
            ("{source}/a.flac", "OIga X", "misspelling"),
        ],
        ids=["other_source", "excluded_path", "not_flac", "uncorrected_cluster"],
    )
    def test_a_piano_file_outside_the_stratum_is_refused(
        self,
        tmp_path: Path,
        piano: ArchiveItemStratum,
        identifier: str,
        cluster: str,
        message: str,
    ) -> None:
        files = (ListedFile(identifier.format(source=piano.source), cluster, None, None, None),)
        path, digest = frozen(tmp_path, canonical(piano, files))
        with pytest.raises(CandidateUniverseError, match=message):
            candidate_universe._read_universe(path, digest, piano)

    def test_a_chamber_file_without_exactly_its_ids_is_refused(
        self, tmp_path: Path, chamber: ZenodoArchiveStratum
    ) -> None:
        path, digest = frozen(tmp_path, canonical(chamber, chamber_files()[:1]))
        with pytest.raises(CandidateUniverseError, match="exactly the stratum's candidate ids"):
            candidate_universe._read_universe(path, digest, chamber)

    @pytest.mark.parametrize(
        "change",
        [
            lambda e: e.update(extra=1),
            lambda e: e.pop("sha1"),
            lambda e: e.update(size=True),
            lambda e: e.update(size=-1),
            lambda e: e.update(md5="xyz"),
            lambda e: e.update(cluster=""),
        ],
        ids=["extra_key", "missing_key", "bool_size", "negative_size", "bad_md5", "empty_cluster"],
    )
    def test_a_malformed_entry_is_refused(
        self, tmp_path: Path, piano: ArchiveItemStratum, change: Any
    ) -> None:
        data = json.loads(canonical(piano, piano_files(piano)))
        change(data["candidates"][0])
        path, digest = frozen(tmp_path, (json.dumps(data, indent=2) + "\n").encode())
        with pytest.raises(CandidateUniverseError):
            candidate_universe._read_universe(path, digest, piano)

    def test_a_missing_file_is_refused(self, tmp_path: Path, piano: ArchiveItemStratum) -> None:
        with pytest.raises(CandidateUniverseError, match="does not exist"):
            candidate_universe._read_universe(tmp_path / "absent.json", "0" * 64, piano)

    def test_an_unpinned_stratum_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, piano: ArchiveItemStratum
    ) -> None:
        monkeypatch.setitem(candidate_universe.UNIVERSE_SHA256, piano.name, None)
        with pytest.raises(CandidateUniverseError, match="not been listed"):
            candidate_universe.load_universe(piano)

    def test_the_pinned_file_loads(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, piano: ArchiveItemStratum
    ) -> None:
        data = canonical(piano, piano_files(piano))
        monkeypatch.setattr(candidate_universe, "_MANIFESTS", tmp_path)
        candidate_universe.universe_path(piano.name).write_bytes(data)
        monkeypatch.setitem(
            candidate_universe.UNIVERSE_SHA256, piano.name, hashlib.sha256(data).hexdigest()
        )
        assert candidate_universe.load_universe(piano).files == piano_files(piano)


class Served(io.BytesIO):
    """A stand-in response with a status and a declared length."""

    def __init__(self, body: bytes, status: int = 200, length: int | None = None) -> None:
        super().__init__(body)
        self.status = status
        self.headers = Message()
        if length is not None:
            self.headers["Content-Length"] = str(length)


def serving(bodies: dict[str, bytes], **response: Any) -> tuple[Opener, list[str]]:
    """An opener serving each URL's body, and the URLs it was asked for."""
    asked: list[str] = []

    def opener(request: urllib.request.Request, timeout_s: float) -> IO[bytes]:
        asked.append(request.full_url)
        return Served(bodies[request.full_url], **response)

    return opener, asked


def piano_listing(piano: ArchiveItemStratum) -> dict[str, bytes]:
    """The item's metadata, served at its URL."""
    body = json.dumps(item(flac("b.flac"), flac("a.flac"), {"name": "c.mp3"})).encode()
    return {candidate_universe.metadata_url(piano.source): body}


def chamber_listing(chamber: ZenodoArchiveStratum) -> dict[str, bytes]:
    """MusicNet's metadata file, served at its Zenodo URL."""
    return {candidate_universe.zenodo_file_url(chamber.zenodo_record, chamber.metadata): CSV}


class TestListStratum:
    """The one live listing per stratum: fetched once, whole, and frozen."""

    def test_each_stratum_becomes_a_file_the_loader_accepts(
        self, tmp_path: Path, piano: ArchiveItemStratum, chamber: ZenodoArchiveStratum
    ) -> None:
        for stratum, bodies in ((piano, piano_listing(piano)), (chamber, chamber_listing(chamber))):
            opener, asked = serving(bodies)
            data = list_stratum(stratum, tmp_path / "u.json", opener=opener)
            assert asked == list(bodies)
            path, digest = frozen(tmp_path, data)
            universe = candidate_universe._read_universe(path, digest, stratum)
            assert universe.query == asked[0]
            assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", universe.enumerated_at)

    def test_an_existing_file_is_not_replaced_and_nothing_is_fetched(
        self, tmp_path: Path, piano: ArchiveItemStratum
    ) -> None:
        existing = tmp_path / "u.json"
        existing.write_text("frozen")
        opener, asked = serving(piano_listing(piano))
        with pytest.raises(ListingError, match="listed once"):
            list_stratum(piano, existing, opener=opener)
        assert (asked, existing.read_text()) == ([], "frozen")

    @pytest.mark.parametrize(
        "response", [{"status": 503}, {"length": 10**6}], ids=["server_error", "ended_short"]
    )
    def test_a_listing_that_is_not_a_whole_200_is_refused(
        self, tmp_path: Path, piano: ArchiveItemStratum, response: dict[str, Any]
    ) -> None:
        opener, _ = serving(piano_listing(piano), **response)
        with pytest.raises(ListingError):
            list_stratum(piano, tmp_path / "u.json", opener=opener)

    def test_a_listing_over_the_cap_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, piano: ArchiveItemStratum
    ) -> None:
        monkeypatch.setattr(candidate_listing, "MAX_LISTING_BYTES", 10)
        opener, _ = serving(piano_listing(piano))
        with pytest.raises(ListingError, match="cap exceeded True"):
            list_stratum(piano, tmp_path / "u.json", opener=opener)


class TestMain:
    """Both files are written only after both strata listed."""

    def run_main(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        strata: tuple[ArchiveItemStratum, ZenodoArchiveStratum],
        bodies: dict[str, bytes],
    ) -> int:
        monkeypatch.setattr(candidate_universe, "_MANIFESTS", tmp_path)
        opener, _ = serving(bodies)
        real = candidate_listing.list_stratum
        monkeypatch.setattr(
            candidate_listing,
            "list_stratum",
            lambda stratum, path: real(stratum, path, opener=opener),
        )

        class Selection:
            pass

        class Protocol:
            selection = Selection()

        Protocol.selection.strata = strata  # type: ignore[attr-defined]
        monkeypatch.setattr(candidate_listing, "load_protocol", lambda: Protocol)
        return candidate_listing.main()

    def test_both_listings_are_written_with_their_digests(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        piano: ArchiveItemStratum,
        chamber: ZenodoArchiveStratum,
    ) -> None:
        bodies = {**piano_listing(piano), **chamber_listing(chamber)}
        assert self.run_main(tmp_path, monkeypatch, (piano, chamber), bodies) == 0
        printed = capsys.readouterr().out
        for name in (piano.name, chamber.name):
            data = candidate_universe.universe_path(name).read_bytes()
            assert f"sha256 {hashlib.sha256(data).hexdigest()}" in printed
        assert "cap 5 allows 2 admissions; partitions need 36" in printed

    def test_a_failed_second_listing_writes_neither_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        piano: ArchiveItemStratum,
        chamber: ZenodoArchiveStratum,
    ) -> None:
        bodies = {**piano_listing(piano), **chamber_listing(chamber)}
        url = candidate_universe.zenodo_file_url(chamber.zenodo_record, chamber.metadata)
        bodies[url] = CSV + b"altered"
        assert self.run_main(tmp_path, monkeypatch, (piano, chamber), bodies) == 1
        assert list(tmp_path.iterdir()) == []


class TestSummary:
    """What the listing prints about each universe."""

    def test_capacity_counts_at_most_the_cap_per_cluster(self, piano: ArchiveItemStratum) -> None:
        files = tuple(
            ListedFile(f"{piano.source}/{i}.flac", "Ann" if i < 7 else "Bo", None, None, None)
            for i in range(9)
        )
        lines = candidate_listing.summary(piano, canonical(piano, files))
        assert lines[0] == "solo_piano: 9 candidates in 2 clusters"
        assert lines[1] == "  cap 5 allows 7 admissions; partitions need 42"
        assert lines[3:] == ["     7  Ann", "     2  Bo"]


class TestTheCommittedUniverses:
    """The frozen files in the repository load against their pinned digests."""

    def test_each_stratum_loads_with_the_counts_listed(
        self, strata: tuple[ArchiveItemStratum, ZenodoArchiveStratum]
    ) -> None:
        piano, chamber = (candidate_universe.load_universe(stratum) for stratum in strata)
        assert (len(piano.files), len({f.cluster for f in piano.files})) == (206, 12)
        assert (len(chamber.files), len({f.cluster for f in chamber.files})) == (80, 19)
        assert all(f.md5 is not None and f.sha1 is not None for f in piano.files)
        assert all(f.size is None for f in chamber.files)
