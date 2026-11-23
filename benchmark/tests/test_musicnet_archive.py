"""Tests for the MusicNet archive: its fetch and the safe extraction of the frozen ids.

Small gzip tar files are built in the test directory; no test reads the network.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import tarfile
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from benchmark.fetch import ListingMismatch, Staged
from benchmark.https_get import Response
from benchmark.musicnet_archive import ArchiveError, extract_members, fetch_archive
from benchmark.protocol import ZenodoArchiveStratum, load_protocol


def wav(identifier: int) -> bytes:
    """Distinct stand-in bytes for one member."""
    return f"RIFF{identifier}".encode() * 50


def build(path: Path, entries: list[tuple[str, Any]]) -> Path:
    """A gzip tar at *path*; each entry is (name, bytes) or (name, ('symlink'|'dir', target))."""
    with tarfile.open(path, "w:gz") as tar:
        for name, content in entries:
            info = tarfile.TarInfo(name)
            if isinstance(content, tuple):
                kind, target = content
                info.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.DIRTYPE
                info.linkname = target
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return path


def extract(
    tmp_path: Path, entries: list[tuple[str, Any]], ids: tuple[int, ...], cap: int = 10_000
) -> Any:
    """Build an archive from *entries* and extract *ids* into tmp_path/out."""
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    archive = build(tmp_path / "a.tar.gz", entries)
    return extract_members(archive, ids, out, member_cap=cap)


class TestExtraction:
    """Only each frozen id's one regular file is written, under a name built from the id."""

    def test_each_frozen_id_is_written_once_and_nothing_else(self, tmp_path: Path) -> None:
        entries = [
            ("musicnet", ("dir", "")),
            ("musicnet/train_data/11.wav", wav(11)),
            ("musicnet/test_data/12.wav", wav(12)),
            ("musicnet/train_data/10.wav", wav(10)),
            ("musicnet/train_labels/11.csv", b"labels"),
        ]
        members = extract(tmp_path, entries, (11, 12))
        assert sorted(members) == [11, 12]
        for identifier, member in members.items():
            assert member.path == tmp_path / "out" / f"{identifier}.wav"
            assert member.path.read_bytes() == wav(identifier)
            assert member.sha256 == hashlib.sha256(wav(identifier)).hexdigest()
            assert (member.size, member.oversize) == (len(wav(identifier)), False)
        assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["11.wav", "12.wav"]

    def test_a_member_name_that_climbs_out_is_written_inside_by_its_id(
        self, tmp_path: Path
    ) -> None:
        members = extract(tmp_path, [("../../11.wav", wav(11))], (11,))
        assert members[11].path == tmp_path / "out" / "11.wav"
        assert not (tmp_path.parent / "11.wav").exists()

    def test_a_link_is_never_followed_so_its_id_counts_as_missing(self, tmp_path: Path) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("outside")
        entries = [("musicnet/11.wav", wav(11)), ("musicnet/12.wav", ("symlink", str(secret)))]
        with pytest.raises(ArchiveError, match=r"\(12, 0\)"):
            extract(tmp_path, entries, (11, 12))
        assert list((tmp_path / "out").iterdir()) == []

    @pytest.mark.parametrize(
        ("entries", "expected"),
        [
            ([("a/11.wav", wav(11))], r"\(12, 0\)"),
            ([("a/11.wav", wav(11)), ("a/12.wav", wav(12)), ("b/12.wav", wav(99))], r"\(12, 2\)"),
        ],
        ids=["missing", "twice"],
    )
    def test_an_id_not_found_exactly_once_refuses_the_archive_and_removes_what_was_written(
        self, tmp_path: Path, entries: list[tuple[str, Any]], expected: str
    ) -> None:
        with pytest.raises(ArchiveError, match=expected):
            extract(tmp_path, entries, (11, 12))
        assert list((tmp_path / "out").iterdir()) == []

    def test_an_oversize_member_is_kept_as_a_prefix_without_a_digest(self, tmp_path: Path) -> None:
        members = extract(tmp_path, [("a/11.wav", b"x" * 500)], (11,), cap=100)
        assert (members[11].size, members[11].oversize, members[11].sha256) == (101, True, None)
        assert members[11].path.read_bytes() == b"x" * 101

    def test_a_relative_output_directory_is_refused(self, tmp_path: Path) -> None:
        archive = build(tmp_path / "a.tar.gz", [("a/11.wav", wav(11))])
        with pytest.raises(ValueError, match="absolute"):
            extract_members(archive, (11,), Path("out"), member_cap=100)


ARCHIVE = b"\x1f\x8b" + bytes(range(256)) * 8


@pytest.fixture
def chamber() -> ZenodoArchiveStratum:
    """The chamber stratum with archive values matching ARCHIVE."""
    stratum = load_protocol().selection.strata[1]
    assert isinstance(stratum, ZenodoArchiveStratum)
    return dataclasses.replace(
        stratum, archive_bytes=len(ARCHIVE), archive_md5=hashlib.md5(ARCHIVE).hexdigest()
    )


class Serve:
    """A stand-in GET serving one body per call, recording the URLs asked for."""

    def __init__(self, *bodies: bytes) -> None:
        self.bodies, self.urls = list(bodies), []

    def __call__(self, url: str, sink: BinaryIO, *, max_bytes: int, timeout_s: float) -> Response:
        self.urls.append(url)
        body = self.bodies.pop(0)[:max_bytes]
        sink.write(body)
        return Response(
            200,
            len(body),
            False,
            len(body),
            hashlib.sha256(body).hexdigest(),
            hashlib.md5(body).hexdigest(),
            hashlib.sha1(body).hexdigest(),
        )


class TestFetchArchive:
    """The archive is one fetch, checked against its record's size and md5."""

    def test_the_archive_is_fetched_from_its_record_and_staged(
        self, tmp_path: Path, chamber: ZenodoArchiveStratum
    ) -> None:
        serve = Serve(ARCHIVE)
        selection = load_protocol().selection
        outcome = fetch_archive(chamber, selection, tmp_path, get_=serve, sleep=lambda s: None)
        assert serve.urls == [
            "https://zenodo.org/api/records/5120004/files/musicnet.tar.gz/content"
        ]
        assert isinstance(outcome, Staged)
        assert outcome.sha256 == hashlib.sha256(ARCHIVE).hexdigest()
        assert outcome.compared == ("size", "md5")
        assert outcome.path.name.endswith(".tar.gz")

    def test_an_archive_with_another_md5_on_every_attempt_mismatches(
        self, tmp_path: Path, chamber: ZenodoArchiveStratum
    ) -> None:
        other = ARCHIVE[:-1] + b"\x00"
        serve = Serve(other, other, other, other)
        selection = load_protocol().selection
        outcome = fetch_archive(chamber, selection, tmp_path, get_=serve, sleep=lambda s: None)
        assert isinstance(outcome, ListingMismatch)
        assert all(a.mismatched == ("md5",) for a in outcome.attempts)

    def test_a_staged_archive_is_reused_on_a_rerun(
        self, tmp_path: Path, chamber: ZenodoArchiveStratum
    ) -> None:
        selection = load_protocol().selection
        fetch_archive(chamber, selection, tmp_path, get_=Serve(ARCHIVE), sleep=lambda s: None)
        again = Serve()
        outcome = fetch_archive(chamber, selection, tmp_path, get_=again, sleep=lambda s: None)
        assert isinstance(outcome, Staged)
        assert (outcome.reused, again.urls) == (True, [])
