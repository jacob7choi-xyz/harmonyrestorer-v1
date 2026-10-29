"""Tests for the checkpoint download.

A failure here must leave nothing at the destination and must not touch a file that is
already there, so each failure path checks the directory as well as the error.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path
from typing import IO

import pytest

from scripts import fetch_checkpoint as fc

PAYLOAD = b"checkpoint bytes"
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class FakeOpener:
    """Serves fixed bytes, or raises, and records the URLs it was asked for."""

    def __init__(self, payload: bytes = PAYLOAD, error: OSError | None = None) -> None:
        self.payload = payload
        self.error = error
        self.urls: list[str] = []

    def __call__(self, url: str, timeout: float) -> IO[bytes]:
        self.urls.append(url)
        if self.error is not None:
            raise self.error
        return io.BytesIO(self.payload)


def test_expected_digest_reads_the_manifest_field(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"checkpoint_sha256": PAYLOAD_SHA256, "count": 1}))

    assert fc.expected_digest(manifest) == PAYLOAD_SHA256


def test_matching_download_is_moved_into_place(tmp_path: Path) -> None:
    destination = tmp_path / "final.pt"
    opener = FakeOpener()

    assert fc.fetch(destination, PAYLOAD_SHA256, opener=opener) is True

    assert destination.read_bytes() == PAYLOAD
    assert opener.urls == [fc.RELEASE_URL]
    assert list(tmp_path.iterdir()) == [destination]


def test_missing_directory_is_created(tmp_path: Path) -> None:
    destination = tmp_path / "checkpoints" / "final.pt"

    fc.fetch(destination, PAYLOAD_SHA256, opener=FakeOpener())

    assert destination.read_bytes() == PAYLOAD


def test_mismatched_download_leaves_nothing(tmp_path: Path) -> None:
    destination = tmp_path / "final.pt"

    with pytest.raises(fc.FetchError, match=f"expected {PAYLOAD_SHA256}"):
        fc.fetch(destination, PAYLOAD_SHA256, opener=FakeOpener(payload=b"other bytes"))

    assert list(tmp_path.iterdir()) == []


def test_oversized_download_leaves_nothing(tmp_path: Path) -> None:
    destination = tmp_path / "final.pt"

    with pytest.raises(fc.FetchError, match="exceeded 4 bytes"):
        fc.download(fc.RELEASE_URL, destination, PAYLOAD_SHA256, opener=FakeOpener(), max_bytes=4)

    assert list(tmp_path.iterdir()) == []


def test_failed_connection_leaves_nothing(tmp_path: Path) -> None:
    destination = tmp_path / "final.pt"
    opener = FakeOpener(error=urllib.error.URLError("no route to host"))

    with pytest.raises(OSError, match="no route to host"):
        fc.fetch(destination, PAYLOAD_SHA256, opener=opener)

    assert list(tmp_path.iterdir()) == []


def test_existing_matching_file_is_kept_without_downloading(tmp_path: Path) -> None:
    destination = tmp_path / "final.pt"
    destination.write_bytes(PAYLOAD)
    opener = FakeOpener()

    assert fc.fetch(destination, PAYLOAD_SHA256, opener=opener) is False

    assert opener.urls == []
    assert destination.read_bytes() == PAYLOAD


def test_existing_mismatched_file_is_left_unchanged(tmp_path: Path) -> None:
    destination = tmp_path / "final.pt"
    destination.write_bytes(b"a retrained model")
    opener = FakeOpener()

    with pytest.raises(fc.FetchError, match="left unchanged"):
        fc.fetch(destination, PAYLOAD_SHA256, opener=opener)

    assert opener.urls == []
    assert destination.read_bytes() == b"a retrained model"
    assert list(tmp_path.iterdir()) == [destination]


def test_downloaded_file_is_readable_by_the_image_user(tmp_path: Path) -> None:
    # The runtime image copies the file and runs as a non-root user, and COPY keeps the
    # mode, so a 0600 file from mkstemp could not be read there
    destination = tmp_path / "final.pt"

    fc.fetch(destination, PAYLOAD_SHA256, opener=FakeOpener())

    assert destination.stat().st_mode & 0o777 == 0o644


def test_dangling_symlink_is_not_replaced(tmp_path: Path) -> None:
    destination = tmp_path / "final.pt"
    destination.symlink_to(tmp_path / "missing.pt")
    opener = FakeOpener()

    with pytest.raises(OSError):
        fc.fetch(destination, PAYLOAD_SHA256, opener=opener)

    assert opener.urls == []
    assert destination.is_symlink()
    assert list(tmp_path.iterdir()) == [destination]


def test_main_reports_an_unreadable_manifest_without_a_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "final.pt"
    monkeypatch.setattr(fc, "MANIFEST", tmp_path / "missing.json")
    monkeypatch.setattr("sys.argv", ["fetch_checkpoint.py", "--destination", str(destination)])

    assert fc.main() == 1

    assert "missing.json" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_main_names_the_release_url_when_the_download_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(destination: Path, expected: str) -> bool:
        raise urllib.error.HTTPError(fc.RELEASE_URL, 404, "Not Found", None, None)

    monkeypatch.setattr(fc, "fetch", fail)
    monkeypatch.setattr("sys.argv", ["fetch_checkpoint.py", "--destination", str(tmp_path / "f")])

    assert fc.main() == 1

    err = capsys.readouterr().err
    assert fc.RELEASE_URL in err
    assert "404" in err
