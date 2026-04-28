"""Tests for dataset.acquire manifest downloader."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dataset.acquire import acquire_from_manifest


def _make_manifest(tmp_path: Path, entries: list) -> Path:
    """Write a JSON manifest to tmp_path and return its path."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(entries))
    return manifest


def _fake_urlopen(content: bytes = b"audio data") -> MagicMock:
    """Return a mock for urllib.request.urlopen that yields fake response bytes."""
    resp = MagicMock()
    resp.read.return_value = content
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=resp)


class TestAcquireFromManifest:
    """Tests for acquire_from_manifest security and behavior."""

    def test_downloads_valid_entry(self, tmp_path: Path) -> None:
        """Valid manifest entry is downloaded to the output directory."""
        manifest = _make_manifest(
            tmp_path, [{"url": "https://example.com/song.wav", "filename": "song.wav"}]
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("urllib.request.urlopen", _fake_urlopen()):
            count = acquire_from_manifest(manifest, output_dir)

        assert count == 1
        assert (output_dir / "song.wav").exists()

    def test_path_traversal_filename_cannot_escape_output_dir(self, tmp_path: Path) -> None:
        """Manifest filename with path traversal components cannot write outside output_dir.

        '../../evil.wav' is sanitized (slashes replaced, then .name stripped)
        before being joined to output_dir, so no file reaches the parent directory.
        """
        manifest = _make_manifest(
            tmp_path,
            [{"url": "https://example.com/evil.wav", "filename": "../../evil.wav"}],
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("urllib.request.urlopen", _fake_urlopen()):
            acquire_from_manifest(manifest, output_dir)

        # The attack target must not be created
        assert not (tmp_path / "evil.wav").exists(), (
            "Path traversal must not write files outside output_dir"
        )

    def test_non_http_url_is_skipped(self, tmp_path: Path) -> None:
        """file:// and ftp:// URLs are rejected; urlopen is never called."""
        manifest = _make_manifest(
            tmp_path,
            [
                {"url": "file:///etc/passwd", "filename": "passwd.wav"},
                {"url": "ftp://evil.com/song.wav", "filename": "song.wav"},
            ],
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("urllib.request.urlopen") as mock_open:
            count = acquire_from_manifest(manifest, output_dir)

        assert count == 0
        mock_open.assert_not_called()

    def test_missing_url_entry_is_skipped(self, tmp_path: Path) -> None:
        """Entries without a 'url' key and non-dict entries are silently skipped."""
        manifest = _make_manifest(
            tmp_path,
            [
                {"filename": "no_url.wav"},  # missing url
                "not a dict",
                42,
            ],
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        count = acquire_from_manifest(manifest, output_dir)
        assert count == 0

    def test_corrupt_json_returns_zero(self, tmp_path: Path) -> None:
        """Corrupt or non-JSON manifest returns 0 without raising."""
        manifest = tmp_path / "bad.json"
        manifest.write_text("{not valid json")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        count = acquire_from_manifest(manifest, output_dir)
        assert count == 0

    def test_exclude_keyword_filters_files(self, tmp_path: Path) -> None:
        """Files matching exclude keywords are not downloaded."""
        manifest = _make_manifest(
            tmp_path,
            [
                {"url": "https://example.com/bach.wav", "filename": "bach.wav"},
                {"url": "https://example.com/mozart.wav", "filename": "mozart.wav"},
            ],
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("urllib.request.urlopen", _fake_urlopen()):
            count = acquire_from_manifest(manifest, output_dir, exclude_keywords=["bach"])

        assert count == 1
        assert not (output_dir / "bach.wav").exists()
        assert (output_dir / "mozart.wav").exists()

    def test_format_filter_skips_non_matching_extension(self, tmp_path: Path) -> None:
        """Only files with matching extensions are downloaded when formats is set."""
        manifest = _make_manifest(
            tmp_path,
            [
                {"url": "https://example.com/song.mp3", "filename": "song.mp3"},
                {"url": "https://example.com/song.wav", "filename": "song.wav"},
            ],
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with patch("urllib.request.urlopen", _fake_urlopen()):
            count = acquire_from_manifest(manifest, output_dir, formats={".wav"})

        assert count == 1
        assert (output_dir / "song.wav").exists()
        assert not (output_dir / "song.mp3").exists()

    def test_existing_file_is_not_re_downloaded(self, tmp_path: Path) -> None:
        """Files already present in the output directory are skipped (idempotent)."""
        manifest = _make_manifest(
            tmp_path, [{"url": "https://example.com/song.wav", "filename": "song.wav"}]
        )
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        existing = output_dir / "song.wav"
        existing.write_bytes(b"original content")

        with patch("urllib.request.urlopen") as mock_open:
            count = acquire_from_manifest(manifest, output_dir)

        mock_open.assert_not_called()
        assert count == 1
        assert existing.read_bytes() == b"original content"
