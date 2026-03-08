"""Download public domain classical recordings for training data.

Uses the Internet Archive API to find and download CC/public domain
classical music recordings. Also supports a local manifest file for
custom sources.

Usage:
    python -m dataset.acquire --output data/raw/
    python -m dataset.acquire --output data/raw/ --max-tracks 100
    python -m dataset.acquire --manifest my_urls.json --output data/raw/
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)

# Internet Archive search API
_IA_SEARCH_URL = "https://archive.org/advancedsearch.php"
_IA_METADATA_URL = "https://archive.org/metadata"
_IA_DOWNLOAD_URL = "https://archive.org/download"

# Audio file extensions we want
_AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg"}

# Search queries for finding classical recordings on Internet Archive
_SEARCH_QUERIES = [
    # Musopen: specifically recorded to be free/CC
    'collection:musopen AND mediatype:audio',
    # Public domain classical recordings
    'subject:"classical music" AND licenseurl:*creativecommons* AND mediatype:audio',
]


def search_archive(
    query: str,
    max_results: int = 50,
) -> list[dict[str, str]]:
    """Search Internet Archive for audio recordings.

    Args:
        query: Archive.org advanced search query.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with 'identifier' and 'title' keys.
    """
    params = (
        f"?q={urllib.request.quote(query)}"
        f"&fl[]=identifier&fl[]=title"
        f"&rows={max_results}"
        f"&output=json"
    )
    url = f"{_IA_SEARCH_URL}{params}"

    logger.info("Searching: %s", query)

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            docs = data.get("response", {}).get("docs", [])
            logger.info("Found %d results", len(docs))
            return docs
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        logger.error("Search failed: %s", e)
        return []


def get_audio_files(identifier: str) -> list[str]:
    """Get list of audio files in an Internet Archive item.

    Args:
        identifier: Archive.org item identifier.

    Returns:
        List of audio filenames available for download.
    """
    url = f"{_IA_METADATA_URL}/{identifier}"

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            files = data.get("files", [])
            audio_files = [
                f["name"]
                for f in files
                if Path(f.get("name", "")).suffix.lower() in _AUDIO_EXTENSIONS
            ]
            return audio_files
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        logger.error("Metadata fetch failed for %s: %s", identifier, e)
        return []


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from a URL.

    Args:
        url: Source URL.
        output_path: Local path to save to.

    Returns:
        True if download succeeded.
    """
    if output_path.exists():
        logger.debug("Skipping (exists): %s", output_path.name)
        return True

    try:
        logger.info("Downloading: %s", output_path.name)
        urllib.request.urlretrieve(url, output_path)
        return True
    except (urllib.error.URLError, OSError) as e:
        logger.error("Download failed: %s -- %s", url, e)
        if output_path.exists():
            output_path.unlink()
        return False


def acquire_from_archive(
    output_dir: Path,
    max_tracks: int = 50,
) -> int:
    """Download classical recordings from Internet Archive.

    Args:
        output_dir: Directory to save downloaded files.
        max_tracks: Maximum number of tracks to download.

    Returns:
        Number of files successfully downloaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for query in _SEARCH_QUERIES:
        if downloaded >= max_tracks:
            break

        items = search_archive(query, max_results=max_tracks - downloaded)

        for item in items:
            if downloaded >= max_tracks:
                break

            identifier = item.get("identifier", "")
            title = item.get("title", identifier)

            audio_files = get_audio_files(identifier)
            if not audio_files:
                continue

            logger.info("Processing: %s (%d audio files)", title, len(audio_files))

            for filename in audio_files:
                if downloaded >= max_tracks:
                    break

                # Sanitize filename: use identifier prefix for uniqueness
                safe_name = f"{identifier}__{filename}".replace("/", "_")
                output_path = output_dir / safe_name

                url = f"{_IA_DOWNLOAD_URL}/{identifier}/{urllib.request.quote(filename)}"
                if download_file(url, output_path):
                    downloaded += 1

    logger.info("Downloaded %d tracks to %s", downloaded, output_dir)
    return downloaded


def acquire_from_manifest(
    manifest_path: Path,
    output_dir: Path,
) -> int:
    """Download files from a JSON manifest.

    Manifest format:
        [
            {"url": "https://...", "filename": "recording.wav"},
            ...
        ]

    Args:
        manifest_path: Path to JSON manifest file.
        output_dir: Directory to save downloaded files.

    Returns:
        Number of files successfully downloaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        entries = json.load(f)

    downloaded = 0
    for entry in entries:
        url = entry["url"]
        filename = entry.get("filename", Path(url).name)
        output_path = output_dir / filename

        if download_file(url, output_path):
            downloaded += 1

    logger.info("Downloaded %d files from manifest", downloaded)
    return downloaded


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Download public domain classical recordings for training data."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw"),
        help="Output directory for downloaded files (default: data/raw/)",
    )
    parser.add_argument(
        "--max-tracks", type=int, default=50,
        help="Maximum number of tracks to download (default: 50)",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Path to JSON manifest with custom download URLs",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.manifest:
        acquire_from_manifest(args.manifest, args.output)
    else:
        acquire_from_archive(args.output, max_tracks=args.max_tracks)


if __name__ == "__main__":
    main()
