"""Download public domain classical recordings for training data.

Uses the `internetarchive` library (the official IA Python client) to
download from curated Musopen collections on archive.org. Also supports
a local manifest file for custom sources.

Known Musopen collections on Internet Archive:
    - musopen-lossless-dvd       Lossless DVD bundle
    - MusopenCollectionAsFlac    Full catalog as FLAC
    - Musopen-Libre              10.2 GB mixed collection

Usage:
    python -m dataset.acquire --output data/raw/
    python -m dataset.acquire --output data/raw/ --max-tracks 100
    python -m dataset.acquire --collection MusopenCollectionAsFlac --output data/raw/
    python -m dataset.acquire --manifest my_urls.json --output data/raw/

Requires:
    pip install internetarchive
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import internetarchive as ia

logger = logging.getLogger(__name__)

# Audio file extensions we want to download
_AUDIO_EXTENSIONS = {".mp3", ".flac", ".wav", ".ogg"}

# Curated Musopen collection identifiers on Internet Archive.
# These are explicitly public domain / CC-licensed recordings.
_DEFAULT_COLLECTIONS = [
    "MusopenCollectionAsFlac",
]


def list_audio_files(identifier: str) -> list[dict[str, str]]:
    """List audio files in an Internet Archive item.

    Args:
        identifier: Archive.org item identifier.

    Returns:
        List of dicts with 'name' and 'size' keys for each audio file.
    """
    try:
        item = ia.get_item(identifier)
    except Exception as e:
        logger.error("Failed to fetch item %s: %s", identifier, e)
        return []

    audio_files = []
    for f in item.files:
        name = f.get("name", "")
        if Path(name).suffix.lower() in _AUDIO_EXTENSIONS:
            audio_files.append({
                "name": name,
                "size": f.get("size", "0"),
            })

    return audio_files


def download_item(
    identifier: str,
    output_dir: Path,
    max_files: int | None = None,
    formats: set[str] | None = None,
    exclude_keywords: list[str] | None = None,
) -> int:
    """Download audio files from a single Internet Archive item.

    Args:
        identifier: Archive.org item identifier.
        output_dir: Directory to save downloaded files.
        max_files: Maximum number of files to download (None = all).
        formats: File extensions to download (default: all audio).
        exclude_keywords: Skip files whose name contains any of these
            (case-insensitive). Filtering happens before download to
            avoid wasting bandwidth.

    Returns:
        Number of files successfully downloaded.
    """
    if formats is None:
        formats = _AUDIO_EXTENSIONS

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        item = ia.get_item(identifier)
    except Exception as e:
        logger.error("Failed to fetch item %s: %s", identifier, e)
        return 0

    # Filter to audio files only, then exclude keywords -- all before downloading
    exclude_lower = [kw.lower() for kw in (exclude_keywords or [])]
    audio_files = [
        f for f in item.files
        if Path(f.get("name", "")).suffix.lower() in formats
        and not any(kw in f.get("name", "").lower() for kw in exclude_lower)
    ]

    if exclude_lower:
        total_audio = sum(
            1 for f in item.files
            if Path(f.get("name", "")).suffix.lower() in formats
        )
        logger.info(
            "Excluded %d files matching keywords %s",
            total_audio - len(audio_files), exclude_keywords,
        )

    if not audio_files:
        logger.warning("No audio files found in %s", identifier)
        return 0

    if max_files is not None:
        audio_files = audio_files[:max_files]

    logger.info(
        "Downloading %d audio files from %s (%s)",
        len(audio_files), identifier, item.metadata.get("title", identifier),
    )

    downloaded = 0
    for f in audio_files:
        filename = f["name"]
        # Flatten nested paths and prefix with identifier for uniqueness
        safe_name = f"{identifier}__{filename}".replace("/", "_")
        output_path = output_dir / safe_name

        if output_path.exists():
            logger.debug("Skipping (exists): %s", safe_name)
            downloaded += 1
            continue

        try:
            logger.info("Downloading: %s", safe_name)
            item.download(
                files=[filename],
                destdir=str(output_dir / "_tmp_ia"),
                no_directory=False,
                retries=3,
            )

            # Move from IA's nested structure to flat output
            ia_path = output_dir / "_tmp_ia" / identifier / filename
            if ia_path.exists():
                ia_path.rename(output_path)
                downloaded += 1
            else:
                logger.warning("Expected file not found after download: %s", ia_path)

        except Exception as e:
            logger.error("Download failed for %s: %s", filename, e)
            if output_path.exists():
                output_path.unlink()

    # Clean up temp directory
    tmp_dir = output_dir / "_tmp_ia"
    if tmp_dir.exists():
        _rmtree_safe(tmp_dir)

    logger.info("Downloaded %d files from %s", downloaded, identifier)
    return downloaded


def search_and_download(
    query: str,
    output_dir: Path,
    max_tracks: int = 50,
) -> int:
    """Search Internet Archive and download matching audio files.

    Args:
        query: Archive.org search query (e.g. 'collection:musopen').
        output_dir: Directory to save downloaded files.
        max_tracks: Maximum total files to download.

    Returns:
        Number of files successfully downloaded.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Searching: %s", query)

    try:
        results = list(ia.search_items(query))
    except Exception as e:
        logger.error("Search failed: %s", e)
        return 0

    logger.info("Found %d items", len(results))

    downloaded = 0
    for result in results:
        if downloaded >= max_tracks:
            break

        identifier = result.get("identifier", "")
        remaining = max_tracks - downloaded
        downloaded += download_item(identifier, output_dir, max_files=remaining)

    logger.info("Total downloaded: %d tracks", downloaded)
    return downloaded


def acquire_from_collections(
    output_dir: Path,
    collections: list[str] | None = None,
    max_tracks: int = 50,
) -> int:
    """Download from curated Musopen collections.

    Args:
        output_dir: Directory to save downloaded files.
        collections: List of IA item identifiers (default: Musopen collections).
        max_tracks: Maximum total files to download.

    Returns:
        Number of files successfully downloaded.
    """
    if collections is None:
        collections = _DEFAULT_COLLECTIONS

    downloaded = 0
    for identifier in collections:
        if downloaded >= max_tracks:
            break

        remaining = max_tracks - downloaded
        downloaded += download_item(identifier, output_dir, max_files=remaining)

    logger.info("Total: %d tracks from %d collections", downloaded, len(collections))
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
    import urllib.error
    import urllib.request

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path) as f:
        entries = json.load(f)

    downloaded = 0
    for entry in entries:
        url = entry["url"]
        filename = entry.get("filename", Path(url).name)
        output_path = output_dir / filename

        if output_path.exists():
            logger.debug("Skipping (exists): %s", filename)
            downloaded += 1
            continue

        try:
            logger.info("Downloading: %s", filename)
            urllib.request.urlretrieve(url, output_path)
            downloaded += 1
        except (urllib.error.URLError, OSError) as e:
            logger.error("Download failed: %s -- %s", url, e)
            if output_path.exists():
                output_path.unlink()

    logger.info("Downloaded %d files from manifest", downloaded)
    return downloaded


def _rmtree_safe(path: Path) -> None:
    """Remove a directory tree, ignoring errors."""
    import shutil

    try:
        shutil.rmtree(path)
    except OSError:
        pass


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
        "--collection", type=str, default=None,
        help="Specific IA item identifier to download from",
    )
    parser.add_argument(
        "--search", type=str, default=None,
        help="IA search query (e.g. 'collection:musopen')",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Path to JSON manifest with custom download URLs",
    )
    parser.add_argument(
        "--formats", type=str, default=None,
        help="Comma-separated extensions to download (e.g. '.flac' or '.flac,.wav')",
    )
    parser.add_argument(
        "--exclude", type=str, nargs="+", default=None,
        help="Keywords to exclude from filenames (case-insensitive, e.g. --exclude goldberg bach)",
    )

    args = parser.parse_args()

    formats = None
    if args.formats:
        formats = {ext.strip() for ext in args.formats.split(",")}

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.manifest:
        acquire_from_manifest(args.manifest, args.output)
    elif args.collection:
        download_item(
            args.collection, args.output,
            max_files=args.max_tracks, formats=formats,
            exclude_keywords=args.exclude,
        )
    elif args.search:
        search_and_download(args.search, args.output, max_tracks=args.max_tracks)
    else:
        acquire_from_collections(args.output, max_tracks=args.max_tracks)


if __name__ == "__main__":
    main()
