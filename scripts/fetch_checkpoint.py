"""Download the OpGAN checkpoint from its GitHub release and check its digest.

The checkpoint is not in the repository. It is published as an asset of the immutable
release opgan-checkpoint-v1 (docs/adr/008-checkpoint-release.md). The expected SHA-256 is
checkpoint_sha256 in benchmark/manifests/training_corpus.json, which a test keeps equal to
OPGAN_CHECKPOINT_SHA256 in backend/app/config.py.

This check only makes a wrong download fail early. The backend checks the digest again
before it loads the file (docs/adr/007-checkpoint-digest.md), and that is the check that
guards loading.

The download is written to a temporary file beside the destination, hashed as it is
written, and moved into place only if the digest matches, so a failed or wrong download
leaves nothing at the destination. The file is made readable by all users, because the
runtime image copies it with its mode and runs as a non-root user. An existing file or
link at the destination is not replaced: a file that matches is kept, and anything else
is reported and left for a person to look at. This is checked before the download, so a
file created at the destination while the download runs would still be replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import IO

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_URL = (
    "https://github.com/jacob7choi-xyz/harmonyrestorer-v1/releases/download/"
    "opgan-checkpoint-v1/final.pt"
)
DEFAULT_DESTINATION = REPO_ROOT / "checkpoints" / "final.pt"
MANIFEST = REPO_ROOT / "benchmark" / "manifests" / "training_corpus.json"

# The published file is 18,027,171 bytes. The cap bounds what a wrong response can write.
MAX_BYTES = 32 * 1024 * 1024
# Bounds each socket operation, not the whole download; the CI job's own timeout bounds that
TIMEOUT_S = 120.0
# mkstemp creates the file 0600, which the runtime image's non-root user could not read
FILE_MODE = 0o644
_CHUNK_BYTES = 1024 * 1024

Opener = Callable[[str, float], IO[bytes]]


class FetchError(Exception):
    """Raised when the checkpoint cannot be fetched or does not have the expected digest."""


def expected_digest(manifest: Path) -> str:
    """Return the checkpoint SHA-256 recorded in the training corpus manifest."""
    return str(json.loads(manifest.read_text())["checkpoint_sha256"])


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _urlopen(url: str, timeout: float) -> IO[bytes]:
    """Open a URL with urllib, which follows the release's redirect to its storage host."""
    return urllib.request.urlopen(url, timeout=timeout)


def download(
    url: str,
    destination: Path,
    expected: str,
    *,
    opener: Opener = _urlopen,
    max_bytes: int = MAX_BYTES,
) -> None:
    """Download url to destination only if its SHA-256 equals expected.

    Args:
        url: The URL to download.
        destination: The final path. Its directory must exist.
        expected: Lowercase hex SHA-256 the download must have.
        opener: Opens the URL; replaced in tests.
        max_bytes: The download is refused once it exceeds this many bytes.

    Raises:
        FetchError: If the download is too large or has a different digest.
        OSError: If the download or a write fails. urllib's URLError is an OSError.
    """
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    tmp = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        written = 0
        with os.fdopen(fd, "wb") as out, opener(url, TIMEOUT_S) as response:
            for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                written += len(chunk)
                if written > max_bytes:
                    raise FetchError(f"download exceeded {max_bytes} bytes")
                digest.update(chunk)
                out.write(chunk)
            out.flush()
            os.fchmod(out.fileno(), FILE_MODE)
            os.fsync(out.fileno())
        actual = digest.hexdigest()
        if actual != expected:
            raise FetchError(f"download has SHA-256 {actual}, expected {expected}")
        tmp.replace(destination)
    except BaseException:  # broad: also removes the temporary file on Ctrl+C, then re-raises
        tmp.unlink(missing_ok=True)
        raise


def fetch(destination: Path, expected: str, *, opener: Opener = _urlopen) -> bool:
    """Make destination hold the checkpoint with the expected digest.

    Args:
        destination: Where the checkpoint should be.
        expected: Lowercase hex SHA-256 the checkpoint must have.
        opener: Opens the release URL; replaced in tests.

    Returns:
        True if the checkpoint was downloaded, False if a matching file was already there.

    Raises:
        FetchError: If an existing file has a different digest (it is left unchanged), or
            the download is too large or has a different digest (nothing is written).
        OSError: If the download or a write fails, or the destination is a link whose
            target does not exist (it is left unchanged).
    """
    # is_symlink as well, because exists() is False for a dangling link, which the
    # replace below would otherwise swap for a regular file
    if destination.exists() or destination.is_symlink():
        actual = file_digest(destination)
        if actual != expected:
            raise FetchError(
                f"{destination} has SHA-256 {actual}, expected {expected}; it was left unchanged"
            )
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    download(RELEASE_URL, destination, expected, opener=opener)
    return True


def main() -> int:
    """Fetch the checkpoint and report the outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()

    try:
        expected = expected_digest(MANIFEST)
    except (OSError, ValueError, KeyError) as e:
        print(f"expected digest not read from {MANIFEST}: {e!r}", file=sys.stderr)
        return 1
    try:
        downloaded = fetch(args.destination, expected)
    except (FetchError, OSError) as e:
        print(f"checkpoint not fetched from {RELEASE_URL}: {e}", file=sys.stderr)
        return 1
    action = "downloaded" if downloaded else "already present"
    print(f"{args.destination}: {action}, SHA-256 {expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
