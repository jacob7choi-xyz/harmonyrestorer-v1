"""List each v12 stratum's candidate universe once from its source and write the frozen file.

This is the only step that reads the sources' live metadata. For `solo_piano` it fetches the
Internet Archive item's metadata; for `chamber` it fetches MusicNet's metadata file from its
Zenodo record, whose md5 the protocol freezes. Each fetch goes once through
`benchmark.https_get`. It refuses to overwrite an existing universe file, because each
universe is frozen once. For each stratum it prints the file's SHA-256, which is then pinned in
`benchmark.candidate_universe.UNIVERSE_SHA256` by hand in the same commit, the number of
candidates per cluster, and how many admissions the cluster cap allows against how many the
stratum's partitions need. No audio is fetched.

Run from the repository root, with network access:
`uv run --frozen python -m benchmark.measurements.candidate_listing`.
"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from benchmark import candidate_universe
from benchmark.atomic_write import write_atomically
from benchmark.candidate_universe import (
    CandidateUniverseError,
    ListedFile,
    archive_item_candidates,
    chamber_candidates,
    universe_bytes,
)
from benchmark.https_get import Opener, get
from benchmark.protocol import ArchiveItemStratum, Stratum, load_protocol

logger = logging.getLogger(__name__)

# Each source's metadata is a small document; this bounds it far above its expected size.
MAX_LISTING_BYTES = 64 * 1024 * 1024
TIMEOUT_S = 120.0
_OK = 200


class ListingError(RuntimeError):
    """Raised when a listing cannot be fetched whole or its file already exists."""


def _fetch(url: str, opener: Opener | None) -> bytes:
    """One whole 200 response from *url*, or a ListingError."""
    body = io.BytesIO()
    kwargs = {} if opener is None else {"opener": opener}
    response = get(url, body, max_bytes=MAX_LISTING_BYTES, timeout_s=TIMEOUT_S, **kwargs)
    short = response.content_length is not None and response.size != response.content_length
    if response.status != _OK or response.exceeded or short:
        raise ListingError(
            f"{url} returned status {response.status}, {response.size} bytes, "
            f"cap exceeded {response.exceeded}, declared length {response.content_length}"
        )
    return body.getvalue()


def list_stratum(stratum: Stratum, destination: Path, *, opener: Opener | None = None) -> bytes:
    """Fetch the stratum's metadata once and return its frozen file's bytes.

    Raises:
        ListingError: If *destination* exists, or the fetch is not a whole 200 response.
        CandidateUniverseError: If the metadata does not describe the stratum as frozen.
        OSError: On a network failure.
    """
    if destination.exists():
        raise ListingError(f"{destination} exists; each universe is listed once")
    files: tuple[ListedFile, ...]
    if isinstance(stratum, ArchiveItemStratum):
        url = candidate_universe.metadata_url(stratum.source)
        files = archive_item_candidates(json.loads(_fetch(url, opener)), stratum)
    else:
        url = candidate_universe.zenodo_file_url(stratum.zenodo_record, stratum.metadata)
        files = chamber_candidates(_fetch(url, opener), stratum)
    enumerated_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return universe_bytes(
        files, stratum=stratum.name, source=stratum.source, query=url, enumerated_at=enumerated_at
    )


def summary(stratum: Stratum, data: bytes) -> list[str]:
    """Lines describing a listed universe: digest, clusters, and capacity under the cap."""
    candidates = json.loads(data)["candidates"]
    clusters = collections.Counter(c["cluster"] for c in candidates)
    capacity = sum(min(n, stratum.cluster_cap) for n in clusters.values())
    needed = sum(stratum.partitions.values())
    lines = [
        f"{stratum.name}: {len(candidates)} candidates in {len(clusters)} clusters",
        f"  cap {stratum.cluster_cap} allows {capacity} admissions; partitions need {needed}",
        f"  sha256 {hashlib.sha256(data).hexdigest()}",
    ]
    lines += [f"  {n:4d}  {name}" for name, n in clusters.most_common()]
    return lines


def main() -> int:
    """List every stratum, write each file, and print what was listed; 0 on success."""
    strata = load_protocol().selection.strata
    listed: list[tuple[Stratum, Path, bytes]] = []
    try:
        for stratum in strata:
            path = candidate_universe.universe_path(stratum.name)
            listed.append((stratum, path, list_stratum(stratum, path)))
    except (ListingError, CandidateUniverseError, OSError) as error:
        logger.error("Nothing written: %s", error)
        return 1
    for stratum, path, data in listed:
        write_atomically(path, data)
        print("\n".join(summary(stratum, data)))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())
