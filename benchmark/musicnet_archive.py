"""The chamber stratum's one archive: fetched whole, then only the frozen ids extracted.

Protocol v12 publishes MusicNet as one archive and treats it as one fetch under the fetch
rules (`selection.strata.chamber.archive_note`): its size must be `archive_bytes` and its md5
`archive_md5`, and it is bounded by that size rather than by `max_identity_stream_bytes`,
since it is not itself a candidate source. `fetch_archive` applies `benchmark.fetch` to it, so
the attempts, waits, staging and reuse are the same as for a candidate file, and the archive's
SHA-256, computed while streaming, is recorded in its sidecar for the audit trail and the
publication.

`extract_members` reads the gzip stream once. For each frozen id it takes the one regular
file whose final path component is `<id>.wav` and writes it to `<id>.wav` in the output
directory. A member's name never becomes a path, and a link, device or directory member is
never followed, so nothing in the archive can write outside that directory. Each member is a
candidate source: it is capped at `member_cap + 1` bytes like any fetched file, and an oversize
member keeps no digest. If any id is found zero times or more than once, everything written is
removed and the archive is refused. The members are extracted again on every run rather than
reused, since the archive they come from is checked on every run.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tarfile
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import IO, NoReturn

from benchmark.candidate_universe import zenodo_file_url
from benchmark.fetch import WAITS_S, Expected, FetchOutcome, Getter, fetch_file, staged_paths
from benchmark.https_get import get
from benchmark.protocol import Selection, ZenodoArchiveStratum

__all__ = ["ArchiveError", "Member", "extract_members", "fetch_archive"]

logger = logging.getLogger(__name__)

_MEMBER = re.compile(r"(\d+)\.wav")
_CHUNK_BYTES = 1 << 20
_FILE_MODE = 0o600


class ArchiveError(ValueError):
    """Raised when the archive does not hold each frozen id exactly once."""


@dataclass(frozen=True)
class Member:
    """One extracted candidate. `sha256` is None for an oversize prefix."""

    identifier: int
    path: Path
    size: int
    sha256: str | None
    oversize: bool


def fetch_archive(
    stratum: ZenodoArchiveStratum,
    selection: Selection,
    staging: Path,
    *,
    get_: Getter = get,
    sleep: Callable[[float], None] = time.sleep,
    waits: tuple[float, ...] = WAITS_S,
) -> FetchOutcome:
    """Fetch the stratum's archive whole, or reuse a staged copy that still matches.

    Raises:
        RetriesExhausted: If every attempt failed transiently, not all by listing mismatch.
    """
    identifier = f"{stratum.source}/{stratum.archive}"
    return fetch_file(
        identifier,
        zenodo_file_url(stratum.zenodo_record, stratum.archive),
        Expected(stratum.archive_bytes, stratum.archive_md5, None, None),
        staged_paths(staging, identifier, ".tar.gz"),
        retries=selection.acquisition_retries,
        permanent=selection.permanent_failure_codes,
        get_=get_,
        sleep=sleep,
        waits=waits,
    )


def _refuse(written: list[Path], message: str) -> NoReturn:
    """Remove what was written, log, and raise *message* as an ArchiveError."""
    for path in written:
        path.unlink(missing_ok=True)
    logger.error("MusicNet archive: %s", message)
    raise ArchiveError(message)


def _copy_capped(source: IO[bytes], out_dir: Path, name: str, cap: int) -> tuple[Path, int, str]:
    """Copy at most *cap* bytes of *source* to *name* atomically; path, size, sha256."""
    fd, tmp_name = tempfile.mkstemp(prefix=".member.", dir=out_dir)
    tmp, digest, size = Path(tmp_name), hashlib.sha256(), 0
    try:
        with os.fdopen(fd, "wb") as out:
            while size < cap and (chunk := source.read(min(_CHUNK_BYTES, cap - size))):
                digest.update(chunk)
                out.write(chunk)
                size += len(chunk)
            out.flush()
            os.fchmod(out.fileno(), _FILE_MODE)
            os.fsync(out.fileno())
        target = out_dir / name
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target, size, digest.hexdigest()


def extract_members(
    archive: Path, ids: tuple[int, ...], out_dir: Path, *, member_cap: int
) -> dict[int, Member]:
    """Extract each frozen id's one regular-file member, in one pass over the stream.

    Args:
        archive: The staged archive, checked by `fetch_archive`.
        ids: The stratum's `candidate_ids`.
        out_dir: An absolute, existing directory for the members.
        member_cap: `max_identity_stream_bytes`; a longer member is kept as a prefix.

    Raises:
        ArchiveError: If an id is missing from the archive or appears more than once.
        ValueError: If *out_dir* is not absolute.
    """
    if not out_dir.is_absolute():
        raise ValueError(f"output directory must be absolute, got {out_dir}")
    wanted, seen = set(ids), dict.fromkeys(ids, 0)
    members: dict[int, Member] = {}
    written: list[Path] = []
    try:
        with tarfile.open(archive, mode="r|gz") as tar:
            for info in tar:
                match = _MEMBER.fullmatch(PurePosixPath(info.name).name)
                if not info.isreg() or match is None or int(match.group(1)) not in wanted:
                    continue
                identifier = int(match.group(1))
                seen[identifier] += 1
                if seen[identifier] > 1:
                    continue
                source = tar.extractfile(info)
                if source is None:
                    _refuse(written, f"member for id {identifier} cannot be read")
                path, size, sha256 = _copy_capped(
                    source, out_dir, f"{identifier}.wav", member_cap + 1
                )
                written.append(path)
                over = size > member_cap
                members[identifier] = Member(identifier, path, size, None if over else sha256, over)
    except BaseException:
        for path in written:
            path.unlink(missing_ok=True)
        raise
    bad = sorted(i for i, n in seen.items() if n != 1)
    if bad:
        _refuse(written, f"ids not found exactly once: {[(i, seen[i]) for i in bad]}")
    return members
