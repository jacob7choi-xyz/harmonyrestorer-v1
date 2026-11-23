"""Fetching and staging one candidate file under the protocol's fetch rules.

Protocol v10 to v12 fix what a fetch failure means, because a skipped candidate would let
network conditions choose the test set (`selection.transient_failure`,
`selection.permanent_failure_codes`, `selection.listing_check`, `selection.oversize_fetch`):

- **Attempts.** At most `1 + acquisition_retries`. A status in `permanent_failure_codes`
  ends the fetch at once as `PermanentlyUnavailable`. Every other failure is transient: any
  other status, a network error, a timeout, a transfer that ends short, a listing mismatch.
- **Listing check.** After each attempt, every value the frozen universe recorded (size, md5,
  sha1) is compared with the bytes received; an absent value is not compared. If every
  attempt mismatched, the result is `ListingMismatch`. If the attempts run out with any other
  transient failure among them, `RetriesExhausted` is raised and the run is invalid.
- **Oversize.** A body longer than `oversize_bound` is stopped at `oversize_bound + 1` bytes,
  the prefix is staged so the source-outcome stages exclude it without reading it, and no raw
  digest is reported for it. Its listed size, if recorded, is checked only to be above the bound.
  An attempt writes at most one byte past the smaller of its listed size and the bound
  (`_attempt_cap`), so a body longer than a listed size that is under the bound stops one byte
  past that size, and the listing check then names the size.

**Waits between attempts** are a harness setting, since they cannot change which candidates
are admitted, but they decide whether a brief rate limit invalidates the run. `WAITS_S` holds
30, 120 and 600 seconds, and a numeric Retry-After from the server is honoured up to
`MAX_WAIT_S`.

**Staging.** Each file is written to a temporary file in the staging directory, flushed and
fsynced, then moved to a name built from the SHA-256 of the identifier, never from the listed
path. A sidecar JSON records what was accepted. On a rerun a staged file is reused, with no
request made, only if its sidecar names the same identifier and its recomputed size and
digests still match the sidecar and the listing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.https_get import Response, get

__all__ = [
    "MAX_WAIT_S",
    "WAITS_S",
    "Attempt",
    "Expected",
    "FetchOutcome",
    "Getter",
    "ListingMismatch",
    "PermanentlyUnavailable",
    "RetriesExhausted",
    "Staged",
    "download_url",
    "fetch_file",
]

logger = logging.getLogger(__name__)

WAITS_S = (30.0, 120.0, 600.0)
MAX_WAIT_S = 600.0
TIMEOUT_S = 120.0
_OK = 200
_CHUNK_BYTES = 1 << 20
_FILE_MODE = 0o600

Getter = Callable[..., Response]


@dataclass(frozen=True)
class Expected:
    """What the bytes must be: the listed values (None where absent) and the size bound.

    `oversize_bound` is the size above which the oversize rule applies (a candidate file), or
    None when any size other than `size` is a mismatch (an archive fetched whole).
    """

    size: int | None
    md5: str | None
    sha1: str | None
    oversize_bound: int | None


@dataclass(frozen=True)
class Attempt:
    """One attempt: its status or error class, and the listing values compared and failed."""

    status: int | None
    error: str | None
    compared: tuple[str, ...]
    mismatched: tuple[str, ...]


@dataclass(frozen=True)
class Staged:
    """A staged file. `sha256` is None for an oversize prefix, which has no raw digest."""

    identifier: str
    path: Path
    size: int
    sha256: str | None
    oversize: bool
    compared: tuple[str, ...]
    attempts: tuple[Attempt, ...]
    reused: bool


@dataclass(frozen=True)
class PermanentlyUnavailable:
    """The source answered with a permanent status; the candidate is excluded."""

    identifier: str
    status: int
    attempts: tuple[Attempt, ...]


@dataclass(frozen=True)
class ListingMismatch:
    """Every attempt delivered bytes the listing does not describe; the candidate is excluded."""

    identifier: str
    attempts: tuple[Attempt, ...]


class RetriesExhausted(RuntimeError):
    """Every attempt failed transiently, not all by listing mismatch; the run is invalid."""

    def __init__(self, identifier: str, attempts: tuple[Attempt, ...]) -> None:
        super().__init__(f"{identifier}: {len(attempts)} attempts failed transiently")
        self.identifier = identifier
        self.attempts = attempts


FetchOutcome = Staged | PermanentlyUnavailable | ListingMismatch


def download_url(item: str, path: str) -> str:
    """The Internet Archive download URL, each path segment percent-encoded from UTF-8."""
    segments = (urllib.parse.quote(segment, safe="") for segment in path.split("/"))
    return f"https://archive.org/download/{urllib.parse.quote(item, safe='')}/{'/'.join(segments)}"


def staged_paths(staging: Path, identifier: str, suffix: str) -> tuple[Path, Path]:
    """The staged file and its sidecar, named from the identifier's SHA-256."""
    stem = hashlib.sha256(identifier.encode("utf-8", "surrogatepass")).hexdigest()
    return staging / f"{stem}{suffix}", staging / f"{stem}{suffix}.json"


def _listing_check(
    expected: Expected, response: Response
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The listing values compared with *response* and the ones that did not match."""
    bound = expected.oversize_bound
    if bound is not None and response.size > bound:
        listed_ok = expected.size is None or expected.size > bound
        return (("size",) if expected.size is not None else ()), (() if listed_ok else ("size",))
    received = {"size": response.size, "md5": response.md5, "sha1": response.sha1}
    listed = {"size": expected.size, "md5": expected.md5, "sha1": expected.sha1}
    compared = tuple(k for k, v in listed.items() if v is not None)
    return compared, tuple(k for k in compared if listed[k] != received[k])


def _transfer_failure(response: Response, expected: Expected) -> str | None:
    """Why a 200 response is not a whole transfer, or None if it is.

    A body over the size bound, or longer than the attempt's cap, is not short: an oversize
    candidate is staged as a prefix, and an archive longer than its listed size fails the listing
    check on its size. The cap can be below the bound, so `exceeded` counts on its own.
    """
    bound = expected.oversize_bound
    over = response.exceeded or (bound is not None and response.size > bound)
    declared = response.content_length
    if not over and declared is not None and response.size != declared:
        return "transfer ended short"
    return None


def _write_sidecar(sidecar: Path, record: dict[str, Any]) -> None:
    """Write the sidecar atomically: temporary file, fsync, then replace."""
    fd, tmp = tempfile.mkstemp(prefix=".sidecar.", dir=sidecar.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(record, out, sort_keys=True)
            out.flush()
            os.fsync(out.fileno())
        Path(tmp).replace(sidecar)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _digests(path: Path) -> tuple[int, str, str, str]:
    """Size, sha256, md5 and sha1 of the file at *path*, read once."""
    hashes = (
        hashlib.sha256(),
        hashlib.md5(usedforsecurity=False),
        hashlib.sha1(usedforsecurity=False),
    )
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            size += len(chunk)
            for h in hashes:
                h.update(chunk)
    return size, hashes[0].hexdigest(), hashes[1].hexdigest(), hashes[2].hexdigest()


def _reuse(identifier: str, expected: Expected, staged: Path, sidecar: Path) -> Staged | None:
    """The staged file, if its sidecar and recomputed digests still match; otherwise None."""
    if not staged.is_file() or not sidecar.is_file():
        return None
    try:
        record = json.loads(sidecar.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("identifier") != identifier:
        return None
    size, sha256, md5, sha1 = _digests(staged)
    if (size, sha256) != (record.get("size"), record.get("written_sha256")):
        return None
    response = Response(_OK, size, False, size, sha256, md5, sha1)
    compared, mismatched = _listing_check(expected, response)
    if mismatched:
        return None
    oversize = expected.oversize_bound is not None and size > expected.oversize_bound
    logger.info("Reusing staged %s", identifier)
    return Staged(
        identifier, staged, size, None if oversize else sha256, oversize, compared, (), True
    )


def _attempt_cap(expected: Expected) -> int:
    """The most bytes one attempt writes: the listed size, capped at the oversize bound, plus one.

    The extra byte lets the transfer show that it is longer than its listing. With no listed size,
    the cap is the bound, or zero when there is no bound, plus one.
    """
    bound = expected.oversize_bound
    if expected.size is None:
        return (bound if bound is not None else 0) + 1
    return (expected.size if bound is None else min(expected.size, bound)) + 1


def _attempt_once(
    url: str, staging: Path, expected: Expected, get_: Getter
) -> tuple[Attempt, Response | None, Path | None]:
    """One attempt into a temporary file; the record, the response, and the temp if kept."""
    cap = _attempt_cap(expected)
    fd, tmp_name = tempfile.mkstemp(prefix=".fetch.", dir=staging)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out:
            response = get_(url, out, max_bytes=cap, timeout_s=TIMEOUT_S)
            out.flush()
            os.fchmod(out.fileno(), _FILE_MODE)
            os.fsync(out.fileno())
    except OSError as error:
        tmp.unlink(missing_ok=True)
        logger.warning("GET %s failed: %r", url, error)
        return Attempt(None, type(error).__name__, (), ()), None, None
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    failure = None if response.status != _OK else _transfer_failure(response, expected)
    if response.status != _OK or failure is not None:
        tmp.unlink(missing_ok=True)
        return Attempt(response.status, failure, (), ()), response, None
    compared, mismatched = _listing_check(expected, response)
    if mismatched:
        tmp.unlink(missing_ok=True)
        return Attempt(response.status, None, compared, mismatched), response, None
    return Attempt(response.status, None, compared, ()), response, tmp


def _wait_s(attempt_index: int, response: Response | None, waits: tuple[float, ...]) -> float:
    """How long to wait before attempt *attempt_index* (1-based retries)."""
    base = waits[min(attempt_index - 1, len(waits) - 1)]
    hinted = response.retry_after if response is not None else None
    return min(max(base, hinted or 0.0), MAX_WAIT_S)


def fetch_file(
    identifier: str,
    url: str,
    expected: Expected,
    staged_as: tuple[Path, Path],
    *,
    retries: int,
    permanent: tuple[int, ...],
    get_: Getter = get,
    sleep: Callable[[float], None] = time.sleep,
    waits: tuple[float, ...] = WAITS_S,
) -> FetchOutcome:
    """Fetch one file under the protocol's attempt rules and stage it, or reuse it.

    Args:
        identifier: The candidate's identifier, recorded in the sidecar.
        url: The HTTPS URL to fetch.
        expected: The listed values and the size bound.
        staged_as: The staged file and its sidecar (`staged_paths`), in an absolute directory.
        retries: `selection.acquisition_retries`.
        permanent: `selection.permanent_failure_codes`.
        get_: The GET; tests replace it.
        sleep: Waits between attempts; tests replace it.
        waits: The wait before each retry, in seconds.

    Returns:
        The staged file, or why the candidate is excluded.

    Raises:
        RetriesExhausted: If every attempt failed transiently, not all by listing mismatch.
        ValueError: If the staging directory is not absolute.
    """
    staged, sidecar = staged_as
    if not staged.parent.is_absolute():
        raise ValueError(f"staging directory must be absolute, got {staged.parent}")
    reused = _reuse(identifier, expected, staged, sidecar)
    if reused is not None:
        return reused
    return _run_attempts(
        identifier, url, expected, staged_as, retries, permanent, get_, sleep, waits
    )


def _run_attempts(
    identifier: str,
    url: str,
    expected: Expected,
    staged_as: tuple[Path, Path],
    retries: int,
    permanent: tuple[int, ...],
    get_: Getter,
    sleep: Callable[[float], None],
    waits: tuple[float, ...],
) -> FetchOutcome:
    """Up to `retries + 1` attempts, classified as `fetch_file` describes."""
    staged, sidecar = staged_as
    attempts: list[Attempt] = []
    response: Response | None = None
    for index in range(retries + 1):
        if index:
            sleep(_wait_s(index, response, waits))
        attempt, response, tmp = _attempt_once(url, staged.parent, expected, get_)
        attempts.append(attempt)
        if attempt.status in permanent:
            logger.info("%s permanently unavailable: %d", identifier, attempt.status)
            return PermanentlyUnavailable(identifier, attempt.status, tuple(attempts))
        if tmp is not None and response is not None:
            return _accept(identifier, expected, tmp, (staged, sidecar), response, tuple(attempts))
    if all(a.mismatched for a in attempts):
        logger.info("%s: every attempt mismatched its listing", identifier)
        return ListingMismatch(identifier, tuple(attempts))
    logger.error("%s: attempts exhausted; the run is invalid", identifier)
    raise RetriesExhausted(identifier, tuple(attempts))


def _accept(
    identifier: str,
    expected: Expected,
    tmp: Path,
    staged_as: tuple[Path, Path],
    response: Response,
    attempts: tuple[Attempt, ...],
) -> Staged:
    """Move an accepted temporary file into place and record it in its sidecar."""
    staged, sidecar = staged_as
    oversize = expected.oversize_bound is not None and response.size > expected.oversize_bound
    try:
        tmp.replace(staged)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    record = {
        "identifier": identifier,
        "size": response.size,
        "written_sha256": response.sha256,
        "md5": response.md5,
        "sha1": response.sha1,
        "oversize": oversize,
    }
    _write_sidecar(sidecar, record)
    compared = attempts[-1].compared
    sha256 = None if oversize else response.sha256
    return Staged(identifier, staged, response.size, sha256, oversize, compared, attempts, False)
