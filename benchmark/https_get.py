"""One bounded HTTPS GET, shared by the candidate listing and the candidate fetch.

The listing reads the collection's metadata once; the fetch downloads each candidate. Both go
through `get`, so they share one set of guarantees:

- **HTTPS only,** for the request and for every redirect. The archive redirects a download to
  a storage host, and a redirect to plain HTTP is refused rather than followed. Certificates
  are verified by the standard library's default context.
- **A byte cap.** At most `max_bytes` bytes are written to the sink. If the server offers more,
  the response is marked `exceeded` and the transfer stops there, so an oversize file costs
  at most `max_bytes` plus one read.
- **No retries.** One call is one attempt. The protocol counts attempts (`acquisition_retries`),
  so a retry inside this layer would make more attempts than the run records.
- **The response is reported, not judged.** An HTTP error status is returned as `status`;
  whether it is permanent or transient is the caller's rule. Network errors, timeouts and a
  refused redirect raise `OSError`, which the caller classifies too.

The digests of the bytes written are computed as they stream, so a large file is read once.
"""

from __future__ import annotations

import hashlib
import http.client
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from email.message import Message
from typing import IO, Any, BinaryIO

__all__ = ["InsecureRedirectError", "Response", "TransferError", "get"]

logger = logging.getLogger(__name__)

# A generic client name; no contact details are sent.
USER_AGENT = "harmonyrestorer-benchmark"
_CHUNK_BYTES = 1 << 20
_HTTPS = "https"
# ASCII digits only. str.isdigit also accepts superscripts and other scripts, which int and
# float then refuse, so a header carrying them would raise instead of being ignored.
_ASCII_DIGITS = re.compile(r"[0-9]+")


class InsecureRedirectError(urllib.error.URLError):
    """Raised when a server redirects an HTTPS request to another scheme."""


class TransferError(OSError):
    """Raised when the response is malformed or the connection ends inside it."""


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follows redirects only to HTTPS URLs."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        """Refuse a redirect whose target is not HTTPS; otherwise redirect as usual."""
        if urllib.parse.urlsplit(newurl).scheme != _HTTPS:
            logger.error("Refused a redirect from %s to %s", req.full_url, newurl)
            raise InsecureRedirectError(f"redirect to a non-HTTPS URL: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


@dataclass(frozen=True)
class Response:
    """What one GET returned: its status, and the size and digests of the bytes written.

    `content_length` is the length the server declared, or None, so a caller can tell a
    transfer that ended short. `exceeded` means the server offered more than the cap.
    `retry_after` is the server's Retry-After in seconds when it gave one as a number.
    """

    status: int
    size: int
    exceeded: bool
    content_length: int | None
    sha256: str
    md5: str
    sha1: str
    retry_after: float | None = None


Opener = Callable[[urllib.request.Request, float], IO[bytes]]


def _open(request: urllib.request.Request, timeout_s: float) -> IO[bytes]:
    """Open *request* with redirects restricted to HTTPS and the default certificate checks."""
    opener = urllib.request.build_opener(_HttpsOnlyRedirects)
    response: IO[bytes] = opener.open(request, timeout=timeout_s)
    return response


def _retry_after(headers: Message | None) -> float | None:
    """The Retry-After header as seconds, or None if absent or not a number of ASCII digits."""
    value = None if headers is None else headers.get("Retry-After")
    if value is None or not _ASCII_DIGITS.fullmatch(value.strip()):
        return None
    return float(value.strip())


def _declared_length(headers: Message) -> int | None:
    """The Content-Length header as a non-negative int, or None if absent or malformed.

    A malformed value is treated as absent, as is one too long for `int` to convert.
    """
    value = headers.get("Content-Length")
    if value is None or not _ASCII_DIGITS.fullmatch(value):
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning(
            "Content-Length of %d digits is too long to read, so it is ignored", len(value)
        )
        return None


def get(
    url: str,
    sink: BinaryIO,
    *,
    max_bytes: int,
    timeout_s: float,
    user_agent: str | None = None,
    opener: Opener = _open,
) -> Response:
    """GET *url*, writing at most *max_bytes* bytes of the body to *sink*.

    Args:
        url: An HTTPS URL.
        sink: Where the body is written.
        max_bytes: The most bytes written; more offered sets `exceeded`.
        timeout_s: The bound on each socket operation, not on the whole transfer.
        user_agent: The User-Agent header to send. None sends `USER_AGENT`, the default.
        opener: Opens the request; tests replace it.

    Returns:
        The status, size, declared length, and digests of what was written. An HTTP error
        status is returned with nothing written.

    Raises:
        ValueError: If *url* is not HTTPS or *max_bytes* is negative.
        OSError: On a network error, a timeout, or a redirect to a non-HTTPS URL.
    """
    if urllib.parse.urlsplit(url).scheme != _HTTPS:
        raise ValueError(f"only HTTPS URLs are fetched, got {url}")
    if max_bytes < 0:
        raise ValueError(f"max_bytes must be non-negative, got {max_bytes}")
    if user_agent is not None and not user_agent.strip():
        raise ValueError("user_agent must not be blank when given")
    agent = USER_AGENT if user_agent is None else user_agent
    digests = _new_digests()
    request = urllib.request.Request(url, headers={"User-Agent": agent})
    try:
        return _transfer(url, request, sink, max_bytes, timeout_s, opener, digests)
    except http.client.HTTPException as error:
        logger.warning("GET %s failed inside the response: %r", url, error)
        raise TransferError(f"GET {url} failed inside the response: {error!r}") from error


def _new_digests() -> tuple[Any, ...]:
    """The SHA-256, MD5 and SHA-1 hashes, which the bytes written are fed to as they stream."""
    return (
        hashlib.sha256(),
        hashlib.md5(usedforsecurity=False),
        hashlib.sha1(usedforsecurity=False),
    )


def _hexdigests(digests: tuple[Any, ...]) -> tuple[str, str, str]:
    """The SHA-256, MD5 and SHA-1 hex digests of the hashes, in that order."""
    sha256, md5, sha1 = (d.hexdigest() for d in digests)
    return sha256, md5, sha1


def _transfer(
    url: str,
    request: urllib.request.Request,
    sink: BinaryIO,
    max_bytes: int,
    timeout_s: float,
    opener: Opener,
    digests: tuple[Any, ...],
) -> Response:
    """Open the request and stream its body to *sink*. An HTTP error status writes nothing."""
    try:
        response = opener(request, timeout_s)
    except urllib.error.HTTPError as error:
        logger.info("GET %s returned %d", url, error.code)
        error.close()
        sha256, md5, sha1 = _hexdigests(digests)
        return Response(error.code, 0, False, None, sha256, md5, sha1, _retry_after(error.headers))
    with response:
        status = getattr(response, "status", 200)
        headers = getattr(response, "headers", Message())
        declared = _declared_length(headers)
        size, exceeded = _stream(response, sink, max_bytes, digests)
    logger.info("GET %s returned %d, %d bytes%s", url, status, size, " (cap)" if exceeded else "")
    sha256, md5, sha1 = _hexdigests(digests)
    return Response(status, size, exceeded, declared, sha256, md5, sha1, _retry_after(headers))


def _stream(
    response: IO[bytes], sink: BinaryIO, max_bytes: int, digests: tuple[Any, ...]
) -> tuple[int, bool]:
    """Copy at most *max_bytes* of *response* to *sink*; the size and whether more was offered.

    Each read asks for at most one byte past the cap, so an oversize body is detected
    without reading much more of it.
    """
    size = 0
    while chunk := response.read(min(_CHUNK_BYTES, max_bytes - size + 1)):
        exceeded = size + len(chunk) > max_bytes
        if exceeded:
            chunk = chunk[: max_bytes - size]
        for digest in digests:
            digest.update(chunk)
        sink.write(chunk)
        size += len(chunk)
        if exceeded:
            return size, True
    return size, False
