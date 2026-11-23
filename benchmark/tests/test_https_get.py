"""Tests for the bounded HTTPS GET, with a stand-in opener in place of the network."""

from __future__ import annotations

import hashlib
import http.client
import io
import urllib.error
import urllib.request
from email.message import Message
from typing import IO

import pytest

from benchmark import https_get
from benchmark.https_get import InsecureRedirectError, TransferError, get

URL = "https://archive.org/download/item/file.flac"


class FakeResponse(io.BytesIO):
    """A response body with a status and headers, counting the bytes read from it."""

    def __init__(self, body: bytes, status: int = 200, length: str | None = None) -> None:
        super().__init__(body)
        self.status = status
        self.headers = Message()
        if length is not None:
            self.headers["Content-Length"] = length
        self.bytes_read = 0

    def read(self, size: int | None = -1) -> bytes:  # type: ignore[override]
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk


def serving(
    response: IO[bytes], seen: list[urllib.request.Request] | None = None
) -> https_get.Opener:
    """An opener that records the request and returns *response*."""

    def opener(request: urllib.request.Request, timeout_s: float) -> IO[bytes]:
        if seen is not None:
            seen.append(request)
        return response

    return opener


def fetch(
    body: bytes, max_bytes: int, **response: object
) -> tuple[https_get.Response, bytes, FakeResponse]:
    """GET a fake *body* under *max_bytes*; the response, what was written, and the fake."""
    fake = FakeResponse(body, **response)  # type: ignore[arg-type]
    sink = io.BytesIO()
    result = get(URL, sink, max_bytes=max_bytes, timeout_s=5.0, opener=serving(fake))
    return result, sink.getvalue(), fake


class TestBody:
    """What is written, and the size and digests reported for it."""

    def test_a_body_under_the_cap_is_written_whole_with_its_digests(self) -> None:
        body = bytes(range(256)) * 10
        result, written, _ = fetch(body, 10_000, length=str(len(body)))
        assert written == body
        assert (result.status, result.size, result.exceeded) == (200, len(body), False)
        assert result.content_length == len(body)
        assert result.sha256 == hashlib.sha256(body).hexdigest()
        assert result.md5 == hashlib.md5(body).hexdigest()
        assert result.sha1 == hashlib.sha1(body).hexdigest()

    def test_a_body_of_exactly_the_cap_is_not_exceeded(self) -> None:
        result, written, _ = fetch(b"x" * 64, 64)
        assert (result.size, result.exceeded, written) == (64, False, b"x" * 64)

    def test_a_body_over_the_cap_stops_one_byte_past_it(self) -> None:
        body = b"y" * 10_000
        result, written, fake = fetch(body, 100)
        assert (result.size, result.exceeded) == (100, True)
        assert written == body[:100]
        assert result.sha256 == hashlib.sha256(body[:100]).hexdigest()
        assert fake.bytes_read == 101

    @pytest.mark.parametrize(
        "length",
        [None, "12abc", "-5", "\u00b2", "\u0663", "1" * 5000],
        ids=["absent", "letters", "sign", "superscript", "arabic_indic", "too_long_for_int"],
    )
    def test_a_missing_or_malformed_length_is_reported_as_none(self, length: str | None) -> None:
        result, _, _ = fetch(b"abc", 10, length=length)
        assert result.content_length is None


class TestRequest:
    """What is sent, and which requests are refused before anything is sent."""

    def test_the_request_names_a_generic_client(self) -> None:
        seen: list[urllib.request.Request] = []
        get(URL, io.BytesIO(), max_bytes=10, timeout_s=5.0, opener=serving(FakeResponse(b""), seen))
        assert seen[0].get_header("User-agent") == https_get.USER_AGENT

    @pytest.mark.parametrize(
        "url", ["http://archive.org/x", "ftp://archive.org/x", "file:///etc/passwd"]
    )
    def test_a_url_that_is_not_https_is_refused(self, url: str) -> None:
        with pytest.raises(ValueError, match="only HTTPS"):
            get(url, io.BytesIO(), max_bytes=10, timeout_s=5.0, opener=serving(FakeResponse(b"")))

    def test_a_negative_cap_is_refused(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            get(URL, io.BytesIO(), max_bytes=-1, timeout_s=5.0, opener=serving(FakeResponse(b"")))


class TestFailures:
    """An HTTP status is reported; a broken transfer raises OSError."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("120", 120.0),
            (" 5 ", 5.0),
            ("Wed, 21 Oct 2026 07:28:00 GMT", None),
            ("\u00b2", None),
            ("\u0663", None),
        ],
    )
    def test_a_numeric_retry_after_is_reported_in_seconds(
        self, value: str, expected: float | None
    ) -> None:
        headers = Message()
        headers["Retry-After"] = value

        def limited(request: urllib.request.Request, timeout_s: float) -> IO[bytes]:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many", headers, io.BytesIO())

        result = get(URL, io.BytesIO(), max_bytes=10, timeout_s=5.0, opener=limited)
        assert (result.status, result.retry_after) == (429, expected)

    def test_a_superscript_retry_after_on_a_whole_response_is_ignored(self) -> None:
        fake = FakeResponse(b"abc")
        fake.headers["Retry-After"] = "\u00b2"

        result = get(URL, io.BytesIO(), max_bytes=10, timeout_s=5.0, opener=serving(fake))

        assert (result.status, result.retry_after) == (200, None)

    def test_an_http_error_status_is_returned_with_nothing_written(self) -> None:
        def refusing(request: urllib.request.Request, timeout_s: float) -> IO[bytes]:
            raise urllib.error.HTTPError(
                request.full_url, 404, "Not Found", Message(), io.BytesIO(b"page")
            )

        sink = io.BytesIO()
        result = get(URL, sink, max_bytes=10, timeout_s=5.0, opener=refusing)
        assert (result.status, result.size, sink.getvalue()) == (404, 0, b"")

    def test_a_connection_that_ends_inside_the_body_raises_an_os_error(self) -> None:
        class CutOff(FakeResponse):
            def read(self, size: int | None = -1) -> bytes:  # type: ignore[override]
                raise http.client.IncompleteRead(b"partial", 100)

        with pytest.raises(TransferError) as raised:
            get(URL, io.BytesIO(), max_bytes=1000, timeout_s=5.0, opener=serving(CutOff(b"")))
        assert isinstance(raised.value, OSError)


class TestRedirects:
    """Redirects are followed only to HTTPS."""

    def handler_redirect(self, target: str) -> urllib.request.Request | None:
        handler = https_get._HttpsOnlyRedirects()
        request = urllib.request.Request(URL)
        return handler.redirect_request(
            request, io.BytesIO(), 302, "Found", http.client.HTTPMessage(), target
        )

    def test_a_redirect_to_plain_http_is_refused(self) -> None:
        with pytest.raises(InsecureRedirectError):
            self.handler_redirect("http://ia800.us.archive.org/file.flac")

    def test_a_redirect_to_https_is_followed(self) -> None:
        redirected = self.handler_redirect("https://ia800.us.archive.org/file.flac")
        assert redirected is not None
        assert redirected.full_url == "https://ia800.us.archive.org/file.flac"

    def test_the_default_opener_uses_the_https_only_handler(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        built: list[tuple[object, ...]] = []

        class Recorder:
            def open(self, request: urllib.request.Request, timeout: float) -> IO[bytes]:
                return FakeResponse(b"")

        def build_opener(*handlers: object) -> Recorder:
            built.append(handlers)
            return Recorder()

        monkeypatch.setattr(https_get.urllib.request, "build_opener", build_opener)
        get(URL, io.BytesIO(), max_bytes=10, timeout_s=5.0)
        assert built == [(https_get._HttpsOnlyRedirects,)]
