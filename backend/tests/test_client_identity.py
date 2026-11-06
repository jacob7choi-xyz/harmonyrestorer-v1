"""Tests for client_identity."""

import json

import pytest
from app.client_identity import LAMBDA_REQUEST_CONTEXT_HEADER, client_identity
from app.config import ClientIdentitySource, settings
from starlette.requests import Request


def _request(client: tuple[str, int] | None, headers: dict[str, str] | None = None) -> Request:
    """Build a bare HTTP request with the given client address and headers."""
    raw_headers = [(k.encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": raw_headers,
        "client": client,
    }
    return Request(scope)


def _context(source_ip: object) -> str:
    """Serialize a request context shaped like the one the Lambda Web Adapter sends."""
    return json.dumps({"accountId": "anonymous", "http": {"method": "POST", "sourceIp": source_ip}})


@pytest.fixture
def lambda_mode(monkeypatch) -> None:
    """Read the client address from the Lambda request context."""
    monkeypatch.setattr(
        settings, "client_identity_source", ClientIdentitySource.LAMBDA_REQUEST_CONTEXT
    )


# --- CLIENT_IDENTITY=connection (default) ---


def test_returns_the_client_host() -> None:
    """The identity is the host part of the client address."""
    assert client_identity(_request(("203.0.113.7", 51234))) == "203.0.113.7"


def test_returns_none_without_a_client_address() -> None:
    """A request without a client address has no identity."""
    assert client_identity(_request(None)) is None


def test_connection_mode_ignores_a_request_context_header() -> None:
    """Outside Lambda a caller can send the header, so the default mode never reads it."""
    request = _request(
        ("203.0.113.7", 51234), {LAMBDA_REQUEST_CONTEXT_HEADER: _context("198.51.100.9")}
    )
    assert client_identity(request) == "203.0.113.7"


# --- CLIENT_IDENTITY=lambda_request_context ---


def test_lambda_mode_returns_the_source_ip(lambda_mode) -> None:
    """The identity is http.sourceIp from the request context."""
    request = _request(
        ("127.0.0.1", 40000), {LAMBDA_REQUEST_CONTEXT_HEADER: _context("198.51.100.9")}
    )
    assert client_identity(request) == "198.51.100.9"


def test_lambda_mode_ignores_the_connection_address(lambda_mode) -> None:
    """An address uvicorn took from a caller's X-Forwarded-For does not change the identity."""
    request = _request(
        ("203.0.113.66", 40000),
        {
            "x-forwarded-for": "203.0.113.66",
            LAMBDA_REQUEST_CONTEXT_HEADER: _context("198.51.100.9"),
        },
    )
    assert client_identity(request) == "198.51.100.9"


def test_lambda_mode_returns_the_canonical_address(lambda_mode) -> None:
    """Two spellings of one IPv6 address give one identity."""
    request = _request(None, {LAMBDA_REQUEST_CONTEXT_HEADER: _context("2001:DB8:0::0001")})
    assert client_identity(request) == "2001:db8::1"


@pytest.mark.parametrize(
    "header",
    [
        None,
        "not json",
        json.dumps(["198.51.100.9"]),
        json.dumps({"accountId": "anonymous"}),
        json.dumps({"http": "198.51.100.9"}),
        json.dumps({"http": {"method": "POST"}}),
        _context(1234),
        _context("not-an-address"),
    ],
    ids=[
        "missing header",
        "not JSON",
        "not an object",
        "no http field",
        "http not an object",
        "no sourceIp",
        "sourceIp not a string",
        "sourceIp not an address",
    ],
)
def test_lambda_mode_without_a_usable_source_ip_has_no_identity(
    lambda_mode, header: str | None
) -> None:
    """An unusable request context gives no identity, so uploads fail closed with 503."""
    headers = {} if header is None else {LAMBDA_REQUEST_CONTEXT_HEADER: header}
    assert client_identity(_request(("127.0.0.1", 40000), headers)) is None
