"""Tests for client_identity."""

from app.client_identity import client_identity
from starlette.requests import Request


def _request(client: tuple[str, int] | None) -> Request:
    """Build a bare HTTP request with the given client address."""
    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "client": client}
    return Request(scope)


def test_returns_the_client_host() -> None:
    """The identity is the host part of the client address."""
    assert client_identity(_request(("203.0.113.7", 51234))) == "203.0.113.7"


def test_returns_none_without_a_client_address() -> None:
    """A request without a client address has no identity."""
    assert client_identity(_request(None)) is None
