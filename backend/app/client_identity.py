"""The client address that per-client controls attribute a request to."""

from __future__ import annotations

from starlette.requests import Request


def client_identity(request: Request) -> str | None:
    """Return the address this backend attributes a request to.

    The address is request.client.host. Uvicorn sets it from the connection,
    or from X-Forwarded-For when the request comes from a proxy it trusts
    (the proxy-headers and forwarded-allow-ips settings in the Dockerfile
    CMD). The rate limiter and the per-IP job cap both read it through this
    function, so they agree on who the client is.

    Args:
        request: The incoming request.

    Returns:
        The client's address, or None when the server has no client address
        for the request, which points to a proxy misconfiguration.
    """
    if request.client is None:
        return None
    return request.client.host
