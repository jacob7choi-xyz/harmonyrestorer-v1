"""The client address that per-client controls attribute a request to."""

from __future__ import annotations

import ipaddress
import json

from starlette.requests import Request

from app.config import ClientIdentitySource, settings

# Set by the Lambda Web Adapter on each request it forwards. In adapter 1.1.0 (src/lib.rs,
# fetch_response) it is added with HeaderMap::insert, which removes any value the caller
# sent under the same name.
LAMBDA_REQUEST_CONTEXT_HEADER = "x-amzn-request-context"


def client_identity(request: Request) -> str | None:
    """Return the address this backend attributes a request to.

    With CLIENT_IDENTITY=connection (the default), the address is
    request.client.host. Uvicorn sets it from the connection, or from
    X-Forwarded-For when the request comes from a proxy it trusts (the
    proxy-headers and forwarded-allow-ips settings in the Dockerfile CMD).

    With CLIENT_IDENTITY=lambda_request_context, the address is http.sourceIp
    from the request context the Lambda Web Adapter forwards. Only a host that
    runs the adapter may use this mode: elsewhere a caller can send the header
    themselves.

    The rate limiter and the per-IP job cap both read the address through this
    function, so they agree on who the client is.

    Args:
        request: The incoming request.

    Returns:
        The client's address, or None when the server has no usable client
        address for the request, which points to a proxy misconfiguration.
    """
    if settings.client_identity_source is ClientIdentitySource.LAMBDA_REQUEST_CONTEXT:
        return _lambda_source_ip(request.headers.get(LAMBDA_REQUEST_CONTEXT_HEADER))
    if request.client is None:
        return None
    return request.client.host


def _lambda_source_ip(header: str | None) -> str | None:
    """Return http.sourceIp from a Lambda request context, or None if it is unusable."""
    if header is None:
        return None
    try:
        context = json.loads(header)
    except json.JSONDecodeError:
        return None
    http = context.get("http") if isinstance(context, dict) else None
    source_ip = http.get("sourceIp") if isinstance(http, dict) else None
    if not isinstance(source_ip, str):
        return None
    try:
        # The canonical form, so that one address is always one rate-limit key
        return str(ipaddress.ip_address(source_ip))
    except ValueError:
        return None
