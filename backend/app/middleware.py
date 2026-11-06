"""Custom middleware."""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.client_identity import client_identity
from app.services.rate_limit import RateLimiter

logger = logging.getLogger(__name__)


# The routes with a per-client limit
_UPLOAD_PATH = "/api/v1/denoise"
_DOWNLOAD_PATH_PREFIX = "/api/v1/download/"


def _limited_route(request: Request) -> tuple[str, RateLimiter] | None:
    """Return a name for the request's limited route and that route's limiter, or None."""
    path = request.url.path
    if request.method == "POST" and path == _UPLOAD_PATH:
        return "upload", request.app.state.rate_limiter
    if request.method == "GET" and path.startswith(_DOWNLOAD_PATH_PREFIX):
        return "download", request.app.state.download_rate_limiter
    return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client rate limits for uploads and downloads.

    Rejects POST /api/v1/denoise with 429 when the client is over the limit kept
    by the RateLimiter on app.state.rate_limiter, and GET /api/v1/download/{job_id}
    when it is over the separate limit on app.state.download_rate_limiter. Both
    are installed by app.composition. Returns 503 when the request has no client
    address, which indicates a proxy misconfiguration rather than a client error.

    The client is identified by client_identity(). Behind a proxy, that is the
    real client only if Uvicorn trusts the proxy's forwarded headers (the
    forwarded-allow-ips setting in the Dockerfile CMD) and clients cannot
    reach the backend without passing through the proxy.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Rate limit uploads and downloads by client identity.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler in the stack.

        Returns:
            503 if the client address is absent (proxy misconfiguration).
            429 if the client has exceeded the route's rate limit.
            Otherwise the response from the next handler.
        """
        limited = _limited_route(request)
        if limited is not None:
            route, limiter = limited
            client_ip = client_identity(request)
            if client_ip is None:
                logger.warning(
                    "Rejecting %s request with no client address (proxy misconfiguration)", route
                )
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Service unavailable: proxy misconfiguration."},
                )

            logger.debug("Rate-limit check for %s from client_ip=%s", route, client_ip)
            if not await limiter.allow(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Try again later."},
                )

        return await call_next(request)
