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


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client rate limit for uploads.

    Rejects POST /api/v1/denoise with 429 when the client is over the limit
    kept by the RateLimiter on app.state.rate_limiter, which app.composition
    installs. Returns 503 when the request has no client address, which
    indicates a proxy misconfiguration rather than a client error.

    The client is identified by client_identity(). Behind a proxy, that is the
    real client only if Uvicorn trusts the proxy's forwarded headers (the
    forwarded-allow-ips setting in the Dockerfile CMD) and clients cannot
    reach the backend without passing through the proxy.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Rate limit POST /api/v1/denoise by client identity.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler in the stack.

        Returns:
            503 if the client address is absent (proxy misconfiguration).
            429 if the client has exceeded the rate limit.
            Otherwise the response from the next handler.
        """
        if request.method == "POST" and request.url.path == "/api/v1/denoise":
            client_ip = client_identity(request)
            if client_ip is None:
                logger.warning(
                    "Rejecting upload request with no client address (proxy misconfiguration)"
                )
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Service unavailable: proxy misconfiguration."},
                )

            logger.debug("Rate-limit check for client_ip=%s", client_ip)
            limiter: RateLimiter = request.app.state.rate_limiter
            if not await limiter.allow(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Try again later."},
                )

        return await call_next(request)
