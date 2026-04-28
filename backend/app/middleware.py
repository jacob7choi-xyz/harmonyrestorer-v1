"""Custom middleware."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from app.config import settings
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple per-IP rate limiter for upload endpoints.

    Rejects requests on POST /api/v1/denoise with 429 when a client exceeds
    the configured request rate. Returns 503 when the client IP cannot be
    determined, which indicates a proxy misconfiguration rather than a
    client error.

    Safe to use only when the backend is not directly reachable from the
    internet. In production, the backend must sit behind a trusted reverse
    proxy (Nginx) that overwrites X-Forwarded-For with $remote_addr, and
    Uvicorn must be started with --proxy-headers so request.client is
    populated with the real client IP before this middleware runs.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize the rate limiter.

        Args:
            app: The ASGI application to wrap.
        """
        super().__init__(app)
        self.max_requests: int = settings.rate_limit_max_requests
        self.window_seconds: int = settings.rate_limit_window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Rate limit POST /api/v1/denoise by client IP.

        Args:
            request: The incoming HTTP request.
            call_next: The next middleware or route handler in the stack.

        Returns:
            503 if the client IP is absent (proxy misconfiguration).
            429 if the client has exceeded the rate limit.
            Otherwise the response from the next handler.
        """
        if request.method == "POST" and request.url.path == "/api/v1/denoise":
            if request.client is None:
                logger.warning(
                    "Rejecting upload request with no client address (proxy misconfiguration)"
                )
                return JSONResponse(
                    status_code=503,
                    content={"detail": "Service unavailable: proxy misconfiguration."},
                )

            client_ip = request.client.host
            logger.debug("Rate-limit check for client_ip=%s", client_ip)

            now = time.time()
            cutoff = now - self.window_seconds

            active = [t for t in self._requests[client_ip] if t > cutoff]
            if not active:
                del self._requests[client_ip]
                active = []
            else:
                self._requests[client_ip] = active

            if len(active) >= self.max_requests:
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many requests. Try again later."},
                )

            self._requests[client_ip].append(now)

        return await call_next(request)
