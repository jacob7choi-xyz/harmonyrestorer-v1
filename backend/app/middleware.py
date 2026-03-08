"""Custom middleware."""

import time
from collections import defaultdict

from app.config import settings
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple per-IP rate limiter for upload endpoints."""

    def __init__(self, app):
        super().__init__(app)
        self.max_requests = settings.rate_limit_max_requests
        self.window_seconds = settings.rate_limit_window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        # Only rate-limit the upload endpoint
        if request.method == "POST" and request.url.path == "/api/v1/denoise":
            client_ip = request.client.host if request.client else "unknown"
            now = time.time()
            cutoff = now - self.window_seconds

            # Evict expired timestamps and clean up empty entries
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
