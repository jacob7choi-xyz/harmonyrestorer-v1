"""Tests for security features: rate limiting, error sanitization, input validation."""

import time
from unittest.mock import MagicMock, patch

from app.config import settings
from app.main import app
from fastapi.testclient import TestClient

SMALL_WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 32


# --- Rate limiting ---


def test_rate_limit_none_client_returns_503():
    """POST /api/v1/denoise returns 503 when request.client is None.

    Covers the proxy misconfiguration case where the backend receives a
    request without a validated peer address. The fail-closed response
    prevents all such requests from sharing a single "unknown" bucket.
    """
    from tests.conftest import _clear_rate_limiter

    class _NullClientApp:
        """ASGI wrapper that strips the client from the request scope."""

        def __init__(self, wrapped_app):
            self._app = wrapped_app

        async def __call__(self, scope, receive, send):
            if scope["type"] in ("http", "websocket"):
                scope = {**scope, "client": None}
            await self._app(scope, receive, send)

    _clear_rate_limiter(app)
    with TestClient(_NullClientApp(app)) as c:
        r = c.post(
            "/api/v1/denoise",
            files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
        )
    assert r.status_code == 503
    assert "proxy misconfiguration" in r.json()["detail"]


def test_rate_limit_different_ips_have_independent_buckets(mock_denoiser, monkeypatch):
    """Two different client IPs have fully independent rate-limit buckets.

    Proves that rate limiting is keyed on the real client IP, not on a shared
    bucket. If proxy normalization were broken and every request appeared as
    the Nginx container IP, exhausting the limit from one IP would block all
    users.
    """
    from tests.conftest import _clear_rate_limiter

    class _FixedClientApp:
        """ASGI wrapper that pins a specific client IP in the request scope."""

        def __init__(self, wrapped_app, ip: str):
            self._app = wrapped_app
            self._ip = ip

        async def __call__(self, scope, receive, send):
            if scope["type"] in ("http", "websocket"):
                scope = {**scope, "client": (self._ip, 12345)}
            await self._app(scope, receive, send)

    monkeypatch.setattr(settings, "max_jobs_per_ip", 100)
    limit = settings.rate_limit_max_requests
    _clear_rate_limiter(app)

    # Exhaust the limit for IP A
    with TestClient(_FixedClientApp(app, "10.0.0.1")) as c:
        for i in range(limit):
            r = c.post(
                "/api/v1/denoise",
                files={"file": (f"test{i}.wav", SMALL_WAV, "audio/wav")},
            )
            assert r.status_code == 200, f"Request {i + 1} from IP A should succeed"
        r = c.post(
            "/api/v1/denoise",
            files={"file": ("overflow.wav", SMALL_WAV, "audio/wav")},
        )
        assert r.status_code == 429, "IP A should be rate-limited after exhausting its bucket"

    # IP B must have a completely independent bucket
    with TestClient(_FixedClientApp(app, "10.0.0.2")) as c:
        r = c.post(
            "/api/v1/denoise",
            files={"file": ("from_b.wav", SMALL_WAV, "audio/wav")},
        )
    assert r.status_code == 200, "IP B must not be affected by IP A's exhausted bucket"


def test_rate_limiter_returns_429_after_limit(client, mock_denoiser, monkeypatch):
    """Uploads exceeding rate_limit_max_requests should be rejected with 429."""
    monkeypatch.setattr(settings, "max_jobs_per_ip", 100)
    limit = settings.rate_limit_max_requests

    for i in range(limit):
        r = client.post(
            "/api/v1/denoise",
            files={"file": (f"test{i}.wav", SMALL_WAV, "audio/wav")},
        )
        assert r.status_code == 200, f"Request {i + 1} should succeed"

    # Next request should be rate-limited
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("overflow.wav", SMALL_WAV, "audio/wav")},
    )
    assert r.status_code == 429
    assert "Too many requests" in r.json()["detail"]


def test_rate_limit_window_expiry_allows_new_requests(client, mock_denoiser, monkeypatch) -> None:
    """After the rate-limit window expires, stale timestamps are pruned and
    the client can upload again without a server restart.

    Covers the sliding-window prune branch in middleware.py:
        active = [t for t in self._requests[ip] if t > cutoff]
    """
    monkeypatch.setattr(settings, "max_jobs_per_ip", 100)
    limit = settings.rate_limit_max_requests
    window = settings.rate_limit_window_seconds

    # Fill the bucket to exhaustion
    for i in range(limit):
        r = client.post(
            "/api/v1/denoise",
            files={"file": (f"t{i}.wav", SMALL_WAV, "audio/wav")},
        )
        assert r.status_code == 200, f"Request {i + 1} should succeed before limit"

    r = client.post(
        "/api/v1/denoise",
        files={"file": ("over.wav", SMALL_WAV, "audio/wav")},
    )
    assert r.status_code == 429

    # Advance time past the window so all stored timestamps are stale
    future = time.time() + window + 1
    with patch("app.middleware.time.time", return_value=future):
        r = client.post(
            "/api/v1/denoise",
            files={"file": ("renewed.wav", SMALL_WAV, "audio/wav")},
        )

    assert r.status_code == 200, "Request after window expiry must succeed"


# --- Stack trace suppression ---


def test_unhandled_exception_returns_generic_error():
    """Unhandled exceptions must return 'Internal server error', never a stack trace."""
    from tests.conftest import _clear_rate_limiter

    mock_info = MagicMock()
    mock_info.duration = 1.0

    with TestClient(app, raise_server_exceptions=False) as c:
        _clear_rate_limiter(app)
        with (
            patch("app.routes.denoise.sf.info", return_value=mock_info),
            patch(
                "app.routes.denoise.job_manager.create_job",
                side_effect=RuntimeError("secret database password exposed"),
            ),
        ):
            r = c.post(
                "/api/v1/denoise",
                files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
            )

    assert r.status_code == 500
    body = r.json()
    assert body["detail"] == "Internal server error"
    # Ensure no stack trace or internal details leak
    assert "secret" not in str(body)
    assert "Traceback" not in str(body)


# --- Magic byte validation ---


def test_upload_rejects_mismatched_magic_bytes(client):
    """A PDF disguised as .wav should be rejected."""
    pdf_bytes = b"%PDF-1.4" + b"\x00" * 40
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("fake.wav", pdf_bytes, "audio/wav")},
    )
    assert r.status_code == 400
    assert "does not match" in r.json()["detail"]


def test_upload_rejects_empty_file(client):
    """Empty files should be rejected (no magic bytes to validate)."""
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert r.status_code == 400


def test_path_traversal_filename_uses_uuid(client, mock_denoiser):
    """Path traversal in filename is neutralized by UUID-based storage."""
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("../../etc/passwd.wav", SMALL_WAV, "audio/wav")},
    )
    # Should succeed: the malicious filename is ignored, UUID is used
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    # Verify the job_id is a valid UUID, not a path
    import uuid

    uuid.UUID(job_id)  # Raises if not valid UUID
