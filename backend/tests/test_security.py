"""Tests for security features: rate limiting, error sanitization, input validation."""

import asyncio
import json
from unittest.mock import patch

from app.client_identity import LAMBDA_REQUEST_CONTEXT_HEADER
from app.config import ClientIdentitySource, Settings, settings
from app.main import _add_cors, app
from app.services.rate_limit import InMemoryRateLimiter
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from tests.conftest import _clear_job_store
from tests.rate_limiter_contract import FakeClock

SMALL_WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 32


# --- Rate limiting ---


def test_rate_limit_none_client_returns_503() -> None:
    """POST /api/v1/denoise returns 503 when request.client is None.

    Covers the proxy misconfiguration case where the backend receives a
    request without a validated peer address. The fail-closed response
    prevents all such requests from sharing a single "unknown" bucket. The 503
    carries the CORS header so the frontend can read it.
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
    origin = settings.cors_origins[0]
    with TestClient(_NullClientApp(app)) as c:
        r = c.post(
            "/api/v1/denoise",
            headers={"Origin": origin},
            files={"file": ("test.wav", SMALL_WAV, "audio/wav")},
        )
    assert r.status_code == 503
    assert "proxy misconfiguration" in r.json()["detail"]
    assert r.headers.get("access-control-allow-origin") == origin


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
    _clear_job_store(app)

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


def test_rate_limiter_returns_429_after_limit(client, mock_denoiser, monkeypatch) -> None:
    """Uploads exceeding rate_limit_max_requests should be rejected with 429."""
    monkeypatch.setattr(settings, "max_jobs_per_ip", 100)
    _clear_job_store(app)
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


def _lambda_context_header(source_ip: str) -> dict[str, str]:
    """The request context header the Lambda Web Adapter sets, for one source address."""
    return {LAMBDA_REQUEST_CONTEXT_HEADER: json.dumps({"http": {"sourceIp": source_ip}})}


def test_lambda_identity_keys_the_upload_limit_on_the_source_ip(
    client, mock_denoiser, monkeypatch
) -> None:
    """With CLIENT_IDENTITY=lambda_request_context, each source address has its own bucket.

    Every request here reaches the app from the same connection address, which is what
    the mode is for: then only the request context can tell the callers apart.
    """
    from tests.conftest import _clear_rate_limiter

    monkeypatch.setattr(
        settings, "client_identity_source", ClientIdentitySource.LAMBDA_REQUEST_CONTEXT
    )
    monkeypatch.setattr(settings, "max_jobs_per_ip", 100)
    _clear_rate_limiter(app)
    _clear_job_store(app)

    for i in range(settings.rate_limit_max_requests):
        r = client.post(
            "/api/v1/denoise",
            headers=_lambda_context_header("198.51.100.1"),
            files={"file": (f"a{i}.wav", SMALL_WAV, "audio/wav")},
        )
        assert r.status_code == 200, f"Request {i + 1} from address A should succeed"
    r = client.post(
        "/api/v1/denoise",
        headers=_lambda_context_header("198.51.100.1"),
        files={"file": ("a_over.wav", SMALL_WAV, "audio/wav")},
    )
    assert r.status_code == 429

    r = client.post(
        "/api/v1/denoise",
        headers=_lambda_context_header("198.51.100.2"),
        files={"file": ("b.wav", SMALL_WAV, "audio/wav")},
    )
    assert r.status_code == 200


def test_lambda_identity_without_a_request_context_returns_503(
    client, mock_denoiser, monkeypatch
) -> None:
    """With CLIENT_IDENTITY=lambda_request_context, an upload without the header fails closed."""
    from tests.conftest import _clear_rate_limiter

    monkeypatch.setattr(
        settings, "client_identity_source", ClientIdentitySource.LAMBDA_REQUEST_CONTEXT
    )
    _clear_rate_limiter(app)
    _clear_job_store(app)

    r = client.post("/api/v1/denoise", files={"file": ("none.wav", SMALL_WAV, "audio/wav")})
    assert r.status_code == 503
    assert "proxy misconfiguration" in r.json()["detail"]
    assert asyncio.run(app.state.job_store.counts()) == {}


def test_rate_limited_upload_carries_cors_header(client, mock_denoiser, monkeypatch) -> None:
    """A 429 from the rate limiter carries the CORS header for an allowed origin.

    The deployed frontend calls the backend from another origin. Without the header
    the browser's fetch rejects, so frontend/src/api/client.ts cannot read the 429 detail.
    """
    monkeypatch.setattr(settings, "max_jobs_per_ip", 100)
    _clear_job_store(app)
    origin = settings.cors_origins[0]
    headers = {"Origin": origin}

    for i in range(settings.rate_limit_max_requests):
        r = client.post(
            "/api/v1/denoise",
            headers=headers,
            files={"file": (f"cors{i}.wav", SMALL_WAV, "audio/wav")},
        )
        assert r.status_code == 200, f"Request {i + 1} should succeed"

    r = client.post(
        "/api/v1/denoise",
        headers=headers,
        files={"file": ("cors_over.wav", SMALL_WAV, "audio/wav")},
    )
    assert r.status_code == 429
    assert "Too many requests" in r.json()["detail"]
    assert r.headers.get("access-control-allow-origin") == origin


def _app_with_cors(origins: list[str]) -> TestClient:
    """Build a one-route app with the CORS handling main.py uses."""
    cors_app = FastAPI()

    @cors_app.get("/ping")
    def ping() -> dict[str, str]:
        return {"status": "ok"}

    _add_cors(cors_app, origins)
    return TestClient(cors_app)


def _cors_headers(response: Response) -> dict[str, str]:
    return {k: v for k, v in response.headers.items() if k.startswith("access-control-")}


def test_empty_cors_origins_adds_no_cors_headers_to_a_request() -> None:
    """With no origins, a cross-origin GET gets no Access-Control headers.

    ADR 009 plans to set CORS_ORIGINS empty on Lambda, where the function URL adds
    the CORS headers.
    """
    r = _app_with_cors([]).get("/ping", headers={"Origin": "https://frontend.example"})
    assert r.status_code == 200
    assert _cors_headers(r) == {}


def test_empty_cors_origins_leaves_preflight_unanswered() -> None:
    """With no origins, the app does not answer a preflight with CORS headers.

    CORSMiddleware with an empty origin list still answers a preflight with 400 and
    Access-Control-Allow-Methods, -Allow-Headers and -Max-Age, so the app adds no
    middleware at all instead. The OPTIONS request then reaches routing, which has no
    OPTIONS route.
    """
    r = _app_with_cors([]).options(
        "/ping",
        headers={"Origin": "https://frontend.example", "Access-Control-Request-Method": "GET"},
    )
    assert r.status_code == 405
    assert _cors_headers(r) == {}


def test_listed_cors_origin_gets_allow_origin_header() -> None:
    """A listed origin is allowed to read responses."""
    origin = "https://frontend.example"
    r = _app_with_cors([origin]).get("/ping", headers={"Origin": origin})
    assert r.headers.get("access-control-allow-origin") == origin


def test_rate_limit_window_expiry_allows_new_requests(client, mock_denoiser, monkeypatch) -> None:
    """After the rate-limit window expires, stale timestamps are pruned and
    the client can upload again without a server restart.

    Covers the sliding-window prune in InMemoryRateLimiter.allow
    (app/services/rate_limit.py), driven by a clock the test advances.
    """
    monkeypatch.setattr(settings, "max_jobs_per_ip", 100)
    _clear_job_store(app)
    limit = settings.rate_limit_max_requests
    window = settings.rate_limit_window_seconds
    clock = FakeClock()
    app.state.rate_limiter = InMemoryRateLimiter(limit, window, clock=clock)

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
    clock.advance(window + 1)
    r = client.post(
        "/api/v1/denoise",
        files={"file": ("renewed.wav", SMALL_WAV, "audio/wav")},
    )

    assert r.status_code == 200, "Request after window expiry must succeed"


# --- Docs gating ---


def test_docs_disabled_by_default(client) -> None:
    """/api/docs is unreachable when ENABLE_DOCS is not set."""
    assert client.get("/api/docs").status_code == 404


def test_redoc_disabled_by_default(client) -> None:
    """/api/redoc is unreachable when ENABLE_DOCS is not set."""
    assert client.get("/api/redoc").status_code == 404


def test_openapi_json_disabled_by_default(client) -> None:
    """/api/openapi.json is unreachable when ENABLE_DOCS is not set."""
    assert client.get("/api/openapi.json").status_code == 404


def test_openapi_json_default_route_disabled(client) -> None:
    """FastAPI's implicit /openapi.json default route is also unreachable.

    Setting openapi_url explicitly in the FastAPI constructor removes the
    default /openapi.json route. This test guards against a regression where
    the schema is hidden from /api/openapi.json but still accessible at the
    FastAPI default.
    """
    assert client.get("/openapi.json").status_code == 404


def test_enable_docs_setting_parses_true(monkeypatch) -> None:
    """ENABLE_DOCS=true is parsed correctly as True."""
    monkeypatch.setenv("ENABLE_DOCS", "true")
    assert Settings().enable_docs is True


def test_enable_docs_setting_defaults_false(monkeypatch) -> None:
    """enable_docs is False when ENABLE_DOCS is not set."""
    monkeypatch.delenv("ENABLE_DOCS", raising=False)
    assert Settings().enable_docs is False


# --- Stack trace suppression ---


def test_unhandled_exception_returns_generic_error() -> None:
    """Unhandled exceptions must return 'Internal server error', never a stack trace.

    The failure here comes after the upload was decoded, so the decoded file
    must also be removed rather than left counting against the byte budget.
    """
    from tests.conftest import _clear_rate_limiter, _fake_decode

    with TestClient(app, raise_server_exceptions=False) as c:
        _clear_rate_limiter(app)
        _clear_job_store(app)
        uploads_before = set(settings.upload_dir.iterdir())
        with (
            patch("app.routes.denoise.decode_to_wav", side_effect=_fake_decode),
            patch.object(
                app.state.job_store,
                "create",
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
    assert set(settings.upload_dir.iterdir()) == uploads_before


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
