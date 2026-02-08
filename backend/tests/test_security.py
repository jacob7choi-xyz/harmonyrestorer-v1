"""Tests for security features: rate limiting, error sanitization."""

from unittest.mock import patch

from app.config import settings
from app.main import app
from fastapi.testclient import TestClient

SMALL_WAV = b"RIFF" + b"\x00" * 40


# --- Rate limiting ---


def test_rate_limiter_returns_429_after_limit(client, mock_denoiser):
    """Uploads exceeding rate_limit_max_requests should be rejected with 429."""
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


# --- Stack trace suppression ---


def test_unhandled_exception_returns_generic_error():
    """Unhandled exceptions must return 'Internal server error', never a stack trace."""
    from tests.conftest import _clear_rate_limiter

    with TestClient(app, raise_server_exceptions=False) as c:
        _clear_rate_limiter(app)
        with patch(
            "app.routes.denoise.job_manager.create_job",
            side_effect=RuntimeError("secret database password exposed"),
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
