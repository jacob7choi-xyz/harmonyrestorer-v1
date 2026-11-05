"""Composition root: builds the configured adapters and attaches them to the app (ADR 010)."""

from __future__ import annotations

from fastapi import FastAPI

from app.config import settings
from app.services.file_store import build_file_store
from app.services.job_store import build_job_store
from app.services.rate_limit import build_rate_limiter


def install_adapters(app: FastAPI) -> None:
    """Build the configured adapters and attach them to app.state.

    Args:
        app: The application whose requests will use the adapters.
    """
    app.state.rate_limiter = build_rate_limiter(settings.rate_limiter)
    app.state.job_store = build_job_store(settings.job_store)
    app.state.file_store = build_file_store(settings.file_store)
