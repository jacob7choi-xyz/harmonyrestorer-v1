"""Health and info endpoints."""

import shutil
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.config import DenoiserEngine, settings
from app.services.jobs import job_manager

router = APIRouter(tags=["health"])


@router.get("/")
async def root() -> dict:
    """API root endpoint."""
    return {
        "platform": "HarmonyRestorer v1",
        "version": "1.0.0",
        "description": "AI-powered audio denoising",
        "status": "ready",
        "docs": "/api/docs",
        "supported_formats": sorted(settings.supported_formats),
    }


@router.get("/health", response_model=None)
async def health_check(request: Request) -> JSONResponse:
    """Health check with real system verification."""
    disk = shutil.disk_usage(settings.processed_dir)
    disk_free_mb = disk.free // (1024 * 1024)
    disk_ok = disk.free > settings.min_disk_bytes

    dirs_ok = settings.upload_dir.is_dir() and settings.processed_dir.is_dir()

    # Set by the lifespan; if it is missing, no startup result was recorded, so it is not ok
    ffmpeg_ok = getattr(request.app.state, "ffmpeg_at_startup_ok", False)

    checks: dict[str, dict[str, bool | int]] = {
        "disk_space": {"ok": disk_ok, "free_mb": disk_free_mb},
        "directories": {"ok": dirs_ok},
        "ffmpeg_at_startup": {"ok": ffmpeg_ok},
    }
    all_ok = disk_ok and dirs_ok and ffmpeg_ok

    if settings.denoiser_engine is DenoiserEngine.OPGAN:
        # Set by the lifespan for OpGAN; if it is missing, no startup result was recorded
        checkpoint_ok = getattr(request.app.state, "opgan_checkpoint_at_startup_ok", False)
        checks["opgan_checkpoint_at_startup"] = {"ok": checkpoint_ok}
        all_ok = all_ok and checkpoint_ok

    status = "healthy" if all_ok else "degraded"
    payload = {
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
        "checks": checks,
        "jobs": job_manager.job_counts,
    }
    return JSONResponse(content=payload, status_code=200 if all_ok else 503)
