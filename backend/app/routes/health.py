"""Health and info endpoints."""

import shutil
from datetime import UTC, datetime

from app.config import settings
from app.services.jobs import job_manager
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/")
async def root() -> dict:
    """API root endpoint."""
    return {
        "platform": "HarmonyRestorer v1",
        "version": "1.0.0",
        "description": "AI-powered audio denoising using UVR",
        "status": "ready",
        "docs": "/api/docs",
        "supported_formats": sorted(settings.supported_formats),
    }


@router.get("/health")
async def health_check() -> dict:
    """Health check with real system verification."""
    disk = shutil.disk_usage(settings.processed_dir)
    disk_free_mb = disk.free // (1024 * 1024)
    disk_ok = disk.free > settings.min_disk_bytes

    dirs_ok = settings.upload_dir.is_dir() and settings.processed_dir.is_dir()

    all_ok = disk_ok and dirs_ok
    status = "healthy" if all_ok else "degraded"

    return {
        "status": status,
        "timestamp": datetime.now(UTC).isoformat(),
        "checks": {
            "disk_space": {"ok": disk_ok, "free_mb": disk_free_mb},
            "directories": {"ok": dirs_ok},
        },
        "jobs": job_manager.job_counts,
    }
