"""API route aggregation."""

from fastapi import APIRouter

from app.routes.denoise import router as denoise_router
from app.routes.health import router as health_router

router = APIRouter()
router.include_router(health_router)
router.include_router(denoise_router)
