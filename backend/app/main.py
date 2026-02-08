"""HarmonyRestorer v1 — FastAPI application."""

import asyncio
import logging
from contextlib import asynccontextmanager

from app.config import settings
from app.exceptions import unhandled_exception_handler
from app.middleware import RateLimitMiddleware
from app.routes import router
from app.services.jobs import job_manager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def _cleanup_loop() -> None:
    """Periodically evict expired jobs and their output files."""
    while True:
        await asyncio.sleep(settings.cleanup_interval_seconds)
        try:
            job_manager.cleanup_expired()
        except Exception:
            logger.exception("Error during job cleanup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    logger.info("HarmonyRestorer v1 starting up")
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Processed directory: {settings.processed_dir}")
    logger.info(
        f"Job TTL: {settings.job_ttl_seconds}s, "
        f"cleanup every {settings.cleanup_interval_seconds}s"
    )

    cleanup_task = asyncio.create_task(_cleanup_loop())
    yield
    cleanup_task.cancel()

    logger.info("HarmonyRestorer v1 shutting down")


app = FastAPI(
    title="HarmonyRestorer v1",
    description="AI-powered audio denoising",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# --- Middleware (outermost runs first) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitMiddleware)

# --- Exception handlers ---
app.add_exception_handler(Exception, unhandled_exception_handler)

# --- Routes ---
app.include_router(router)
