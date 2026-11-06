"""HarmonyRestorer v1: FastAPI application."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.composition import install_adapters
from app.config import OPGAN_CHECKPOINT_SHA256, DenoiserEngine, settings
from app.exceptions import unhandled_exception_handler
from app.middleware import RateLimitMiddleware
from app.routes import router
from app.services.audio_decode import ffmpeg_version
from app.services.jobs import job_manager
from app.services.opgan_denoiser import CheckpointMismatchError, read_verified_checkpoint


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production environments."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string."""
        log_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def _configure_logging() -> None:
    """Set up logging based on LOG_FORMAT setting."""
    handler = logging.StreamHandler()
    if settings.log_format == "json":
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
    logging.root.handlers.clear()
    logging.root.addHandler(handler)
    logging.root.setLevel(settings.log_level)


_configure_logging()
logger = logging.getLogger(__name__)


async def _cleanup_loop(app: FastAPI) -> None:
    """Periodically evict expired jobs and their output files."""
    while True:
        await asyncio.sleep(settings.cleanup_interval_seconds)
        try:
            # Read on each pass, so the loop always uses the store the app has now
            await job_manager.cleanup_expired(app.state.job_store, app.state.file_store)
        except Exception:
            logger.exception("Error during job cleanup")


def _opgan_checkpoint_ok() -> bool:
    """Check once at startup that the OpGAN checkpoint can be read and has the expected digest.

    This only reports; the service checks the digest again when it loads the model.
    """
    try:
        read_verified_checkpoint(settings.opgan_checkpoint, OPGAN_CHECKPOINT_SHA256)
    except (OSError, CheckpointMismatchError) as e:
        logger.error("OpGAN checkpoint failed its startup check; OpGAN jobs will fail: %s", e)
        return False
    logger.info("OpGAN checkpoint matches the expected digest")
    return True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan management."""
    logger.info("HarmonyRestorer v1 starting up")
    logger.info("Upload directory: %s", settings.upload_dir)
    logger.info("Processed directory: %s", settings.processed_dir)
    logger.info(
        "Job TTL: %ss, cleanup every %ss",
        settings.job_ttl_seconds,
        settings.cleanup_interval_seconds,
    )
    version = await run_in_threadpool(ffmpeg_version, timeout_s=settings.decode_timeout_seconds)
    # Checked once; /health reports this result rather than running ffmpeg on every probe
    app.state.ffmpeg_at_startup_ok = version is not None
    if version is None:
        # Every upload is decoded with ffmpeg first, so without it uploads fail with 500
        logger.error("ffmpeg is not available; uploads cannot be decoded")
    else:
        logger.info("Upload decoder: %s", version)
    if settings.denoiser_engine is DenoiserEngine.OPGAN:
        # Reported by /health; the bytes read here are not kept
        app.state.opgan_checkpoint_at_startup_ok = await run_in_threadpool(_opgan_checkpoint_ok)

    cleanup_task = asyncio.create_task(_cleanup_loop(app))
    yield
    cleanup_task.cancel()

    logger.info("HarmonyRestorer v1 shutting down")


def _add_cors(app: FastAPI, origins: list[str]) -> None:
    """Add CORS handling for the given origins, or none at all when the list is empty.

    With no origins, CORSMiddleware still answers preflight requests with 400 and
    some Access-Control headers. ADR 009 plans to set CORS_ORIGINS empty on Lambda,
    where the function URL adds the CORS headers, so the app adds none of its own.

    Args:
        app: The application to add the middleware to.
        origins: Origins allowed to read responses; empty disables CORS handling.
    """
    if not origins:
        logger.info("CORS_ORIGINS is empty; the app adds no CORS headers")
        return
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Accept"],
    )


app = FastAPI(
    title="HarmonyRestorer v1",
    description="AI-powered audio denoising",
    version="1.0.0",
    docs_url="/api/docs" if settings.enable_docs else None,
    redoc_url="/api/redoc" if settings.enable_docs else None,
    openapi_url="/api/openapi.json" if settings.enable_docs else None,
    lifespan=lifespan,
)

# --- Adapters ---
# Installed at import rather than in the lifespan, so they exist even if a host
# serves requests without running the lifespan
install_adapters(app)

# --- Middleware ---
# Each add_middleware call wraps the middleware added before it, so the last one added runs
# first. CORS is added last so that responses returned early by the rate limiter (429, 503)
# also carry CORS headers; otherwise the browser blocks the frontend from reading them.
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(RateLimitMiddleware)
_add_cors(app, settings.cors_origins)

# --- Exception handlers ---
app.add_exception_handler(Exception, unhandled_exception_handler)

# --- Routes ---
app.include_router(router)
