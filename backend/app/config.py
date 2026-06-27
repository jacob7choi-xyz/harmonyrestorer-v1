"""Application settings loaded from environment variables."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VALID_LOG_FORMATS: frozenset[str] = frozenset({"text", "json"})


class Settings:
    """Central configuration. Override any value via env vars or .env file."""

    def __init__(self) -> None:
        self.base_dir: Path = Path(__file__).parent.parent

        # Directories
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", str(self.base_dir / "uploads")))
        self.processed_dir = Path(os.getenv("PROCESSED_DIR", str(self.base_dir / "processed")))

        # Audio
        self.supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        self.max_audio_duration_seconds = int(
            os.getenv("MAX_AUDIO_DURATION_SECONDS", str(10 * 60))  # 10 minutes
        )
        if self.max_audio_duration_seconds < 1:
            raise ValueError("MAX_AUDIO_DURATION_SECONDS must be at least 1")
        self.denoiser_engine = os.getenv("DENOISER_ENGINE", "opgan")
        if self.denoiser_engine not in ("opgan", "uvr"):
            raise ValueError(
                f"Invalid DENOISER_ENGINE={self.denoiser_engine!r}, must be 'opgan' or 'uvr'"
            )
        self.uvr_model_name = os.getenv("UVR_MODEL_NAME", "UVR-DeNoise.pth")
        self.opgan_checkpoint = Path(
            os.getenv("OPGAN_CHECKPOINT", str(self.base_dir.parent / "checkpoints" / "final.pt"))
        )

        # Security
        self.max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
        if self.max_upload_bytes < 1:
            raise ValueError("MAX_UPLOAD_BYTES must be at least 1")
        self.rate_limit_max_requests = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
        if self.rate_limit_max_requests < 1:
            raise ValueError("RATE_LIMIT_MAX_REQUESTS must be at least 1")
        self.rate_limit_window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
        if self.rate_limit_window_seconds < 1:
            raise ValueError("RATE_LIMIT_WINDOW_SECONDS must be at least 1")
        self.cors_origins: list[str] = [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
            if o.strip()
        ]

        # Health
        self.min_disk_bytes: int = int(
            os.getenv("MIN_DISK_BYTES", str(100 * 1024 * 1024))  # 100 MB
        )
        if self.min_disk_bytes < 0:
            raise ValueError("MIN_DISK_BYTES must be 0 or greater")

        # Job lifecycle
        self.job_ttl_seconds = int(os.getenv("JOB_TTL_SECONDS", str(60 * 60)))  # 1 hour
        if self.job_ttl_seconds < 1:
            raise ValueError("JOB_TTL_SECONDS must be at least 1")
        self.cleanup_interval_seconds = int(os.getenv("CLEANUP_INTERVAL_SECONDS", str(5 * 60)))
        if self.cleanup_interval_seconds < 1:
            raise ValueError("CLEANUP_INTERVAL_SECONDS must be at least 1")
        self.max_total_jobs = int(os.getenv("MAX_TOTAL_JOBS", "100"))
        if self.max_total_jobs < 1:
            raise ValueError("MAX_TOTAL_JOBS must be at least 1")
        self.max_jobs_per_ip = int(os.getenv("MAX_JOBS_PER_IP", "3"))
        if self.max_jobs_per_ip < 1:
            raise ValueError("MAX_JOBS_PER_IP must be at least 1")
        if self.max_jobs_per_ip > self.max_total_jobs:
            raise ValueError("MAX_JOBS_PER_IP cannot exceed MAX_TOTAL_JOBS")
        self.download_ttl_seconds = int(os.getenv("DOWNLOAD_TTL_SECONDS", "300"))
        if self.download_ttl_seconds < 1:
            raise ValueError("DOWNLOAD_TTL_SECONDS must be at least 1")

        # Server
        self.log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper()
        if self.log_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid LOG_LEVEL={self.log_level!r}, must be one of {sorted(_VALID_LOG_LEVELS)}"
            )
        self.log_format = os.getenv("LOG_FORMAT", "text").strip().lower()
        if self.log_format not in _VALID_LOG_FORMATS:
            raise ValueError(
                f"Invalid LOG_FORMAT={self.log_format!r}, must be one of {sorted(_VALID_LOG_FORMATS)}"
            )
        self.enable_docs: bool = os.getenv("ENABLE_DOCS", "false").strip().lower() == "true"

        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


settings = get_settings()
