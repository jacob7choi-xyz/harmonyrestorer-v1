"""Application settings loaded from environment variables."""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration. Override any value via env vars or .env file."""

    def __init__(self) -> None:
        self.base_dir: Path = Path(__file__).parent.parent

        # Directories
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", str(self.base_dir / "uploads")))
        self.processed_dir = Path(os.getenv("PROCESSED_DIR", str(self.base_dir / "processed")))

        # Audio
        self.supported_formats = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
        self.uvr_model_name = os.getenv("UVR_MODEL_NAME", "UVR-DeNoise.pth")

        # Security
        self.max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
        self.rate_limit_max_requests = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", "10"))
        self.rate_limit_window_seconds = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
        self.cors_origins: list[str] = [
            o.strip()
            for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
            if o.strip()
        ]

        # Job lifecycle
        self.job_ttl_seconds = int(os.getenv("JOB_TTL_SECONDS", str(60 * 60)))  # 1 hour
        self.cleanup_interval_seconds = int(os.getenv("CLEANUP_INTERVAL_SECONDS", str(5 * 60)))

        # Server
        self.log_level = os.getenv("LOG_LEVEL", "INFO")

        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()


settings = get_settings()
