"""Application settings loaded from environment variables."""

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_VALID_LOG_LEVELS: frozenset[str] = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_VALID_LOG_FORMATS: frozenset[str] = frozenset({"text", "json"})


class DenoiserEngine(StrEnum):
    """Denoising engines the backend can run, selected with DENOISER_ENGINE."""

    OPGAN = "opgan"
    UVR = "uvr"


# UVR model names checked to run here. UVR-DeNoise ran end to end in images built from
# this Dockerfile. Names of two other audio-separator types are kept out on evidence: the
# MDX architecture module failed to import in those images (torchvision's compiled
# extension did not load), and the Demucs loader calls torch.load with
# weights_only=False on downloaded files. Other types, such as MDXC, are kept out because
# they were not tested here. This limits the name only: audio-separator decides a model's
# architecture from a list it fetches over the network, so the check is a local
# allowlist rather than a lookup.
_DEFAULT_UVR_MODEL = "UVR-DeNoise.pth"
_SUPPORTED_UVR_MODELS: frozenset[str] = frozenset({_DEFAULT_UVR_MODEL})


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
        engine = os.getenv("DENOISER_ENGINE", DenoiserEngine.OPGAN)
        try:
            # Exact match on the value; case is not normalized
            self.denoiser_engine = DenoiserEngine(engine)
        except ValueError as e:
            raise ValueError(
                f"Invalid DENOISER_ENGINE={engine!r}, "
                f"must be one of {sorted(member.value for member in DenoiserEngine)}"
            ) from e
        # Checked whatever the engine, so a bad value surfaces at startup rather than on
        # the first UVR job, which would fail while /health still reports healthy.
        self.uvr_model_name = os.getenv("UVR_MODEL_NAME", _DEFAULT_UVR_MODEL)
        if self.uvr_model_name not in _SUPPORTED_UVR_MODELS:
            raise ValueError(
                f"Invalid UVR_MODEL_NAME={self.uvr_model_name!r}, "
                f"must be one of {sorted(_SUPPORTED_UVR_MODELS)}"
            )
        # Wall-clock limit for decoding one upload with ffmpeg. A full decode of 600 s of
        # MP3 or M4A took 0.4 to 0.9 s on macOS and in the emulated linux/amd64 image.
        self.decode_timeout_seconds = int(os.getenv("DECODE_TIMEOUT_SECONDS", "60"))
        if self.decode_timeout_seconds < 1:
            raise ValueError("DECODE_TIMEOUT_SECONDS must be at least 1")
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
        # Process-local bound on concurrently executing inferences; the
        # deployment runs one application worker per instance, so this is
        # also the instance-wide bound.
        self.inference_concurrency = int(os.getenv("INFERENCE_CONCURRENCY", "1"))
        if self.inference_concurrency < 1:
            raise ValueError("INFERENCE_CONCURRENCY must be at least 1")
        # Byte budget for managed artifacts. The Cloud Run filesystem is
        # memory-backed, so this bounds instance memory, not disk. Default
        # derived from the 4 GiB instance: measured inference peak plus
        # transients plus a 1 GiB safety reserve leaves ~1.5 GiB.
        self.max_artifact_bytes = int(os.getenv("MAX_ARTIFACT_BYTES", str(1536 * 1024 * 1024)))
        if self.max_artifact_bytes < 1:
            raise ValueError("MAX_ARTIFACT_BYTES must be at least 1")

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
