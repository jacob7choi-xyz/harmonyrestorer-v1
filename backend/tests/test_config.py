"""Tests for Settings validation: every env var that has a validation guard."""

import pytest
from app.config import Settings


def _base_dirs(monkeypatch, tmp_path) -> None:
    """Pin upload and processed dirs to tmp_path so acceptance tests stay hermetic."""
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))


# --- Numeric rejection ---


@pytest.mark.parametrize(
    ("env_name", "message"),
    [
        ("MAX_AUDIO_DURATION_SECONDS", "MAX_AUDIO_DURATION_SECONDS must be at least 1"),
        ("MAX_UPLOAD_BYTES", "MAX_UPLOAD_BYTES must be at least 1"),
        ("RATE_LIMIT_MAX_REQUESTS", "RATE_LIMIT_MAX_REQUESTS must be at least 1"),
        ("RATE_LIMIT_WINDOW_SECONDS", "RATE_LIMIT_WINDOW_SECONDS must be at least 1"),
        ("JOB_TTL_SECONDS", "JOB_TTL_SECONDS must be at least 1"),
        ("CLEANUP_INTERVAL_SECONDS", "CLEANUP_INTERVAL_SECONDS must be at least 1"),
        ("MAX_TOTAL_JOBS", "MAX_TOTAL_JOBS must be at least 1"),
        ("MAX_JOBS_PER_IP", "MAX_JOBS_PER_IP must be at least 1"),
        ("DOWNLOAD_TTL_SECONDS", "DOWNLOAD_TTL_SECONDS must be at least 1"),
        ("INFERENCE_CONCURRENCY", "INFERENCE_CONCURRENCY must be at least 1"),
        ("MAX_ARTIFACT_BYTES", "MAX_ARTIFACT_BYTES must be at least 1"),
    ],
)
def test_numeric_settings_reject_zero(monkeypatch, env_name, message) -> None:
    """Settings raises ValueError when any numeric setting is set to zero."""
    monkeypatch.setenv(env_name, "0")
    with pytest.raises(ValueError, match=message):
        Settings()


# --- Numeric acceptance at minimum boundary ---


@pytest.mark.parametrize(
    "env_name",
    [
        "MAX_AUDIO_DURATION_SECONDS",
        "MAX_UPLOAD_BYTES",
        "RATE_LIMIT_MAX_REQUESTS",
        "RATE_LIMIT_WINDOW_SECONDS",
        "JOB_TTL_SECONDS",
        "CLEANUP_INTERVAL_SECONDS",
    ],
)
def test_numeric_settings_accept_one(monkeypatch, tmp_path, env_name) -> None:
    """Settings accepts 1 as the minimum valid value for independent numeric settings."""
    _base_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv(env_name, "1")
    Settings()  # must not raise


# --- MIN_DISK_BYTES ---


def test_min_disk_bytes_rejects_negative(monkeypatch) -> None:
    """MIN_DISK_BYTES rejects negative values."""
    monkeypatch.setenv("MIN_DISK_BYTES", "-1")
    with pytest.raises(ValueError, match="MIN_DISK_BYTES must be 0 or greater"):
        Settings()


def test_min_disk_bytes_accepts_zero(monkeypatch, tmp_path) -> None:
    """MIN_DISK_BYTES=0 is valid and disables the disk-space guard."""
    _base_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("MIN_DISK_BYTES", "0")
    Settings()  # must not raise


# --- Job cap cross-constraint ---


def test_max_jobs_per_ip_cannot_exceed_max_total_jobs(monkeypatch) -> None:
    """MAX_JOBS_PER_IP > MAX_TOTAL_JOBS raises ValueError."""
    monkeypatch.setenv("MAX_TOTAL_JOBS", "2")
    monkeypatch.setenv("MAX_JOBS_PER_IP", "3")
    with pytest.raises(ValueError, match="MAX_JOBS_PER_IP cannot exceed MAX_TOTAL_JOBS"):
        Settings()


def test_job_cap_settings_accept_boundary_values(monkeypatch, tmp_path) -> None:
    """MAX_TOTAL_JOBS=1, MAX_JOBS_PER_IP=1, DOWNLOAD_TTL_SECONDS=1 is a valid config."""
    _base_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("MAX_TOTAL_JOBS", "1")
    monkeypatch.setenv("MAX_JOBS_PER_IP", "1")
    monkeypatch.setenv("DOWNLOAD_TTL_SECONDS", "1")
    Settings()  # must not raise


# --- LOG_LEVEL ---


def test_log_level_rejects_unknown_value(monkeypatch) -> None:
    """An unrecognized LOG_LEVEL raises ValueError."""
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValueError, match="Invalid LOG_LEVEL"):
        Settings()


@pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_log_level_accepts_valid_values(monkeypatch, tmp_path, level) -> None:
    """Each valid log level is accepted without error."""
    _base_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("LOG_LEVEL", level)
    Settings()  # must not raise


def test_log_level_normalizes_lowercase(monkeypatch, tmp_path) -> None:
    """LOG_LEVEL=debug is normalized to DEBUG and accepted."""
    _base_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert Settings().log_level == "DEBUG"


# --- LOG_FORMAT ---


def test_log_format_rejects_unknown_value(monkeypatch) -> None:
    """An unrecognized LOG_FORMAT raises ValueError."""
    monkeypatch.setenv("LOG_FORMAT", "xml")
    with pytest.raises(ValueError, match="Invalid LOG_FORMAT"):
        Settings()


@pytest.mark.parametrize("fmt", ["text", "json"])
def test_log_format_accepts_valid_values(monkeypatch, tmp_path, fmt) -> None:
    """Each valid log format is accepted without error."""
    _base_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("LOG_FORMAT", fmt)
    Settings()  # must not raise


def test_log_format_normalizes_uppercase(monkeypatch, tmp_path) -> None:
    """LOG_FORMAT=JSON is normalized to json and accepted."""
    _base_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("LOG_FORMAT", "JSON")
    assert Settings().log_format == "json"


# --- DENOISER_ENGINE ---


@pytest.mark.parametrize(
    "engine",
    [
        "UVR",  # matching is exact; unlike LOG_LEVEL, case is not normalized
        "demucs",  # not an engine this backend provides
    ],
)
def test_denoiser_engine_rejects_unknown_value(monkeypatch, engine) -> None:
    """A DENOISER_ENGINE other than opgan or uvr raises ValueError."""
    monkeypatch.setenv("DENOISER_ENGINE", engine)
    with pytest.raises(ValueError, match="Invalid DENOISER_ENGINE"):
        Settings()


@pytest.mark.parametrize("engine", ["opgan", "uvr"])
def test_denoiser_engine_accepts_valid_values(monkeypatch, tmp_path, engine) -> None:
    """Each supported engine is accepted and stored as given."""
    _base_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DENOISER_ENGINE", engine)
    assert Settings().denoiser_engine == engine


def test_denoiser_engine_defaults_to_opgan(monkeypatch, tmp_path) -> None:
    """With DENOISER_ENGINE unset, the engine is opgan."""
    _base_dirs(monkeypatch, tmp_path)
    monkeypatch.delenv("DENOISER_ENGINE", raising=False)
    assert Settings().denoiser_engine == "opgan"


# --- UVR_MODEL_NAME ---


@pytest.mark.parametrize(
    "model_name",
    [
        "UVR-MDX-NET-Inst_full_292.onnx",  # MDX: its architecture module fails to import
        "htdemucs_ft.yaml",  # Demucs: loads its weights with weights_only=False
        "UVR-Denoise.pth",  # a case typo of the supported name
    ],
)
def test_uvr_model_name_rejects_unsupported_value(monkeypatch, model_name) -> None:
    """A UVR model outside the supported set raises ValueError."""
    monkeypatch.setenv("UVR_MODEL_NAME", model_name)
    with pytest.raises(ValueError, match="Invalid UVR_MODEL_NAME"):
        Settings()


def test_uvr_model_name_accepts_default(monkeypatch, tmp_path) -> None:
    """The default UVR model is accepted."""
    _base_dirs(monkeypatch, tmp_path)
    monkeypatch.delenv("UVR_MODEL_NAME", raising=False)
    assert Settings().uvr_model_name == "UVR-DeNoise.pth"
