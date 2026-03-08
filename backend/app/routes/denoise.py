"""Audio denoising endpoints."""

import io
import logging
import uuid
from pathlib import Path

import soundfile as sf
from app.config import settings
from app.schemas import DenoiseUploadResponse, JobStatus, JobStatusEnum
from app.services.jobs import job_manager
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["denoise"])

_MAX_MB = settings.max_upload_bytes / (1024 * 1024)
_MAX_DURATION_MIN = settings.max_audio_duration_seconds / 60

# Formats where soundfile can read duration from raw bytes
_SOUNDFILE_FORMATS = {".wav", ".flac", ".ogg"}

# Magic byte signatures for supported audio formats
_MAGIC_BYTES: dict[str, list[tuple[int, bytes]]] = {
    ".wav": [(0, b"RIFF")],
    ".mp3": [(0, b"\xff\xfb"), (0, b"\xff\xf3"), (0, b"\xff\xf2"), (0, b"ID3")],
    ".flac": [(0, b"fLaC")],
    ".ogg": [(0, b"OggS")],
    ".m4a": [(4, b"ftyp")],
    ".aac": [(0, b"\xff\xf1"), (0, b"\xff\xf9"), (4, b"ftyp")],
}


def _validate_audio_magic(content: bytes, extension: str) -> bool:
    """Check that file content matches expected magic bytes for the extension.

    Args:
        content: Raw file bytes (at least first 12 bytes needed).
        extension: Lowercase file extension including dot (e.g. ".wav").

    Returns:
        True if any known signature matches, False otherwise.
    """
    signatures = _MAGIC_BYTES.get(extension)
    if signatures is None:
        return False
    for offset, magic in signatures:
        end = offset + len(magic)
        if len(content) >= end and content[offset:end] == magic:
            return True
    return False


def _validate_job_id(job_id: str) -> None:
    """Reject anything that isn't a valid UUID (prevents path traversal)."""
    try:
        uuid.UUID(job_id)
    except ValueError as err:
        raise HTTPException(status_code=400, detail="Invalid job ID") from err


@router.post("/denoise", response_model=DenoiseUploadResponse)
async def denoise_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> DenoiseUploadResponse:
    """Upload and denoise an audio file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate extension (use only the final suffix to ignore path tricks)
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}",
        )

    # Read with size limit
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {_MAX_MB:.0f} MB)",
        )

    # Validate file content matches declared format
    if not _validate_audio_magic(content, file_ext):
        raise HTTPException(
            status_code=400,
            detail=f"File content does not match {file_ext} format",
        )

    # Validate audio duration for formats soundfile can parse
    if file_ext in _SOUNDFILE_FORMATS:
        try:
            info = sf.info(io.BytesIO(content))
            if info.duration > settings.max_audio_duration_seconds:
                raise HTTPException(
                    status_code=400,
                    detail=f"Audio too long ({info.duration:.0f}s). Max {_MAX_DURATION_MIN:.0f} minutes.",
                )
        except HTTPException:
            raise
        except Exception:
            logger.warning("Could not read audio duration for %s file, skipping check", file_ext)

    job_id = str(uuid.uuid4())
    input_path = settings.upload_dir / f"{job_id}{file_ext}"

    try:
        with open(input_path, "wb") as f:
            f.write(content)
        logger.info("Saved upload %s (%d bytes)", job_id, len(content))
    except Exception as err:
        logger.exception("Failed to save upload %s", job_id)
        raise HTTPException(status_code=500, detail="Failed to save file") from err

    job_manager.create_job(job_id)
    background_tasks.add_task(job_manager.process, job_id, input_path)

    return DenoiseUploadResponse(
        job_id=job_id,
        status=JobStatusEnum.QUEUED,
        message="Audio uploaded and queued for denoising",
    )


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """Get processing job status."""
    _validate_job_id(job_id)
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/download/{job_id}")
async def download_audio(job_id: str):
    """Download denoised audio."""
    _validate_job_id(job_id)
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatusEnum.COMPLETED:
        raise HTTPException(status_code=400, detail="Processing not completed")

    # Guard against cleanup deleting the file mid-transfer
    if not job_manager.mark_downloading(job_id):
        raise HTTPException(status_code=404, detail="Job not found")

    try:
        output_path = settings.processed_dir / f"{job_id}_denoised.wav"
        if not output_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=output_path,
            filename="denoised_audio.wav",
            media_type="audio/wav",
            background=BackgroundTask(job_manager.unmark_downloading, job_id),
        )
    except Exception:
        job_manager.unmark_downloading(job_id)
        raise
