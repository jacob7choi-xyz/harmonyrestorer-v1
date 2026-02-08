"""Audio denoising endpoints."""

import logging
import uuid
from pathlib import Path

from app.config import settings
from app.services.jobs import job_manager
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["denoise"])

_MAX_MB = settings.max_upload_bytes / (1024 * 1024)


def _validate_job_id(job_id: str) -> None:
    """Reject anything that isn't a valid UUID (prevents path traversal)."""
    try:
        uuid.UUID(job_id)
    except ValueError as err:
        raise HTTPException(status_code=400, detail="Invalid job ID") from err


@router.post("/denoise")
async def denoise_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
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

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Audio uploaded and queued for denoising",
    }


@router.get("/status/{job_id}")
async def get_job_status(job_id: str):
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

    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")

    output_path = settings.processed_dir / f"{job_id}_denoised.wav"
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(output_path),
        filename="denoised_audio.wav",
        media_type="audio/wav",
    )
