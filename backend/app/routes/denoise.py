"""Audio denoising endpoints."""

import asyncio
import io
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path

import audioread
import librosa
import soundfile as sf
from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.config import settings
from app.schemas import DenoiseUploadResponse, JobStatus, JobStatusEnum
from app.services.admission import InferenceAdmission
from app.services.jobs import IPJobCapError, JobCapError, job_manager
from app.services.transcode import (
    DOWNLOAD_FORMATS,
    TranscodeBusyError,
    TranscodeError,
    transcode_output,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["denoise"])

admission = InferenceAdmission(settings.inference_concurrency)

# Strong references keep lifecycle tasks alive after a cancelled request;
# the event loop itself holds only weak references to tasks.
_inference_tasks: set[asyncio.Task[None]] = set()


def _observe_inference_task(task: asyncio.Task[None]) -> None:
    """Drop the registry reference and surface unexpected lifecycle failures."""
    _inference_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        # process() converts processing errors to a FAILED job internally,
        # so an exception escaping the lifecycle means the pipeline itself
        # is broken, not just one job.
        logger.error("Inference lifecycle task failed unexpectedly", exc_info=exc)


async def _run_inference(job_id: str, input_path: Path) -> None:
    """Run one inference and release its admission slot on true completion.

    The slot is owned by this task, not the HTTP handler: a cancelled
    request must not free inference capacity while the worker thread is
    still executing.
    """
    try:
        await run_in_threadpool(job_manager.process, job_id, input_path)
    finally:
        admission.release()


_MAX_MB = settings.max_upload_bytes / (1024 * 1024)
_MAX_DURATION_MIN = settings.max_audio_duration_seconds / 60

# Formats where soundfile can read duration from raw bytes
_SOUNDFILE_FORMATS = {".wav", ".flac", ".ogg"}

# Formats that need a file path for librosa to read duration
_LIBROSA_FORMATS = {".mp3", ".m4a", ".aac"}

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


def _check_audio_duration(tmp_path: str, file_ext: str, content: bytes) -> None:
    """Validate audio duration before processing.

    Security invariant: duration validation must fail closed. If duration cannot
    be validated, the upload must not proceed to inference because malformed or
    highly compressed audio can otherwise bypass the max-duration limit and
    amplify CPU use.

    Args:
        tmp_path: Path to the temporary file on disk (used for librosa formats).
        file_ext: Lowercase file extension including dot (e.g. ".mp3").
        content: Raw file bytes (used for soundfile formats).

    Raises:
        HTTPException: 400 if audio duration exceeds the configured limit or
            cannot be determined due to a known decode or parser error.
        HTTPException: 500 if an unexpected error occurs during duration reading.
    """
    duration: float | None = None
    try:
        if file_ext in _SOUNDFILE_FORMATS:
            duration = sf.info(io.BytesIO(content)).duration
        elif file_ext in _LIBROSA_FORMATS:
            duration = librosa.get_duration(path=tmp_path)
    except (
        sf.SoundFileError,
        librosa.util.exceptions.ParameterError,
        audioread.exceptions.DecodeError,
        audioread.exceptions.NoBackendError,
    ) as exc:
        logger.info("Rejected upload: duration unreadable: %s", exc)
        raise HTTPException(
            status_code=400,
            detail="Could not validate audio duration. Re-export the file and try again.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error reading audio duration")
        raise HTTPException(
            status_code=500,
            detail="Upload processing failed",
        ) from exc

    if duration is not None and duration > settings.max_audio_duration_seconds:
        raise HTTPException(
            status_code=400,
            detail=f"Audio too long ({duration:.0f}s). Max {_MAX_DURATION_MIN:.0f} minutes.",
        )


def _validate_job_id(job_id: str) -> None:
    """Reject anything that isn't a valid UUID (prevents path traversal)."""
    try:
        uuid.UUID(job_id)
    except ValueError as err:
        raise HTTPException(status_code=400, detail="Invalid job ID") from err


_STEM_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9 ._-]+")
_MAX_STEM_LENGTH = 80


def _sanitize_download_stem(filename: str) -> str:
    """Reduce a client filename to a header-safe download stem.

    Client filenames are never used in filesystem paths; this stem appears
    only in the download Content-Disposition header, restricted to a
    conservative character set so header injection is impossible.

    Args:
        filename: Raw client-supplied filename.

    Returns:
        A safe stem, or "audio" when nothing safe remains.
    """
    stem = Path(filename).stem
    stem = _STEM_UNSAFE_CHARS.sub("", stem).strip(" .")
    return stem[:_MAX_STEM_LENGTH] or "audio"


@router.post("/denoise", response_model=DenoiseUploadResponse)
async def denoise_audio(
    request: Request,
    file: UploadFile = File(...),
) -> DenoiseUploadResponse:
    """Upload an audio file and denoise it within the request.

    Transitional contract: processing completes before this endpoint
    responds, while the status and download endpoints are retained so
    existing polling clients keep working unchanged. A processing failure
    returns HTTP 200 with status "failed", matching the job API.

    Cancellation: if the request is cancelled mid-inference the worker
    thread cannot be interrupted; the detached lifecycle task keeps the
    admission slot until the thread finishes, bounding orphaned work to
    the configured inference concurrency. In-flight work and in-memory
    job state do not survive process or instance termination.
    """
    client_ip = request.client.host if request.client else ""

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate extension (use only the final suffix to ignore path tricks)
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}",
        )

    # Read at most max_upload_bytes + 1 so the server never allocates an unbounded
    # body in application memory. Exactly one extra byte is read to distinguish
    # "at the limit" from "over the limit" without a second read.
    content = await file.read(settings.max_upload_bytes + 1)
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

    # Fail-fast admission before any application-owned artifact exists.
    # Inference runs inside this request, so inference capacity is the
    # scarce resource protected here; busy rejection creates no tempfile
    # and no job record (framework multipart spooling may already have
    # occurred before this handler ran).
    if not admission.try_acquire():
        logger.warning("Inference capacity busy; rejecting upload from %s", client_ip or "unknown")
        raise HTTPException(
            status_code=503,
            detail="Server is busy processing another file. Try again shortly.",
            headers={"Retry-After": "30"},
        )

    job_id = str(uuid.uuid4())
    input_path = settings.upload_dir / f"{job_id}{file_ext}"

    # Slot ownership transfers to the lifecycle task once it is created;
    # any failure before that point must release the slot here.
    slot_transferred = False
    try:
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=f".{job_id}.", suffix=f"{file_ext}.tmp", dir=settings.upload_dir
        )
        os.close(tmp_fd)
        try:
            with open(tmp_path_str, "wb") as f:
                f.write(content)
                os.fsync(f.fileno())
            _check_audio_duration(tmp_path_str, file_ext, content)
            Path(tmp_path_str).replace(input_path)
            logger.info("Saved upload %s (%d bytes)", job_id, len(content))
        except HTTPException:
            Path(tmp_path_str).unlink(missing_ok=True)
            raise
        except Exception as err:
            Path(tmp_path_str).unlink(missing_ok=True)
            logger.exception("Failed to save upload %s", job_id)
            raise HTTPException(status_code=500, detail="Failed to save file") from err

        try:
            job_manager.create_job(
                job_id,
                client_ip=client_ip,
                download_stem=_sanitize_download_stem(file.filename),
            )
        except IPJobCapError as exc:
            input_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=429,
                detail="Too many active jobs for your IP. Try again later.",
            ) from exc
        except JobCapError as exc:
            input_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=503,
                detail="Server is busy. Try again later.",
            ) from exc

        try:
            task = asyncio.create_task(_run_inference(job_id, input_path))
        except Exception as err:
            # Transaction rollback at the ownership-transfer boundary: the
            # client never received this job_id, so no record or file may
            # survive; the outer finally returns the admission slot
            job_manager.discard_job(job_id)
            input_path.unlink(missing_ok=True)
            logger.exception("Failed to start inference lifecycle for %s", job_id)
            raise HTTPException(status_code=500, detail="Failed to start processing") from err
        slot_transferred = True
        _inference_tasks.add(task)
        task.add_done_callback(_observe_inference_task)
    finally:
        if not slot_transferred:
            admission.release()

    # Shield: a cancelled request must not cancel the inference lifecycle.
    await asyncio.shield(task)

    job = job_manager.get_job(job_id)
    if job is None:
        # Reachable only if TTL cleanup raced within milliseconds of completion
        return DenoiseUploadResponse(
            job_id=job_id,
            status=JobStatusEnum.COMPLETED,
            message="Denoising complete",
        )
    return DenoiseUploadResponse(job_id=job_id, status=job.status, message=job.message)


@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str) -> JobStatus:
    """Get processing job status."""
    _validate_job_id(job_id)
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/download/{job_id}")
async def download_audio(job_id: str, format: str = Query("wav")) -> FileResponse:
    """Download restored audio in the requested format.

    The restoration is always rendered at 16kHz mono; the format choice
    changes the container and encoding, not the audio resolution. Non-WAV
    formats are converted on first request and cached for the job's lifetime.
    """
    _validate_job_id(job_id)
    fmt = format.strip().lower()
    if fmt not in DOWNLOAD_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Choose one of: {', '.join(sorted(DOWNLOAD_FORMATS))}",
        )

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

        try:
            serve_path = await run_in_threadpool(transcode_output, output_path, fmt)
        except TranscodeBusyError as err:
            raise HTTPException(
                status_code=503,
                detail="Another conversion is in progress. Try again shortly.",
                headers={"Retry-After": "15"},
            ) from err
        except TranscodeError as err:
            logger.error("Download conversion failed for %s: %s", job_id, err)
            raise HTTPException(status_code=500, detail="Format conversion failed") from err

        return FileResponse(
            path=serve_path,
            filename=f"{job.download_stem}_restored.{fmt}",
            media_type=DOWNLOAD_FORMATS[fmt],
            background=BackgroundTask(job_manager.unmark_downloading, job_id),
        )
    except Exception:
        job_manager.unmark_downloading(job_id)
        raise
