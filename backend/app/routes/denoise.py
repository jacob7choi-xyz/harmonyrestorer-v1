"""Audio denoising endpoints."""

import asyncio
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.config import settings
from app.schemas import DenoiseUploadResponse, JobStatus, JobStatusEnum
from app.services.admission import InferenceAdmission
from app.services.artifacts import artifact_budget
from app.services.audio_decode import (
    AudioDecodeError,
    AudioTooLongError,
    DecoderUnavailableError,
    decode_to_wav,
    decoded_size_cap,
    demuxer_for,
)
from app.services.engines import input_sample_rate, peak_output_bytes
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


async def _run_inference(job_id: str, input_path: Path, reservation_bytes: int) -> None:
    """Run one inference; release its slot and byte reservation on completion.

    Both resources are owned by this task, not the HTTP handler: a
    cancelled request must not free inference capacity or budget headroom
    while the worker thread is still executing. The reservation is
    released after process() has materialized or removed its files, so
    subsequent budget scans see the persisted truth.
    """
    try:
        await run_in_threadpool(job_manager.process, job_id, input_path)
    finally:
        admission.release()
        artifact_budget.release(reservation_bytes)


_MAX_MB = settings.max_upload_bytes / (1024 * 1024)
_MAX_DURATION_MIN = settings.max_audio_duration_seconds / 60

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


async def _decode_upload(raw: Path, destination: Path, demuxer: str, sample_rate: int) -> float:
    """Decode the upload in a worker thread and return its decoded duration.

    ffmpeg cannot be interrupted from here. If the request is cancelled while it runs,
    this waits for it to stop and removes its output before letting the cancellation
    through, because the caller's finally then releases the admission slot and the
    byte reservation, which must not happen while the decode still uses them.

    Args:
        raw: The upload as received.
        destination: Path for the decoded WAV.
        demuxer: ffmpeg demuxer from demuxer_for.
        sample_rate: Rate the configured engine reads input at.

    Returns:
        Duration in seconds, from the decoded frame count.
    """
    decode = asyncio.ensure_future(
        run_in_threadpool(
            decode_to_wav,
            raw,
            destination,
            demuxer=demuxer,
            sample_rate=sample_rate,
            max_seconds=settings.max_audio_duration_seconds,
            timeout_s=settings.decode_timeout_seconds,
        )
    )
    try:
        return await asyncio.shield(decode)
    except asyncio.CancelledError:
        await asyncio.wait({decode})
        if not decode.cancelled() and decode.exception() is None:
            destination.unlink(missing_ok=True)
        raise


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
    # Engines read only the decoded WAV, never the upload as received
    input_path = settings.upload_dir / f"{job_id}.wav"
    engine = settings.denoiser_engine
    sample_rate = input_sample_rate(engine)
    max_seconds = settings.max_audio_duration_seconds

    # Bytes this job's lifecycle is expected to hold at once: the upload as received,
    # its decoded WAV at the size cap, and the configured engine's peak output
    upload_reservation = (
        len(content)
        + decoded_size_cap(sample_rate, max_seconds)
        + peak_output_bytes(engine, max_seconds)
    )

    # Slot and reservation ownership transfer to the lifecycle task once it
    # is created; any failure before that point must release both here.
    slot_transferred = False
    reservation_made = False
    try:
        if not artifact_budget.try_reserve(upload_reservation):
            raise HTTPException(
                status_code=503,
                detail="Server storage is at capacity. Try again shortly.",
                headers={"Retry-After": "60"},
            )
        reservation_made = True
        tmp_fd, tmp_path_str = tempfile.mkstemp(
            prefix=f".{job_id}.", suffix=f"{file_ext}.tmp", dir=settings.upload_dir
        )
        os.close(tmp_fd)
        raw_path = Path(tmp_path_str)
        try:
            with open(raw_path, "wb") as f:
                f.write(content)
                os.fsync(f.fileno())
            # Fail closed: the only duration accepted is the decoded one, so no upload
            # reaches inference without its decoded length checked against the cap
            duration = await _decode_upload(
                raw_path, input_path, demuxer_for(file_ext, content), sample_rate
            )
            logger.info("Saved upload %s (%d bytes, %.1fs decoded)", job_id, len(content), duration)
        except AudioTooLongError as exc:
            logger.info("Rejected upload %s: %s", job_id, exc)
            raise HTTPException(
                status_code=400,
                detail=f"Audio too long. Max {_MAX_DURATION_MIN:.0f} minutes.",
            ) from exc
        except AudioDecodeError as exc:
            logger.info("Rejected upload %s: %s", job_id, exc)
            raise HTTPException(
                status_code=400,
                detail="Could not decode the audio. Re-export the file and try again.",
            ) from exc
        except DecoderUnavailableError as exc:
            # The server could not decode, which is not the uploader's fault
            logger.error("Decoder unavailable for upload %s: %s", job_id, exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Upload processing failed") from exc
        except Exception as err:
            logger.exception("Failed to save upload %s", job_id)
            raise HTTPException(status_code=500, detail="Failed to save file") from err
        finally:
            raw_path.unlink(missing_ok=True)

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
            task = asyncio.create_task(_run_inference(job_id, input_path, upload_reservation))
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
            if reservation_made:
                artifact_budget.release(upload_reservation)

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
