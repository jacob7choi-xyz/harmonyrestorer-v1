"""Convert restored WAV output into user-selected download formats."""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404 - fixed argv, no shell, inputs validated below
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscodeError(Exception):
    """Raised when converting restored audio to a download format fails."""


class TranscodeBusyError(TranscodeError):
    """Raised when the single conversion slot is already occupied."""


_FFMPEG_TIMEOUT_SECONDS = 120

# At most one ffmpeg conversion at a time, mirroring inference admission:
# expensive work gets an explicit fail-fast bound instead of competing
# threadpool-wide for the instance's two vCPUs. The slot is acquired and
# released inside the worker thread that runs the subprocess, so a
# cancelled HTTP request cannot free capacity while ffmpeg still runs.
_TRANSCODE_SLOTS = threading.BoundedSemaphore(1)

# Formats servable for download, mapped to media types. wav is the model's
# native output and is served directly without conversion.
DOWNLOAD_FORMATS: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "m4a": "audio/mp4",
}


def transcode_output(wav_path: Path, fmt: str) -> Path:
    """Convert a restored WAV into the requested format, cached beside it.

    The encoder is inferred by ffmpeg from the output extension. Results are
    written atomically and cached, so repeat downloads of the same format
    never reconvert.

    Args:
        wav_path: Path to the restored 16kHz mono WAV.
        fmt: Target format key from DOWNLOAD_FORMATS.

    Returns:
        Path to the file to serve (the WAV itself for fmt "wav").

    Raises:
        TranscodeError: If the format is unknown, ffmpeg fails, or times out.
    """
    if fmt == "wav":
        return wav_path
    if fmt not in DOWNLOAD_FORMATS:
        raise TranscodeError(f"Unsupported download format: {fmt}")

    target = wav_path.with_suffix(f".{fmt}")
    if target.exists():
        return target

    if not _TRANSCODE_SLOTS.acquire(blocking=False):
        logger.warning("Conversion slot busy; rejecting %s to %s", wav_path.name, fmt)
        raise TranscodeBusyError("Another conversion is in progress")
    tmp_path: Path | None = None
    try:
        tmp_fd, tmp_str = tempfile.mkstemp(suffix=f".{fmt}", dir=wav_path.parent)
        os.close(tmp_fd)
        tmp_path = Path(tmp_str)
        # Single encoder thread: the admitted conversion must not itself
        # consume uncontrolled parallel CPU next to a running inference
        argv = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
            str(wav_path),
            "-threads",
            "1",
            str(tmp_path),
        ]
        # nosec B603 - argv is fixed; paths are server-generated (UUID-derived)
        # and fmt is allowlisted above; no shell is involved
        result = subprocess.run(  # nosec
            argv,
            capture_output=True,
            timeout=_FFMPEG_TIMEOUT_SECONDS,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[:500]
            logger.error("ffmpeg failed converting %s to %s: %s", wav_path.name, fmt, stderr)
            raise TranscodeError(f"Conversion to {fmt} failed")
        tmp_path.replace(target)
        logger.info("Converted %s to %s", wav_path.name, target.name)
    except subprocess.TimeoutExpired as err:
        logger.error("ffmpeg timed out converting %s to %s", wav_path.name, fmt)
        raise TranscodeError(f"Conversion to {fmt} timed out") from err
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        _TRANSCODE_SLOTS.release()
    return target
