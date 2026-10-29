"""Decode uploads into a bounded 16-bit PCM WAV with ffmpeg before anything else reads them.

The duration an audio file reports can differ from what it holds: a VBR MP3 without a
Xing header read as 19.16 s of a 30.09 s file through soundfile, and an M4A whose header
claimed 10 s decoded to 1200 s through ffmpeg. Decoding once, with the length counted
from the decoded frames, gives the duration check and both engines the same audio.

The upload is attacker-controlled, so ffmpeg is not left to probe it. The demuxer is
pinned from the extension and magic bytes the route has already validated, only the file
protocol is allowed, MP4 external references are refused explicitly, and the output is
capped by length and size.
"""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404 - argv only, no shell; see decode_to_wav
import tempfile
from pathlib import Path

import soundfile as sf

logger = logging.getLogger(__name__)

# Sources with more channels are mixed down to stereo; mono and stereo pass through
# unchanged, so each engine mixes channels as it did before decoding was added. ffmpeg's
# own mono-to-stereo upmix lowered the level by 3 dB in testing.
_MAX_CHANNELS = 2
_SAMPLE_BYTES = 2  # pcm_s16le
# WAV header and the one packet ffmpeg may write past -fs
_SIZE_CAP_SLACK_BYTES = 64 * 1024
_STDERR_LOG_CHARS = 500


class AudioDecodeError(Exception):
    """The upload could not be decoded as audio."""


class AudioTooLongError(AudioDecodeError):
    """The decoded audio is longer than the allowed maximum."""

    def __init__(self, duration_s: float) -> None:
        super().__init__(f"Decoded audio is {duration_s:.1f}s")
        self.duration_s = duration_s


class DecoderUnavailableError(Exception):
    """ffmpeg could not be run or did not finish in time."""


def ffmpeg_version(timeout_s: float) -> str | None:
    """Return the first line of ``ffmpeg -version``, or None if ffmpeg cannot run.

    The image installs ffmpeg without a version pin, and decoded lengths differed
    slightly between the two installs measured (Debian's 7.1.5 in the linux/amd64
    image, Homebrew's 9.0.2 on macOS), so the build in use is worth recording.

    Args:
        timeout_s: Wall-clock limit for the version query.

    Returns:
        The version line, or None if ffmpeg is missing, cannot be started (for example
        without execute permission), fails, or times out.
    """
    try:
        # Fixed argv, no shell
        result = subprocess.run(  # nosec B603 B607
            ["ffmpeg", "-hide_banner", "-version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    # OSError covers a missing binary (FileNotFoundError), one without execute permission
    # (PermissionError) and one that is not a valid program (exec format error)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.decode(errors="replace").splitlines()[:1]
    return first_line[0] if first_line else None


def demuxer_for(extension: str, content: bytes) -> str:
    """Return the ffmpeg demuxer for an upload whose magic bytes the route has checked.

    Args:
        extension: Lowercase extension including the dot.
        content: The upload's bytes; only the first few are read.

    Returns:
        The demuxer name to pass to ffmpeg with -f.

    Raises:
        AudioDecodeError: If the extension has no demuxer here.
    """
    if extension == ".aac":
        # .aac uploads may be ADTS or AAC in an MP4 container; the route accepts both
        return "mov" if content[4:8] == b"ftyp" else "aac"
    demuxers = {".wav": "wav", ".mp3": "mp3", ".flac": "flac", ".ogg": "ogg", ".m4a": "mov"}
    if extension not in demuxers:
        raise AudioDecodeError(f"No demuxer for {extension}")
    return demuxers[extension]


def decoded_size_cap(sample_rate: int, max_seconds: int) -> int:
    """Return the largest decoded WAV accepted: one second past the cap, two channels.

    Args:
        sample_rate: Output sample rate.
        max_seconds: Maximum accepted duration.

    Returns:
        Size cap in bytes.
    """
    return (max_seconds + 1) * sample_rate * _MAX_CHANNELS * _SAMPLE_BYTES + _SIZE_CAP_SLACK_BYTES


def _argv(
    source: Path, output: Path, demuxer: str, sample_rate: int, max_seconds: int
) -> list[str]:
    """Build the ffmpeg command. Every input-handling control is explicit."""
    input_options = ["-protocol_whitelist", "file"]
    if demuxer == "mov":
        # Both default to false in the image's ffmpeg 7.1.5; stated so a default change
        # cannot turn them on
        input_options += ["-enable_drefs", "0", "-use_absolute_path", "0"]
    return [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-threads",
        "1",
        *input_options,
        "-f",
        demuxer,
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-sn",
        "-dn",
        "-map_metadata",
        "-1",
        "-map_chapters",
        "-1",
        "-af",
        "aformat=channel_layouts=mono|stereo",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-threads",
        "1",
        "-t",
        str(max_seconds + 1),
        "-fs",
        str(decoded_size_cap(sample_rate, max_seconds)),
        "-fflags",
        "+bitexact",
        "-f",
        "wav",
        "-y",
        str(output),
    ]


def _run(argv: list[str], source: Path, timeout_s: float) -> None:
    """Run ffmpeg, mapping its failures to decode errors."""
    try:
        # argv is built above from server-generated paths and a fixed demuxer map; no
        # shell is involved
        result = subprocess.run(  # nosec B603 B607
            argv,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except OSError as e:
        # Missing, without execute permission, or not a valid program
        raise DecoderUnavailableError(f"ffmpeg could not be started: {e}") from e
    except subprocess.TimeoutExpired as e:
        raise DecoderUnavailableError(f"ffmpeg did not finish within {timeout_s}s") from e
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[:_STDERR_LOG_CHARS]
        logger.info("ffmpeg could not decode %s: %s", source.name, stderr)
        raise AudioDecodeError("ffmpeg could not decode the upload")


def decode_to_wav(
    source: Path,
    destination: Path,
    *,
    demuxer: str,
    sample_rate: int,
    max_seconds: int,
    timeout_s: float,
) -> float:
    """Decode *source* into a 16-bit PCM WAV at *destination* and return its duration.

    The WAV is written to a temp file beside *destination* and moved into place only
    after its length is checked, so a rejected upload leaves nothing behind.

    Args:
        source: The validated upload.
        destination: Final WAV path.
        demuxer: ffmpeg demuxer from demuxer_for.
        sample_rate: Output sample rate.
        max_seconds: Maximum accepted duration.
        timeout_s: Wall-clock limit for ffmpeg.

    Returns:
        Duration in seconds, from the decoded frame count.

    Raises:
        AudioTooLongError: If the decoded audio is longer than max_seconds.
        AudioDecodeError: If the upload cannot be decoded or holds no audio.
        DecoderUnavailableError: If ffmpeg is missing, cannot be started, or times out.
    """
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".wav.tmp", dir=destination.parent
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        _run(_argv(source, tmp, demuxer, sample_rate, max_seconds), source, timeout_s)
        try:
            info = sf.info(str(tmp))
        except sf.SoundFileError as e:
            raise AudioDecodeError("ffmpeg produced no readable audio") from e
        if info.frames == 0:
            raise AudioDecodeError("The upload holds no audio")
        duration = float(info.frames / info.samplerate)
        if duration > max_seconds or tmp.stat().st_size >= decoded_size_cap(
            sample_rate, max_seconds
        ):
            raise AudioTooLongError(duration)
        tmp.replace(destination)
        return duration
    finally:
        # After a successful replace the temp path no longer exists
        tmp.unlink(missing_ok=True)
