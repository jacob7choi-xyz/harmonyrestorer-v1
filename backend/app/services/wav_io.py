"""WAV reading and writing shared by the denoiser services."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# The soundfile default for WAV, passed explicitly so the written sample width does not
# depend on that default
_SUBTYPE = "PCM_16"


def load_mono(path: Path, target_sr: int) -> np.ndarray:
    """Read an audio file as float32 mono at *target_sr*.

    Args:
        path: Audio file readable by soundfile.
        target_sr: Sample rate to resample to when the file differs.

    Returns:
        One-dimensional float32 samples at target_sr.
    """
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        logger.debug("Resampling from %dHz to %dHz", sr, target_sr)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    # soundfile and librosa are untyped; the samples are already float32, so this
    # restates the type for mypy without copying
    return np.asarray(audio, dtype=np.float32)


def read_duration_seconds(path: Path) -> float:
    """Return the duration of an audio file in seconds, as librosa reads it.

    librosa.get_duration asks soundfile first and falls back to audioread for formats
    soundfile cannot open, such as m4a and aac.

    Args:
        path: Audio file to measure.

    Returns:
        Duration in seconds.
    """
    return float(librosa.get_duration(path=str(path)))


def write_wav_atomic(output_path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write *audio* as a 16-bit WAV at *output_path*, or leave it untouched.

    The samples go to a temp file in the same directory, which is synced and
    then moved into place, so a crash never leaves a partial file at the target.

    Args:
        output_path: Final file path.
        audio: Samples to write.
        sample_rate: Sample rate recorded in the WAV header.
    """
    tmp_fd, tmp_str = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".wav.tmp", dir=output_path.parent
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_str)
    try:
        sf.write(tmp_path, audio, sample_rate, format="WAV", subtype=_SUBTYPE)
        with open(tmp_path, "rb") as f:
            os.fsync(f.fileno())
        tmp_path.replace(output_path)
    finally:
        # After a successful replace the temp path no longer exists
        tmp_path.unlink(missing_ok=True)
