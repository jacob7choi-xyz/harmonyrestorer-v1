"""WAV reading and writing shared by the denoiser services."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import audioread
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


# audioread yields 16-bit PCM buffers, which librosa.load converts with buf_to_float
_AUDIOREAD_SAMPLE_BYTES = 2


def read_duration_seconds(path: Path) -> float:
    """Return the decoded duration of an audio file in seconds.

    Reads the way librosa.load does: soundfile first, then audioread for formats
    soundfile cannot open, such as m4a and aac. For audioread the decoded samples are
    counted, because the duration it reports from the container can be an estimate: in
    the linux/amd64 image, an ADTS AAC file reported 66.5 s and decoded to 65.69 s.

    Args:
        path: Audio file to measure.

    Returns:
        Duration in seconds.
    """
    try:
        return float(sf.info(str(path)).duration)
    except sf.SoundFileRuntimeError:
        pass
    with audioread.audio_open(str(path)) as source:
        decoded_bytes = sum(len(buffer) for buffer in source)
        frames = decoded_bytes // (_AUDIOREAD_SAMPLE_BYTES * source.channels)
        return float(frames / source.samplerate)


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
