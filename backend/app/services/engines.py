"""The contract each denoising engine meets, and how the configured one is built."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, assert_never

from app.config import OPGAN_CHECKPOINT_SHA256, DenoiserEngine, settings
from app.services.denoiser import DenoiserService
from app.services.opgan_denoiser import OpGANDenoiserService

logger = logging.getLogger(__name__)

# Header allowance per WAV file, as in the fixed output cap this replaces
_WAV_HEADER_BYTES = 1024

# OpGAN: the intermediate and the final 16 kHz mono 16-bit WAV, the allowance
# reserved before engines declared their own.
_OPGAN_FILES = 2
_OPGAN_FILE_BYTES_PER_S = 16_000 * 1 * 2

# UVR-DeNoise through audio-separator 0.44.2. Each stem is 44.1 kHz stereo at up to
# 32 bits (PCM_32 is the widest subtype either of its writers chooses on the VR path
# this model uses). This figure was derived for its pydub writer, which per stem also
# writes an int16 staging file and an ffmpeg output file to the system temp directory:
# the first stem, the second stem, and both staging files present at once, 81,056,990
# bytes measured on one 65.65 s PCM_32 input. The service now uses the soundfile
# writer, which writes no staging files and peaked at 46,318,168 bytes on the same
# input, so this figure is kept as a conservative bound.
_UVR_STEM_BYTES_PER_S = 44_100 * 2 * 4
_UVR_STAGING_BYTES_PER_S = 44_100 * 2 * 2
_UVR_STEM_SIZED_FILES = 3


class Denoiser(Protocol):
    """What the job pipeline needs from an engine.

    An engine is expected to return a 16 kHz mono 16-bit WAV inside its output
    directory and to remove the other files it writes there. The output size cap and
    the download path in routes/denoise.py rely on this; it is not checked at runtime.
    """

    def denoise(self, input_path: Path) -> Path:
        """Denoise *input_path* and return the path of the restored WAV."""
        ...


def peak_output_bytes(engine: DenoiserEngine, duration_s: int) -> int:
    """Return the bytes to reserve for one job of *engine* with *duration_s* of input.

    The figure is the expected peak of files present at once, derived from each engine's
    output formats. It counts files outside the directories the artifact budget scans,
    such as UVR's staging files in the system temp directory, because on a memory-backed
    filesystem they use the same memory.

    Args:
        engine: The configured engine.
        duration_s: Input duration in seconds.

    Returns:
        Bytes to reserve, not counting the input file itself.
    """
    match engine:
        case DenoiserEngine.OPGAN:
            return _OPGAN_FILES * (duration_s * _OPGAN_FILE_BYTES_PER_S + _WAV_HEADER_BYTES)
        case DenoiserEngine.UVR:
            stems = _UVR_STEM_SIZED_FILES * (duration_s * _UVR_STEM_BYTES_PER_S + _WAV_HEADER_BYTES)
            return stems + duration_s * _UVR_STAGING_BYTES_PER_S + _WAV_HEADER_BYTES
        case _:
            assert_never(engine)


def input_sample_rate(engine: DenoiserEngine) -> int:
    """Return the sample rate uploads are decoded to for *engine*.

    OpGAN resamples its input to 16 kHz, so an input already at 16 kHz is not resampled
    again (wav_io.load_mono). For UVR this is audio-separator's default sample_rate,
    44.1 kHz, which DenoiserService does not override; its VR model still loads each
    band at that band's rate (vr_separator.py:273).

    Args:
        engine: The configured engine.

    Returns:
        Sample rate in Hz.
    """
    match engine:
        case DenoiserEngine.OPGAN:
            return 16_000
        case DenoiserEngine.UVR:
            return 44_100
        case _:
            assert_never(engine)


def build_denoiser(engine: DenoiserEngine) -> Denoiser:
    """Build the denoiser for *engine* from the current settings.

    Args:
        engine: The configured engine.

    Returns:
        A denoiser writing into settings.processed_dir.
    """
    # match with assert_never lets mypy report an engine that has no branch here
    match engine:
        case DenoiserEngine.OPGAN:
            denoiser: Denoiser = OpGANDenoiserService(
                output_dir=settings.processed_dir,
                checkpoint_path=settings.opgan_checkpoint,
                expected_sha256=OPGAN_CHECKPOINT_SHA256,
            )
        case DenoiserEngine.UVR:
            denoiser = DenoiserService(
                output_dir=settings.processed_dir,
                model_name=settings.uvr_model_name,
            )
        case _:
            assert_never(engine)
    logger.info("Denoiser initialized for engine %s", engine)
    return denoiser
