"""The contract each denoising engine meets, and how the configured one is built."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, assert_never

from app.config import DenoiserEngine, settings
from app.services.denoiser import DenoiserService
from app.services.opgan_denoiser import OpGANDenoiserService

logger = logging.getLogger(__name__)


class Denoiser(Protocol):
    """What the job pipeline needs from an engine.

    An engine is expected to return a 16 kHz mono 16-bit WAV inside its output
    directory and to remove the other files it writes there. The output size cap and
    the download path in routes/denoise.py rely on this; it is not checked at runtime.
    """

    def denoise(self, input_path: Path) -> Path:
        """Denoise *input_path* and return the path of the restored WAV."""
        ...


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
