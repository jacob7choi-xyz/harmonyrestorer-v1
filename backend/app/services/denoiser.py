"""Audio denoising service using UVR (Ultimate Vocal Remover)."""

import logging
from pathlib import Path

from audio_separator.separator import Separator

logger = logging.getLogger(__name__)

# UVR marks the denoised track with this substring in the filename
DENOISED_OUTPUT_MARKER = "No Noise"


class DenoiserService:
    """Audio denoising using UVR-DeNoise."""

    def __init__(self, output_dir: Path, model_name: str = "UVR-DeNoise.pth") -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._separator: Separator | None = None

    def _get_separator(self) -> Separator:
        """Lazy-load the UVR separator model."""
        if self._separator is None:
            logger.info("Loading UVR denoising model...")
            self._separator = Separator(output_dir=str(self.output_dir), output_format="WAV")
            self._separator.load_model(self.model_name)
            logger.info("UVR model loaded")
        return self._separator

    def denoise(self, input_path: Path) -> Path:
        """Run UVR denoising on *input_path* and return the cleaned output path."""
        separator = self._get_separator()

        logger.info("Denoising: %s", input_path)
        outputs = separator.separate(str(input_path))

        clean_file: Path | None = None
        for output in outputs:
            if DENOISED_OUTPUT_MARKER in output:
                candidate = (self.output_dir / Path(output).name).resolve()
                # Prevent path traversal via unexpected UVR output names
                if not candidate.is_relative_to(self.output_dir):
                    raise RuntimeError(f"UVR output escapes output_dir: {candidate}")
                clean_file = candidate
                break

        if clean_file is None:
            raise RuntimeError(f"Could not find denoised output in {outputs}")

        logger.info("Denoised output: %s", clean_file)
        return clean_file
