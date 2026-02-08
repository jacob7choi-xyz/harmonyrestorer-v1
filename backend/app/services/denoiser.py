"""
Audio Denoising Service using UVR
"""

import logging
from pathlib import Path

from audio_separator.separator import Separator

logger = logging.getLogger(__name__)


class DenoiserService:
    """Audio denoising using UVR-DeNoise"""

    def __init__(self, output_dir: Path, model_name: str = "UVR-DeNoise.pth"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._separator: Separator | None = None

    def _get_separator(self) -> Separator:
        """Lazy load the separator"""
        if self._separator is None:
            logger.info("Loading UVR denoising model...")
            self._separator = Separator(output_dir=str(self.output_dir), output_format="WAV")
            self._separator.load_model(self.model_name)
            logger.info("UVR model loaded")
        return self._separator

    def denoise(self, input_path: Path) -> Path:
        separator = self._get_separator()

        logger.info(f"Denoising: {input_path}")
        outputs = separator.separate(str(input_path))

        clean_file = None
        for output in outputs:
            if "No Noise" in output:
                clean_file = self.output_dir / Path(output).name
                break

        if clean_file is None:
            raise RuntimeError(f"Could not find denoised output in {outputs}")

        logger.info(f"Denoised output: {clean_file}")
        return clean_file
