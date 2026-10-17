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
            sep = Separator(output_dir=str(self.output_dir), output_format="WAV")
            sep.load_model(self.model_name)
            # Only assign after load_model succeeds so a failed load is retried
            self._separator = sep
            logger.info("UVR model loaded")
        return self._separator

    def _contained_output(self, output: str) -> Path:
        """Resolve a separator output name inside output_dir, or raise."""
        candidate = (self.output_dir / Path(output).name).resolve()
        # Prevent path traversal via unexpected UVR output names
        if not candidate.is_relative_to(self.output_dir):
            raise RuntimeError(f"UVR output escapes output_dir: {candidate}")
        return candidate

    def denoise(self, input_path: Path) -> Path:
        """Run UVR denoising on *input_path* and return the cleaned output path."""
        separator = self._get_separator()

        logger.info("Denoising: %s", input_path)
        outputs = separator.separate(str(input_path))
        written = [self._contained_output(output) for output in outputs]

        clean_file = next((path for path in written if DENOISED_OUTPUT_MARKER in path.name), None)
        # UVR writes one file per stem. Job cleanup removes only {job_id}_denoised.*, so a
        # stem left here would stay in output_dir and count against the artifact budget.
        for path in written:
            if path != clean_file:
                path.unlink(missing_ok=True)

        if clean_file is None:
            raise RuntimeError(f"Could not find denoised output in {outputs}")

        logger.info("Denoised output: %s", clean_file)
        return clean_file
