"""Audio denoising service using UVR (Ultimate Vocal Remover)."""

import logging
from pathlib import Path

import soundfile as sf
from audio_separator.separator import Separator

from app.services.wav_io import load_mono, read_duration_seconds, write_wav_atomic

logger = logging.getLogger(__name__)

# UVR marks the denoised track with this substring in the filename
DENOISED_OUTPUT_MARKER = "No Noise"
# UVR-DeNoise wrote 44.1 kHz stereo in the image run. The output size cap in
# routes/denoise.py and the download path assume 16 kHz mono, as OpGAN writes.
_OUTPUT_SR = 16_000
# audio-separator catches an OSError from a stem export, logs it, and still returns the stem
# path (common_separator.py:388-390, vr_separator.py:228-229 and 246-247), and a WAV cut
# short reads as valid audio with fewer frames. In every real run measured here (2 s to
# 600 s; WAV, FLAC, MP3, M4A, AAC and OGG inputs), stems were 0 to 9.43 ms shorter than the
# input and never longer, so a larger difference is treated as a failed write.
_STEM_LENGTH_TOLERANCE_S = 0.1


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

    def _check_stem_length(self, stem: Path, input_path: Path) -> None:
        """Raise if *stem* is missing or its length differs from the input's beyond tolerance."""
        if not stem.is_file():
            raise RuntimeError(f"UVR stem {stem.name} was not written")
        stem_s = sf.info(str(stem)).duration
        input_s = read_duration_seconds(input_path)
        if abs(stem_s - input_s) > _STEM_LENGTH_TOLERANCE_S:
            raise RuntimeError(
                f"UVR stem length {stem_s:.3f}s does not match input length {input_s:.3f}s"
            )

    def denoise(self, input_path: Path) -> Path:
        """Run UVR denoising on *input_path* and return the cleaned output path.

        The denoised stem is converted to a 16 kHz mono 16-bit WAV. Every file that
        separate() returned is removed once all of them pass the containment check,
        whether or not the conversion succeeds. Files from a separate() call that
        raises are not known here and are not removed.

        Args:
            input_path: Path to the input audio file.

        Returns:
            Path to the 16 kHz mono WAV in output_dir.

        Raises:
            RuntimeError: If an output escapes output_dir or no denoised stem is found.
            Exception: Errors from separate(), from reading the stem, and from writing
                the WAV propagate.
        """
        separator = self._get_separator()

        logger.info("Denoising: %s", input_path)
        outputs = separator.separate(str(input_path))
        written = [self._contained_output(output) for output in outputs]

        clean_file = next((path for path in written if DENOISED_OUTPUT_MARKER in path.name), None)
        # With output_single_stem unset, the VR separator writes and returns both stems.
        # Job cleanup removes only {job_id}_denoised.*, so a stem left here would count
        # against the artifact budget.
        for path in written:
            if path != clean_file:
                path.unlink(missing_ok=True)

        if clean_file is None:
            raise RuntimeError(f"Could not find denoised output in {outputs}")

        output_path = self._contained_output(f"{input_path.stem}_uvr.wav")
        try:
            self._check_stem_length(clean_file, input_path)
            write_wav_atomic(output_path, load_mono(clean_file, _OUTPUT_SR), _OUTPUT_SR)
        finally:
            clean_file.unlink(missing_ok=True)

        logger.info("Denoised output: %s", output_path)
        return output_path
