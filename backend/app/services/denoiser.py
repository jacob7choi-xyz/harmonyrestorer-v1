"""Audio denoising service using UVR (Ultimate Vocal Remover)."""

import logging
import shutil
import threading
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
# audio-separator writes every stem into its output_dir and, after a failed write, can
# return without naming the stems it already wrote (separator.py:978-987). The service
# gives it a directory of its own and empties it around each job, so no stem outlives the
# job whatever the separator returns. It sits inside output_dir so the artifact budget
# counts what is in it.
STAGING_DIR_NAME = ".uvr-staging"


class DenoiserService:
    """Audio denoising using UVR-DeNoise."""

    def __init__(self, output_dir: Path, model_name: str = "UVR-DeNoise.pth") -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._staging_dir = self.output_dir / STAGING_DIR_NAME
        self._separator: Separator | None = None
        # One job at a time: a job empties the shared staging directory, which must not
        # happen while another job's stems are in it
        self._lock = threading.Lock()

    def _get_separator(self) -> Separator:
        """Lazy-load the UVR separator model."""
        if self._separator is None:
            logger.info("Loading UVR denoising model...")
            sep = Separator(output_dir=str(self._staging_dir), output_format="WAV")
            sep.load_model(self.model_name)
            # Only assign after load_model succeeds so a failed load is retried
            self._separator = sep
            logger.info("UVR model loaded")
        return self._separator

    @staticmethod
    def _contained(directory: Path, name: str) -> Path:
        """Resolve *name* inside *directory*, or raise."""
        candidate = (directory / Path(name).name).resolve()
        # Prevent path traversal via unexpected UVR output names
        if not candidate.is_relative_to(directory):
            raise RuntimeError(f"UVR output escapes {directory}: {candidate}")
        return candidate

    def _prepare_staging(self) -> None:
        """Create the staging directory if needed and empty it. Caller holds the lock.

        Raises:
            RuntimeError: If the staging path is a symlink.
            OSError: If the directory cannot be created or emptied.
        """
        if self._staging_dir.is_symlink():
            raise RuntimeError(f"UVR staging directory is a symlink: {self._staging_dir}")
        self._staging_dir.mkdir(exist_ok=True)
        self._empty_staging()

    def _empty_staging(self) -> None:
        """Remove everything inside the staging directory, never following symlinks."""
        for entry in self._staging_dir.iterdir():
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()

    def _separate_and_convert(self, input_path: Path) -> Path:
        """Run the separator and convert its denoised stem. Caller holds the lock."""
        separator = self._get_separator()
        outputs = separator.separate(str(input_path))
        stems = [self._contained(self._staging_dir, output) for output in outputs]
        clean_file = next((path for path in stems if DENOISED_OUTPUT_MARKER in path.name), None)
        if clean_file is None:
            raise RuntimeError(
                f"UVR returned no denoised stem for {input_path.name} (returned {outputs}); "
                "audio-separator logs the cause"
            )
        self._check_stem_length(clean_file, input_path)
        output_path = self._contained(self.output_dir, f"{input_path.stem}_uvr.wav")
        write_wav_atomic(output_path, load_mono(clean_file, _OUTPUT_SR), _OUTPUT_SR)
        return output_path

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

        The separator writes into a staging directory that is emptied before and
        after the job, so its stems are removed whether or not separate() names them
        and whether or not the conversion succeeds. The denoised stem is converted to
        a 16 kHz mono 16-bit WAV in output_dir.

        Args:
            input_path: Path to the input audio file.

        Returns:
            Path to the 16 kHz mono WAV in output_dir.

        Raises:
            RuntimeError: If the staging path is a symlink, an output escapes its
                directory, no denoised stem is returned, or the stem is missing or the
                wrong length.
            Exception: Errors from separate(), from reading the stem, and from writing
                the WAV propagate.
        """
        logger.info("Denoising: %s", input_path)
        with self._lock:
            self._prepare_staging()
            try:
                output_path = self._separate_and_convert(input_path)
            finally:
                try:
                    self._empty_staging()
                except OSError:
                    # The job's own result stands, but what is left in staging keeps
                    # counting against the artifact budget until the next job empties it
                    logger.error(
                        "Could not empty UVR staging directory %s", self._staging_dir, exc_info=True
                    )
        logger.info("Denoised output: %s", output_path)
        return output_path
