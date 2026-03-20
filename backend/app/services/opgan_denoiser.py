"""Audio denoising service using trained OpGAN model.

Loads a trained OpGAN checkpoint and restores noisy audio using
overlap-add chunking. Resamples input to 16kHz mono and output
back to the original sample rate.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from app.models.chunking import chunk_audio, overlap_add
from app.models.op_gan import OpGANGenerator

logger = logging.getLogger(__name__)

_TARGET_SR = 16_000
_BATCH_SIZE = 8  # frames per GPU batch


class OpGANDenoiserService:
    """Audio denoising using trained OpGAN model.

    Lazily loads the checkpoint on first call. Resamples input audio to
    16kHz mono for processing, then writes the restored output at 16kHz.
    """

    def __init__(self, output_dir: Path, checkpoint_path: Path) -> None:
        """Initialize the OpGAN denoiser service.

        Args:
            output_dir: Directory to write denoised output files.
            checkpoint_path: Path to the trained OpGAN checkpoint (.pt).
        """
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self._lock = threading.Lock()
        self._generator: OpGANGenerator | None = None
        self._device: torch.device | None = None

    def _select_device(self) -> torch.device:
        """Select the best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device

    def _get_generator(self) -> tuple[OpGANGenerator, torch.device]:
        """Lazy-load the OpGAN generator model.

        Returns:
            Tuple of (generator, device).

        Raises:
            FileNotFoundError: If checkpoint doesn't exist.
            KeyError: If checkpoint is missing generator_state_dict.
        """
        if self._generator is not None:
            return self._generator, self._device  # type: ignore[return-value]

        with self._lock:
            # Re-check after acquiring lock (double-checked locking)
            if self._generator is not None:
                return self._generator, self._device

            if not self.checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

            logger.info("Loading OpGAN model from %s...", self.checkpoint_path)
            device = self._select_device()
            checkpoint = torch.load(self.checkpoint_path, map_location=device, weights_only=True)

            generator = OpGANGenerator()
            state_dict = {
                k: v
                for k, v in checkpoint["generator_state_dict"].items()
                if not k.endswith(("_operator_weights_cache", "_cache_valid"))
            }
            generator.load_state_dict(state_dict, strict=False)
            generator.to(device)
            generator.eval()

            self._device = device
            self._generator = generator
            logger.info("OpGAN model loaded (epoch %d)", checkpoint.get("epoch", -1))

        return self._generator, self._device

    @torch.no_grad()
    def _restore_audio(self, audio: np.ndarray) -> np.ndarray:
        """Restore a single audio signal using the trained generator.

        Frames are batched for GPU throughput. Output is clamped to [-1, 1]
        to prevent clipping artifacts in the output WAV.

        Args:
            audio: 1D float32 array at 16kHz.

        Returns:
            Restored audio as 1D float32 array, same length as input.
        """
        generator, device = self._get_generator()
        original_length = len(audio)
        chunks = chunk_audio(audio)

        total_batches = (len(chunks) + _BATCH_SIZE - 1) // _BATCH_SIZE
        processed = []
        for batch_idx, batch_start in enumerate(range(0, len(chunks), _BATCH_SIZE)):
            batch = chunks[batch_start : batch_start + _BATCH_SIZE]
            logger.debug("Batch %d/%d (%d frames)", batch_idx + 1, total_batches, len(batch))
            frames = (
                torch.stack([torch.from_numpy(frame) for _, frame in batch]).unsqueeze(1).to(device)
            )  # [B, 1, frame_len]

            restored = generator(frames)  # [B, 1, frame_len]

            for i, (start, _) in enumerate(batch):
                out = restored[i].squeeze(0).cpu().numpy()
                processed.append((start, out))

        result = overlap_add(processed, original_length)
        np.clip(result, -1.0, 1.0, out=result)
        return result

    def denoise(self, input_path: Path) -> Path:
        """Run OpGAN denoising on *input_path* and return the cleaned output path.

        Loads the audio, resamples to 16kHz mono, runs inference, and
        writes the restored output to the output directory.

        Args:
            input_path: Path to the input audio file.

        Returns:
            Path to the denoised output file.

        Raises:
            FileNotFoundError: If input file or checkpoint doesn't exist.
            RuntimeError: If inference fails.
        """
        logger.info("Denoising: %s", input_path)

        audio, sr = sf.read(input_path, dtype="float32")

        # Convert stereo to mono
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != _TARGET_SR:
            logger.debug("Resampling from %dHz to %dHz", sr, _TARGET_SR)
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_TARGET_SR)

        start_time = time.perf_counter()
        restored = self._restore_audio(audio)
        elapsed = time.perf_counter() - start_time

        # Release GPU memory between jobs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Write to temp file then atomic rename
        output_name = f"{input_path.stem}_opgan.wav"
        output_path = (self.output_dir / output_name).resolve()
        if not output_path.is_relative_to(self.output_dir):
            raise RuntimeError(f"Output escapes output_dir: {output_path}")

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=self.output_dir)
        os.close(tmp_fd)
        try:
            sf.write(tmp_path, restored, _TARGET_SR)
            Path(tmp_path).replace(output_path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        duration = len(audio) / _TARGET_SR
        logger.info(
            "Denoised %s (%.1fs audio in %.2fs, %.1fx realtime)",
            input_path.name,
            duration,
            elapsed,
            duration / max(elapsed, 0.001),
        )
        return output_path
