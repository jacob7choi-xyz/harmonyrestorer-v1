"""Inference script for OpGAN audio restoration.

Loads a trained checkpoint and restores noisy audio files. Supports both
single-file and batch directory modes. Long files are automatically chunked
into 2-second frames with overlap-add for smooth transitions.

Usage:
    # Single file
    python -m dataset.infer --checkpoint checkpoints/final.pt --input noisy.wav --output restored.wav

    # Batch mode (directory of noisy WAVs)
    python -m dataset.infer --checkpoint checkpoints/final.pt --input-dir data/test/noisy/ --output-dir results/opgan/
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

# Import OpGAN components from archive
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app._archive.op_gan import OpGANGenerator  # noqa: E402

logger = logging.getLogger(__name__)

_EXPECTED_SR = 16_000
_FRAME_LEN = 32_000  # 2 seconds at 16kHz
_OVERLAP = 1600  # 100ms overlap for crossfade (5% of frame)


def _load_generator(checkpoint_path: Path, device: torch.device) -> OpGANGenerator:
    """Load a trained generator from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Torch device to load the model onto.

    Returns:
        Generator model in eval mode.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist.
        KeyError: If checkpoint is missing generator_state_dict.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generator = OpGANGenerator()
    state_dict = {
        k: v
        for k, v in checkpoint["generator_state_dict"].items()
        if not k.endswith(("_operator_weights_cache", "_cache_valid"))
    }
    generator.load_state_dict(state_dict, strict=False)
    generator.to(device)
    generator.eval()

    logger.info("Loaded generator from %s (epoch %d)", checkpoint_path, checkpoint.get("epoch", -1))
    return generator


def _chunk_audio(audio: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Split audio into overlapping frames for processing.

    Args:
        audio: 1D float32 array of audio samples.

    Returns:
        List of (start_index, frame) tuples.
    """
    total = len(audio)
    if total == 0:
        raise ValueError("Audio is empty (zero samples)")

    if total <= _FRAME_LEN:
        padded = np.zeros(_FRAME_LEN, dtype=np.float32)
        padded[:total] = audio
        return [(0, padded)]

    step = _FRAME_LEN - _OVERLAP
    chunks = []

    for start in range(0, total, step):
        end = start + _FRAME_LEN
        if end <= total:
            chunks.append((start, audio[start:end]))
        else:
            remaining = total - start
            if remaining <= _OVERLAP and chunks:
                # Tail is too short to justify a new chunk -- skip it,
                # the previous chunk's overlap already covers this region
                break
            padded = np.zeros(_FRAME_LEN, dtype=np.float32)
            padded[:remaining] = audio[start:]
            chunks.append((start, padded))
            break

    return chunks


def _overlap_add(chunks: list[tuple[int, np.ndarray]], original_length: int) -> np.ndarray:
    """Reassemble processed chunks using overlap-add with linear crossfade.

    Only interior overlap regions get crossfaded. The first chunk keeps
    its original start, and the last chunk keeps its original end.

    Args:
        chunks: List of (start_index, processed_frame) tuples.
        original_length: Length of the original audio to trim to.

    Returns:
        Reconstructed audio as 1D float32 array.
    """
    if len(chunks) == 1:
        start, frame = chunks[0]
        return frame[:original_length].copy()

    output = np.zeros(original_length, dtype=np.float32)
    weights = np.zeros(original_length, dtype=np.float32)
    last_idx = len(chunks) - 1

    for i, (start, frame) in enumerate(chunks):
        end = min(start + len(frame), original_length)
        length = end - start

        window = np.ones(length, dtype=np.float32)
        if _OVERLAP > 0:
            fade_len = min(_OVERLAP, length)
            # First chunk: no fade-in, only fade-out
            # Last chunk: only fade-in, no fade-out
            # Middle chunks: both fade-in and fade-out
            if i > 0:
                window[:fade_len] *= np.linspace(0, 1, fade_len, dtype=np.float32)
            if i < last_idx:
                window[-fade_len:] *= np.linspace(1, 0, fade_len, dtype=np.float32)

        output[start:end] += frame[:length] * window
        weights[start:end] += window

    # Avoid division by zero
    nonzero = weights > 0
    output[nonzero] /= weights[nonzero]

    return output


@torch.no_grad()
def restore_audio(
    generator: OpGANGenerator,
    audio: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Restore a single audio signal using the trained generator.

    Handles arbitrary-length audio by chunking into 2-second frames
    and reassembling with overlap-add.

    Args:
        generator: Trained OpGAN generator in eval mode.
        audio: 1D float32 array at 16kHz.
        device: Torch device.

    Returns:
        Restored audio as 1D float32 array, same length as input.
    """
    original_length = len(audio)
    chunks = _chunk_audio(audio)

    processed = []
    for start, frame in chunks:
        tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).to(device)
        restored = generator(tensor)
        processed.append((start, restored.squeeze(0).squeeze(0).cpu().numpy()))

    return _overlap_add(processed, original_length)


def restore_file(
    generator: OpGANGenerator,
    input_path: Path,
    output_path: Path,
    device: torch.device,
) -> None:
    """Load, restore, and save a single audio file.

    Args:
        generator: Trained OpGAN generator in eval mode.
        input_path: Path to noisy input WAV.
        output_path: Path to save restored WAV.
        device: Torch device.

    Raises:
        ValueError: If sample rate doesn't match expected 16kHz.
    """
    audio, sr = sf.read(input_path, dtype="float32")

    # Convert stereo to mono before resampling (half the work)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if sr != _EXPECTED_SR:
        logger.info("Resampling from %dHz to %dHz", sr, _EXPECTED_SR)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=_EXPECTED_SR)

    start = time.monotonic()
    restored = restore_audio(generator, audio, device)
    elapsed = time.monotonic() - start

    # Write to temp file then atomic rename to avoid corrupt files on crash
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=output_path.parent)
    os.close(tmp_fd)
    try:
        sf.write(tmp_path, restored, _EXPECTED_SR)
        Path(tmp_path).rename(output_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    duration = len(audio) / _EXPECTED_SR
    logger.info(
        "Restored %s (%.1fs audio in %.2fs, %.1fx realtime)",
        input_path.name,
        duration,
        elapsed,
        duration / max(elapsed, 0.001),
    )


def _select_device() -> torch.device:
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


def main() -> None:
    """CLI entry point for OpGAN inference."""
    parser = argparse.ArgumentParser(description="Restore noisy audio using trained OpGAN")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to trained checkpoint (.pt file)",
    )

    # Single file mode
    parser.add_argument("--input", type=Path, help="Path to single noisy WAV file")
    parser.add_argument("--output", type=Path, help="Path to save restored WAV file")

    # Batch mode
    parser.add_argument("--input-dir", type=Path, help="Directory of noisy WAV files")
    parser.add_argument("--output-dir", type=Path, help="Directory to save restored WAV files")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Validate args
    single_mode = args.input is not None
    batch_mode = args.input_dir is not None

    if not single_mode and not batch_mode:
        parser.error("Provide either --input (single file) or --input-dir (batch mode)")
    if single_mode and batch_mode:
        parser.error("Cannot use both --input and --input-dir")
    if single_mode and args.output is None:
        parser.error("--output is required with --input")
    if batch_mode and args.output_dir is None:
        parser.error("--output-dir is required with --input-dir")

    device = _select_device()
    generator = _load_generator(args.checkpoint, device)

    if single_mode:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        restore_file(generator, args.input, args.output, device)
    else:
        wav_files = sorted(args.input_dir.glob("*.wav"))
        if not wav_files:
            logger.error("No WAV files found in %s", args.input_dir)
            sys.exit(1)

        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Skip files that already exist (allows resuming interrupted runs)
        existing_names = {p.name for p in args.output_dir.glob("*.wav")}
        remaining = [f for f in wav_files if f.name not in existing_names]
        skipped = len(wav_files) - len(remaining)
        if skipped:
            logger.info("Skipping %d already-processed files", skipped)

        logger.info("Processing %d files from %s", len(remaining), args.input_dir)
        failed = 0
        for i, wav_file in enumerate(remaining):
            output_path = args.output_dir / wav_file.name
            try:
                restore_file(generator, wav_file, output_path, device)
            except Exception as e:
                logger.error("Failed to restore %s: %s", wav_file.name, e)
                failed += 1

            if (i + 1) % 1000 == 0:
                logger.info("Progress: %d/%d files", i + 1, len(remaining))

        succeeded = len(remaining) - failed
        logger.info(
            "Done. Restored %d/%d files to %s (skipped %d existing)",
            succeeded,
            len(remaining),
            args.output_dir,
            skipped,
        )


if __name__ == "__main__":
    main()
