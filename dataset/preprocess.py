"""Preprocess raw audio into fixed-length frames for training.

Takes raw audio files in any common format, resamples to the model's
target sample rate (16 kHz), converts to mono, normalizes, and slices
into fixed-length frames (2 seconds = 32,000 samples).

Usage:
    python -m dataset.preprocess data/raw/ --output data/clean_frames/
    python -m dataset.preprocess data/raw/ --output data/clean_frames/ --sr 16000 --frame-len 32000
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}

# Minimum RMS energy to keep a frame (skip silence)
_MIN_RMS = 0.01


def load_and_resample(
    path: Path,
    target_sr: int,
) -> tuple[np.ndarray, int] | None:
    """Load an audio file and convert to mono float32.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate.

    Returns:
        Tuple of (mono_signal, sample_rate) or None if loading fails.
    """
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path.name, e)
        return None

    # Convert to mono by averaging channels
    if audio.shape[1] > 1:
        audio = np.mean(audio, axis=1)
    else:
        audio = audio[:, 0]

    # Resample with librosa (polyphase Kaiser-windowed sinc filter)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr

    return audio, sr


def normalize(signal: np.ndarray, headroom: float = 0.95) -> np.ndarray:
    """Peak-normalize a signal to [-headroom, headroom].

    Args:
        signal: Audio signal.
        headroom: Target peak level.

    Returns:
        Normalized signal.
    """
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal * (headroom / peak)
    return signal


def slice_frames(
    signal: np.ndarray,
    frame_len: int,
    hop_len: int | None = None,
    min_rms: float = _MIN_RMS,
) -> list[np.ndarray]:
    """Slice a signal into fixed-length frames, discarding silence.

    Args:
        signal: Audio signal.
        frame_len: Frame length in samples.
        hop_len: Hop between frames (default: frame_len, no overlap).
        min_rms: Minimum RMS energy to keep a frame.

    Returns:
        List of frame arrays, each of length frame_len.
    """
    if hop_len is None:
        hop_len = frame_len

    frames = []
    for start in range(0, len(signal) - frame_len + 1, hop_len):
        frame = signal[start : start + frame_len]

        # Skip near-silent frames
        rms = np.sqrt(np.mean(frame**2))
        if rms < min_rms:
            continue

        frames.append(frame)

    return frames


def preprocess_directory(
    input_dir: Path,
    output_dir: Path,
    target_sr: int = 16_000,
    frame_len: int = 32_000,
    hop_len: int | None = None,
) -> int:
    """Preprocess all audio files in a directory into training frames.

    Args:
        input_dir: Directory containing raw audio files.
        output_dir: Directory to save frame .wav files.
        target_sr: Target sample rate.
        frame_len: Frame length in samples.
        hop_len: Hop between frames (default: frame_len).

    Returns:
        Total number of frames generated.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [
        f for f in sorted(input_dir.rglob("*")) if f.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]

    if not audio_files:
        logger.warning("No audio files found in %s", input_dir)
        return 0

    logger.info("Found %d audio files in %s", len(audio_files), input_dir)

    total_frames = 0

    for file_path in audio_files:
        result = load_and_resample(file_path, target_sr)
        if result is None:
            continue

        audio, sr = result
        audio = normalize(audio)
        frames = slice_frames(audio, frame_len, hop_len)

        if not frames:
            duration = len(audio) / sr
            if len(audio) < frame_len:
                reason = f"too short for frame_len={frame_len} ({duration:.1f}s audio)"
            else:
                reason = f"all frames below silence threshold ({duration:.1f}s audio)"
            logger.info("Skipping %s: %s", file_path.name, reason)
            continue

        # Save each frame as a separate .wav file
        stem = file_path.stem
        for i, frame in enumerate(frames):
            frame_path = output_dir / f"{stem}__frame_{i:04d}.wav"
            sf.write(frame_path, frame, sr, subtype="FLOAT")

        total_frames += len(frames)
        logger.info(
            "Processed %s: %d frames (%.1fs audio)",
            file_path.name,
            len(frames),
            len(audio) / sr,
        )

    logger.info("Total: %d frames from %d files", total_frames, len(audio_files))
    return total_frames


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocess raw audio into fixed-length training frames."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing raw audio files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/clean_frames"),
        help="Output directory for frame files (default: data/clean_frames/)",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16_000,
        help="Target sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--frame-len",
        type=int,
        default=32_000,
        help="Frame length in samples (default: 32000 = 2s at 16kHz)",
    )
    parser.add_argument(
        "--hop-len",
        type=int,
        default=None,
        help="Hop length between frames (default: same as frame-len)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    preprocess_directory(
        args.input_dir,
        args.output,
        target_sr=args.sr,
        frame_len=args.frame_len,
        hop_len=args.hop_len,
    )


if __name__ == "__main__":
    main()
