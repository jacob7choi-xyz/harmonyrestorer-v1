"""Batch inference script for UVR (Ultimate Vocal Remover) denoising.

Runs the UVR-DeNoise model on a directory of noisy WAV files and saves
denoised outputs with matching filenames for evaluation with evaluate.py.

Usage:
    python -m dataset.infer_uvr --input-dir data/pairs/noisy/ --output-dir results/uvr/
    python -m dataset.infer_uvr --input-dir data/pairs/noisy/ --output-dir results/uvr/ --model-name UVR-DeNoise.pth
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import librosa
import soundfile as sf
from audio_separator.separator import Separator

logger = logging.getLogger(__name__)

_TARGET_SR = 16_000

# UVR marks denoised output with this substring in the filename
_DENOISED_MARKER = "No Noise"


def _resample_if_needed(input_path: Path, temp_dir: Path) -> Path:
    """Resample input to 16 kHz mono if needed, returning the path to use.

    If the file is already 16 kHz mono, returns the original path unchanged.
    Otherwise writes a resampled copy to temp_dir and returns that path.

    Args:
        input_path: Original input audio file.
        temp_dir: Directory for writing resampled scratch files.

    Returns:
        Path to use as input for the separator.
    """
    info = sf.info(str(input_path))
    if info.samplerate == _TARGET_SR and info.channels == 1:
        return input_path

    logger.debug(
        "Converting %s (%d Hz, %d ch) to %d Hz mono",
        input_path.name,
        info.samplerate,
        info.channels,
        _TARGET_SR,
    )
    audio, _ = librosa.load(input_path, sr=_TARGET_SR, mono=True)
    resampled_path = temp_dir / f"_resampled_{input_path.name}"
    sf.write(str(resampled_path), audio, _TARGET_SR, subtype="FLOAT")
    return resampled_path


def _find_denoised_output(temp_dir: Path) -> Path | None:
    """Find the denoised output file among audio-separator's outputs.

    audio-separator writes files like ``{stem}_(No Noise).wav`` into the
    output directory. This function locates that file.

    Args:
        temp_dir: Directory where audio-separator wrote its outputs.

    Returns:
        Path to the denoised file, or None if not found.
    """
    for candidate in temp_dir.iterdir():
        if _DENOISED_MARKER in candidate.name:
            return candidate
    return None


def restore_file(
    separator: Separator,
    input_path: Path,
    output_path: Path,
    temp_dir: Path,
) -> None:
    """Process one file through UVR and save with the original filename.

    Uses a temp directory for separator scratch space, then moves the
    denoised file to the final output path.

    Args:
        separator: Pre-loaded audio-separator instance.
        input_path: Path to noisy input WAV.
        output_path: Path to save denoised WAV.
        temp_dir: Scratch directory for separator outputs.

    Raises:
        RuntimeError: If UVR fails to produce a denoised output.
    """
    # Clear temp dir from any previous run
    for f in temp_dir.iterdir():
        if f.is_dir():
            shutil.rmtree(f)
        else:
            f.unlink()

    # Ensure input is 16 kHz mono before passing to UVR
    actual_input = _resample_if_needed(input_path, temp_dir)

    start = time.monotonic()
    separator.separate(str(actual_input))
    elapsed = time.monotonic() - start

    denoised = _find_denoised_output(temp_dir)
    if denoised is None:
        existing = [f.name for f in temp_dir.iterdir()]
        raise RuntimeError(
            f"No denoised output for {input_path.name}. Files in temp dir: {existing}"
        )

    # Resample UVR output back to 16 kHz mono to match clean references
    audio, _ = librosa.load(str(denoised), sr=_TARGET_SR, mono=True)

    # Atomic write: temp file then rename to avoid corrupt files on crash
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=output_path.parent)
    os.close(tmp_fd)
    try:
        sf.write(tmp_path, audio, _TARGET_SR, subtype="FLOAT")
        Path(tmp_path).rename(output_path)
    except BaseException:
        Path(tmp_path).unlink(missing_ok=True)
        raise

    logger.info("Restored %s in %.2fs", input_path.name, elapsed)


def main() -> None:
    """CLI entry point for UVR batch inference."""
    parser = argparse.ArgumentParser(description="Denoise audio files using UVR-DeNoise")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of noisy WAV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save denoised WAV files",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="UVR-DeNoise.pth",
        help="UVR model filename (default: UVR-DeNoise.pth)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if not args.input_dir.is_dir():
        logger.error("Input directory does not exist: %s", args.input_dir)
        sys.exit(1)

    wav_files = sorted(args.input_dir.glob("*.wav"))
    if not wav_files:
        logger.error("No WAV files found in %s", args.input_dir)
        sys.exit(1)

    # Create a temp directory for separator scratch space
    with tempfile.TemporaryDirectory(prefix="uvr_") as tmp:
        temp_dir = Path(tmp)

        logger.info("Loading UVR model: %s", args.model_name)
        separator = Separator(output_dir=str(temp_dir), output_format="WAV")
        separator.load_model(args.model_name)
        logger.info("UVR model loaded")

        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Skip files that already exist (allows resuming interrupted runs)
        existing_names = {p.name for p in args.output_dir.glob("*.wav")}
        remaining = [f for f in wav_files if f.name not in existing_names]
        skipped = len(wav_files) - len(remaining)
        if skipped:
            logger.info("Skipping %d already-processed files", skipped)

        logger.info("Processing %d files from %s", len(remaining), args.input_dir)
        failed = 0
        total_start = time.monotonic()

        for i, wav_file in enumerate(remaining):
            output_path = args.output_dir / wav_file.name
            try:
                restore_file(separator, wav_file, output_path, temp_dir)
            except Exception as e:
                logger.error("Failed to restore %s: %s", wav_file.name, e, exc_info=True)
                failed += 1

            if (i + 1) % 1000 == 0:
                logger.info("Progress: %d/%d files", i + 1, len(remaining))

        total_elapsed = time.monotonic() - total_start
        succeeded = len(remaining) - failed
        logger.info(
            "Done. Restored %d/%d files to %s in %.1fs (skipped %d existing)",
            succeeded,
            len(remaining),
            args.output_dir,
            total_elapsed,
            skipped,
        )


if __name__ == "__main__":
    main()
