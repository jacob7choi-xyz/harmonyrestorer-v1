"""Generate paired noisy/clean training data.

Takes clean audio frames and applies randomized analog degradation to
create paired training examples for the OpGAN denoiser.

Usage:
    python -m dataset.generate_pairs data/clean_frames/ --output data/pairs/
    python -m dataset.generate_pairs data/clean_frames/ --output data/pairs/ --variants 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from dataset.noise import DegradationParams, random_degradation

logger = logging.getLogger(__name__)


def generate_pairs(
    clean_dir: Path,
    output_dir: Path,
    variants_per_frame: int = 5,
    snr_range: tuple[float, float] = (5.0, 30.0),
    sr: int = 16_000,
) -> int:
    """Generate noisy/clean paired training data.

    For each clean frame, generates N noisy variants with randomized
    analog degradation. Saves pairs as:
        output_dir/
            clean/frame_name.wav
            noisy/frame_name__v0.wav
            noisy/frame_name__v1.wav
            ...
            metadata/frame_name__v0.json  (degradation params)

    Args:
        clean_dir: Directory containing clean .wav frames.
        output_dir: Output directory for paired data.
        variants_per_frame: Number of noisy variants per clean frame.
        snr_range: Min/max SNR in dB for tape hiss.
        sr: Expected sample rate of clean frames.

    Returns:
        Total number of pairs generated.
    """
    clean_out = output_dir / "clean"
    noisy_out = output_dir / "noisy"
    meta_out = output_dir / "metadata"

    clean_out.mkdir(parents=True, exist_ok=True)
    noisy_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    clean_files = sorted(clean_dir.glob("*.wav"))

    if not clean_files:
        logger.warning("No .wav files found in %s", clean_dir)
        return 0

    logger.info(
        "Generating %d variants for %d clean frames (= %d pairs)",
        variants_per_frame, len(clean_files), variants_per_frame * len(clean_files),
    )

    total_pairs = 0

    for clean_path in clean_files:
        try:
            clean_audio, file_sr = sf.read(clean_path, dtype="float32")
        except Exception as e:
            logger.warning("Failed to read %s: %s", clean_path.name, e)
            continue

        if file_sr != sr:
            logger.warning(
                "Skipping %s: expected %d Hz, got %d Hz", clean_path.name, sr, file_sr,
            )
            continue

        stem = clean_path.stem

        # Copy clean frame to output
        clean_dest = clean_out / clean_path.name
        if not clean_dest.exists():
            sf.write(clean_dest, clean_audio, sr, subtype="FLOAT")

        # Generate noisy variants
        for v in range(variants_per_frame):
            noisy_audio, params = random_degradation(
                clean_audio, sr, snr_range=snr_range,
            )

            variant_name = f"{stem}__v{v:02d}"
            noisy_path = noisy_out / f"{variant_name}.wav"
            meta_path = meta_out / f"{variant_name}.json"

            sf.write(noisy_path, noisy_audio, sr, subtype="FLOAT")

            # Save degradation params for reproducibility
            meta = {
                "clean_frame": clean_path.name,
                "variant": v,
                "params": _params_to_dict(params),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            total_pairs += 1

        if total_pairs % 100 == 0 and total_pairs > 0:
            logger.info("Generated %d pairs so far...", total_pairs)

    logger.info("Done: %d total pairs in %s", total_pairs, output_dir)
    return total_pairs


def _params_to_dict(params: DegradationParams) -> dict:
    """Convert DegradationParams to a JSON-serializable dict."""
    return {
        "hiss_snr_db": params.hiss_snr_db,
        "hiss_color": params.hiss_color,
        "crackle_density": params.crackle_density,
        "crackle_amplitude": params.crackle_amplitude,
        "hum_amplitude": params.hum_amplitude,
        "hum_freq": params.hum_freq,
        "rolloff_hz": params.rolloff_hz,
        "saturation_drive": params.saturation_drive,
        "enable_hiss": params.enable_hiss,
        "enable_crackle": params.enable_crackle,
        "enable_hum": params.enable_hum,
        "enable_rolloff": params.enable_rolloff,
        "enable_saturation": params.enable_saturation,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate paired noisy/clean training data."
    )
    parser.add_argument(
        "clean_dir", type=Path,
        help="Directory containing clean .wav frames",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/pairs"),
        help="Output directory for paired data (default: data/pairs/)",
    )
    parser.add_argument(
        "--variants", type=int, default=5,
        help="Number of noisy variants per clean frame (default: 5)",
    )
    parser.add_argument(
        "--snr-min", type=float, default=5.0,
        help="Minimum SNR in dB (default: 5.0 = heavy noise)",
    )
    parser.add_argument(
        "--snr-max", type=float, default=30.0,
        help="Maximum SNR in dB (default: 30.0 = mild noise)",
    )
    parser.add_argument(
        "--sr", type=int, default=16_000,
        help="Expected sample rate (default: 16000)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    generate_pairs(
        args.clean_dir,
        args.output,
        variants_per_frame=args.variants,
        snr_range=(args.snr_min, args.snr_max),
        sr=args.sr,
    )


if __name__ == "__main__":
    main()
