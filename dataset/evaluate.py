"""Evaluation script for comparing audio restoration quality.

Computes SDR, PESQ, and STOI between restored audio and clean references.
Outputs per-file results and aggregate statistics.

Usage:
    python -m dataset.evaluate --restored results/opgan/ --clean data/pairs/clean/ --output metrics.json

    # Compare two models
    python -m dataset.evaluate --restored results/opgan/ --clean data/pairs/clean/ --output opgan_metrics.json
    python -m dataset.evaluate --restored results/uvr/   --clean data/pairs/clean/ --output uvr_metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from mir_eval.separation import bss_eval_sources
from pesq import pesq
from pystoi import stoi

logger = logging.getLogger(__name__)

_EXPECTED_SR = 16_000


def compute_sdr(clean: np.ndarray, restored: np.ndarray) -> float:
    """Compute Signal-to-Distortion Ratio.

    Args:
        clean: Reference clean audio (1D float32).
        restored: Restored audio (1D float32).

    Returns:
        SDR in dB.
    """
    sdr, _, _, _ = bss_eval_sources(
        clean.reshape(1, -1),
        restored.reshape(1, -1),
    )
    return float(sdr[0])


def compute_pesq(clean: np.ndarray, restored: np.ndarray, sr: int) -> float:
    """Compute Perceptual Evaluation of Speech Quality.

    Args:
        clean: Reference clean audio (1D float32).
        restored: Restored audio (1D float32).
        sr: Sample rate (must be 8000 or 16000).

    Returns:
        PESQ score (-0.5 to 4.5, higher is better).
    """
    mode = "wb" if sr == _EXPECTED_SR else "nb"
    return float(pesq(sr, clean, restored, mode))


def compute_stoi(clean: np.ndarray, restored: np.ndarray, sr: int) -> float:
    """Compute Short-Time Objective Intelligibility.

    Args:
        clean: Reference clean audio (1D float32).
        restored: Restored audio (1D float32).
        sr: Sample rate.

    Returns:
        STOI score (0.0 to 1.0, higher is better).
    """
    return float(stoi(clean, restored, sr, extended=False))


def evaluate_pair(
    clean_path: Path,
    restored_path: Path,
) -> dict[str, float]:
    """Evaluate a single restored/clean pair on all metrics.

    Args:
        clean_path: Path to reference clean WAV.
        restored_path: Path to restored WAV.

    Returns:
        Dict with sdr, pesq, stoi scores.

    Raises:
        ValueError: If sample rates don't match or aren't 16kHz.
    """
    clean, sr_clean = sf.read(clean_path, dtype="float32")
    restored, sr_restored = sf.read(restored_path, dtype="float32")

    if sr_clean != _EXPECTED_SR:
        raise ValueError(f"Clean file not {_EXPECTED_SR}Hz: {clean_path} ({sr_clean}Hz)")
    if sr_restored != _EXPECTED_SR:
        raise ValueError(f"Restored file not {_EXPECTED_SR}Hz: {restored_path} ({sr_restored}Hz)")

    # Convert stereo to mono if needed
    if clean.ndim == 2:
        clean = clean.mean(axis=1)
    if restored.ndim == 2:
        restored = restored.mean(axis=1)

    # Trim to equal length, warn if mismatch is significant
    min_len = min(len(clean), len(restored))
    max_len = max(len(clean), len(restored))
    if max_len > 0 and (max_len - min_len) / max_len > 0.01:
        logger.warning(
            "Length mismatch: clean=%d, restored=%d (%.1f%% diff)",
            len(clean),
            len(restored),
            (max_len - min_len) / max_len * 100,
        )
    clean = clean[:min_len]
    restored = restored[:min_len]

    # Guard against silent/corrupt files that would cause division by zero
    if np.max(np.abs(clean)) < 1e-10:
        raise ValueError(f"Clean file is silent: {clean_path}")
    if np.max(np.abs(restored)) < 1e-10:
        raise ValueError(f"Restored file is silent: {restored_path}")

    return {
        "sdr": compute_sdr(clean, restored),
        "pesq": compute_pesq(clean, restored, _EXPECTED_SR),
        "stoi": compute_stoi(clean, restored, _EXPECTED_SR),
    }


def evaluate_directory(
    restored_dir: Path,
    clean_dir: Path,
) -> dict[str, list[dict] | dict[str, float]]:
    """Evaluate all restored files against clean references.

    Matches files by stem name. For noisy variants (stem__v00.wav),
    matches against the base clean file (stem.wav).

    Args:
        restored_dir: Directory of restored WAV files.
        clean_dir: Directory of clean reference WAV files.

    Returns:
        Dict with 'per_file' results and 'summary' statistics.
    """
    if not restored_dir.is_dir():
        logger.error("Restored directory does not exist: %s", restored_dir)
        sys.exit(1)
    if not clean_dir.is_dir():
        logger.error("Clean directory does not exist: %s", clean_dir)
        sys.exit(1)

    restored_files = sorted(restored_dir.glob("*.wav"))
    if not restored_files:
        logger.error("No WAV files in %s", restored_dir)
        sys.exit(1)

    clean_stems = {p.stem: p for p in clean_dir.glob("*.wav")}

    results: list[dict] = []
    skipped = 0

    for restored_path in restored_files:
        # Match variant to clean: "frame__v00" -> "frame"
        stem = restored_path.stem
        clean_stem = stem.split("__v")[0] if "__v" in stem else stem

        clean_path = clean_stems.get(clean_stem)
        if clean_path is None:
            logger.warning("No clean reference for %s (looked for %s)", stem, clean_stem)
            skipped += 1
            continue

        try:
            scores = evaluate_pair(clean_path, restored_path)
            scores["file"] = restored_path.name
            results.append(scores)
        except Exception as e:
            logger.error("Failed to evaluate %s: %s", restored_path.name, e)
            skipped += 1

    if not results:
        logger.error("No files successfully evaluated")
        sys.exit(1)

    # Aggregate statistics
    sdrs = [r["sdr"] for r in results]
    pesqs = [r["pesq"] for r in results]
    stois = [r["stoi"] for r in results]

    summary = {
        "count": len(results),
        "skipped": skipped,
        "sdr_mean": float(np.mean(sdrs)),
        "sdr_std": float(np.std(sdrs)),
        "sdr_median": float(np.median(sdrs)),
        "pesq_mean": float(np.mean(pesqs)),
        "pesq_std": float(np.std(pesqs)),
        "pesq_median": float(np.median(pesqs)),
        "stoi_mean": float(np.mean(stois)),
        "stoi_std": float(np.std(stois)),
        "stoi_median": float(np.median(stois)),
    }

    return {"per_file": results, "summary": summary}


def _print_summary(summary: dict[str, float]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 50)
    print(f"{'Metric':<10} {'Mean':>10} {'Std':>10} {'Median':>10}")
    print("-" * 50)
    print(
        f"{'SDR (dB)':<10} {summary['sdr_mean']:>10.2f} {summary['sdr_std']:>10.2f} {summary['sdr_median']:>10.2f}"
    )
    print(
        f"{'PESQ':<10} {summary['pesq_mean']:>10.2f} {summary['pesq_std']:>10.2f} {summary['pesq_median']:>10.2f}"
    )
    print(
        f"{'STOI':<10} {summary['stoi_mean']:>10.3f} {summary['stoi_std']:>10.3f} {summary['stoi_median']:>10.3f}"
    )
    print("=" * 50)
    print(f"Evaluated {summary['count']} files, skipped {summary['skipped']}")


def main() -> None:
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate restored audio quality")
    parser.add_argument(
        "--restored",
        type=Path,
        required=True,
        help="Directory of restored WAV files",
    )
    parser.add_argument(
        "--clean",
        type=Path,
        required=True,
        help="Directory of clean reference WAV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save metrics JSON (optional)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Evaluating %s against %s", args.restored, args.clean)

    results = evaluate_directory(args.restored, args.clean)
    _print_summary(results["summary"])

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved metrics to %s", args.output)


if __name__ == "__main__":
    main()
