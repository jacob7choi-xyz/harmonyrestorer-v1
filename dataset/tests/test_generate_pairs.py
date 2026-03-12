"""Tests for dataset.generate_pairs module."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dataset.generate_pairs import generate_pairs
from dataset.noise import DegradationParams

SR = 16_000
NUM_SAMPLES = 32_000  # 2 seconds


def _make_sine_wav(path: Path, sr: int = SR, num_samples: int = NUM_SAMPLES) -> None:
    """Write a mono float32 sine-wave WAV file.

    Args:
        path: Destination file path.
        sr: Sample rate in Hz.
        num_samples: Number of samples to generate.
    """
    t = np.arange(num_samples, dtype=np.float32) / sr
    audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    sf.write(path, audio, sr, subtype="FLOAT")


@pytest.fixture()
def clean_dir(tmp_path: Path) -> Path:
    """Provide a temp directory with three 16 kHz sine-wave WAV files."""
    d = tmp_path / "clean_frames"
    d.mkdir()
    for name in ("frame_001.wav", "frame_002.wav", "frame_003.wav"):
        _make_sine_wav(d / name)
    return d


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Provide a temp output directory for paired data."""
    return tmp_path / "pairs"


class TestGeneratePairs:
    """Tests for the generate_pairs function."""

    def test_creates_directory_structure(self, clean_dir: Path, output_dir: Path) -> None:
        """Verify that clean/, noisy/, and metadata/ subdirectories are created."""
        generate_pairs(clean_dir, output_dir, variants_per_frame=1, base_seed=42)

        assert (output_dir / "clean").is_dir()
        assert (output_dir / "noisy").is_dir()
        assert (output_dir / "metadata").is_dir()

    def test_correct_number_of_variants(self, clean_dir: Path, output_dir: Path) -> None:
        """Verify the number of noisy files equals frames * variants_per_frame."""
        variants = 3
        generate_pairs(clean_dir, output_dir, variants_per_frame=variants, base_seed=0)

        noisy_files = list((output_dir / "noisy").glob("*.wav"))
        assert len(noisy_files) == 3 * variants

    def test_clean_files_copied_correctly(self, clean_dir: Path, output_dir: Path) -> None:
        """Verify that clean output files match the original source content."""
        generate_pairs(clean_dir, output_dir, variants_per_frame=1, base_seed=0)

        for src in sorted(clean_dir.glob("*.wav")):
            dest = output_dir / "clean" / src.name
            assert dest.exists(), f"Clean copy missing: {src.name}"

            src_audio, _ = sf.read(src, dtype="float32")
            dest_audio, _ = sf.read(dest, dtype="float32")
            np.testing.assert_array_equal(src_audio, dest_audio)

    def test_noisy_naming_pattern(self, clean_dir: Path, output_dir: Path) -> None:
        """Verify noisy files follow the {stem}__v{nn}.wav naming convention."""
        variants = 2
        generate_pairs(clean_dir, output_dir, variants_per_frame=variants, base_seed=0)

        noisy_names = sorted(f.name for f in (output_dir / "noisy").glob("*.wav"))
        expected = sorted(
            f"{stem}__v{v:02d}.wav"
            for stem in ("frame_001", "frame_002", "frame_003")
            for v in range(variants)
        )
        assert noisy_names == expected

    def test_metadata_contains_expected_keys(self, clean_dir: Path, output_dir: Path) -> None:
        """Verify each metadata JSON has the required top-level keys."""
        generate_pairs(clean_dir, output_dir, variants_per_frame=1, base_seed=0)

        for meta_path in (output_dir / "metadata").glob("*.json"):
            with open(meta_path) as f:
                meta = json.load(f)

            assert "clean_frame" in meta
            assert "variant" in meta
            assert "seed" in meta
            assert "params" in meta

    def test_metadata_params_match_degradation_fields(
        self, clean_dir: Path, output_dir: Path
    ) -> None:
        """Verify metadata params dict contains all DegradationParams fields."""
        generate_pairs(clean_dir, output_dir, variants_per_frame=1, base_seed=0)

        expected_fields = {f.name for f in fields(DegradationParams)}

        meta_path = next((output_dir / "metadata").glob("*.json"))
        with open(meta_path) as f:
            meta = json.load(f)

        assert set(meta["params"].keys()) == expected_fields

    def test_different_seeds_produce_different_noise(self, clean_dir: Path, tmp_path: Path) -> None:
        """Verify that different base_seeds lead to different noisy output."""
        out_a = tmp_path / "pairs_a"
        out_b = tmp_path / "pairs_b"

        generate_pairs(clean_dir, out_a, variants_per_frame=1, base_seed=0)
        generate_pairs(clean_dir, out_b, variants_per_frame=1, base_seed=999)

        noisy_a = sorted((out_a / "noisy").glob("*.wav"))
        noisy_b = sorted((out_b / "noisy").glob("*.wav"))

        any_different = False
        for fa, fb in zip(noisy_a, noisy_b, strict=True):
            audio_a, _ = sf.read(fa, dtype="float32")
            audio_b, _ = sf.read(fb, dtype="float32")
            if not np.array_equal(audio_a, audio_b):
                any_different = True
                break

        assert any_different, "Different seeds should produce different noisy audio"

    def test_same_seed_produces_identical_output(self, clean_dir: Path, tmp_path: Path) -> None:
        """Verify that the same base_seed produces bit-identical noisy output."""
        out_a = tmp_path / "pairs_a"
        out_b = tmp_path / "pairs_b"

        generate_pairs(clean_dir, out_a, variants_per_frame=2, base_seed=42)
        generate_pairs(clean_dir, out_b, variants_per_frame=2, base_seed=42)

        noisy_a = sorted((out_a / "noisy").glob("*.wav"))
        noisy_b = sorted((out_b / "noisy").glob("*.wav"))

        assert len(noisy_a) == len(noisy_b)
        for fa, fb in zip(noisy_a, noisy_b, strict=True):
            audio_a, _ = sf.read(fa, dtype="float32")
            audio_b, _ = sf.read(fb, dtype="float32")
            np.testing.assert_array_equal(audio_a, audio_b)

    def test_skips_non_16khz_files_with_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify that files with wrong sample rate are skipped and warned."""
        clean = tmp_path / "clean_frames"
        clean.mkdir()

        # Write a 44100 Hz WAV file
        t = np.arange(44100, dtype=np.float32) / 44100
        audio = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(clean / "wrong_sr.wav", audio, 44100, subtype="FLOAT")

        output = tmp_path / "pairs"
        with caplog.at_level("WARNING", logger="dataset.generate_pairs"):
            count = generate_pairs(clean, output, variants_per_frame=1, base_seed=0)

        assert count == 0
        assert any("expected 16000 Hz" in msg for msg in caplog.messages)

    def test_skips_unreadable_files(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Verify that files failing to load are skipped with a warning."""
        clean = tmp_path / "clean_frames"
        clean.mkdir()

        # Write invalid data with a .wav extension
        (clean / "corrupt.wav").write_bytes(b"not a wav file at all")

        output = tmp_path / "pairs"
        with caplog.at_level("WARNING", logger="dataset.generate_pairs"):
            count = generate_pairs(clean, output, variants_per_frame=1, base_seed=0)

        assert count == 0
        assert any("Failed to read" in msg for msg in caplog.messages)

    def test_returns_correct_pair_count(self, clean_dir: Path, output_dir: Path) -> None:
        """Verify the return value matches frames * variants_per_frame."""
        variants = 4
        count = generate_pairs(clean_dir, output_dir, variants_per_frame=variants, base_seed=0)
        assert count == 3 * variants

    def test_empty_input_directory_returns_zero(self, tmp_path: Path) -> None:
        """Verify that an empty clean directory returns 0 pairs."""
        empty_clean = tmp_path / "empty"
        empty_clean.mkdir()
        output = tmp_path / "pairs"

        count = generate_pairs(empty_clean, output, variants_per_frame=5, base_seed=0)
        assert count == 0

    def test_skip_existing_does_not_duplicate(self, clean_dir: Path, output_dir: Path) -> None:
        """Verify re-running does not overwrite existing clean files.

        The function checks clean_dest.exists() before copying. On a second
        run, the clean file should not be rewritten, confirming idempotent
        behavior for the clean copy step.
        """
        generate_pairs(clean_dir, output_dir, variants_per_frame=2, base_seed=42)

        # Record modification times of clean output files
        clean_out = output_dir / "clean"
        mtimes_before = {f.name: f.stat().st_mtime for f in sorted(clean_out.glob("*.wav"))}

        # Re-run with the same seed
        generate_pairs(clean_dir, output_dir, variants_per_frame=2, base_seed=42)

        mtimes_after = {f.name: f.stat().st_mtime for f in sorted(clean_out.glob("*.wav"))}

        # Clean files should not have been rewritten
        for name in mtimes_before:
            assert (
                mtimes_before[name] == mtimes_after[name]
            ), f"Clean file {name} was rewritten on second run"
