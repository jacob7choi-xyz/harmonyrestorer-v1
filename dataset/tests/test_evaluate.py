"""Tests for dataset.evaluate module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dataset.evaluate import (
    compute_pesq,
    compute_sdr,
    compute_stoi,
    evaluate_directory,
    evaluate_pair,
)

_SR = 16_000
_DURATION = 1.0  # seconds
_NUM_SAMPLES = int(_SR * _DURATION)


def _sine(freq: float = 440.0, duration: float = _DURATION, sr: int = _SR) -> np.ndarray:
    """Generate a sine wave at the given frequency."""
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _write_wav(path: Path, data: np.ndarray, sr: int = _SR) -> None:
    """Write audio data to a WAV file."""
    sf.write(str(path), data, sr)


class TestComputeSdr:
    """Tests for compute_sdr."""

    def test_identical_signals_return_inf(self) -> None:
        """Identical clean and restored signals should give infinite SDR."""
        signal = _sine()
        result = compute_sdr(signal, signal)
        assert result == float("inf")

    def test_zero_noise_returns_inf(self) -> None:
        """When noise power is below the epsilon threshold, SDR should be inf."""
        signal = _sine()
        # Add negligible noise well below 1e-10 threshold
        restored = signal + np.float32(1e-12)
        result = compute_sdr(signal, restored)
        assert result == float("inf")

    def test_known_sdr_value(self) -> None:
        """Adding noise at known power should produce predictable SDR.

        If noise has the same power as the signal, SDR should be ~0 dB.
        We add noise scaled to produce a known SDR of ~20 dB.
        """
        signal = _sine()
        signal_power = np.sum(signal**2)
        # Target SDR = 20 dB => noise_power = signal_power / 10^(20/10) = signal_power / 100
        target_sdr = 20.0
        target_noise_power = signal_power / (10.0 ** (target_sdr / 10.0))
        # Create noise with exact target power
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(len(signal)).astype(np.float32)
        noise = noise * np.sqrt(target_noise_power / np.sum(noise**2))
        restored = signal + noise
        result = compute_sdr(signal, restored)
        assert abs(result - target_sdr) < 0.1

    def test_uncorrelated_signals_give_low_sdr(self) -> None:
        """Uncorrelated signals should give low or negative SDR."""
        signal = _sine(freq=440.0)
        other = _sine(freq=1000.0)
        result = compute_sdr(signal, other)
        assert result < 5.0


class TestComputePesq:
    """Tests for compute_pesq."""

    def test_returns_value_in_valid_range(self) -> None:
        """PESQ score should fall within the standard range."""
        signal = _sine()
        rng = np.random.default_rng(42)
        noisy = signal + 0.01 * rng.standard_normal(len(signal)).astype(np.float32)
        result = compute_pesq(signal, noisy, _SR)
        assert -0.5 <= result <= 4.5

    def test_identical_signals_give_high_score(self) -> None:
        """Identical signals should produce a high PESQ score."""
        signal = _sine()
        result = compute_pesq(signal, signal, _SR)
        assert result > 4.0


class TestComputeStoi:
    """Tests for compute_stoi."""

    def test_returns_value_in_valid_range(self) -> None:
        """STOI score should be between 0 and 1."""
        signal = _sine()
        rng = np.random.default_rng(42)
        noisy = signal + 0.05 * rng.standard_normal(len(signal)).astype(np.float32)
        result = compute_stoi(signal, noisy, _SR)
        assert 0.0 <= result <= 1.0

    def test_identical_signals_give_high_score(self) -> None:
        """Identical signals should produce a STOI score close to 1.0."""
        signal = _sine()
        result = compute_stoi(signal, signal, _SR)
        assert result > 0.95

    def test_heavily_degraded_signal_gives_lower_score(self) -> None:
        """A heavily degraded signal should score lower than a clean copy."""
        signal = _sine()
        rng = np.random.default_rng(42)
        degraded = signal + 1.0 * rng.standard_normal(len(signal)).astype(np.float32)
        clean_score = compute_stoi(signal, signal, _SR)
        degraded_score = compute_stoi(signal, degraded, _SR)
        assert degraded_score < clean_score


class TestEvaluatePair:
    """Tests for evaluate_pair."""

    def test_matching_wav_files(self, tmp_path: Path) -> None:
        """Matching clean and restored WAV files should return all three metrics."""
        signal = _sine()
        rng = np.random.default_rng(42)
        restored = signal + 0.01 * rng.standard_normal(len(signal)).astype(np.float32)

        clean_path = tmp_path / "clean.wav"
        restored_path = tmp_path / "restored.wav"
        _write_wav(clean_path, signal)
        _write_wav(restored_path, restored)

        result = evaluate_pair(clean_path, restored_path)
        assert "sdr" in result
        assert "pesq" in result
        assert "stoi" in result
        assert isinstance(result["sdr"], float)
        assert isinstance(result["pesq"], float)
        assert isinstance(result["stoi"], float)

    def test_raises_on_silent_clean_file(self, tmp_path: Path) -> None:
        """A silent clean file should raise ValueError."""
        silent = np.zeros(_NUM_SAMPLES, dtype=np.float32)
        signal = _sine()

        clean_path = tmp_path / "clean.wav"
        restored_path = tmp_path / "restored.wav"
        _write_wav(clean_path, silent)
        _write_wav(restored_path, signal)

        with pytest.raises(ValueError, match="silent"):
            evaluate_pair(clean_path, restored_path)

    def test_raises_on_silent_restored_file(self, tmp_path: Path) -> None:
        """A silent restored file should raise ValueError."""
        signal = _sine()
        silent = np.zeros(_NUM_SAMPLES, dtype=np.float32)

        clean_path = tmp_path / "clean.wav"
        restored_path = tmp_path / "restored.wav"
        _write_wav(clean_path, signal)
        _write_wav(restored_path, silent)

        with pytest.raises(ValueError, match="silent"):
            evaluate_pair(clean_path, restored_path)

    def test_raises_on_non_16khz_sample_rate(self, tmp_path: Path) -> None:
        """Files not at 16kHz should raise ValueError."""
        signal = _sine(sr=44100)

        clean_path = tmp_path / "clean.wav"
        restored_path = tmp_path / "restored.wav"
        _write_wav(clean_path, signal, sr=44100)
        _write_wav(restored_path, signal, sr=44100)

        with pytest.raises(ValueError, match="16000Hz"):
            evaluate_pair(clean_path, restored_path)

    def test_raises_when_only_restored_is_wrong_sr(self, tmp_path: Path) -> None:
        """Should raise even if only the restored file has wrong sample rate."""
        clean_signal = _sine()
        restored_signal = _sine(sr=44100)

        clean_path = tmp_path / "clean.wav"
        restored_path = tmp_path / "restored.wav"
        _write_wav(clean_path, clean_signal, sr=_SR)
        _write_wav(restored_path, restored_signal, sr=44100)

        with pytest.raises(ValueError, match="16000Hz"):
            evaluate_pair(clean_path, restored_path)

    def test_handles_stereo_files(self, tmp_path: Path) -> None:
        """Stereo files should be converted to mono and evaluated successfully."""
        mono = _sine()
        stereo = np.column_stack([mono, mono])

        clean_path = tmp_path / "clean.wav"
        restored_path = tmp_path / "restored.wav"
        _write_wav(clean_path, stereo)
        _write_wav(restored_path, stereo)

        result = evaluate_pair(clean_path, restored_path)
        assert "sdr" in result
        assert result["sdr"] == float("inf")  # identical signals

    def test_handles_length_mismatch(self, tmp_path: Path) -> None:
        """Files of different lengths should be trimmed to the shorter one."""
        short = _sine(duration=1.0)
        long = _sine(duration=1.5)

        clean_path = tmp_path / "clean.wav"
        restored_path = tmp_path / "restored.wav"
        _write_wav(clean_path, short)
        _write_wav(restored_path, long)

        result = evaluate_pair(clean_path, restored_path)
        assert "sdr" in result
        # Same signal content in the overlapping region, should be inf
        assert result["sdr"] == float("inf")


class TestEvaluateDirectory:
    """Tests for evaluate_directory."""

    def _populate_dirs(
        self,
        clean_dir: Path,
        restored_dir: Path,
        *,
        use_variants: bool = False,
    ) -> None:
        """Create matching clean and restored WAV files in the given directories.

        Args:
            clean_dir: Directory for clean reference files.
            restored_dir: Directory for restored files.
            use_variants: If True, name restored files with __v00 suffix.
        """
        rng = np.random.default_rng(42)
        for i in range(3):
            signal = _sine(freq=440.0 + i * 100)
            noise = 0.01 * rng.standard_normal(len(signal)).astype(np.float32)
            restored = signal + noise

            _write_wav(clean_dir / f"frame{i}.wav", signal)
            suffix = "__v00" if use_variants else ""
            _write_wav(restored_dir / f"frame{i}{suffix}.wav", restored)

    def test_matches_restored_to_clean_by_stem(self, tmp_path: Path) -> None:
        """Files with matching stems should be paired and evaluated."""
        clean_dir = tmp_path / "clean"
        restored_dir = tmp_path / "restored"
        clean_dir.mkdir()
        restored_dir.mkdir()
        self._populate_dirs(clean_dir, restored_dir)

        result = evaluate_directory(restored_dir, clean_dir)
        assert "per_file" in result
        assert "summary" in result
        assert len(result["per_file"]) == 3
        assert result["summary"]["count"] == 3

    def test_handles_variant_matching(self, tmp_path: Path) -> None:
        """Restored files with __v00 suffix should match to clean stems."""
        clean_dir = tmp_path / "clean"
        restored_dir = tmp_path / "restored"
        clean_dir.mkdir()
        restored_dir.mkdir()
        self._populate_dirs(clean_dir, restored_dir, use_variants=True)

        result = evaluate_directory(restored_dir, clean_dir)
        assert len(result["per_file"]) == 3

    def test_raises_on_missing_restored_dir(self, tmp_path: Path) -> None:
        """A nonexistent restored directory should raise FileNotFoundError."""
        clean_dir = tmp_path / "clean"
        clean_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="Restored directory"):
            evaluate_directory(tmp_path / "nonexistent", clean_dir)

    def test_raises_on_missing_clean_dir(self, tmp_path: Path) -> None:
        """A nonexistent clean directory should raise FileNotFoundError."""
        restored_dir = tmp_path / "restored"
        restored_dir.mkdir()
        _write_wav(restored_dir / "frame.wav", _sine())

        with pytest.raises(FileNotFoundError, match="Clean directory"):
            evaluate_directory(restored_dir, tmp_path / "nonexistent")

    def test_raises_on_empty_restored_dir(self, tmp_path: Path) -> None:
        """An empty restored directory should raise FileNotFoundError."""
        clean_dir = tmp_path / "clean"
        restored_dir = tmp_path / "restored"
        clean_dir.mkdir()
        restored_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No WAV files"):
            evaluate_directory(restored_dir, clean_dir)

    def test_raises_when_no_files_match(self, tmp_path: Path) -> None:
        """When no restored files match a clean reference, raise RuntimeError."""
        clean_dir = tmp_path / "clean"
        restored_dir = tmp_path / "restored"
        clean_dir.mkdir()
        restored_dir.mkdir()

        _write_wav(clean_dir / "alpha.wav", _sine())
        _write_wav(restored_dir / "beta.wav", _sine())

        with pytest.raises(RuntimeError, match="No files successfully evaluated"):
            evaluate_directory(restored_dir, clean_dir)

    def test_skips_files_without_clean_reference(self, tmp_path: Path) -> None:
        """Files without a matching clean reference are skipped and counted."""
        clean_dir = tmp_path / "clean"
        restored_dir = tmp_path / "restored"
        clean_dir.mkdir()
        restored_dir.mkdir()

        # Create one matching pair and one orphan
        signal = _sine()
        rng = np.random.default_rng(42)
        restored = signal + 0.01 * rng.standard_normal(len(signal)).astype(np.float32)

        _write_wav(clean_dir / "matched.wav", signal)
        _write_wav(restored_dir / "matched.wav", restored)
        _write_wav(restored_dir / "orphan.wav", _sine(freq=880.0))

        result = evaluate_directory(restored_dir, clean_dir)
        assert result["summary"]["count"] == 1
        assert result["summary"]["skipped"] == 1

    def test_skips_files_that_fail_evaluation(self, tmp_path: Path) -> None:
        """Files that raise during evaluate_pair are skipped, not fatal."""
        clean_dir = tmp_path / "clean"
        restored_dir = tmp_path / "restored"
        clean_dir.mkdir()
        restored_dir.mkdir()

        # One good pair
        signal = _sine()
        rng = np.random.default_rng(42)
        restored = signal + 0.01 * rng.standard_normal(len(signal)).astype(np.float32)
        _write_wav(clean_dir / "good.wav", signal)
        _write_wav(restored_dir / "good.wav", restored)

        # One pair where restored is silent (triggers ValueError)
        _write_wav(clean_dir / "bad.wav", signal)
        _write_wav(restored_dir / "bad.wav", np.zeros(_NUM_SAMPLES, dtype=np.float32))

        result = evaluate_directory(restored_dir, clean_dir)
        assert result["summary"]["count"] == 1
        assert result["summary"]["skipped"] == 1

    def test_skips_empty_audio_files(self, tmp_path: Path) -> None:
        """Empty audio files (0 samples) are skipped, not fatal."""
        clean_dir = tmp_path / "clean"
        restored_dir = tmp_path / "restored"
        clean_dir.mkdir()
        restored_dir.mkdir()

        # One good pair
        signal = _sine()
        rng = np.random.default_rng(42)
        restored = signal + 0.01 * rng.standard_normal(len(signal)).astype(np.float32)
        _write_wav(clean_dir / "good.wav", signal)
        _write_wav(restored_dir / "good.wav", restored)

        # One pair with empty restored file (0 samples)
        _write_wav(clean_dir / "empty.wav", signal)
        _write_wav(restored_dir / "empty.wav", np.array([], dtype=np.float32))

        result = evaluate_directory(restored_dir, clean_dir)
        assert result["summary"]["count"] == 1
        assert result["summary"]["skipped"] == 1

    def test_inf_sdr_in_summary_does_not_crash(self, tmp_path: Path) -> None:
        """Identical files produce inf SDR; summary aggregation should not crash."""
        clean_dir = tmp_path / "clean"
        restored_dir = tmp_path / "restored"
        clean_dir.mkdir()
        restored_dir.mkdir()

        # All identical pairs -> inf SDR
        for i in range(2):
            signal = _sine(freq=440.0 + i * 100)
            _write_wav(clean_dir / f"frame{i}.wav", signal)
            _write_wav(restored_dir / f"frame{i}.wav", signal)

        result = evaluate_directory(restored_dir, clean_dir)
        summary = result["summary"]
        assert summary["count"] == 2
        assert summary["sdr_mean"] == float("inf")

    def test_returns_correct_summary_statistics(self, tmp_path: Path) -> None:
        """Summary should contain mean, std, and median for all three metrics."""
        clean_dir = tmp_path / "clean"
        restored_dir = tmp_path / "restored"
        clean_dir.mkdir()
        restored_dir.mkdir()
        self._populate_dirs(clean_dir, restored_dir)

        result = evaluate_directory(restored_dir, clean_dir)
        summary = result["summary"]

        expected_keys = {
            "count",
            "skipped",
            "sdr_mean",
            "sdr_std",
            "sdr_median",
            "pesq_mean",
            "pesq_std",
            "pesq_median",
            "stoi_mean",
            "stoi_std",
            "stoi_median",
        }
        assert set(summary.keys()) == expected_keys
        assert summary["count"] == 3
        assert summary["skipped"] == 0
        # SDR should be high since we only added small noise
        assert summary["sdr_mean"] > 20.0
