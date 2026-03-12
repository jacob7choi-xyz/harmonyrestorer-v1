"""Tests for dataset.preprocess module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from dataset.preprocess import (
    load_and_resample,
    normalize,
    preprocess_directory,
    slice_frames,
)


def _sine_wave(freq: float = 440.0, sr: int = 16000, duration: float = 1.0) -> np.ndarray:
    """Generate a mono sine wave at the given frequency.

    Args:
        freq: Frequency in Hz.
        sr: Sample rate.
        duration: Duration in seconds.

    Returns:
        Float32 numpy array of the sine wave.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def _write_wav(path: Path, signal: np.ndarray, sr: int = 16000) -> Path:
    """Write a signal to a WAV file and return the path.

    Args:
        path: Output file path.
        signal: Audio signal array.
        sr: Sample rate.

    Returns:
        The path that was written.
    """
    sf.write(path, signal, sr, subtype="FLOAT")
    return path


class TestLoadAndResample:
    """Tests for load_and_resample."""

    def test_loads_wav_correctly(self, tmp_path: Path) -> None:
        """Test that a valid WAV file is loaded and returned."""
        signal = _sine_wave(sr=16000, duration=0.5)
        wav_path = _write_wav(tmp_path / "test.wav", signal, sr=16000)

        result = load_and_resample(wav_path, target_sr=16000)

        assert result is not None
        audio, sr = result
        assert sr == 16000
        assert len(audio) > 0

    def test_resamples_from_44100_to_16000(self, tmp_path: Path) -> None:
        """Test that audio at 44100 Hz is resampled to 16000 Hz."""
        duration = 1.0
        signal = _sine_wave(sr=44100, duration=duration)
        wav_path = _write_wav(tmp_path / "hi_sr.wav", signal, sr=44100)

        result = load_and_resample(wav_path, target_sr=16000)

        assert result is not None
        audio, sr = result
        assert sr == 16000
        expected_len = int(16000 * duration)
        assert abs(len(audio) - expected_len) <= 1

    def test_converts_stereo_to_mono(self, tmp_path: Path) -> None:
        """Test that stereo input is downmixed to a mono signal."""
        mono = _sine_wave(sr=16000, duration=0.5)
        stereo = np.column_stack([mono, mono * 0.5])
        wav_path = tmp_path / "stereo.wav"
        sf.write(wav_path, stereo, 16000, subtype="FLOAT")

        result = load_and_resample(wav_path, target_sr=16000)

        assert result is not None
        audio, sr = result
        assert audio.ndim == 1

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        """Test that a missing file path returns None."""
        result = load_and_resample(tmp_path / "does_not_exist.wav", target_sr=16000)
        assert result is None

    def test_returns_none_for_corrupt_file(self, tmp_path: Path) -> None:
        """Test that a corrupt file returns None."""
        bad_path = tmp_path / "corrupt.wav"
        bad_path.write_bytes(b"not a real audio file at all")

        result = load_and_resample(bad_path, target_sr=16000)
        assert result is None

    def test_output_is_float32(self, tmp_path: Path) -> None:
        """Test that the returned audio array has float32 dtype."""
        signal = _sine_wave(sr=16000, duration=0.5)
        wav_path = _write_wav(tmp_path / "f32.wav", signal, sr=16000)

        result = load_and_resample(wav_path, target_sr=16000)

        assert result is not None
        audio, _ = result
        assert audio.dtype == np.float32


class TestNormalize:
    """Tests for normalize."""

    def test_peak_normalizes_to_headroom(self) -> None:
        """Test that the signal peak equals the headroom value after normalization."""
        signal = np.array([0.0, 0.25, -0.5, 0.1], dtype=np.float32)
        result = normalize(signal, headroom=0.95)

        assert np.isclose(np.max(np.abs(result)), 0.95, atol=1e-6)

    def test_silent_signal_unchanged(self) -> None:
        """Test that an all-zeros signal is returned unchanged."""
        signal = np.zeros(100, dtype=np.float32)
        result = normalize(signal, headroom=0.95)

        np.testing.assert_array_equal(result, signal)

    def test_already_normalized_stays_same(self) -> None:
        """Test that a signal already at the target peak is unchanged."""
        signal = np.array([0.0, 0.95, -0.95, 0.0], dtype=np.float32)
        result = normalize(signal, headroom=0.95)

        np.testing.assert_allclose(result, signal, atol=1e-6)

    def test_custom_headroom_value(self) -> None:
        """Test that a custom headroom value is applied correctly."""
        signal = np.array([0.0, 1.0, -0.5], dtype=np.float32)
        result = normalize(signal, headroom=0.5)

        assert np.isclose(np.max(np.abs(result)), 0.5, atol=1e-6)


class TestSliceFrames:
    """Tests for slice_frames."""

    def test_correct_number_of_frames(self) -> None:
        """Test that a known-length signal produces the expected number of frames."""
        # 3 * 100 = 300 samples, frame_len=100, no overlap -> 3 frames
        signal = _sine_wave(freq=440.0, sr=16000, duration=300 / 16000)
        frames = slice_frames(signal, frame_len=100)

        assert len(frames) == 3

    def test_each_frame_has_correct_length(self) -> None:
        """Test that every returned frame has exactly frame_len samples."""
        signal = _sine_wave(sr=16000, duration=0.5)
        frame_len = 1000
        frames = slice_frames(signal, frame_len=frame_len)

        for frame in frames:
            assert len(frame) == frame_len

    def test_silent_frames_skipped(self) -> None:
        """Test that frames below the min_rms threshold are discarded."""
        frame_len = 100
        loud = _sine_wave(freq=440.0, sr=16000, duration=frame_len / 16000)
        silent = np.zeros(frame_len, dtype=np.float32)
        signal = np.concatenate([loud, silent, loud])

        frames = slice_frames(signal, frame_len=frame_len)

        assert len(frames) == 2

    def test_custom_hop_len_produces_overlapping_frames(self) -> None:
        """Test that a hop_len smaller than frame_len creates overlapping frames."""
        signal = _sine_wave(sr=16000, duration=0.5)
        frame_len = 4000
        hop_len = 2000
        frames_no_overlap = slice_frames(signal, frame_len=frame_len)
        frames_overlap = slice_frames(signal, frame_len=frame_len, hop_len=hop_len)

        assert len(frames_overlap) > len(frames_no_overlap)

    def test_signal_shorter_than_frame_len_produces_no_frames(self) -> None:
        """Test that a signal shorter than frame_len yields zero frames."""
        signal = _sine_wave(sr=16000, duration=0.01)  # 160 samples
        frames = slice_frames(signal, frame_len=32000)

        assert len(frames) == 0

    def test_exact_multiple_of_frame_len(self) -> None:
        """Test that a signal exactly N * frame_len produces N frames."""
        frame_len = 1000
        n_frames = 5
        signal = _sine_wave(sr=16000, duration=(frame_len * n_frames) / 16000)

        frames = slice_frames(signal, frame_len=frame_len)

        assert len(frames) == n_frames


class TestPreprocessDirectory:
    """Tests for preprocess_directory."""

    def test_end_to_end(self, tmp_path: Path) -> None:
        """Test full pipeline: create WAVs, preprocess, verify output frames."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "frames"
        input_dir.mkdir()

        sr = 16000
        frame_len = 1600  # 0.1s frames for speed
        # 2 frames worth of audio
        signal = _sine_wave(freq=440.0, sr=sr, duration=(frame_len * 2) / sr)
        _write_wav(input_dir / "track_a.wav", signal, sr=sr)
        _write_wav(input_dir / "track_b.wav", signal, sr=sr)

        total = preprocess_directory(input_dir, output_dir, target_sr=sr, frame_len=frame_len)

        assert total == 4  # 2 frames per file * 2 files
        output_files = sorted(output_dir.glob("*.wav"))
        assert len(output_files) == 4

    def test_skips_files_too_short(self, tmp_path: Path) -> None:
        """Test that files shorter than one frame are skipped."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "frames"
        input_dir.mkdir()

        sr = 16000
        frame_len = 32000
        short_signal = _sine_wave(sr=sr, duration=0.1)  # 1600 samples < 32000
        _write_wav(input_dir / "short.wav", short_signal, sr=sr)

        total = preprocess_directory(input_dir, output_dir, target_sr=sr, frame_len=frame_len)

        assert total == 0

    def test_skips_silent_files(self, tmp_path: Path) -> None:
        """Test that files containing only silence produce no frames."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "frames"
        input_dir.mkdir()

        sr = 16000
        frame_len = 1600
        silent = np.zeros(frame_len * 3, dtype=np.float32)
        _write_wav(input_dir / "silence.wav", silent, sr=sr)

        total = preprocess_directory(input_dir, output_dir, target_sr=sr, frame_len=frame_len)

        assert total == 0

    def test_output_filenames(self, tmp_path: Path) -> None:
        """Test that output files follow the {stem}__frame_{i:04d}.wav pattern."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "frames"
        input_dir.mkdir()

        sr = 16000
        frame_len = 1600
        signal = _sine_wave(sr=sr, duration=(frame_len * 3) / sr)
        _write_wav(input_dir / "my_song.wav", signal, sr=sr)

        preprocess_directory(input_dir, output_dir, target_sr=sr, frame_len=frame_len)

        output_files = sorted(output_dir.glob("*.wav"))
        expected_names = [
            "my_song__frame_0000.wav",
            "my_song__frame_0001.wav",
            "my_song__frame_0002.wav",
        ]
        assert [f.name for f in output_files] == expected_names

    def test_returns_correct_frame_count(self, tmp_path: Path) -> None:
        """Test that the returned count matches the number of output files."""
        input_dir = tmp_path / "raw"
        output_dir = tmp_path / "frames"
        input_dir.mkdir()

        sr = 16000
        frame_len = 1600
        signal = _sine_wave(sr=sr, duration=(frame_len * 4) / sr)
        _write_wav(input_dir / "track.wav", signal, sr=sr)

        total = preprocess_directory(input_dir, output_dir, target_sr=sr, frame_len=frame_len)

        output_files = list(output_dir.glob("*.wav"))
        assert total == len(output_files)

    def test_empty_directory_returns_zero(self, tmp_path: Path) -> None:
        """Test that an empty input directory returns 0."""
        input_dir = tmp_path / "empty"
        output_dir = tmp_path / "frames"
        input_dir.mkdir()

        total = preprocess_directory(input_dir, output_dir)

        assert total == 0
