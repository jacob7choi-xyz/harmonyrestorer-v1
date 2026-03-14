"""Tests for UVR batch inference logic in dataset.infer_uvr."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import soundfile as sf

from dataset.infer_uvr import (
    _DENOISED_MARKER,
    _TARGET_SR,
    _find_denoised_output,
    _resample_if_needed,
    restore_file,
)


class TestFindDenoisedOutput:
    """Tests for _find_denoised_output file discovery."""

    def test_finds_file_with_marker(self, tmp_path: Path) -> None:
        """Finds the file containing the 'No Noise' marker."""
        noise_file = tmp_path / "audio_(Noise)_UVR.wav"
        denoised_file = tmp_path / "audio_(No Noise)_UVR.wav"
        noise_file.touch()
        denoised_file.touch()

        result = _find_denoised_output(tmp_path)
        assert result == denoised_file

    def test_returns_none_when_no_marker(self, tmp_path: Path) -> None:
        """Returns None when no file has the denoised marker."""
        (tmp_path / "audio_(Noise)_UVR.wav").touch()
        (tmp_path / "other_file.wav").touch()

        result = _find_denoised_output(tmp_path)
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        """Returns None for an empty directory."""
        result = _find_denoised_output(tmp_path)
        assert result is None

    def test_marker_value(self) -> None:
        """Verify the denoised marker constant matches audio-separator convention."""
        assert _DENOISED_MARKER == "No Noise"


class TestResampleIfNeeded:
    """Tests for _resample_if_needed conditional resampling."""

    def test_returns_original_when_already_16khz_mono(self, tmp_path: Path) -> None:
        """Returns original path when file is already 16 kHz mono."""
        audio = np.zeros(16000, dtype=np.float32)
        input_path = tmp_path / "input.wav"
        sf.write(str(input_path), audio, _TARGET_SR)

        result = _resample_if_needed(input_path, tmp_path)
        assert result == input_path

    def test_resamples_44100_to_16000(self, tmp_path: Path) -> None:
        """Resamples 44.1 kHz file to 16 kHz."""
        audio = np.random.default_rng(42).standard_normal(44100).astype(np.float32)
        input_path = tmp_path / "input.wav"
        sf.write(str(input_path), audio, 44100)

        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        result = _resample_if_needed(input_path, temp_dir)

        assert result != input_path
        assert result.parent == temp_dir
        info = sf.info(str(result))
        assert info.samplerate == _TARGET_SR

    def test_converts_stereo_to_mono(self, tmp_path: Path) -> None:
        """Resamples stereo file to mono."""
        audio = np.random.default_rng(42).standard_normal((16000, 2)).astype(np.float32)
        input_path = tmp_path / "stereo.wav"
        sf.write(str(input_path), audio, _TARGET_SR)

        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        result = _resample_if_needed(input_path, temp_dir)

        assert result != input_path
        info = sf.info(str(result))
        assert info.channels == 1

    def test_resampled_filename_includes_original_name(self, tmp_path: Path) -> None:
        """Resampled file name includes original filename for traceability."""
        audio = np.zeros(44100, dtype=np.float32)
        input_path = tmp_path / "my_song.wav"
        sf.write(str(input_path), audio, 44100)

        temp_dir = tmp_path / "temp"
        temp_dir.mkdir()

        result = _resample_if_needed(input_path, temp_dir)
        assert "my_song" in result.name


class TestRestoreFile:
    """Tests for restore_file end-to-end with mocked Separator."""

    @pytest.fixture()
    def noisy_wav(self, tmp_path: Path) -> Path:
        """Create a 16 kHz mono WAV file for testing."""
        audio = np.random.default_rng(42).standard_normal(32000).astype(np.float32)
        path = tmp_path / "noisy.wav"
        sf.write(str(path), audio, _TARGET_SR)
        return path

    @pytest.fixture()
    def temp_dir(self, tmp_path: Path) -> Path:
        """Create a temp directory for separator scratch space."""
        d = tmp_path / "scratch"
        d.mkdir()
        return d

    @pytest.fixture()
    def output_dir(self, tmp_path: Path) -> Path:
        """Create an output directory."""
        d = tmp_path / "output"
        d.mkdir()
        return d

    def _make_separator(self, temp_dir: Path) -> Mock:
        """Create a mock Separator that writes fake denoised output."""
        separator = Mock()

        def fake_separate(path: str) -> list[str]:
            # Simulate audio-separator writing output at 44.1 kHz stereo
            stem = Path(path).stem
            denoised_path = temp_dir / f"{stem}_({_DENOISED_MARKER})_UVR.wav"
            noise_path = temp_dir / f"{stem}_(Noise)_UVR.wav"
            audio = np.random.default_rng(0).standard_normal(88200).astype(np.float32)
            sf.write(str(denoised_path), audio, 44100)
            sf.write(str(noise_path), audio, 44100)
            return [str(denoised_path), str(noise_path)]

        separator.separate.side_effect = fake_separate
        return separator

    def test_produces_output_file(
        self, noisy_wav: Path, temp_dir: Path, output_dir: Path
    ) -> None:
        """Restore produces an output file at the expected path."""
        separator = self._make_separator(temp_dir)
        output_path = output_dir / "noisy.wav"

        restore_file(separator, noisy_wav, output_path, temp_dir)

        assert output_path.exists()

    def test_output_is_16khz_mono(
        self, noisy_wav: Path, temp_dir: Path, output_dir: Path
    ) -> None:
        """Output file is resampled back to 16 kHz mono."""
        separator = self._make_separator(temp_dir)
        output_path = output_dir / "noisy.wav"

        restore_file(separator, noisy_wav, output_path, temp_dir)

        info = sf.info(str(output_path))
        assert info.samplerate == _TARGET_SR
        assert info.channels == 1

    def test_calls_separator_with_input_path(
        self, noisy_wav: Path, temp_dir: Path, output_dir: Path
    ) -> None:
        """Separator.separate is called with a string path."""
        separator = self._make_separator(temp_dir)
        output_path = output_dir / "noisy.wav"

        restore_file(separator, noisy_wav, output_path, temp_dir)

        separator.separate.assert_called_once()
        call_arg = separator.separate.call_args[0][0]
        assert isinstance(call_arg, str)

    def test_raises_when_no_denoised_output(
        self, noisy_wav: Path, temp_dir: Path, output_dir: Path
    ) -> None:
        """Raises RuntimeError when separator produces no denoised file."""
        separator = Mock()
        separator.separate.return_value = []

        output_path = output_dir / "noisy.wav"

        with pytest.raises(RuntimeError, match="No denoised output"):
            restore_file(separator, noisy_wav, output_path, temp_dir)

    def test_cleans_temp_dir_before_processing(
        self, noisy_wav: Path, temp_dir: Path, output_dir: Path
    ) -> None:
        """Temp dir is cleared of leftover files before each call."""
        # Plant a leftover file
        leftover = temp_dir / "old_junk.wav"
        leftover.write_bytes(b"fake")

        separator = self._make_separator(temp_dir)
        output_path = output_dir / "noisy.wav"

        restore_file(separator, noisy_wav, output_path, temp_dir)

        # Leftover should be gone (temp dir is cleared then repopulated by separator)
        assert not leftover.exists()

    def test_no_partial_file_on_failure(
        self, noisy_wav: Path, temp_dir: Path, output_dir: Path
    ) -> None:
        """Output path should not exist if separator fails."""
        separator = Mock()
        separator.separate.side_effect = RuntimeError("UVR crashed")

        output_path = output_dir / "noisy.wav"

        with pytest.raises(RuntimeError, match="UVR crashed"):
            restore_file(separator, noisy_wav, output_path, temp_dir)

        assert not output_path.exists()
