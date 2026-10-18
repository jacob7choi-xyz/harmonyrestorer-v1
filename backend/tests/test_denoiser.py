"""Tests for DenoiserService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from app.services.denoiser import DENOISED_OUTPUT_MARKER, DenoiserService


def _write_stem(path: Path, level: float = 0.5, seconds: float = 0.1) -> None:
    """Write a stem in the format UVR-DeNoise wrote in real runs: 44.1 kHz stereo."""
    frames = round(seconds * 44_100)
    sf.write(str(path), np.full((frames, 2), level, dtype=np.float32), 44_100, subtype="PCM_16")


def _write_input(path: Path, seconds: float = 0.1) -> None:
    """Write a 16 kHz mono input whose length the service compares the stem against."""
    sf.write(str(path), np.zeros(round(seconds * 16_000), dtype=np.float32), 16_000)


class TestDenoiserInit:
    """Tests for DenoiserService initialization."""

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Output directory is created if it does not exist."""
        output_dir = tmp_path / "new_dir"
        service = DenoiserService(output_dir=output_dir)
        assert output_dir.is_dir()
        assert service.output_dir == output_dir.resolve()

    def test_default_model_name(self, tmp_path: Path) -> None:
        """Default model name is UVR-DeNoise.pth."""
        service = DenoiserService(output_dir=tmp_path)
        assert service.model_name == "UVR-DeNoise.pth"

    def test_custom_model_name(self, tmp_path: Path) -> None:
        """Custom model name is stored correctly."""
        service = DenoiserService(output_dir=tmp_path, model_name="custom.pth")
        assert service.model_name == "custom.pth"

    def test_separator_not_loaded_on_init(self, tmp_path: Path) -> None:
        """Separator is not loaded until denoise is called."""
        service = DenoiserService(output_dir=tmp_path)
        assert service._separator is None


class TestGetSeparator:
    """Tests for lazy model loading."""

    @patch("app.services.denoiser.Separator")
    def test_loads_model_on_first_call(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """Model is loaded on the first call to _get_separator."""
        service = DenoiserService(output_dir=tmp_path)
        sep = service._get_separator()

        mock_sep_cls.assert_called_once_with(
            output_dir=str(tmp_path.resolve()), output_format="WAV"
        )
        sep.load_model.assert_called_once_with("UVR-DeNoise.pth")

    @patch("app.services.denoiser.Separator")
    def test_returns_same_instance_on_subsequent_calls(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Subsequent calls return the same separator instance."""
        service = DenoiserService(output_dir=tmp_path)
        first = service._get_separator()
        second = service._get_separator()

        assert first is second
        mock_sep_cls.assert_called_once()

    @patch("app.services.denoiser.Separator")
    def test_load_model_failure_allows_retry(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """If load_model fails, _separator stays None so next call retries."""
        mock_separator = mock_sep_cls.return_value
        mock_separator.load_model.side_effect = [RuntimeError("model not found"), None]

        service = DenoiserService(output_dir=tmp_path)

        with pytest.raises(RuntimeError, match="model not found"):
            service._get_separator()

        assert service._separator is None

        # Second call should retry and succeed
        sep = service._get_separator()
        assert sep is mock_separator
        assert mock_separator.load_model.call_count == 2


class TestDenoise:
    """Tests for the denoise method."""

    @patch("app.services.denoiser.Separator")
    def test_returns_converted_output_path(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """Returns the converted file named after the input, inside output_dir."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        # Simulate separator creating a denoised file
        denoised_path = output_dir / "track_(No Noise).wav"
        _write_stem(denoised_path)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [str(denoised_path)]

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        result = service.denoise(input_path)
        assert result == (output_dir / "track_uvr.wav").resolve()

    @patch("app.services.denoiser.Separator")
    def test_output_is_16k_mono_16_bit(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """A 44.1 kHz stereo stem is returned as a 16 kHz mono 16-bit WAV."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        denoised_path = output_dir / "track_(No Noise).wav"
        _write_stem(denoised_path)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [str(denoised_path)]

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        info = sf.info(str(service.denoise(input_path)))
        assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")

    @patch("app.services.denoiser.Separator")
    def test_raises_when_no_denoised_output(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """Raises RuntimeError when separator produces no denoised file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [str(tmp_path / "track_(Noise).wav")]

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(RuntimeError, match="Could not find denoised output"):
            service.denoise(input_path)

    @patch("app.services.denoiser.Separator")
    def test_path_traversal_neutralized(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """Path traversal in output name is neutralized by .name stripping."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        # Simulate a malicious output path with traversal
        malicious_name = "../../etc/passwd_(No Noise).wav"
        # The service uses Path(output).name, stripping the traversal
        safe_name = Path(malicious_name).name
        safe_path = output_dir / safe_name
        _write_stem(safe_path)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [malicious_name]

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        result = service.denoise(input_path)
        # Should resolve safely inside output_dir, not escape
        assert result.is_relative_to(output_dir.resolve())

    @patch("app.services.denoiser.Separator")
    def test_calls_separate_with_string_path(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """Separator.separate receives a string path, not a Path object."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        denoised_path = output_dir / "track_(No Noise).wav"
        _write_stem(denoised_path)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [str(denoised_path)]

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        service.denoise(input_path)
        mock_separator.separate.assert_called_once_with(str(input_path))

    @patch("app.services.denoiser.Separator")
    def test_selects_correct_output_among_multiple(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Converts the file with the denoised marker when multiple outputs exist."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        # Distinct levels show which stem the output came from
        noise_path = output_dir / "track_(Noise).wav"
        _write_stem(noise_path, level=0.0)
        denoised_path = output_dir / "track_(No Noise).wav"
        _write_stem(denoised_path, level=0.5)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [
            str(noise_path),
            str(denoised_path),
        ]

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        audio, _ = sf.read(str(service.denoise(input_path)), dtype="float32")
        # Away from the resampler's edge effects
        assert audio[len(audio) // 2] == pytest.approx(0.5, abs=1e-2)

    @patch("app.services.denoiser.Separator")
    def test_removes_noise_stem_after_success(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Only the returned file remains in output_dir after denoise; both stems are gone."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        noise_path = output_dir / "track_(Noise).wav"
        _write_stem(noise_path)
        denoised_path = output_dir / "track_(No Noise).wav"
        _write_stem(denoised_path)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [str(noise_path), str(denoised_path)]

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        result = service.denoise(input_path)
        assert sorted(output_dir.iterdir()) == [result]

    @patch("app.services.denoiser.Separator")
    def test_removes_outputs_when_no_denoised_stem(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A run with no denoised stem raises and leaves no output behind."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        noise_path = output_dir / "track_(Noise).wav"
        noise_path.write_bytes(b"RIFF" + b"\x00" * 40)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [str(noise_path)]

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(RuntimeError, match="Could not find denoised output"):
            service.denoise(input_path)
        assert list(output_dir.iterdir()) == []

    @patch("app.services.denoiser.Separator")
    def test_failed_write_raises_and_leaves_no_output(
        self, mock_sep_cls: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed WAV write raises and leaves output_dir empty, raw stem included."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        denoised_path = output_dir / "track_(No Noise).wav"
        _write_stem(denoised_path)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [str(denoised_path)]

        def failing_write(*args: object, **kwargs: object) -> None:
            # A memory-backed filesystem at its limit fails the write this way
            raise OSError("No space left on device")

        monkeypatch.setattr("app.services.denoiser.write_wav_atomic", failing_write)

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(OSError, match="No space left"):
            service.denoise(input_path)
        assert list(output_dir.iterdir()) == []

    @pytest.mark.parametrize(
        "stem_seconds",
        [
            0.5,  # half the input, as a copy cut short by a full filesystem leaves it
            1.25,  # longer than the input
        ],
    )
    @patch("app.services.denoiser.Separator")
    def test_stem_length_mismatch_raises_and_leaves_no_output(
        self, mock_sep_cls: MagicMock, tmp_path: Path, stem_seconds: float
    ) -> None:
        """A denoised stem whose length differs from the input by more than the
        tolerance raises, and no output is left behind."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        denoised_path = output_dir / "track_(No Noise).wav"
        _write_stem(denoised_path, seconds=stem_seconds)
        mock_sep_cls.return_value.separate.return_value = [str(denoised_path)]

        input_path = tmp_path / "track.wav"
        _write_input(input_path, seconds=1.0)

        with pytest.raises(RuntimeError, match="length"):
            service.denoise(input_path)
        assert list(output_dir.iterdir()) == []

    @patch("app.services.denoiser.Separator")
    def test_stem_within_tolerance_is_accepted(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A stem a few milliseconds short, as UVR wrote in real runs, is accepted."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        denoised_path = output_dir / "track_(No Noise).wav"
        _write_stem(denoised_path, seconds=0.99)  # 10 ms short
        mock_sep_cls.return_value.separate.return_value = [str(denoised_path)]

        input_path = tmp_path / "track.wav"
        _write_input(input_path, seconds=1.0)

        assert service.denoise(input_path).is_file()

    @patch("app.services.denoiser.Separator")
    def test_missing_denoised_stem_raises_and_leaves_no_output(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A returned stem path with no file behind it raises a clear error."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        # audio-separator skips writing a near-silent stem but may still return its path
        mock_sep_cls.return_value.separate.return_value = [str(output_dir / "track_(No Noise).wav")]

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(RuntimeError, match="was not written"):
            service.denoise(input_path)
        assert list(output_dir.iterdir()) == []

    @patch("app.services.denoiser.Separator")
    def test_raises_when_separate_fails(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """Exception from separator.separate propagates to caller."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.side_effect = RuntimeError("corrupted audio")

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(RuntimeError, match="corrupted audio"):
            service.denoise(input_path)

    @patch("app.services.denoiser.Separator")
    def test_raises_when_separate_returns_empty_list(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Empty output list from separator raises RuntimeError."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = []

        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(RuntimeError, match="Could not find denoised output"):
            service.denoise(input_path)


class TestDenoisedOutputMarker:
    """Tests for the DENOISED_OUTPUT_MARKER constant."""

    def test_marker_value(self) -> None:
        """Marker matches expected UVR output naming convention."""
        assert DENOISED_OUTPUT_MARKER == "No Noise"
