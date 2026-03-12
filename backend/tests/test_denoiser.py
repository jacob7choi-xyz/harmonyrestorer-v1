"""Tests for DenoiserService."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.services.denoiser import DENOISED_OUTPUT_MARKER, DenoiserService


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
    def test_load_model_failure_allows_retry(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
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
    def test_returns_denoised_output_path(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Returns path to the denoised output file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        # Simulate separator creating a denoised file
        denoised_name = f"track_(No Noise).wav"
        denoised_path = output_dir / denoised_name
        denoised_path.write_bytes(b"RIFF" + b"\x00" * 40)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [str(denoised_path)]

        input_path = tmp_path / "track.wav"
        input_path.write_bytes(b"RIFF" + b"\x00" * 40)

        result = service.denoise(input_path)
        assert result == denoised_path.resolve()

    @patch("app.services.denoiser.Separator")
    def test_raises_when_no_denoised_output(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Raises RuntimeError when separator produces no denoised file."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [str(tmp_path / "track_(Noise).wav")]

        input_path = tmp_path / "track.wav"
        input_path.write_bytes(b"RIFF" + b"\x00" * 40)

        with pytest.raises(RuntimeError, match="Could not find denoised output"):
            service.denoise(input_path)

    @patch("app.services.denoiser.Separator")
    def test_path_traversal_neutralized(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Path traversal in output name is neutralized by .name stripping."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        # Simulate a malicious output path with traversal
        malicious_name = "../../etc/passwd_(No Noise).wav"
        # The service uses Path(output).name, stripping the traversal
        safe_name = Path(malicious_name).name
        safe_path = output_dir / safe_name
        safe_path.write_bytes(b"RIFF" + b"\x00" * 40)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [malicious_name]

        input_path = tmp_path / "track.wav"
        input_path.write_bytes(b"RIFF" + b"\x00" * 40)

        result = service.denoise(input_path)
        # Should resolve safely inside output_dir, not escape
        assert result.is_relative_to(output_dir.resolve())

    @patch("app.services.denoiser.Separator")
    def test_calls_separate_with_string_path(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Separator.separate receives a string path, not a Path object."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        denoised_name = f"track_(No Noise).wav"
        denoised_path = output_dir / denoised_name
        denoised_path.write_bytes(b"RIFF" + b"\x00" * 40)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [str(denoised_path)]

        input_path = tmp_path / "track.wav"
        input_path.write_bytes(b"RIFF" + b"\x00" * 40)

        service.denoise(input_path)
        mock_separator.separate.assert_called_once_with(str(input_path))

    @patch("app.services.denoiser.Separator")
    def test_selects_correct_output_among_multiple(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Picks the file with the denoised marker when multiple outputs exist."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        service = DenoiserService(output_dir=output_dir)

        noise_path = output_dir / "track_(Noise).wav"
        noise_path.write_bytes(b"RIFF" + b"\x00" * 40)
        denoised_path = output_dir / "track_(No Noise).wav"
        denoised_path.write_bytes(b"RIFF" + b"\x00" * 40)

        mock_separator = mock_sep_cls.return_value
        mock_separator.separate.return_value = [
            str(noise_path),
            str(denoised_path),
        ]

        input_path = tmp_path / "track.wav"
        input_path.write_bytes(b"RIFF" + b"\x00" * 40)

        result = service.denoise(input_path)
        assert result == denoised_path.resolve()


class TestDenoisedOutputMarker:
    """Tests for the DENOISED_OUTPUT_MARKER constant."""

    def test_marker_value(self) -> None:
        """Marker matches expected UVR output naming convention."""
        assert DENOISED_OUTPUT_MARKER == "No Noise"
