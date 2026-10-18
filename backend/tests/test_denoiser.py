"""Tests for DenoiserService."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from app.services.denoiser import DENOISED_OUTPUT_MARKER, STAGING_DIR_NAME, DenoiserService


def _write_stem(path: Path, level: float = 0.5, seconds: float = 0.1) -> None:
    """Write a stem in the format UVR-DeNoise wrote in real runs: 44.1 kHz stereo."""
    frames = round(seconds * 44_100)
    sf.write(str(path), np.full((frames, 2), level, dtype=np.float32), 44_100, subtype="PCM_16")


def _write_input(path: Path, seconds: float = 0.1) -> None:
    """Write a 16 kHz mono input whose length the service compares the stem against."""
    sf.write(str(path), np.zeros(round(seconds * 16_000), dtype=np.float32), 16_000)


def _fake_separation(
    mock_sep_cls: MagicMock,
    written: list[tuple[str, float, float]],
    returned: list[str] | None = None,
) -> None:
    """Make the mocked Separator act like audio-separator: write stems into the
    output_dir it was constructed with and return their bare names.

    Args:
        mock_sep_cls: The patched Separator class.
        written: (name, level, seconds) of each stem to write.
        returned: Names to return; defaults to every written name. A shorter list
            mimics a separation that logged a failed write and returned early.
    """

    def separate(_input_path: str) -> list[str]:
        out = Path(mock_sep_cls.call_args.kwargs["output_dir"])
        for name, level, seconds in written:
            _write_stem(out / name, level=level, seconds=seconds)
        return [name for name, _, _ in written] if returned is None else returned

    mock_sep_cls.return_value.separate.side_effect = separate


def _left_behind(output_dir: Path) -> list[str]:
    """Names of every file under output_dir, including inside the staging directory."""
    return sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if not p.is_dir())


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
            output_dir=str(tmp_path.resolve() / STAGING_DIR_NAME),
            output_format="WAV",
            use_soundfile=True,
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
        service = DenoiserService(output_dir=output_dir)
        _fake_separation(mock_sep_cls, [("track_(No Noise).wav", 0.5, 0.1)])
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        result = service.denoise(input_path)

        assert result == (output_dir / "track_uvr.wav").resolve()
        assert _left_behind(output_dir) == ["track_uvr.wav"]

    @patch("app.services.denoiser.Separator")
    def test_output_is_16k_mono_16_bit(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """A 44.1 kHz stereo stem is returned as a 16 kHz mono 16-bit WAV."""
        service = DenoiserService(output_dir=tmp_path / "output")
        _fake_separation(mock_sep_cls, [("track_(No Noise).wav", 0.5, 0.1)])
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        info = sf.info(str(service.denoise(input_path)))
        assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")

    @patch("app.services.denoiser.Separator")
    def test_calls_separate_with_string_path(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """Separator.separate receives a string path, not a Path object."""
        service = DenoiserService(output_dir=tmp_path / "output")
        _fake_separation(mock_sep_cls, [("track_(No Noise).wav", 0.5, 0.1)])
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        service.denoise(input_path)
        mock_sep_cls.return_value.separate.assert_called_once_with(str(input_path))

    @patch("app.services.denoiser.Separator")
    def test_selects_correct_output_among_multiple(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Converts the stem with the denoised marker and removes both raw stems."""
        output_dir = tmp_path / "output"
        service = DenoiserService(output_dir=output_dir)
        # Distinct levels show which stem the output came from
        _fake_separation(
            mock_sep_cls,
            [("track_(Noise).wav", 0.0, 0.1), ("track_(No Noise).wav", 0.5, 0.1)],
        )
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        audio, _ = sf.read(str(service.denoise(input_path)), dtype="float32")

        # Away from the resampler's edge effects
        assert audio[len(audio) // 2] == pytest.approx(0.5, abs=1e-2)
        assert _left_behind(output_dir) == ["track_uvr.wav"]

    @patch("app.services.denoiser.Separator")
    def test_path_traversal_neutralized(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """A returned name with traversal resolves inside the staging directory."""
        output_dir = tmp_path / "output"
        service = DenoiserService(output_dir=output_dir)
        _fake_separation(
            mock_sep_cls,
            [("passwd_(No Noise).wav", 0.5, 0.1)],
            returned=["../../etc/passwd_(No Noise).wav"],
        )
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        result = service.denoise(input_path)

        assert result.is_relative_to(output_dir.resolve())
        assert _left_behind(output_dir) == ["track_uvr.wav"]

    @patch("app.services.denoiser.Separator")
    def test_raises_when_no_denoised_output(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """A run with no denoised stem raises and leaves no file behind."""
        output_dir = tmp_path / "output"
        service = DenoiserService(output_dir=output_dir)
        _fake_separation(mock_sep_cls, [("track_(Noise).wav", 0.5, 0.1)])
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(RuntimeError, match="no denoised stem"):
            service.denoise(input_path)
        assert _left_behind(output_dir) == []

    @patch("app.services.denoiser.Separator")
    def test_stems_from_a_separation_that_returned_early_are_removed(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """audio-separator logs a failed stem write and returns without the stems it
        already wrote; those files are removed anyway."""
        output_dir = tmp_path / "output"
        service = DenoiserService(output_dir=output_dir)
        _fake_separation(mock_sep_cls, [("track_(Noise).wav", 0.5, 0.1)], returned=[])
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(RuntimeError, match="no denoised stem"):
            service.denoise(input_path)
        assert _left_behind(output_dir) == []

    @patch("app.services.denoiser.Separator")
    def test_files_left_in_staging_by_an_earlier_run_are_removed(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Files left in the staging directory, as a crashed run would leave them,
        are removed before the next separation."""
        output_dir = tmp_path / "output"
        service = DenoiserService(output_dir=output_dir)
        staging = output_dir / STAGING_DIR_NAME
        staging.mkdir()
        (staging / "crashed_(Noise)_UVR-DeNoise.wav").write_bytes(b"x" * 100)
        _fake_separation(mock_sep_cls, [("track_(No Noise).wav", 0.5, 0.1)])
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        service.denoise(input_path)
        assert _left_behind(output_dir) == ["track_uvr.wav"]

    @patch("app.services.denoiser.Separator")
    def test_symlink_in_staging_is_removed_without_touching_its_target(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Clearing the staging directory removes a link itself, never what it points at."""
        output_dir = tmp_path / "output"
        service = DenoiserService(output_dir=output_dir)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "keep.wav").write_bytes(b"x" * 10)
        staging = output_dir / STAGING_DIR_NAME
        staging.mkdir()
        (staging / "file_link").symlink_to(outside / "keep.wav")
        (staging / "dir_link").symlink_to(outside, target_is_directory=True)
        _fake_separation(mock_sep_cls, [("track_(No Noise).wav", 0.5, 0.1)])
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        service.denoise(input_path)

        assert list(staging.iterdir()) == []
        assert (outside / "keep.wav").read_bytes() == b"x" * 10

    @patch("app.services.denoiser.Separator")
    def test_staging_directory_that_is_a_symlink_is_refused(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A staging path that is a symlink raises before anything is cleared or run."""
        output_dir = tmp_path / "output"
        service = DenoiserService(output_dir=output_dir)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "keep.wav").write_bytes(b"x" * 10)
        (output_dir / STAGING_DIR_NAME).symlink_to(elsewhere, target_is_directory=True)
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(RuntimeError, match="symlink"):
            service.denoise(input_path)
        assert (elsewhere / "keep.wav").read_bytes() == b"x" * 10
        mock_sep_cls.return_value.separate.assert_not_called()

    @patch("app.services.denoiser.Separator")
    def test_failed_write_raises_and_leaves_no_output(
        self, mock_sep_cls: MagicMock, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed WAV write raises and leaves no file behind, raw stem included."""
        output_dir = tmp_path / "output"
        service = DenoiserService(output_dir=output_dir)
        _fake_separation(mock_sep_cls, [("track_(No Noise).wav", 0.5, 0.1)])

        def failing_write(*args: object, **kwargs: object) -> None:
            # A memory-backed filesystem at its limit fails the write this way
            raise OSError("No space left on device")

        monkeypatch.setattr("app.services.denoiser.write_wav_atomic", failing_write)
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(OSError, match="No space left"):
            service.denoise(input_path)
        assert _left_behind(output_dir) == []

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
        tolerance raises, and no file is left behind."""
        output_dir = tmp_path / "output"
        service = DenoiserService(output_dir=output_dir)
        _fake_separation(mock_sep_cls, [("track_(No Noise).wav", 0.5, stem_seconds)])
        input_path = tmp_path / "track.wav"
        _write_input(input_path, seconds=1.0)

        with pytest.raises(RuntimeError, match="length"):
            service.denoise(input_path)
        assert _left_behind(output_dir) == []

    @patch("app.services.denoiser.Separator")
    def test_stem_within_tolerance_is_accepted(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A stem a few milliseconds short, as UVR wrote in real runs, is accepted."""
        service = DenoiserService(output_dir=tmp_path / "output")
        _fake_separation(mock_sep_cls, [("track_(No Noise).wav", 0.5, 0.99)])  # 10 ms short
        input_path = tmp_path / "track.wav"
        _write_input(input_path, seconds=1.0)

        assert service.denoise(input_path).is_file()

    @patch("app.services.denoiser.Separator")
    def test_missing_denoised_stem_raises_and_leaves_no_output(
        self, mock_sep_cls: MagicMock, tmp_path: Path
    ) -> None:
        """A returned stem name with no file behind it raises a clear error."""
        output_dir = tmp_path / "output"
        service = DenoiserService(output_dir=output_dir)
        # audio-separator skips writing a near-silent stem but may still return its name
        _fake_separation(mock_sep_cls, [], returned=["track_(No Noise).wav"])
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(RuntimeError, match="was not written"):
            service.denoise(input_path)
        assert _left_behind(output_dir) == []

    @patch("app.services.denoiser.Separator")
    def test_raises_when_separate_fails(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """An exception from separator.separate propagates to the caller."""
        output_dir = tmp_path / "output"
        service = DenoiserService(output_dir=output_dir)
        mock_sep_cls.return_value.separate.side_effect = RuntimeError("corrupted audio")
        input_path = tmp_path / "track.wav"
        _write_input(input_path)

        with pytest.raises(RuntimeError, match="corrupted audio"):
            service.denoise(input_path)
        assert _left_behind(output_dir) == []

    @patch("app.services.denoiser.Separator")
    def test_concurrent_calls_do_not_overlap(self, mock_sep_cls: MagicMock, tmp_path: Path) -> None:
        """Two denoise calls on one service run their separations one at a time, so
        one call never clears stems the other is still using."""
        service = DenoiserService(output_dir=tmp_path / "output")
        active = 0
        peak = 0
        guard = threading.Lock()

        def separate(input_path: str) -> list[str]:
            nonlocal active, peak
            with guard:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            out = Path(mock_sep_cls.call_args.kwargs["output_dir"])
            name = f"{Path(input_path).stem}_(No Noise).wav"
            _write_stem(out / name)
            with guard:
                active -= 1
            return [name]

        mock_sep_cls.return_value.separate.side_effect = separate
        inputs = [tmp_path / "a.wav", tmp_path / "b.wav"]
        for path in inputs:
            _write_input(path)
        threads = [threading.Thread(target=service.denoise, args=(path,)) for path in inputs]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert peak == 1
        assert (tmp_path / "output" / "a_uvr.wav").is_file()
        assert (tmp_path / "output" / "b_uvr.wav").is_file()

    @patch("app.services.denoiser.Separator")
    def test_failed_clear_after_success_is_logged_as_error(
        self,
        mock_sep_cls: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """If the staging directory cannot be emptied after a job, the job's result
        stands and the failure is logged at ERROR, since the leftover counts against
        the artifact budget."""
        service = DenoiserService(output_dir=tmp_path / "output")
        _fake_separation(mock_sep_cls, [("track_(No Noise).wav", 0.5, 0.1)])
        input_path = tmp_path / "track.wav"
        _write_input(input_path)
        calls = 0
        real_unlink = Path.unlink

        def unlink_failing_in_final_clear(self: Path, missing_ok: bool = False) -> None:
            nonlocal calls
            if self.parent.name == STAGING_DIR_NAME:
                calls += 1
                raise PermissionError("denied")
            real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", unlink_failing_in_final_clear)

        with caplog.at_level(logging.ERROR, logger="app.services.denoiser"):
            result = service.denoise(input_path)

        assert result.is_file()
        assert calls >= 1
        assert any("staging" in r.getMessage() for r in caplog.records)


class TestDenoisedOutputMarker:
    """Tests for the DENOISED_OUTPUT_MARKER constant."""

    def test_marker_value(self) -> None:
        """Marker matches expected UVR output naming convention."""
        assert DENOISED_OUTPUT_MARKER == "No Noise"
