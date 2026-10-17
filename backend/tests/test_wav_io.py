"""Tests for the shared WAV helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from app.services import wav_io
from app.services.wav_io import load_mono, write_wav_atomic

_SR = 16_000


class TestLoadMono:
    """Tests for load_mono."""

    def test_averages_stereo_and_resamples(self, tmp_path: Path) -> None:
        """A 44.1 kHz stereo file comes back as one channel at the target rate."""
        path = tmp_path / "stereo.wav"
        stereo = np.zeros((44_100, 2), dtype=np.float32)
        stereo[:, 0] = 0.5
        sf.write(str(path), stereo, 44_100, subtype="FLOAT")

        audio = load_mono(path, _SR)

        assert audio.ndim == 1
        assert len(audio) == _SR
        # The channel mean of 0.5 and 0.0, away from the resampler's edge effects
        assert audio[len(audio) // 2] == pytest.approx(0.25, abs=1e-3)

    def test_keeps_target_rate_mono_samples(self, tmp_path: Path) -> None:
        """A file already at the target rate and mono is returned unchanged."""
        path = tmp_path / "mono.wav"
        samples = np.linspace(-0.5, 0.5, _SR, dtype=np.float32)
        sf.write(str(path), samples, _SR, subtype="FLOAT")

        audio = load_mono(path, _SR)

        np.testing.assert_array_equal(audio, samples)


class TestWriteWavAtomic:
    """Tests for write_wav_atomic."""

    def test_writes_16_bit_wav_at_given_rate(self, tmp_path: Path) -> None:
        """The written file is a 16-bit mono WAV at the requested rate."""
        path = tmp_path / "out.wav"
        write_wav_atomic(path, np.zeros(1600, dtype=np.float32), _SR)

        info = sf.info(str(path))
        assert info.samplerate == _SR
        assert info.channels == 1
        assert info.subtype == "PCM_16"
        assert sorted(tmp_path.iterdir()) == [path]

    @pytest.mark.parametrize(
        "error",
        [
            OSError("disk full"),  # an Exception, such as a full disk
            KeyboardInterrupt(),  # a BaseException that is not an Exception, such as Ctrl+C
        ],
    )
    def test_failed_write_leaves_target_and_directory_unchanged(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: BaseException
    ) -> None:
        """A failing write raises, keeps the existing target, and leaves no temp file."""
        path = tmp_path / "out.wav"
        path.write_bytes(b"previous")

        def failing_write(file: Path, *args: object, **kwargs: object) -> None:
            Path(file).write_bytes(b"partial")
            raise error

        monkeypatch.setattr(wav_io.sf, "write", failing_write)

        with pytest.raises(type(error)):
            write_wav_atomic(path, np.zeros(1600, dtype=np.float32), _SR)

        assert path.read_bytes() == b"previous"
        assert sorted(tmp_path.iterdir()) == [path]
