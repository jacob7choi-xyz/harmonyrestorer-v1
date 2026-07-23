"""Tests for the download format transcoder."""

import shutil
import subprocess
from pathlib import Path

import pytest
from app.services.transcode import DOWNLOAD_FORMATS, TranscodeError, transcode_output


@pytest.fixture()
def wav_file(tmp_path) -> Path:
    path = tmp_path / "job_denoised.wav"
    path.write_bytes(b"RIFF" + b"\x00" * 100)
    return path


class TestTranscodeOutput:
    def test_wav_is_served_directly_without_conversion(self, wav_file, monkeypatch) -> None:
        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("ffmpeg must not run for wav")

        monkeypatch.setattr(subprocess, "run", forbidden)
        assert transcode_output(wav_file, "wav") == wav_file

    def test_unknown_format_is_rejected(self, wav_file) -> None:
        with pytest.raises(TranscodeError, match="Unsupported download format"):
            transcode_output(wav_file, "exe")

    def test_successful_conversion_writes_target_atomically(self, wav_file, monkeypatch) -> None:
        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            Path(argv[-1]).write_bytes(b"encoded")
            return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = transcode_output(wav_file, "mp3")

        assert result == wav_file.with_suffix(".mp3")
        assert result.read_bytes() == b"encoded"
        # No temp files remain beside the outputs
        leftovers = [p for p in wav_file.parent.iterdir() if p not in (wav_file, result)]
        assert leftovers == []

    def test_cached_conversion_skips_ffmpeg(self, wav_file, monkeypatch) -> None:
        cached = wav_file.with_suffix(".mp3")
        cached.write_bytes(b"cached")

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("cached format must not reconvert")

        monkeypatch.setattr(subprocess, "run", forbidden)
        assert transcode_output(wav_file, "mp3") == cached
        assert cached.read_bytes() == b"cached"

    def test_ffmpeg_failure_raises_and_leaves_no_artifacts(self, wav_file, monkeypatch) -> None:
        def failing_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"boom")

        monkeypatch.setattr(subprocess, "run", failing_run)
        with pytest.raises(TranscodeError, match="Conversion to mp3 failed"):
            transcode_output(wav_file, "mp3")

        assert not wav_file.with_suffix(".mp3").exists()
        assert list(wav_file.parent.iterdir()) == [wav_file]

    def test_timeout_raises_and_leaves_no_artifacts(self, wav_file, monkeypatch) -> None:
        def hanging_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            raise subprocess.TimeoutExpired(cmd=argv, timeout=1)

        monkeypatch.setattr(subprocess, "run", hanging_run)
        with pytest.raises(TranscodeError, match="timed out"):
            transcode_output(wav_file, "mp3")

        assert list(wav_file.parent.iterdir()) == [wav_file]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_ffmpeg_roundtrip(tmp_path) -> None:
    """Real conversion of a genuine tiny WAV for every non-wav format."""
    import numpy as np
    import soundfile as sf

    wav = tmp_path / "real_denoised.wav"
    tone = (0.1 * np.sin(np.linspace(0, 440 * 2 * np.pi, 16000))).astype("float32")
    sf.write(wav, tone, 16000)

    for fmt in DOWNLOAD_FORMATS:
        out = transcode_output(wav, fmt)
        assert out.exists()
        assert out.stat().st_size > 0
