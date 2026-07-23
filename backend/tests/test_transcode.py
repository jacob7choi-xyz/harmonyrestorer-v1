"""Tests for the download format transcoder."""

import shutil
import subprocess
import threading
from pathlib import Path

import pytest
from app.services.transcode import (
    _TRANSCODE_SLOTS,
    DOWNLOAD_FORMATS,
    TranscodeBusyError,
    TranscodeError,
    transcode_output,
)


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


class TestTranscodeAdmission:
    def test_busy_when_slot_held(self, wav_file, monkeypatch) -> None:
        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            Path(argv[-1]).write_bytes(b"encoded")
            return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _TRANSCODE_SLOTS.acquire(blocking=False)
        try:
            with pytest.raises(TranscodeBusyError):
                transcode_output(wav_file, "mp3")
        finally:
            _TRANSCODE_SLOTS.release()

        # Slot free again: the same conversion succeeds
        assert transcode_output(wav_file, "mp3").exists()

    def test_cached_format_needs_no_slot(self, wav_file) -> None:
        cached = wav_file.with_suffix(".mp3")
        cached.write_bytes(b"cached")

        assert _TRANSCODE_SLOTS.acquire(blocking=False)
        try:
            assert transcode_output(wav_file, "mp3") == cached
        finally:
            _TRANSCODE_SLOTS.release()

    def test_slot_lifetime_matches_subprocess_not_any_awaiter(self, wav_file, monkeypatch) -> None:
        """The slot is owned by the thread running ffmpeg for exactly its
        lifetime; nothing outside that scope can free capacity early.

        This is the transcode analog of the inference cancellation
        invariant: a second conversion stays rejected until the first
        subprocess actually exits.
        """
        started = threading.Event()
        finish = threading.Event()

        def blocking_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            started.set()
            assert finish.wait(timeout=10), "test never released fake ffmpeg"
            Path(argv[-1]).write_bytes(b"encoded")
            return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", blocking_run)

        worker = threading.Thread(target=transcode_output, args=(wav_file, "mp3"))
        worker.start()
        try:
            assert started.wait(timeout=5)
            # Subprocess is live: capacity must be unavailable
            with pytest.raises(TranscodeBusyError):
                transcode_output(wav_file, "ogg")
        finally:
            finish.set()
            worker.join(timeout=10)

        # Subprocess exited: capacity restored
        assert _TRANSCODE_SLOTS.acquire(blocking=False)
        _TRANSCODE_SLOTS.release()

    def test_budget_exhaustion_rejects_and_releases_the_slot(self, wav_file, monkeypatch) -> None:
        from app.services import transcode as transcode_module
        from app.services.artifacts import ArtifactBudget

        monkeypatch.setattr(
            transcode_module, "artifact_budget", ArtifactBudget(limit_bytes=1, directories=[])
        )

        with pytest.raises(TranscodeBusyError, match="Storage capacity"):
            transcode_output(wav_file, "mp3")

        # The slot must not stay held after a budget rejection
        assert _TRANSCODE_SLOTS.acquire(blocking=False)
        _TRANSCODE_SLOTS.release()

    def test_budget_reservation_released_after_subprocess_failure(
        self, wav_file, monkeypatch
    ) -> None:
        from app.services import transcode as transcode_module
        from app.services.artifacts import ArtifactBudget

        # Budget fits exactly one reservation (2x the source size)
        exact = ArtifactBudget(limit_bytes=wav_file.stat().st_size * 2, directories=[])
        monkeypatch.setattr(transcode_module, "artifact_budget", exact)

        def failing_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"boom")

        monkeypatch.setattr(subprocess, "run", failing_run)
        with pytest.raises(TranscodeError, match="failed"):
            transcode_output(wav_file, "mp3")

        # Reservation returned: the same budget accepts the next attempt
        assert exact.try_reserve(wav_file.stat().st_size * 2) is True

    def test_slot_released_after_subprocess_failure(self, wav_file, monkeypatch) -> None:
        def failing_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"boom")

        monkeypatch.setattr(subprocess, "run", failing_run)
        with pytest.raises(TranscodeError):
            transcode_output(wav_file, "mp3")

        assert _TRANSCODE_SLOTS.acquire(blocking=False)
        _TRANSCODE_SLOTS.release()

    def test_encoder_thread_cap_is_requested(self, wav_file, monkeypatch) -> None:
        """The argv carries the single-thread mechanism; the CPU property
        itself is verified on the candidate revision."""
        captured: list[list[str]] = []

        def fake_run(argv, **kwargs):  # type: ignore[no-untyped-def]
            captured.append(list(argv))
            Path(argv[-1]).write_bytes(b"encoded")
            return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", fake_run)
        transcode_output(wav_file, "mp3")

        assert captured and captured[0].count("-threads") == 2


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
