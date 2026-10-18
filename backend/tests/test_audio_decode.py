"""Tests for decoding uploads into a bounded WAV with ffmpeg.

These run the real ffmpeg binary, which CI installs, because the controls under
test (demuxer pinning, protocol whitelisting, length caps) are ffmpeg behavior that
a mocked subprocess cannot show.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from app.services import audio_decode
from app.services.audio_decode import (
    AudioDecodeError,
    AudioTooLongError,
    DecoderUnavailableError,
    decode_to_wav,
    decoded_size_cap,
    demuxer_for,
)

_TIMEOUT_S = 30.0


def _ffmpeg(*args: str) -> None:
    """Build a test input with the host ffmpeg."""
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args], check=True)


def _tone(path: Path, seconds: float, sr: int = 16_000, channels: int = 1) -> None:
    t = np.arange(round(seconds * sr)) / sr
    mono = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    data = mono if channels == 1 else np.stack([mono] * channels, axis=1)
    sf.write(str(path), data, sr, subtype="FLOAT")


def _decode(
    src: Path, dst: Path, demuxer: str, sample_rate: int = 16_000, max_s: int = 600
) -> float:
    return decode_to_wav(
        src, dst, demuxer=demuxer, sample_rate=sample_rate, max_seconds=max_s, timeout_s=_TIMEOUT_S
    )


class TestDemuxerFor:
    """Tests for choosing the ffmpeg demuxer from the validated upload."""

    @pytest.mark.parametrize(
        ("extension", "content", "expected"),
        [
            (".wav", b"RIFF\x00\x00\x00\x00WAVE", "wav"),
            (".mp3", b"ID3\x04\x00", "mp3"),
            (".flac", b"fLaC\x00", "flac"),
            (".ogg", b"OggS\x00", "ogg"),
            (".m4a", b"\x00\x00\x00\x20ftypM4A ", "mov"),
            (".aac", b"\xff\xf1\x50\x80", "aac"),  # ADTS
            (".aac", b"\x00\x00\x00\x20ftypM4A ", "mov"),  # AAC in an MP4 container
        ],
    )
    def test_maps_each_supported_upload(
        self, extension: str, content: bytes, expected: str
    ) -> None:
        """Every extension the upload route accepts maps to one pinned demuxer."""
        assert demuxer_for(extension, content) == expected

    def test_unknown_extension_is_refused(self) -> None:
        """An extension outside the mapping raises instead of letting ffmpeg probe."""
        with pytest.raises(AudioDecodeError):
            demuxer_for(".m3u8", b"#EXTM3U")


class TestDecodeToWav:
    """Tests for decode_to_wav against the real ffmpeg binary."""

    def test_decodes_to_target_rate_keeping_mono(self, tmp_path: Path) -> None:
        """A mono source stays mono and is resampled to the target rate."""
        src = tmp_path / "in.wav"
        _tone(src, 1.0, sr=48_000, channels=1)
        dst = tmp_path / "out.wav"

        duration = _decode(src, dst, "wav", sample_rate=16_000)

        info = sf.info(str(dst))
        assert (info.samplerate, info.channels, info.subtype) == (16_000, 1, "PCM_16")
        assert duration == pytest.approx(1.0, abs=0.01)

    def test_keeps_stereo_and_mixes_more_channels_down_to_stereo(self, tmp_path: Path) -> None:
        """Stereo passes through as stereo; a source with more channels becomes stereo,
        so channel mixing for mono and stereo stays with each engine."""
        stereo = tmp_path / "stereo.wav"
        _tone(stereo, 0.5, channels=2)
        eight = tmp_path / "eight.wav"
        _tone(eight, 0.5, channels=8)

        _decode(stereo, tmp_path / "stereo_out.wav", "wav")
        _decode(eight, tmp_path / "eight_out.wav", "wav")

        assert sf.info(str(tmp_path / "stereo_out.wav")).channels == 2
        assert sf.info(str(tmp_path / "eight_out.wav")).channels == 2

    def test_mono_level_is_unchanged(self, tmp_path: Path) -> None:
        """Decoding does not change a mono signal's level, as an upmix to stereo would."""
        src = tmp_path / "in.wav"
        _tone(src, 0.5)
        dst = tmp_path / "out.wav"

        _decode(src, dst, "wav")

        a, _ = sf.read(str(src), dtype="float32")
        b, _ = sf.read(str(dst), dtype="float32")
        assert np.sqrt(np.mean(b**2)) == pytest.approx(np.sqrt(np.mean(a**2)), rel=1e-3)

    @pytest.mark.parametrize(
        ("extension", "demuxer", "codec_args"),
        [
            (".mp3", "mp3", ["-c:a", "libmp3lame", "-b:a", "128k"]),
            (".m4a", "mov", ["-c:a", "aac", "-f", "mp4"]),
            (".aac", "aac", ["-c:a", "aac", "-f", "adts"]),
            (".flac", "flac", ["-c:a", "flac"]),
        ],
    )
    def test_decodes_compressed_formats(
        self, tmp_path: Path, extension: str, demuxer: str, codec_args: list[str]
    ) -> None:
        """Formats soundfile cannot open, or opens through an estimate, decode here."""
        wav = tmp_path / "tone.wav"
        _tone(wav, 2.0)
        src = tmp_path / f"in{extension}"
        _ffmpeg("-i", str(wav), *codec_args, str(src))

        duration = _decode(src, tmp_path / "out.wav", demuxer)

        # Encoder delay and padding are decoded too; ADTS cannot signal them for trimming,
        # and a 2 s ADTS file encoded at 16 kHz decoded to 2.112 s here
        assert duration == pytest.approx(2.0, abs=0.15)

    def test_vbr_mp3_without_xing_header_is_measured_by_its_decoded_length(
        self, tmp_path: Path
    ) -> None:
        """soundfile reads a VBR MP3 without a Xing header as shorter than it is; the
        decoded length is the real one, so the length cap applies to all of it."""
        wav = tmp_path / "tone.wav"
        _tone(wav, 30.0)
        src = tmp_path / "vbr.mp3"
        _ffmpeg("-i", str(wav), "-c:a", "libmp3lame", "-q:a", "2", "-write_xing", "0", str(src))
        assert sf.info(str(src)).duration < 25.0  # the estimate this file produces

        with pytest.raises(AudioTooLongError):
            _decode(src, tmp_path / "out.wav", "mp3", max_s=25)
        assert _decode(src, tmp_path / "ok.wav", "mp3", max_s=60) == pytest.approx(30.0, abs=0.2)

    def test_too_long_input_raises_and_leaves_nothing(self, tmp_path: Path) -> None:
        """An input longer than the cap raises, and neither the destination nor a temp
        file is left behind."""
        src = tmp_path / "in.wav"
        _tone(src, 3.0)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with pytest.raises(AudioTooLongError):
            _decode(src, out_dir / "out.wav", "wav", max_s=2)
        assert list(out_dir.iterdir()) == []

    def test_undecodable_input_raises_and_leaves_nothing(self, tmp_path: Path) -> None:
        """Bytes that are not audio raise AudioDecodeError and leave nothing behind."""
        src = tmp_path / "in.mp3"
        src.write_bytes(b"ID3" + b"\x00" * 4096)
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        with pytest.raises(AudioDecodeError):
            _decode(src, out_dir / "out.wav", "mp3")
        assert list(out_dir.iterdir()) == []

    def test_hls_playlist_disguised_as_m4a_does_not_read_its_reference(
        self, tmp_path: Path
    ) -> None:
        """An HLS playlist named .m4a that points at a local audio file is not followed.
        With ffmpeg 9.0.2 it was not followed even without -f, so this guards against a
        version that probes it differently, rather than showing the pin is needed."""
        canary = tmp_path / "canary.wav"
        _tone(canary, 1.0)
        src = tmp_path / "playlist.m4a"
        src.write_text(f"#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n{canary}\n#EXT-X-ENDLIST\n")

        with pytest.raises(AudioDecodeError):
            _decode(src, tmp_path / "out.wav", "mov")
        assert not (tmp_path / "out.wav").exists()

    def test_concat_script_disguised_as_mp3_does_not_read_its_reference(
        self, tmp_path: Path
    ) -> None:
        """An ffconcat script named .mp3 with a relative reference is not followed under
        the pinned mp3 demuxer. Without -f, ffmpeg 9.0.2 followed it and decoded the
        referenced file, even with -protocol_whitelist file."""
        _tone(tmp_path / "canary.wav", 1.0)
        src = tmp_path / "script.mp3"
        src.write_text("ffconcat version 1.0\nfile 'canary.wav'\n")

        with pytest.raises(AudioDecodeError):
            _decode(src, tmp_path / "out.wav", "mp3")
        assert not (tmp_path / "out.wav").exists()

    def test_timeout_raises_decoder_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A decode that does not finish in time is a server-side failure."""
        src = tmp_path / "in.wav"
        _tone(src, 0.5)

        def timing_out(*args: object, **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=_TIMEOUT_S)

        monkeypatch.setattr(audio_decode.subprocess, "run", timing_out)
        with pytest.raises(DecoderUnavailableError):
            _decode(src, tmp_path / "out.wav", "wav")
        assert [p.name for p in tmp_path.iterdir()] == ["in.wav"]

    def test_missing_ffmpeg_raises_decoder_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing ffmpeg binary is a server-side failure, not a bad upload."""
        src = tmp_path / "in.wav"
        _tone(src, 0.5)

        def not_installed(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("ffmpeg")

        monkeypatch.setattr(audio_decode.subprocess, "run", not_installed)
        with pytest.raises(DecoderUnavailableError):
            _decode(src, tmp_path / "out.wav", "wav")
        assert [p.name for p in tmp_path.iterdir()] == ["in.wav"]

    def test_argv_pins_input_handling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ffmpeg command pins the demuxer and the file protocol, refuses MP4
        external references explicitly, and bounds the output by length and size."""
        captured: list[list[str]] = []

        def capture(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            captured.append(argv)
            return subprocess.CompletedProcess(argv, 1, b"", b"stopped for the test")

        monkeypatch.setattr(audio_decode.subprocess, "run", capture)
        with pytest.raises(AudioDecodeError):
            _decode(tmp_path / "in.m4a", tmp_path / "out.wav", "mov", max_s=600)

        argv = captured[0]
        joined = " ".join(argv)
        assert "-nostdin" in argv
        assert "-protocol_whitelist file" in joined
        assert "-enable_drefs 0" in joined
        assert "-use_absolute_path 0" in joined
        assert argv[argv.index("-i") - 2 : argv.index("-i")] == ["-f", "mov"]
        assert "-t 601" in joined
        assert f"-fs {decoded_size_cap(16_000, 600)}" in joined


def test_decoded_size_cap_covers_two_channels_and_one_extra_second() -> None:
    """The size cap allows the length cap plus one second, at two 16-bit channels."""
    assert decoded_size_cap(16_000, 600) >= 601 * 16_000 * 2 * 2
