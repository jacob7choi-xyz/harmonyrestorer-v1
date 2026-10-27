"""Tests for OpGANDenoiserService."""

from __future__ import annotations

import hashlib
import io
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from app.models.chunking import chunk_audio, overlap_add
from app.services.opgan_denoiser import (
    _BATCH_SIZE,
    CheckpointMismatchError,
    OpGANDenoiserService,
)

# Digest of the b"fake" checkpoint bytes the mocked-load tests write
_FAKE_SHA256 = hashlib.sha256(b"fake").hexdigest()


class TestChunkAudio:
    """Tests for the chunk_audio function."""

    def test_short_audio_is_zero_padded(self) -> None:
        """Audio shorter than frame length is zero-padded."""
        audio = np.ones(1000, dtype=np.float32)
        chunks = chunk_audio(audio)

        assert len(chunks) == 1
        start, frame = chunks[0]
        assert start == 0
        assert len(frame) == 32_000
        assert np.all(frame[:1000] == 1.0)
        assert np.all(frame[1000:] == 0.0)

    def test_exact_frame_length(self) -> None:
        """Audio exactly one frame long produces one chunk."""
        audio = np.ones(32_000, dtype=np.float32)
        chunks = chunk_audio(audio)

        assert len(chunks) == 1
        assert chunks[0][0] == 0

    def test_empty_audio_raises(self) -> None:
        """Empty audio raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            chunk_audio(np.array([], dtype=np.float32))

    def test_long_audio_produces_overlapping_chunks(self) -> None:
        """Audio longer than one frame produces multiple overlapping chunks."""
        audio = np.ones(64_000, dtype=np.float32)
        chunks = chunk_audio(audio)

        assert len(chunks) >= 2
        # Second chunk should start before end of first chunk (overlap)
        assert chunks[1][0] < 32_000


class TestOverlapAdd:
    """Tests for the overlap_add function."""

    def test_single_chunk_trimmed(self) -> None:
        """Single chunk is trimmed to original length."""
        frame = np.ones(32_000, dtype=np.float32)
        result = overlap_add([(0, frame)], 1000)

        assert len(result) == 1000
        assert np.all(result == 1.0)

    def test_preserves_constant_signal(self) -> None:
        """Overlap-add of constant signal should remain constant."""
        audio = np.ones(64_000, dtype=np.float32)
        chunks = chunk_audio(audio)
        result = overlap_add(chunks, 64_000)

        np.testing.assert_allclose(result, 1.0, atol=1e-6)


class TestRestoreAudio:
    """Tests for _restore_audio behavior."""

    @patch("app.services.opgan_denoiser.OpGANGenerator")
    @patch("app.services.opgan_denoiser.torch.load")
    def test_output_is_clamped(
        self, mock_load: MagicMock, mock_gen_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Generator output exceeding [-1, 1] is clamped."""
        import torch

        mock_load.return_value = {"generator_state_dict": {}, "epoch": 1}
        mock_gen = mock_gen_cls.return_value
        mock_gen.to.return_value = mock_gen
        mock_gen.eval.return_value = mock_gen

        def amplify(tensor: torch.Tensor) -> torch.Tensor:
            return tensor * 5.0  # push well beyond [-1, 1]

        mock_gen.__call__ = amplify
        mock_gen.side_effect = amplify

        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"fake")
        service = OpGANDenoiserService(
            output_dir=tmp_path, checkpoint_path=checkpoint, expected_sha256=_FAKE_SHA256
        )
        service._generator = mock_gen
        service._device = torch.device("cpu")

        audio = np.random.randn(16_000).astype(np.float32) * 0.5
        result = service._restore_audio(audio)

        assert result.max() <= 1.0
        assert result.min() >= -1.0

    @patch("app.services.opgan_denoiser.OpGANGenerator")
    @patch("app.services.opgan_denoiser.torch.load")
    def test_batched_inference(
        self, mock_load: MagicMock, mock_gen_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Long audio is processed in batches, not one frame at a time."""
        import torch

        mock_load.return_value = {"generator_state_dict": {}, "epoch": 1}
        mock_gen = mock_gen_cls.return_value
        mock_gen.to.return_value = mock_gen
        mock_gen.eval.return_value = mock_gen

        call_shapes: list[tuple[int, ...]] = []

        def track_calls(tensor: torch.Tensor) -> torch.Tensor:
            call_shapes.append(tuple(tensor.shape))
            return tensor

        mock_gen.__call__ = track_calls
        mock_gen.side_effect = track_calls

        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"fake")
        service = OpGANDenoiserService(
            output_dir=tmp_path, checkpoint_path=checkpoint, expected_sha256=_FAKE_SHA256
        )
        service._generator = mock_gen
        service._device = torch.device("cpu")

        # Audio long enough to produce more than _BATCH_SIZE chunks
        samples = (_BATCH_SIZE + 2) * 32_000
        audio = np.random.randn(samples).astype(np.float32) * 0.1
        service._restore_audio(audio)

        # First call should be a full batch
        assert call_shapes[0][0] == _BATCH_SIZE
        # All calls should use batched [B, 1, frame_len] shape
        assert all(s[1] == 1 for s in call_shapes)


class TestOpGANDenoiserInit:
    """Tests for OpGANDenoiserService initialization."""

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        """Output directory is created if it does not exist."""
        output_dir = tmp_path / "new_dir"
        checkpoint = tmp_path / "model.pt"
        service = OpGANDenoiserService(
            output_dir=output_dir, checkpoint_path=checkpoint, expected_sha256=_FAKE_SHA256
        )
        assert output_dir.is_dir()
        assert service.output_dir == output_dir.resolve()

    def test_generator_not_loaded_on_init(self, tmp_path: Path) -> None:
        """Generator is not loaded until denoise is called."""
        service = OpGANDenoiserService(
            output_dir=tmp_path, checkpoint_path=tmp_path / "model.pt", expected_sha256=_FAKE_SHA256
        )
        assert service._generator is None


class TestGetGenerator:
    """Tests for lazy model loading in _get_generator."""

    def test_generator_remains_none_after_failed_load(self, tmp_path: Path) -> None:
        """_generator stays None when checkpoint loading fails so callers can retry.

        If _generator were set before the load completes, a crash mid-load would
        cache bad state and prevent any recovery without a server restart.
        This proves the double-checked locking pattern only sets _generator
        after a fully successful load.
        """
        service = OpGANDenoiserService(
            output_dir=tmp_path / "output",
            checkpoint_path=tmp_path / "nonexistent.pt",
            expected_sha256=_FAKE_SHA256,
        )

        with pytest.raises(FileNotFoundError):
            service._get_generator()

        assert service._generator is None, (
            "_generator must stay None after a failed load so the next call retries"
        )


class TestCheckpointDigest:
    """Tests for the checkpoint digest check in _get_generator."""

    def test_mismatched_checkpoint_is_rejected_before_loading(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A checkpoint whose digest differs is never passed to torch.load.

        In a scratch run, a checkpoint with the same keys and different weights loaded
        without a warning and produced different audio, so the check compares digests.
        """
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"some other checkpoint")
        expected = hashlib.sha256(b"the expected checkpoint").hexdigest()
        service = OpGANDenoiserService(
            output_dir=tmp_path, checkpoint_path=checkpoint, expected_sha256=expected
        )

        with (
            patch("app.services.opgan_denoiser.torch.load") as mock_load,
            caplog.at_level(logging.ERROR, logger="app.services.opgan_denoiser"),
            pytest.raises(CheckpointMismatchError),
        ):
            service._get_generator()

        mock_load.assert_not_called()
        assert service._generator is None
        assert expected in caplog.text
        assert hashlib.sha256(b"some other checkpoint").hexdigest() in caplog.text

    @patch("app.services.opgan_denoiser.OpGANGenerator")
    @patch("app.services.opgan_denoiser.torch.load")
    def test_matching_checkpoint_loads_the_bytes_that_were_hashed(
        self, mock_load: MagicMock, mock_gen_cls: MagicMock, tmp_path: Path
    ) -> None:
        """torch.load receives the bytes that were hashed, not the path read again."""
        mock_load.return_value = {"generator_state_dict": {}, "epoch": 1}
        mock_gen = mock_gen_cls.return_value
        mock_gen.load_state_dict.return_value = MagicMock(missing_keys=[], unexpected_keys=[])
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"the expected checkpoint")
        service = OpGANDenoiserService(
            output_dir=tmp_path,
            checkpoint_path=checkpoint,
            expected_sha256=hashlib.sha256(b"the expected checkpoint").hexdigest(),
        )

        service._get_generator()

        source = mock_load.call_args.args[0]
        assert isinstance(source, io.BytesIO)
        assert source.getvalue() == b"the expected checkpoint"
        assert service._generator is mock_gen


class TestOpGANDenoise:
    """Tests for the denoise method."""

    def _make_wav(self, path: Path, sr: int = 16_000, duration: float = 1.0) -> None:
        """Write a short WAV file for testing."""
        samples = np.random.randn(int(sr * duration)).astype(np.float32) * 0.1
        sf.write(str(path), samples, sr)

    @patch("app.services.opgan_denoiser.OpGANGenerator")
    @patch("app.services.opgan_denoiser.torch.load")
    def test_returns_output_path(
        self, mock_load: MagicMock, mock_gen_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Denoise returns a valid output path."""
        import torch

        # Set up mock checkpoint
        mock_load.return_value = {
            "generator_state_dict": {},
            "epoch": 10,
        }

        # Set up mock generator that returns same-shape tensor
        mock_gen = mock_gen_cls.return_value
        mock_gen.to.return_value = mock_gen
        mock_gen.eval.return_value = mock_gen

        def fake_forward(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        mock_gen.__call__ = fake_forward
        mock_gen.side_effect = fake_forward

        # Create checkpoint file
        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"fake")

        output_dir = tmp_path / "output"
        service = OpGANDenoiserService(
            output_dir=output_dir, checkpoint_path=checkpoint, expected_sha256=_FAKE_SHA256
        )

        input_path = tmp_path / "test.wav"
        self._make_wav(input_path)

        result = service.denoise(input_path)
        assert result.exists()
        assert result.is_relative_to(output_dir.resolve())
        assert result.suffix == ".wav"

    def test_missing_checkpoint_raises(self, tmp_path: Path) -> None:
        """FileNotFoundError raised when checkpoint doesn't exist."""
        service = OpGANDenoiserService(
            output_dir=tmp_path,
            checkpoint_path=tmp_path / "missing.pt",
            expected_sha256=_FAKE_SHA256,
        )

        input_path = tmp_path / "test.wav"
        self._make_wav(input_path)

        with pytest.raises(FileNotFoundError, match="checkpoint not found"):
            service.denoise(input_path)

    @patch("app.services.opgan_denoiser.OpGANGenerator")
    @patch("app.services.opgan_denoiser.torch.load")
    def test_resamples_non_16khz_input(
        self, mock_load: MagicMock, mock_gen_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Input at non-16kHz sample rate is resampled."""
        import torch

        mock_load.return_value = {
            "generator_state_dict": {},
            "epoch": 10,
        }

        mock_gen = mock_gen_cls.return_value
        mock_gen.to.return_value = mock_gen
        mock_gen.eval.return_value = mock_gen

        def fake_forward(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        mock_gen.__call__ = fake_forward
        mock_gen.side_effect = fake_forward

        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"fake")

        output_dir = tmp_path / "output"
        service = OpGANDenoiserService(
            output_dir=output_dir, checkpoint_path=checkpoint, expected_sha256=_FAKE_SHA256
        )

        # Write 44.1kHz input
        input_path = tmp_path / "test_44k.wav"
        self._make_wav(input_path, sr=44_100, duration=1.0)

        result = service.denoise(input_path)
        assert result.exists()

        # Output should be at 16kHz
        info = sf.info(str(result))
        assert info.samplerate == 16_000

    @patch("app.services.opgan_denoiser.OpGANGenerator")
    @patch("app.services.opgan_denoiser.torch.load")
    def test_handles_stereo_input(
        self, mock_load: MagicMock, mock_gen_cls: MagicMock, tmp_path: Path
    ) -> None:
        """Stereo input is converted to mono."""
        import torch

        mock_load.return_value = {
            "generator_state_dict": {},
            "epoch": 10,
        }

        mock_gen = mock_gen_cls.return_value
        mock_gen.to.return_value = mock_gen
        mock_gen.eval.return_value = mock_gen

        def fake_forward(tensor: torch.Tensor) -> torch.Tensor:
            return tensor

        mock_gen.__call__ = fake_forward
        mock_gen.side_effect = fake_forward

        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"fake")

        output_dir = tmp_path / "output"
        service = OpGANDenoiserService(
            output_dir=output_dir, checkpoint_path=checkpoint, expected_sha256=_FAKE_SHA256
        )

        # Write stereo input
        input_path = tmp_path / "stereo.wav"
        stereo = np.random.randn(16_000, 2).astype(np.float32) * 0.1
        sf.write(str(input_path), stereo, 16_000)

        result = service.denoise(input_path)
        assert result.exists()

        # Output should be mono
        info = sf.info(str(result))
        assert info.channels == 1
