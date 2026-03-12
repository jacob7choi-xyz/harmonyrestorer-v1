"""Tests for chunking and overlap-add logic in dataset.infer."""

from __future__ import annotations

import numpy as np
import pytest

from dataset.infer import _FRAME_LEN, _OVERLAP, _chunk_audio, _overlap_add

_STEP = _FRAME_LEN - _OVERLAP


class TestChunkAudio:
    """Tests for _chunk_audio frame-splitting logic."""

    def test_short_audio_returns_single_padded_chunk(self) -> None:
        """Audio shorter than _FRAME_LEN produces one zero-padded chunk."""
        audio = np.ones(100, dtype=np.float32)
        chunks = _chunk_audio(audio)

        assert len(chunks) == 1
        start, frame = chunks[0]
        assert start == 0
        assert len(frame) == _FRAME_LEN
        np.testing.assert_array_equal(frame[:100], audio)
        np.testing.assert_array_equal(frame[100:], 0.0)

    def test_exact_frame_length_returns_single_chunk(self) -> None:
        """Audio exactly _FRAME_LEN long produces one chunk with no padding."""
        audio = np.random.default_rng(42).standard_normal(_FRAME_LEN).astype(np.float32)
        chunks = _chunk_audio(audio)

        assert len(chunks) == 1
        start, frame = chunks[0]
        assert start == 0
        np.testing.assert_array_equal(frame, audio)

    def test_two_frames_exactly(self) -> None:
        """Audio of length _FRAME_LEN + _STEP produces exactly two chunks."""
        length = _FRAME_LEN + _STEP
        audio = np.ones(length, dtype=np.float32)
        chunks = _chunk_audio(audio)

        assert len(chunks) == 2
        assert chunks[0][0] == 0
        assert chunks[1][0] == _STEP

    def test_multi_frame_correct_count_and_starts(self) -> None:
        """Multiple frames have consecutive start indices separated by _STEP."""
        length = _FRAME_LEN + 3 * _STEP
        audio = np.ones(length, dtype=np.float32)
        chunks = _chunk_audio(audio)

        starts = [s for s, _ in chunks]
        for i in range(1, len(starts)):
            assert starts[i] == starts[i - 1] + _STEP

    def test_overlap_between_consecutive_chunks(self) -> None:
        """Consecutive chunks overlap by exactly _OVERLAP samples."""
        length = _FRAME_LEN + 2 * _STEP
        audio = np.ones(length, dtype=np.float32)
        chunks = _chunk_audio(audio)

        for i in range(len(chunks) - 1):
            start_a = chunks[i][0]
            start_b = chunks[i + 1][0]
            end_a = start_a + _FRAME_LEN
            overlap = end_a - start_b
            assert overlap == _OVERLAP

    def test_short_tail_is_dropped(self) -> None:
        """A tail shorter than _OVERLAP is dropped (covered by previous chunk)."""
        # Two full chunks fit at starts 0 and _STEP. The next start is 2*_STEP.
        # remaining = length - 2*_STEP must be <= _OVERLAP for the tail to be dropped.
        length = 2 * _STEP + _OVERLAP
        audio = np.ones(length, dtype=np.float32)
        chunks = _chunk_audio(audio)

        # Should only have 2 chunks -- the short tail is skipped
        assert len(chunks) == 2

    def test_tail_longer_than_overlap_gets_padded_chunk(self) -> None:
        """A tail longer than _OVERLAP gets its own zero-padded chunk."""
        length = _FRAME_LEN + _STEP + _OVERLAP + 100
        audio = np.ones(length, dtype=np.float32)
        chunks = _chunk_audio(audio)

        assert len(chunks) == 3
        # Last chunk should be zero-padded to _FRAME_LEN
        _, last_frame = chunks[-1]
        assert len(last_frame) == _FRAME_LEN

    def test_empty_audio_raises_value_error(self) -> None:
        """Empty audio array raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            _chunk_audio(np.array([], dtype=np.float32))

    def test_all_chunks_have_correct_length(self) -> None:
        """Every chunk is exactly _FRAME_LEN samples, even for long audio."""
        length = _FRAME_LEN * 10 + 5000
        audio = np.random.default_rng(7).standard_normal(length).astype(np.float32)
        chunks = _chunk_audio(audio)

        for _, frame in chunks:
            assert len(frame) == _FRAME_LEN

    def test_single_sample_audio(self) -> None:
        """A single-sample audio produces one padded chunk."""
        audio = np.array([0.5], dtype=np.float32)
        chunks = _chunk_audio(audio)

        assert len(chunks) == 1
        start, frame = chunks[0]
        assert start == 0
        assert frame[0] == 0.5
        assert np.all(frame[1:] == 0.0)


class TestOverlapAdd:
    """Tests for _overlap_add reassembly logic."""

    def test_single_chunk_returns_trimmed(self) -> None:
        """A single chunk is trimmed to original_length."""
        original_length = 500
        frame = np.ones(_FRAME_LEN, dtype=np.float32)
        result = _overlap_add([(0, frame)], original_length)

        assert len(result) == original_length
        np.testing.assert_array_equal(result, 1.0)

    def test_single_chunk_returns_copy(self) -> None:
        """Single-chunk result is a copy, not a view of the input."""
        frame = np.ones(_FRAME_LEN, dtype=np.float32)
        result = _overlap_add([(0, frame)], 100)
        result[0] = 999.0
        assert frame[0] == 1.0

    def test_identity_roundtrip_sine_wave(self) -> None:
        """Chunking then overlap-add with identity processing reconstructs original signal."""
        rng = np.random.default_rng(42)
        length = _FRAME_LEN * 3 + 5000
        t = np.linspace(0, 1, length, dtype=np.float32)
        original = np.sin(2 * np.pi * 440 * t) + 0.3 * rng.standard_normal(length).astype(
            np.float32
        )

        chunks = _chunk_audio(original)
        # Identity "processing" -- pass chunks through unchanged
        result = _overlap_add(chunks, length)

        # In non-overlap regions the signal should be exact; in crossfade
        # regions the weighted average of identical data should still match.
        np.testing.assert_allclose(result, original[:length], atol=1e-5)

    def test_output_length_matches_original(self) -> None:
        """Output length equals the requested original_length."""
        length = _FRAME_LEN * 2 + 1234
        audio = np.ones(length, dtype=np.float32)
        chunks = _chunk_audio(audio)
        result = _overlap_add(chunks, length)

        assert len(result) == length

    def test_first_chunk_no_fade_in(self) -> None:
        """The first chunk's leading samples are not attenuated by a fade-in."""
        length = _FRAME_LEN + _STEP
        # Use a constant signal so any fade would be visible
        audio = np.ones(length, dtype=np.float32)
        chunks = _chunk_audio(audio)
        result = _overlap_add(chunks, length)

        # First few samples (well before any overlap) should be exactly 1.0
        np.testing.assert_allclose(result[:_STEP], 1.0, atol=1e-6)

    def test_last_chunk_no_fade_out(self) -> None:
        """The last chunk's trailing samples are not attenuated by a fade-out."""
        length = _FRAME_LEN + _STEP
        audio = np.ones(length, dtype=np.float32)
        chunks = _chunk_audio(audio)
        result = _overlap_add(chunks, length)

        # Last samples (after the overlap region) should be exactly 1.0
        np.testing.assert_allclose(result[_FRAME_LEN:], 1.0, atol=1e-6)

    def test_constant_signal_energy_preserved(self) -> None:
        """A constant signal is perfectly reconstructed (no gaps or double-counting)."""
        length = _FRAME_LEN * 5 + 7777
        audio = np.full(length, 0.75, dtype=np.float32)
        chunks = _chunk_audio(audio)
        result = _overlap_add(chunks, length)

        np.testing.assert_allclose(result, 0.75, atol=1e-5)

    def test_multi_chunk_signal_energy(self) -> None:
        """Total energy is preserved through chunk-and-reassemble cycle."""
        length = _FRAME_LEN * 4
        t = np.linspace(0, 1, length, dtype=np.float32)
        original = np.sin(2 * np.pi * 100 * t)

        chunks = _chunk_audio(original)
        result = _overlap_add(chunks, length)

        original_energy = np.sum(original**2)
        result_energy = np.sum(result**2)
        # Energy should be very close (within 1%)
        np.testing.assert_allclose(result_energy, original_energy, rtol=0.01)
