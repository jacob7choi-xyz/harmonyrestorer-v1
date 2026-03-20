"""Audio chunking and overlap-add reconstruction for OpGAN inference.

Splits audio into fixed-length overlapping frames for model processing,
then reassembles the results using weighted overlap-add with linear crossfade.
"""

from __future__ import annotations

import numpy as np

FRAME_LEN = 32_000  # 2 seconds at 16kHz
OVERLAP = 1600  # 100ms overlap for crossfade


def chunk_audio(audio: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Split audio into overlapping frames for processing.

    Args:
        audio: 1D float32 array of audio samples.

    Returns:
        List of (start_index, frame) tuples.

    Raises:
        ValueError: If audio is empty.
    """
    total = len(audio)
    if total == 0:
        raise ValueError("Audio is empty (zero samples)")

    if total <= FRAME_LEN:
        padded = np.zeros(FRAME_LEN, dtype=np.float32)
        padded[:total] = audio
        return [(0, padded)]

    step = FRAME_LEN - OVERLAP
    chunks = []

    for start in range(0, total, step):
        end = start + FRAME_LEN
        if end <= total:
            chunks.append((start, audio[start:end]))
        else:
            remaining = total - start
            if remaining <= OVERLAP and chunks:
                # Tail is too short to justify a new chunk -- the previous
                # chunk's overlap already covers this region
                break
            padded = np.zeros(FRAME_LEN, dtype=np.float32)
            padded[:remaining] = audio[start:]
            chunks.append((start, padded))
            break

    return chunks


def overlap_add(chunks: list[tuple[int, np.ndarray]], original_length: int) -> np.ndarray:
    """Reassemble processed chunks using overlap-add with linear crossfade.

    Args:
        chunks: List of (start_index, processed_frame) tuples.
        original_length: Length of the original audio to trim to.

    Returns:
        Reconstructed audio as 1D float32 array.
    """
    if len(chunks) == 1:
        start, frame = chunks[0]
        return frame[:original_length].copy()

    output = np.zeros(original_length, dtype=np.float32)
    weights = np.zeros(original_length, dtype=np.float32)
    last_idx = len(chunks) - 1

    for i, (start, frame) in enumerate(chunks):
        end = min(start + len(frame), original_length)
        length = end - start

        window = np.ones(length, dtype=np.float32)
        if OVERLAP > 0:
            fade_len = min(OVERLAP, length)
            if i > 0:
                window[:fade_len] *= np.linspace(0, 1, fade_len, dtype=np.float32)
            if i < last_idx:
                window[-fade_len:] *= np.linspace(1, 0, fade_len, dtype=np.float32)

        output[start:end] += frame[:length] * window
        weights[start:end] += window

    nonzero = weights > 0
    output[nonzero] /= weights[nonzero]

    return output
