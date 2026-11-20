"""Evaluation frames cut from a canonical waveform, and the remainder that is counted.

v4 cuts frames of `frame_samples` from the canonical waveform with a hop of the same
length, starting at sample 0, and drops a trailing remainder shorter than one frame while
recording its sample count. The loader refuses any other hop, start, or remainder policy
(`protocol._validate_framing`), which is why a reshape of the whole-frame prefix is the
complete operation here rather than a general windowing routine.

**Frames are views, not copies.** The canonical waveform is read-only and owns its memory,
so a view of it cannot be written and cannot change under the digest that identifies the
waveform. Viewing also adds no allocation, where copying would add up to the size of the
waveform.

This module cuts frames only. Frame RMS and the eligibility verdict are computed later,
from these frames. Protocol v5 identifies a frame by its recording's canonical SHA-256
and its row index, both of which the caller already holds, so no identifier is stored
here. Frames are to be cut only after partition assignment; that ordering belongs to the
caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from benchmark.canonicalisation import CanonicalWaveform
from benchmark.protocol import Framing

__all__ = ["FramedWaveform", "frame"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FramedWaveform:
    """The complete frames of one canonical waveform and the samples left over.

    `frames` has shape (frame_count, frame_samples) and is a read-only view of the
    canonical samples; row `i` holds samples `[i * frame_samples, (i + 1) *
    frame_samples)`. `dropped_samples` is the length of the trailing remainder, which is
    not a candidate frame and is recorded so the manifest shows how the frames were formed.
    """

    frames: npt.NDArray[np.float64]
    dropped_samples: int


def frame(waveform: CanonicalWaveform, config: Framing) -> FramedWaveform:
    """Cut the canonical waveform into v4's evaluation frames.

    Args:
        waveform: A canonical waveform, read-only and one-dimensional.
        config: The frozen `framing` section.

    Returns:
        The complete frames and the number of trailing samples dropped.
    """
    samples = waveform.samples
    frame_count = samples.size // config.frame_samples
    complete = frame_count * config.frame_samples
    # A view of the read-only waveform, so it is read-only too and copies nothing.
    frames = samples[:complete].reshape(frame_count, config.frame_samples)
    dropped = samples.size - complete
    logger.debug("Framed %d samples: %d frames, %d dropped", samples.size, frame_count, dropped)
    return FramedWaveform(frames, dropped)
