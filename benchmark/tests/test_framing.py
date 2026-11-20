"""Tests for cutting a canonical waveform into evaluation frames.

Every waveform here comes from `canonicalise` at the 16 kHz target rate, where the
resampler passes the downmix through unchanged, so a waveform of any chosen length is one
a real 16 kHz mono source would produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.canonicalisation import CanonicalWaveform, canonicalise
from benchmark.framing import frame
from benchmark.protocol import load_protocol
from benchmark.source_evidence import ObservedSourceMetadata
from benchmark.source_inspection import InspectedSource

PROTOCOL = load_protocol()
FRAME = PROTOCOL.framing.frame_samples
RATE = PROTOCOL.canonicalisation.resample.target_sample_rate


def waveform(samples: int) -> CanonicalWaveform:
    """A canonical waveform of the given length, from a 16 kHz mono source."""
    decoded = np.random.default_rng(0).uniform(-0.5, 0.5, (samples, 1))
    metadata = ObservedSourceMetadata(
        container="FLAC", subtype="PCM_16", channels=1, sample_rate=RATE, frames=samples
    )
    result = canonicalise(
        decoded,
        InspectedSource(metadata, samples, 0),
        logical_source_id="candidate-1",
        config=PROTOCOL.canonicalisation,
    )
    assert isinstance(result, CanonicalWaveform), result
    return result


class TestFrameCount:
    """Complete frames from sample 0, and the remainder counted rather than kept."""

    @pytest.mark.parametrize(
        ("samples", "frames", "dropped"),
        [
            pytest.param(3 * FRAME, 3, 0, id="exact-multiple"),
            pytest.param(3 * FRAME + 5, 3, 5, id="remainder"),
            pytest.param(FRAME - 1, 0, FRAME - 1, id="shorter-than-one-frame"),
        ],
    )
    def test_frames_and_dropped_samples(self, samples: int, frames: int, dropped: int) -> None:
        """Every sample is either in a complete frame or counted as dropped."""
        framed = frame(waveform(samples), PROTOCOL.framing)

        assert framed.frames.shape == (frames, FRAME)
        assert framed.dropped_samples == dropped


class TestFrameContents:
    """Row i is exactly samples [i * frame, (i + 1) * frame) of the waveform."""

    def test_each_frame_is_its_slice(self) -> None:
        """No sample is moved, padded, or reordered."""
        canonical = waveform(3 * FRAME + 5)

        framed = frame(canonical, PROTOCOL.framing)

        for index in range(3):
            expected = canonical.samples[index * FRAME : (index + 1) * FRAME]
            assert np.array_equal(framed.frames[index], expected)

    def test_frames_are_read_only_views(self) -> None:
        """Views add no allocation, and the frames cannot be written."""
        canonical = waveform(2 * FRAME)

        frames = frame(canonical, PROTOCOL.framing).frames

        assert np.shares_memory(frames, canonical.samples)
        assert not frames.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            frames[0, 0] = 1.0
