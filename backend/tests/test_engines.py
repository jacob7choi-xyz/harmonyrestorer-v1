"""Tests for the per-engine peak output bytes."""

import pytest
from app.config import DenoiserEngine
from app.services.engines import peak_output_bytes


@pytest.mark.parametrize("duration_s", [1, 600])
def test_opgan_peak_matches_the_previous_fixed_cap(duration_s: int) -> None:
    """OpGAN keeps the reservation used before engines declared their own:
    two 16 kHz mono 16-bit WAVs, each with 1024 bytes of header slack."""
    previous_cap = 2 * (duration_s * 16_000 * 2 + 1024)
    assert peak_output_bytes(DenoiserEngine.OPGAN, duration_s) == previous_cap


def test_uvr_peak_covers_the_measured_run() -> None:
    """UVR's figure covers the peak measured on a real 65.65 s PCM_32 input.

    81,056,990 bytes were present at once across the output directory and the
    system temp directory: both stems plus pydub's two staging files.
    """
    measured_peak_bytes = 81_056_990
    assert peak_output_bytes(DenoiserEngine.UVR, 66) >= measured_peak_bytes
