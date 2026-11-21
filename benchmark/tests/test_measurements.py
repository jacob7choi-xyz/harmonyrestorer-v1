"""Tests for the measurement helpers, on small pairs written to a temporary directory.

The measurements themselves read `data/pairs/`, which is not in the repository.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from benchmark.degradation import high_freq_rolloff, tape_saturation
from benchmark.measurements.capacity import parse_result, run_pipeline, write_tone_flac
from benchmark.measurements.d8993b2_equivalence import bit_identical
from benchmark.measurements.training_pairs import (
    expected_hiss_peak,
    hiss_free_signal,
    load_pair,
    sampled_metadata,
    spectral_slope,
)
from benchmark.measurements.worker_exits import CASES, Exit, run_case
from benchmark.protocol import CapacityFixture, load_protocol

RATE = 16000
FRAME = 32000


def write_pairs(root: Path, count: int, saturation: bool = False) -> None:
    """`count` pairs in the layout generate_pairs wrote: metadata, clean, and noisy."""
    for sub in ("metadata", "clean", "noisy"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        stem = f"track__frame_{index:04d}"
        clean = rng.uniform(-0.5, 0.5, FRAME).astype(np.float32)
        sf.write(root / "clean" / f"{stem}.wav", clean, RATE, subtype="FLOAT")
        sf.write(root / "noisy" / f"{stem}__v00.wav", clean * 0.9, RATE, subtype="FLOAT")
        params = {"rolloff_hz": 5000.0, "enable_saturation": saturation, "saturation_drive": 0.3}
        metadata = {"clean_frame": f"{stem}.wav", "variant": 0, "params": params}
        (root / "metadata" / f"{stem}__v00.json").write_text(json.dumps(metadata))


class TestSampling:
    """Pairs are visited in a seeded permutation of sorted filenames."""

    def test_the_order_is_the_seeded_permutation_of_sorted_names(self, tmp_path: Path) -> None:
        write_pairs(tmp_path, 5)
        names = sorted(path.name for path in (tmp_path / "metadata").iterdir())

        visited = [name for name, _ in sampled_metadata(3, tmp_path)]

        assert visited == [names[i] for i in np.random.default_rng(3).permutation(5)]

    def test_a_pair_reads_back_as_float32_with_its_parameters(self, tmp_path: Path) -> None:
        write_pairs(tmp_path, 1)
        name, metadata = next(sampled_metadata(0, tmp_path))

        pair = load_pair(name, metadata, tmp_path)

        assert pair.clean.dtype == np.float32 and pair.noisy.dtype == np.float32
        assert pair.sample_rate == RATE
        assert pair.params["rolloff_hz"] == 5000.0
        np.testing.assert_array_equal(pair.noisy, pair.clean * np.float32(0.9))


class TestHissFreeSignal:
    """The clean frame through the port's saturation, when on, then rolloff."""

    @pytest.mark.parametrize("saturation", [False, True])
    def test_saturation_is_applied_only_when_it_was_on(
        self, tmp_path: Path, saturation: bool
    ) -> None:
        write_pairs(tmp_path, 1, saturation=saturation)
        pair = load_pair(*next(sampled_metadata(0, tmp_path)), tmp_path)

        signal = hiss_free_signal(pair)

        source = tape_saturation(pair.clean, 0.3) if saturation else pair.clean
        assert bit_identical(signal, high_freq_rolloff(source, RATE, 5000.0))


class TestQuantities:
    """The two quantities compared with the training pairs."""

    def test_the_expected_hiss_peak(self) -> None:
        """Power 0.25 at 20 dB SNR: sqrt(0.25 / 100) = 0.05."""
        signal = np.full(FRAME, 0.5, dtype=np.float32)

        assert expected_hiss_peak(signal, 20.0) == pytest.approx(0.05, rel=1e-6)

    @pytest.mark.parametrize("exponent", [1.0, 2.0])
    def test_the_slope_of_a_power_law_spectrum(self, exponent: float) -> None:
        """Power proportional to f ** -exponent inside the band fits -exponent.

        Above the band the spectrum is flat, so a fit that ignored the upper edge would
        not return the in-band exponent.
        """
        freqs = np.fft.rfftfreq(FRAME, 1 / RATE)
        magnitude = np.zeros(freqs.size)
        magnitude[1:] = freqs[1:] ** (-exponent / 2)
        magnitude[freqs > 4000.0] = 1.0
        samples = np.fft.irfft(magnitude, n=FRAME).astype(np.float32)

        slope = spectral_slope(samples, RATE, 100.0, 4000.0)

        assert slope == pytest.approx(-exponent, abs=1e-3)

    def test_bit_identity_needs_dtype_shape_and_bytes(self) -> None:
        a = np.arange(4, dtype=np.float32)

        assert bit_identical(a, a.copy())
        assert not bit_identical(a, a.view(np.int32))
        assert not bit_identical(a, a.reshape(2, 2))
        assert not bit_identical(a, a + np.float32(1e-7))


class TestEngineCall:
    """One engine call is recorded, whether it returns a WAV or raises a job failure."""

    def test_a_successful_call_records_format_and_length(self, tmp_path: Path) -> None:
        from benchmark.measurements.engine_smoke import call

        class Shorter:
            def denoise(self, input_path: Path) -> Path:
                out = tmp_path / "out.wav"
                sf.write(out, np.zeros(FRAME - 130, dtype=np.float32), RATE, subtype="PCM_16")
                return out

        result = call(Shorter(), tmp_path / "frame.wav")

        assert result.error is None
        assert (result.rate, result.subtype, result.channels) == (RATE, "PCM_16", 1)
        assert result.length_difference == -130

    def test_a_job_failure_is_recorded_not_raised(self, tmp_path: Path) -> None:
        from benchmark.measurements.engine_smoke import call

        class Failing:
            def denoise(self, input_path: Path) -> Path:
                raise RuntimeError("no denoised stem")

        result = call(Failing(), tmp_path / "frame.wav")

        assert result.error == "RuntimeError: no denoised stem"

    def test_an_unexpected_exception_stops_the_measurement(self, tmp_path: Path) -> None:
        """Only the engines' failure types are recorded; anything else is a harness defect."""
        from benchmark.measurements.engine_smoke import call

        class Broken:
            def denoise(self, input_path: Path) -> Path:
                raise KeyError("bug")

        with pytest.raises(KeyError):
            call(Broken(), tmp_path / "frame.wav")


class TestCapacityHelpers:
    """The capacity measurement's writer, pipeline, and result parsing on a small fixture."""

    def small_fixture(self) -> CapacityFixture:
        """The frozen fixture's format and tones, at 16 kHz for one second."""
        frozen = load_protocol().runner.capacity_fixture
        return dataclasses.replace(frozen, sample_rate=RATE, frames=RATE)

    def test_the_tones_are_written_in_the_fixture_format(self, tmp_path: Path) -> None:
        path = tmp_path / "small.flac"

        write_tone_flac(path, self.small_fixture())

        info = sf.info(str(path))
        samples, _ = sf.read(path, dtype="float64")
        t = np.arange(RATE) / RATE
        expected = 0.3 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 1234.5 * t)
        assert (info.format, info.subtype, info.channels, info.frames) == (
            "FLAC",
            "PCM_24",
            1,
            RATE,
        )
        np.testing.assert_allclose(samples, expected, atol=2.0**-23)

    def test_the_pipeline_canonicalises_the_fixture(self, tmp_path: Path) -> None:
        path = tmp_path / "small.flac"
        write_tone_flac(path, self.small_fixture())

        result = run_pipeline(path)

        assert result["canonical"] == "CanonicalWaveform"
        assert result["canonical_bytes"] == RATE * 8

    def test_a_result_line_is_parsed_and_its_absence_is_none(self) -> None:
        assert parse_result('noise\nRESULT {"peak_rss": 5}\n') == {"peak_rss": 5}
        assert parse_result("Traceback (most recent call last):\n") is None


class TestWorkerExits:
    """How a child's end reaches the parent, for the cases that are the same on any POSIX."""

    def test_an_exit_code_is_returned(self) -> None:
        assert run_case("print('x'); print('y'); raise SystemExit(3)") == Exit(3, "x\ny\n", "")

    def test_sigkill_is_minus_nine(self) -> None:
        assert run_case(CASES["sigkill"]).returncode == -9
