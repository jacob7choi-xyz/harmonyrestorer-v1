"""Tests for the measurement helpers, on small pairs written to a temporary directory.

The measurements themselves read `data/pairs/`, which is not in the repository.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import librosa
import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from benchmark.degradation import high_freq_rolloff, tape_saturation
from benchmark.measurements.capacity import parse_result, run_pipeline, write_tone_flac
from benchmark.measurements.d8993b2_equivalence import bit_identical
from benchmark.measurements.dead_worker_footprint import Ended
from benchmark.measurements.dead_worker_footprint import run_case as run_footprint_case
from benchmark.measurements.training_pairs import (
    expected_hiss_peak,
    hiss_free_signal,
    load_pair,
    sampled_metadata,
    spectral_slope,
)
from benchmark.measurements.training_screen import (
    TRANSFORMS,
    Corpus,
    Degenerate,
    load_recording,
    offsets,
    report,
    similarity,
    transformed,
    unit_features,
)
from benchmark.measurements.worker_exits import CASES, Exit, run_case
from benchmark.protocol import CapacityFixture, MfccConfig, load_protocol

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


class TestDeadWorkerFootprint:
    """The parent reads how a child ended, what it printed, and on macOS its footprint."""

    def test_the_exit_code_and_output_are_returned(self) -> None:
        ended = run_footprint_case("print('x')\nraise SystemExit(3)")
        assert (ended.returncode, ended.stdout) == (3, "x")

    @pytest.mark.skipif(sys.platform != "darwin", reason="footprint is macOS only")
    def test_a_killed_child_has_a_footprint(self) -> None:
        code = "import numpy as np, os, signal\na = np.ones(100_000_000 // 8)\nos.kill(os.getpid(), signal.SIGKILL)"
        ended = run_footprint_case(code)
        assert ended.returncode == -9
        assert ended.footprint is not None
        assert ended.footprint >= 100_000_000

    @pytest.mark.skipif(sys.platform == "darwin", reason="footprint is read on macOS")
    def test_no_footprint_is_claimed_off_macos(self) -> None:
        assert run_footprint_case("pass") == Ended(0, None, "")


class TestTrainingScreen:
    """The screen's building blocks on synthetic features and a small FLAC."""

    @pytest.fixture(scope="class")
    def config(self) -> MfccConfig:
        """The frozen MFCC configuration."""
        return load_protocol().duplicate_detection.mfcc

    @staticmethod
    def random_unit(frames: int, seed: int) -> npt.NDArray[np.float64]:
        """Unit-length random frames, standing in for normalised MFCCs."""
        values = np.random.default_rng(seed).standard_normal((frames, 20))
        unit: npt.NDArray[np.float64] = values / np.linalg.norm(values, axis=1, keepdims=True)
        return unit

    def test_offsets_are_multiples_of_the_stride_within_the_search(
        self, config: MfccConfig
    ) -> None:
        lags = offsets(config)
        assert (lags[0], lags[-1], len(lags)) == (-156, 156, 157)
        assert 0 in lags
        assert all(lag % config.alignment_stride_frames == 0 for lag in lags)

    def test_an_odd_reach_stops_at_the_last_multiple_of_the_stride(
        self, config: MfccConfig
    ) -> None:
        """At hop 512 and 16 kHz, 4.97 s reaches 155.3 frames: rounded down to 155, and 155 is
        not a multiple of 2, so the last offset is 154."""
        lags = offsets(dataclasses.replace(config, alignment_search_s=4.97))
        assert (lags[0], lags[-1]) == (-154, 154)

    def test_identical_features_score_one(self) -> None:
        a = self.random_unit(500, 0)
        assert similarity(a, a, [0], 100) == pytest.approx(1.0)

    def test_a_shift_inside_the_search_is_found(self, config: MfccConfig) -> None:
        a = self.random_unit(800, 1)
        assert similarity(a, a[40:], offsets(config), 100) == pytest.approx(1.0)
        assert similarity(a[40:], a, offsets(config), 100) == pytest.approx(1.0)

    def test_a_shift_outside_the_search_is_not_found(self, config: MfccConfig) -> None:
        a = self.random_unit(800, 2)
        score = similarity(a, a[200:], offsets(config), 100)
        assert score is not None
        assert score < 0.2

    def test_too_little_overlap_gives_no_score(self) -> None:
        a = self.random_unit(99, 3)
        assert similarity(a, a, [0], 100) is None

    def test_features_are_unit_frames_of_z_normalised_coefficients(
        self, config: MfccConfig
    ) -> None:
        t = np.arange(RATE * 2) / RATE
        waveform = 0.3 * np.sin(2 * np.pi * 440.0 * t * (1.0 + t))
        unit, degenerate = unit_features(waveform, config)
        assert unit.shape[1] == config.n_mfcc
        np.testing.assert_allclose(np.linalg.norm(unit, axis=1), 1.0)
        assert degenerate == Degenerate(0, 0)

    def test_features_match_a_direct_computation(self, config: MfccConfig) -> None:
        """Centred and scaled per coefficient, then each frame scaled to unit length."""
        waveform = np.random.default_rng(5).uniform(-0.5, 0.5, RATE * 2)
        mfcc = librosa.feature.mfcc(y=waveform, sr=RATE, n_mfcc=20, hop_length=512)
        z = ((mfcc - mfcc.mean(axis=1, keepdims=True)) / mfcc.std(axis=1, keepdims=True)).T
        expected = z / np.linalg.norm(z, axis=1, keepdims=True)
        np.testing.assert_allclose(unit_features(waveform, config)[0], expected, atol=1e-12)

    def test_silence_is_counted_as_degenerate_rather_than_divided_by_zero(
        self, config: MfccConfig
    ) -> None:
        unit, degenerate = unit_features(np.zeros(RATE), config)
        assert not np.isnan(unit).any()
        assert degenerate.zero_variance_coefficients == config.n_mfcc
        assert degenerate.zero_norm_frames == len(unit)

    def test_the_transforms_trim_attenuate_and_round_trip(self) -> None:
        protocol = load_protocol()
        waveform = np.random.default_rng(4).uniform(-0.5, 0.5, RATE * 5)
        assert len(transformed(waveform, "trim_3_s", protocol)) == RATE * 2
        np.testing.assert_allclose(
            transformed(waveform, "gain_minus_6_db", protocol), waveform * 10.0 ** (-6.0 / 20.0)
        )
        assert len(transformed(waveform, "round_trip_8_khz", protocol)) == RATE * 5

    def test_a_recording_gives_its_decoded_digest_and_canonical_waveform(
        self, tmp_path: Path
    ) -> None:
        protocol = load_protocol()
        t = np.arange(RATE) / RATE
        stereo = np.stack(
            [0.3 * np.sin(2 * np.pi * 440.0 * t), 0.2 * np.sin(2 * np.pi * 660.0 * t)], 1
        )
        path = tmp_path / "pair.flac"
        sf.write(path, stereo, RATE, format="FLAC", subtype="PCM_24")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()

        loaded = load_recording(path, name="pair.flac", expected_sha256=digest, protocol=protocol)

        assert not isinstance(loaded, str)
        decoded, _ = sf.read(path, dtype="float64", always_2d=True)
        assert loaded[0] == hashlib.sha256(decoded.astype("<f8").tobytes()).hexdigest()
        np.testing.assert_allclose(loaded[1], decoded.mean(axis=1), atol=1e-12)

    def test_a_recording_that_differs_from_its_digest_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "tones.flac"
        sf.write(path, np.zeros(RATE), RATE, format="FLAC", subtype="PCM_24")
        loaded = load_recording(
            path, name="tones.flac", expected_sha256="0" * 64, protocol=load_protocol()
        )
        assert isinstance(loaded, str)
        assert "SOURCE_HASH_MISMATCH" in loaded

    def test_a_run_in_which_every_recording_failed_still_reports_the_failures(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No positive and no pair is scored, so there is no minimum or maximum to print."""
        corpus = Corpus(
            [], {}, [], {t: [] for t in TRANSFORMS}, Counter(), {"a.flac": "decode gave X"}
        )

        report(corpus, [], load_protocol())

        printed = capsys.readouterr().out
        assert "failed a.flac: decode gave X" in printed
        assert "min=none" in printed
        assert "negatives: n=0, unscored=0, max=none" in printed
