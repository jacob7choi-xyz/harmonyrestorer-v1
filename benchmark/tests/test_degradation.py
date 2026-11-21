"""Tests for the port of the noise process that generated the training pairs."""

from __future__ import annotations

import dataclasses
import hashlib
import struct
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from benchmark.degradation import (
    DegradationParams,
    apply_degradation,
    degrade_frame,
    draw_params,
    frame_seed,
    golden_degraded_sha256,
    high_freq_rolloff,
    mains_hum,
    tape_hiss,
    tape_saturation,
    vinyl_crackle,
)
from benchmark.golden_fixture import platform_token
from benchmark.protocol import load_protocol

RATE = 16000
FRAME = 32000
# Not byte-symmetric, so reading its words in the wrong byte order changes them.
DIGEST = hashlib.sha256(b"degradation").hexdigest()
PROTOCOL = load_protocol()
TRAINING = PROTOCOL.degradation.ranges
# Every optional effect off and the cutoff at Nyquist, so only hiss can change the signal.
HISS_ONLY = DegradationParams(
    hiss_snr_db=20.0,
    hiss_color="pink",
    rolloff_hz=8000.0,
    enable_crackle=False,
    crackle_density=0.001,
    crackle_amplitude=0.05,
    enable_hum=False,
    hum_amplitude=0.01,
    hum_freq=60.0,
    enable_saturation=False,
    saturation_drive=0.3,
)


def music(seed: int, peak: float) -> npt.NDArray[np.float64]:
    """A band-limited test frame with the given peak."""
    rng = np.random.default_rng(seed)
    t = np.arange(FRAME) / RATE
    tones = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) for f in (220.0, 440.0, 990.0))
    frame: npt.NDArray[np.float64] = peak * tones / np.max(np.abs(tones))
    return frame


def generator(index: int = 0) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(frame_seed(1, DIGEST, index)))


class TestDrawOrder:
    """The parameters are drawn in random_degradation's order from one generator."""

    def test_draws_follow_the_training_order(self) -> None:
        """Replayed call by call, in the order random_degradation's keywords appear.

        The literal values are d8993b2's, written out independently of the protocol's ranges
        so this test also checks them.
        """
        replay = generator()
        expected = [
            replay.uniform(5.0, 30.0),
            str(replay.choice(["pink", "brown"])),
            replay.uniform(4000.0, 8000.0),
            bool(replay.random() > 0.5),
            replay.uniform(0.0002, 0.002),
            replay.uniform(0.01, 0.1),
            bool(replay.random() > 0.7),
            replay.uniform(0.005, 0.05),
            replay.choice([50.0, 60.0]),
            bool(replay.random() > 0.6),
            replay.uniform(0.1, 0.6),
        ]

        params = draw_params(generator(), TRAINING)

        assert list(dataclasses.astuple(params)) == expected


class TestHissScaling:
    """The property measured on the training pairs holds for the port."""

    @pytest.mark.parametrize("color", ["pink", "brown"])
    def test_the_added_hiss_peaks_at_the_d8993b2_scale(self, color: str) -> None:
        """Unit peak scaled by sqrt(signal power / SNR): peak equals that factor."""
        clean = music(1, 0.5).astype(np.float32)
        params = dataclasses.replace(HISS_ONLY, hiss_color=color)

        added = apply_degradation(clean, RATE, params, generator()) - clean

        expected_peak = np.sqrt(np.mean(clean**2) / 10 ** (params.hiss_snr_db / 10))
        assert np.max(np.abs(added)) == pytest.approx(expected_peak, rel=1e-5)

    def test_unit_peak_hiss_has_less_than_unit_power(self) -> None:
        """Why the training hiss was quieter than recorded."""
        hiss = tape_hiss(FRAME, RATE, "pink", generator())

        assert np.max(np.abs(hiss)) == pytest.approx(1.0)
        assert np.mean(hiss.astype(np.float64) ** 2) < 0.2

    def test_an_unknown_color_is_refused(self) -> None:
        with pytest.raises(ValueError, match="hiss color must be one of"):
            tape_hiss(FRAME, RATE, "white", generator())


class TestLevel:
    """Frames are degraded at training level, set by the recording's peak."""

    def test_the_gain_brings_the_recording_peak_to_the_headroom(self) -> None:
        frame = music(2, 0.4)

        result = degrade_frame(
            frame,
            recording_peak=0.4,
            sample_rate=RATE,
            ranges=TRAINING,
            seed=frame_seed(1, DIGEST, 0),
        )

        assert result.gain == pytest.approx(0.95 / 0.4)

    def test_the_same_frame_at_any_level_degrades_identically_at_training_level(self) -> None:
        """Crackle and hum have fixed amplitudes; scaling to training level first makes
        their loudness relative to the music independent of the recording's level."""
        quiet, loud = music(3, 0.2), music(3, 0.8)
        seed = frame_seed(1, DIGEST, 0)

        a = degrade_frame(quiet, recording_peak=0.2, sample_rate=RATE, ranges=TRAINING, seed=seed)
        b = degrade_frame(loud, recording_peak=0.8, sample_rate=RATE, ranges=TRAINING, seed=seed)

        assert a.params.enable_crackle and a.params.enable_hum
        np.testing.assert_allclose(a.degraded, b.degraded, atol=1e-6)

    @pytest.mark.parametrize("peak", [0.0, float("nan")])
    def test_a_peak_that_sets_no_level_is_refused(self, peak: float) -> None:
        with pytest.raises(ValueError, match="recording_peak must be finite and positive"):
            degrade_frame(
                music(4, 0.5),
                recording_peak=peak,
                sample_rate=RATE,
                ranges=TRAINING,
                seed=frame_seed(1, DIGEST, 0),
            )

    def test_the_result_is_read_only_float32_within_full_scale(self) -> None:
        result = degrade_frame(
            music(5, 0.9),
            recording_peak=0.9,
            sample_rate=RATE,
            ranges=TRAINING,
            seed=frame_seed(1, DIGEST, 3),
        )

        assert result.degraded.dtype == np.float32
        assert np.max(np.abs(result.degraded)) <= 1.0
        with pytest.raises(ValueError, match="read-only"):
            result.degraded[0] = 0


class TestEffects:
    """Optional effects change the signal only when drawn on."""

    def test_crackle_is_added_only_when_enabled(self) -> None:
        clean = music(6, 0.5).astype(np.float32)
        quiet_hiss = dataclasses.replace(HISS_ONLY, hiss_snr_db=200.0)
        on = dataclasses.replace(quiet_hiss, enable_crackle=True)

        without = apply_degradation(clean, RATE, quiet_hiss, generator())
        with_crackle = apply_degradation(clean, RATE, on, generator())

        assert np.max(np.abs(without - clean)) < 1e-6
        assert np.max(np.abs(with_crackle - clean)) > 1e-3

    def test_the_sum_is_clipped_to_full_scale(self) -> None:
        """Hum on a near-full-scale frame passes 1.0, and training clipped it there."""
        clean = np.full(FRAME, 0.99, dtype=np.float32)
        hum_on = dataclasses.replace(
            HISS_ONLY, hiss_snr_db=200.0, enable_hum=True, hum_amplitude=0.05
        )

        degraded = apply_degradation(clean, RATE, hum_on, generator())

        assert np.max(degraded) == 1.0

    def test_a_crackle_with_no_clicks_is_silent(self) -> None:
        assert not vinyl_crackle(FRAME, RATE, 0.0, generator()).any()


class TestHum:
    """Mains hum: a fundamental and three harmonics falling as h ** -1.5."""

    def test_harmonic_amplitudes_follow_the_d8993b2_law(self) -> None:
        """60 Hz over 32,000 samples at 16 kHz lands every harmonic on an exact bin."""
        hum = mains_hum(FRAME, RATE, 60.0, generator()).astype(np.float64)
        spectrum = np.abs(np.fft.rfft(hum))
        bin_hz = RATE / FRAME
        levels = [spectrum[round(60.0 * h / bin_hz)] for h in (1, 2, 3, 4)]

        ratios = [level / levels[0] for level in levels]

        np.testing.assert_allclose(ratios, [h**-1.5 for h in (1, 2, 3, 4)], rtol=1e-3)
        assert np.max(np.abs(hum)) == pytest.approx(1.0)


class TestPrimitives:
    """Properties of each primitive's internals, which the parameter tests do not reach.

    The bounds come from the port itself. On training pairs the added hiss had median
    slopes of -1.000 (pink) and -1.934 (brown) over 100 Hz to 4 kHz.
    """

    @pytest.mark.parametrize(
        ("color", "low", "high"), [("pink", -1.1, -0.9), ("brown", -2.05, -1.8)]
    )
    def test_hiss_spectral_slope(self, color: str, low: float, high: float) -> None:
        """Pink falls as 1/f in power and brown as 1/f**2 above the lowest bins."""
        freqs = np.fft.rfftfreq(FRAME, 1 / RATE)
        band = (freqs >= 100) & (freqs <= 4000)
        slopes = []
        for index in range(10):
            hiss = tape_hiss(FRAME, RATE, color, generator(index)).astype(np.float64)
            power = np.abs(np.fft.rfft(hiss)) ** 2
            slopes.append(np.polyfit(np.log10(freqs[band]), np.log10(power[band]), 1)[0])

        assert low < np.median(slopes) < high

    def test_brown_hiss_has_its_drift_removed(self) -> None:
        """A random walk drifts, and d8993b2 subtracted the mean before normalising."""
        hiss = tape_hiss(FRAME, RATE, "brown", generator()).astype(np.float64)

        assert abs(np.mean(hiss)) < 1e-6

    def test_rolloff_is_fourth_order(self) -> None:
        """At twice a 2 kHz cutoff: about -30.6 dB, where order 2 gives -15.4 and 6 gives -45.9."""
        t = np.arange(64000) / RATE
        tone = np.sin(2 * np.pi * 4000.0 * t).astype(np.float32)

        settled = high_freq_rolloff(tone, RATE, 2000.0)[32000:].astype(np.float64)

        ratio = np.sqrt(np.mean(settled**2) / np.mean(tone[32000:].astype(np.float64) ** 2))
        assert -32.0 < 20 * np.log10(ratio) < -29.0

    def test_saturation_is_peak_matched_tanh(self) -> None:
        """tanh((1 + 4 * drive) * x), rescaled so the output peak equals the input peak."""
        x = music(10, 0.8).astype(np.float32)

        out = tape_saturation(x, 0.3).astype(np.float64)

        shape = np.tanh(2.2 * x.astype(np.float64))
        assert np.max(np.abs(out)) == pytest.approx(np.max(np.abs(x)), rel=1e-6)
        np.testing.assert_allclose(
            out, shape * (np.max(np.abs(x)) / np.max(np.abs(shape))), atol=1e-6
        )

    def test_a_click_decays_over_one_millisecond(self) -> None:
        """One click spreads over 16 samples falling by exp(-1/3) per sample."""

        class OneClick:
            def random(self, length: int) -> npt.NDArray[np.float64]:
                draws = np.ones(length)
                draws[FRAME // 2] = 0.0
                return draws

            def standard_normal(self, count: int) -> npt.NDArray[np.float64]:
                return np.ones(count)

        crackle = vinyl_crackle(FRAME, RATE, 0.5, cast(np.random.Generator, OneClick()))

        tail = crackle[crackle != 0].astype(np.float64)
        assert tail.size == 16
        np.testing.assert_allclose(tail[1:] / tail[:-1], np.exp(-1 / 3), rtol=1e-5)


class TestSeeding:
    """Each frame has its own reproducible stream."""

    def test_the_seed_sequence_is_the_seed_the_digest_words_and_the_index(self) -> None:
        words = struct.unpack(">8I", bytes.fromhex(DIGEST))

        assert frame_seed(7, DIGEST, 3).entropy == [7, *words, 3]

    def test_the_same_seed_reproduces_the_same_bytes(self) -> None:
        frame = music(8, 0.5)
        runs = [
            degrade_frame(
                frame,
                recording_peak=0.5,
                sample_rate=RATE,
                ranges=TRAINING,
                seed=frame_seed(1, DIGEST, 2),
            ).degraded.tobytes()
            for _ in range(2)
        ]

        assert runs[0] == runs[1]

    def test_another_frame_index_draws_differently(self) -> None:
        frame = music(9, 0.5)
        first, second = (
            degrade_frame(
                frame,
                recording_peak=0.5,
                sample_rate=RATE,
                ranges=TRAINING,
                seed=frame_seed(1, DIGEST, index),
            ).degraded
            for index in (0, 1)
        )

        assert not np.array_equal(first, second)

    @pytest.mark.parametrize(
        ("seed", "digest", "index", "message"),
        [
            (-1, DIGEST, 0, "non-negative"),
            (1, DIGEST, -1, "non-negative"),
            (1, DIGEST.upper(), 0, "64 lowercase hex"),
            (1, DIGEST[:-1], 0, "64 lowercase hex"),
        ],
    )
    def test_an_out_of_domain_argument_is_refused(
        self, seed: int, digest: str, index: int, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            frame_seed(seed, digest, index)


class TestGolden:
    """The degradation golden: one frame of the analytic fixture with a fixed seed."""

    def test_the_golden_seed_draws_every_optional_effect(self) -> None:
        """Otherwise the digest would not cover saturation, crackle, or hum."""
        golden = PROTOCOL.degradation.golden
        fixture = next(f for f in PROTOCOL.golden_fixtures.fixtures if f.name == golden.fixture)
        seed = frame_seed(golden.seed, fixture.canonical_sha256, golden.frame_index)

        params = draw_params(np.random.Generator(np.random.PCG64(seed)), TRAINING)

        assert params.enable_saturation and params.enable_crackle and params.enable_hum

    @pytest.mark.skipif(
        platform_token() != PROTOCOL.golden_fixtures.reference_platform,
        reason="the degradation digest is asserted on the reference platform only",
    )
    def test_the_digest_on_the_reference_platform(self) -> None:
        assert golden_degraded_sha256(PROTOCOL) == PROTOCOL.degradation.golden.degraded_sha256
