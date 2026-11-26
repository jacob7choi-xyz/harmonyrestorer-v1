"""Tests for the model v2 measurement of where v1's SI-SNR gain comes from."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from benchmark.degradation import (
    DegradationParams,
    draw_params,
    frame_seed,
    high_freq_rolloff,
)
from benchmark.frame_scoring import degrade_with_reference
from benchmark.measurements import gain_steps as gs
from benchmark.measurements.real_noise_gap import SNR_GRID_DB
from benchmark.protocol import Protocol, load_protocol

_DIGEST = "ab" * 32


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen benchmark protocol."""
    return load_protocol()


def _source(index: int) -> gs.FrameSource:
    """A 2-second frame of a tone with a little noise, at a fixed recording peak."""
    rng = np.random.default_rng(index)
    clean = 0.4 * np.sin(np.arange(32000) * 0.07) + rng.normal(0, 0.02, 32000)
    return gs.FrameSource(clean, 0.6, _DIGEST, index)


def _draw(protocol: Protocol, index: int) -> DegradationParams:
    """The benchmark's parameter draw for a frame index of the test digest."""
    seed = frame_seed(protocol.degradation.seed, _DIGEST, index)
    return draw_params(np.random.Generator(np.random.PCG64(seed)), protocol.degradation.ranges)


def _index_where(protocol: Protocol, wanted: Callable[[DegradationParams], bool]) -> int:
    """The first frame index whose benchmark draw satisfies `wanted`."""
    for index in range(100):
        if wanted(_draw(protocol, index)):
            return index
    raise AssertionError("no such index in the first 100")


class TestDegradeWith:
    """The conditions share one draw and differ only in the parts removed."""

    def test_keeping_both_parts_is_the_benchmarks_degraded_frame(self, protocol: Protocol) -> None:
        source = _source(3)
        expected, _ = degrade_with_reference(
            source.clean, source.recording_peak, _DIGEST, 3, protocol
        )
        full = gs.degrade_with(source, protocol, lowpass=True, saturation=True)
        assert full.tobytes() == expected.degraded.tobytes()

    def test_removing_the_lowpass_changes_the_frame(self, protocol: Protocol) -> None:
        source = _source(3)
        full = gs.degrade_with(source, protocol, lowpass=True, saturation=True)
        no_lowpass = gs.degrade_with(source, protocol, lowpass=False, saturation=True)
        assert not np.array_equal(full, no_lowpass)

    def test_saturation_off_changes_nothing_when_it_was_not_drawn(self, protocol: Protocol) -> None:
        source = _source(_index_where(protocol, lambda p: not p.enable_saturation))
        no_lowpass = gs.degrade_with(source, protocol, lowpass=False, saturation=True)
        noise_only = gs.degrade_with(source, protocol, lowpass=False, saturation=False)
        np.testing.assert_array_equal(no_lowpass, noise_only)

    def test_saturation_off_changes_the_frame_when_it_was_drawn(self, protocol: Protocol) -> None:
        source = _source(_index_where(protocol, lambda p: p.enable_saturation))
        no_lowpass = gs.degrade_with(source, protocol, lowpass=False, saturation=True)
        noise_only = gs.degrade_with(source, protocol, lowpass=False, saturation=False)
        assert not np.array_equal(no_lowpass, noise_only)

    def test_hiss_keeps_its_waveform_and_its_level_follows_the_lowpass(
        self, protocol: Protocol
    ) -> None:
        def hiss_alone(params: DegradationParams) -> bool:
            return not (params.enable_saturation or params.enable_crackle or params.enable_hum)

        index = _index_where(protocol, hiss_alone)
        source = _source(index)
        rate = protocol.canonicalisation.resample.target_sample_rate
        gain = protocol.degradation.ranges.headroom / source.recording_peak
        scaled = (source.clean * gain).astype(np.float32)
        filtered = high_freq_rolloff(scaled, rate, _draw(protocol, index).rolloff_hz)
        full = gs.degrade_with(source, protocol, lowpass=True, saturation=True)
        no_lowpass = gs.degrade_with(source, protocol, lowpass=False, saturation=True)
        hiss_full = full.astype(np.float64) - filtered.astype(np.float64)
        hiss_no_lowpass = no_lowpass.astype(np.float64) - scaled.astype(np.float64)
        assert np.corrcoef(hiss_full, hiss_no_lowpass)[0, 1] == pytest.approx(1.0, abs=1e-5)
        level_ratio = np.sqrt(np.mean(hiss_full**2) / np.mean(hiss_no_lowpass**2))
        power_ratio = np.mean(filtered.astype(np.float64) ** 2) / np.mean(
            scaled.astype(np.float64) ** 2
        )
        assert level_ratio == pytest.approx(np.sqrt(power_ratio), rel=1e-4)


class TestConditionsForFrame:
    """The grid condition rescales the added-noise-only condition's noise to each SNR."""

    def test_grid_inputs_carry_the_same_noise_at_each_snr(self, protocol: Protocol) -> None:
        source = _source(5)
        reference, inputs = gs.conditions_for_frame(source, protocol)
        assert set(inputs) == set(gs.CONDITIONS)
        assert [len(inputs[name]) for name in gs.CONDITIONS] == [1, 1, 1, len(SNR_GRID_DB)]
        gain = protocol.degradation.ranges.headroom / source.recording_peak
        scaled = (source.clean * gain).astype(np.float32).astype(np.float64)
        noise = inputs["added_noise_only"][0] - scaled
        for snr, mixture in zip(SNR_GRID_DB, inputs["snr_grid"], strict=True):
            added = mixture - reference
            measured = 10 * np.log10(np.mean(reference**2) / np.mean(added**2))
            assert measured == pytest.approx(snr, abs=1e-9)
            assert np.corrcoef(added, noise)[0, 1] == pytest.approx(1.0, abs=1e-12)

    def test_a_frame_the_benchmark_degrades_differently_is_refused(
        self, protocol: Protocol, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def shifted(*args: object) -> object:
            degraded, reference = degrade_with_reference(*args)  # type: ignore[arg-type]
            changed = np.roll(degraded.degraded, 1)
            return type(degraded)(changed, degraded.gain, degraded.params), reference

        monkeypatch.setattr(gs, "degrade_with_reference", shifted)
        with pytest.raises(gs.StepsError, match="does not reproduce"):
            gs.conditions_for_frame(_source(5), protocol)


class _Identity:
    """A system whose output is its input, so every improvement is zero."""

    name = "opgan"

    def denoise(self, input_path: Path) -> Path:
        return input_path


class TestScoreInput:
    """A system that changes nothing improves neither SI-SNR nor its shift-tolerant form."""

    def test_identity_output_has_zero_improvement(self, protocol: Protocol, tmp_path: Path) -> None:
        run = gs.StepsRun(protocol=protocol, system=_Identity(), scratch=tmp_path)
        reference, inputs = gs.conditions_for_frame(_source(7), protocol)
        values = gs.score_input(
            run, gs.FrameIdentity(_DIGEST, 7), reference, inputs["benchmark"][0]
        )
        assert values["si_snr_improvement_db"] == pytest.approx(0.0, abs=1e-6)
        assert values["shift_tolerant_si_snr_improvement_db"] == pytest.approx(0.0, abs=1e-6)
        assert values["output_si_snr_db"] == pytest.approx(values["input_si_snr_db"], abs=1e-6)


def _per_recording(differences: list[float]) -> dict[str, dict[str, dict[str, float]]]:
    """Recordings whose `benchmark` condition beats `no_lowpass` by the given amounts."""
    per_recording = {}
    for number, difference in enumerate(differences):
        values = {name: dict.fromkeys(gs.STEP_METRICS, 1.0) for name in gs.CONDITIONS}
        values["benchmark"] = dict.fromkeys(gs.STEP_METRICS, 1.0 + difference)
        per_recording[f"recording-{number}"] = values
    return per_recording


class TestSteps:
    """A step contributes only with the margin and an interval above zero."""

    def test_a_large_consistent_difference_contributes(self, protocol: Protocol) -> None:
        result = gs.steps(_per_recording([2.0, 2.5, 1.8, 2.2, 2.4, 2.1]), protocol)
        lowpass = result["lowpass"]["si_snr_improvement_db"]
        assert lowpass["contributor"] is True
        assert lowpass["mean_difference_db"] == pytest.approx(2.1667, abs=1e-4)
        assert result["saturation"]["si_snr_improvement_db"]["contributor"] is False

    def test_a_difference_below_the_margin_does_not_contribute(self, protocol: Protocol) -> None:
        result = gs.steps(_per_recording([0.5, 0.6, 0.4, 0.7, 0.5, 0.6]), protocol)
        assert result["lowpass"]["si_snr_improvement_db"]["contributor"] is False

    def test_a_large_mean_with_an_interval_reaching_zero_does_not_contribute(
        self, protocol: Protocol
    ) -> None:
        result = gs.steps(_per_recording([12.0, -1.0, -1.0, -1.0, -1.0, -1.0]), protocol)
        lowpass = result["lowpass"]["si_snr_improvement_db"]
        assert lowpass["mean_difference_db"] >= gs.CONTRIBUTOR_MARGIN_DB
        assert lowpass["contributor"] is False


class TestMain:
    """An existing output directory is refused before anything runs."""

    def test_existing_out_dir_is_refused(self, tmp_path: Path) -> None:
        args = ["--walk-dir", "w", "--uvr-model-dir", "u", "--out-dir", str(tmp_path)]
        assert gs.main(args) == 1
