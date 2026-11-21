"""Tests for mapping the measured floors to v6's eligibility codes."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from benchmark.canonicalisation import CanonicalWaveform
from benchmark.eligibility import RecordingFloors, frame_floors
from benchmark.eligibility_outcomes import (
    FrameReason,
    FrameVerdict,
    RecordingReason,
    frame_verdicts,
    recording_floor_reasons,
    recording_frame_reasons,
    unscoreable_metrics,
)
from benchmark.framing import frame
from benchmark.protocol import Protocol, load_protocol

DIGEST = "c" * 64


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol."""
    return load_protocol()


def music(samples: int, seed: int = 0) -> npt.NDArray[np.float64]:
    """Noise at about -17 dBFS (-16.8 at seed 0): above the frame floor and scoreable by every metric."""
    return np.random.default_rng(seed).uniform(-0.25, 0.25, samples)


def verdicts_for(
    rows: list[npt.NDArray[np.float64]], protocol: Protocol
) -> tuple[FrameVerdict, ...]:
    """Frame the concatenated rows as a canonical waveform and give every frame a verdict."""
    samples = np.concatenate(rows)
    samples.flags.writeable = False
    framed = frame(CanonicalWaveform(samples, DIGEST), protocol.framing)
    return frame_verdicts(framed, frame_floors(framed, protocol.eligibility), protocol.metrics)


class TestCodesMatchTheProtocol:
    """The enums mirror v6's lists, and their order is the precedence."""

    def test_recording_codes_are_the_protocol_list_in_order(self, protocol: Protocol) -> None:
        assert tuple(RecordingReason) == protocol.eligibility.recording_ineligible_reasons

    def test_frame_codes_are_the_protocol_list_in_order(self, protocol: Protocol) -> None:
        assert tuple(FrameReason) == protocol.eligibility.frame_ineligible_reasons


class TestRecordingFloorReasons:
    """Both floors are evaluated, and the duration code comes first."""

    @pytest.mark.parametrize(
        ("long_enough", "loud_enough", "expected"),
        [
            (True, True, ()),
            (False, True, (RecordingReason.BELOW_MIN_DURATION,)),
            (True, False, (RecordingReason.BELOW_RMS_FLOOR,)),
            (
                False,
                False,
                (RecordingReason.BELOW_MIN_DURATION, RecordingReason.BELOW_RMS_FLOOR),
            ),
        ],
        ids=["eligible", "too_short", "too_quiet", "both"],
    )
    def test_reasons_follow_the_floors(
        self, long_enough: bool, loud_enough: bool, expected: tuple[RecordingReason, ...]
    ) -> None:
        floors = RecordingFloors(960_000, long_enough, -20.0, loud_enough)
        assert recording_floor_reasons(floors) == expected


class TestUnscoreableMetrics:
    """The metrics' own refusals decide, with the frame as reference and estimate."""

    def test_ordinary_music_is_scoreable_by_every_metric(self, protocol: Protocol) -> None:
        assert unscoreable_metrics(music(32000), protocol.metrics) == ()

    def test_a_constant_frame_is_refused_by_si_snr_only(self, protocol: Protocol) -> None:
        assert unscoreable_metrics(np.full(32000, 0.1), protocol.metrics) == ("si_snr",)

    def test_silence_is_refused_by_every_metric(self, protocol: Protocol) -> None:
        assert unscoreable_metrics(np.zeros(32000), protocol.metrics) == (
            "reconstruction_snr",
            "si_snr",
            "log_spectral_distance",
        )


class TestFrameVerdicts:
    """Every candidate frame gets a verdict, with both checks evaluated."""

    def test_each_frame_is_judged_on_its_own_samples(self, protocol: Protocol) -> None:
        rows = [music(32000, 1), np.full(32000, 1e-4), np.full(32000, 0.1), np.zeros(32000)]

        verdicts = verdicts_for(rows, protocol)

        assert [v.index for v in verdicts] == [0, 1, 2, 3]
        assert [v.reasons for v in verdicts] == [
            (),
            (FrameReason.BELOW_RMS_FLOOR, FrameReason.NOT_SCOREABLE_BY_PRIMARY_METRIC),
            (FrameReason.NOT_SCOREABLE_BY_PRIMARY_METRIC,),
            (FrameReason.BELOW_RMS_FLOOR, FrameReason.NOT_SCOREABLE_BY_PRIMARY_METRIC),
        ]
        assert [v.eligible for v in verdicts] == [True, False, False, False]
        assert verdicts[2].unscoreable_by == ("si_snr",)
        assert verdicts[2].rms_dbfs == pytest.approx(-20.0)
        assert verdicts[3].rms_dbfs == -np.inf

    def test_a_quiet_frame_that_metrics_accept_is_below_the_floor_only(
        self, protocol: Protocol
    ) -> None:
        """About -65 dBFS (-64.8): under the -45 dBFS floor and still scoreable by every metric."""
        verdicts = verdicts_for([music(32000, 2) * 0.004], protocol)
        assert verdicts[0].reasons == (FrameReason.BELOW_RMS_FLOOR,)
        assert verdicts[0].unscoreable_by == ()


class TestRecordingFrameReasons:
    """A recording that cleared its floors but has no eligible frame."""

    def test_no_eligible_frame_gives_the_code(self, protocol: Protocol) -> None:
        verdicts = verdicts_for([np.zeros(32000), np.full(32000, 0.1)], protocol)
        assert recording_frame_reasons(verdicts) == (RecordingReason.WITHOUT_ELIGIBLE_FRAMES,)

    def test_one_eligible_frame_is_enough(self, protocol: Protocol) -> None:
        verdicts = verdicts_for([np.zeros(32000), music(32000, 3)], protocol)
        assert recording_frame_reasons(verdicts) == ()
