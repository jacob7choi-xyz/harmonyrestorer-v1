"""Tests for the clean-input measurement and its blind listening check."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from benchmark.listening.loudness import integrated_loudness
from benchmark.listening.protocol import load_listening_protocol
from benchmark.measurements import clean_input as ci
from benchmark.measurements import clean_input_listen as cl
from benchmark.protocol import Protocol, load_protocol

_RATE = 16000


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen benchmark protocol."""
    return load_protocol()


class _Identity:
    """A system whose output is its input."""

    name = "opgan"

    def denoise(self, input_path: Path) -> Path:
        return input_path


def _tone(seconds: float, seed: int = 0) -> np.ndarray:
    """A tone with a little noise, at a level well inside [-1, 1]."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * _RATE))
    return 0.3 * np.sin(t * 0.05) + rng.normal(0, 0.01, t.size)


class TestScoreClean:
    """An output identical to the clean frame scores no distance from it."""

    def test_identity_output_is_scored_as_unchanged(
        self, protocol: Protocol, tmp_path: Path
    ) -> None:
        run = ci.CleanRun(protocol=protocol, systems={"opgan": _Identity()}, scratch=tmp_path)
        values = ci.score_clean(run, "opgan", ci.FrameIdentity("a" * 64, 0), _tone(2.0))
        # The frame is written as float32, so the round trip leaves a tiny distance.
        assert values["lsd_db"] == pytest.approx(0.0, abs=1e-3)
        assert values["si_snr_db"] >= 50.0
        assert values["shift_tolerant_si_snr_db"] >= values["si_snr_db"]


class TestSummary:
    """The summary gives each system's mean and the per-recording OpGAN minus UVR."""

    def test_means_and_differences(self, protocol: Protocol) -> None:
        per_recording = {
            f"recording-{n}": {
                "opgan": dict.fromkeys(ci.KEYS, 10.0 + n),
                "uvr": dict.fromkeys(ci.KEYS, 20.0),
            }
            for n in range(6)
        }
        result = ci.summary(per_recording, protocol)
        assert result["opgan"]["si_snr_db"]["mean"] == pytest.approx(12.5)
        assert result["uvr"]["lsd_db"]["mean"] == pytest.approx(20.0)
        difference = result["opgan_minus_uvr"]["si_snr_db"]
        assert difference["mean"] == pytest.approx(-7.5)
        low, high = difference["interval"]  # type: ignore[misc]
        assert low <= -7.5 <= high


class TestExcerptStarts:
    """Excerpts start at fixed positions among fully eligible stretches and never overlap."""

    def test_picks_the_third_and_two_thirds_positions(self) -> None:
        eligible = [True] * 40
        # Candidates are frames 0 to 36; positions round(36/3) = 12 and round(72/3) = 24.
        assert cl.excerpt_starts(eligible) == [12, 24]

    def test_ineligible_frames_are_skipped(self) -> None:
        eligible = [False] * 10 + [True] * 30
        # Candidates are frames 10 to 36 (27); positions round(26/3) = 9 and round(52/3) = 17.
        assert cl.excerpt_starts(eligible) == [19, 27]

    def test_overlapping_picks_are_refused(self) -> None:
        with pytest.raises(cl.ListenError, match="overlap"):
            cl.excerpt_starts([True] * 6)

    def test_no_long_stretch_is_refused(self) -> None:
        with pytest.raises(cl.ListenError, match="long enough"):
            cl.excerpt_starts([True, True, True, False] * 5)


class TestExcerptWindow:
    """v1 gets one frame of context each side, cut at the waveform's ends."""

    def test_context_on_both_sides(self) -> None:
        samples = np.arange(100.0)
        excerpt = cl.excerpt_window(samples, start=3, width=10, gain=2.0)
        assert excerpt.padded[excerpt.offset] == 60.0
        assert excerpt.padded.size == (cl.EXCERPT_FRAMES + 2 * cl.CONTEXT_FRAMES) * 10
        middle = excerpt.padded[excerpt.offset : excerpt.offset + excerpt.length]
        np.testing.assert_array_equal(middle, 2.0 * np.arange(30.0, 70.0))

    def test_context_is_cut_at_the_ends(self) -> None:
        samples = np.arange(50.0)
        excerpt = cl.excerpt_window(samples, start=0, width=10, gain=1.0)
        assert excerpt.offset == 0
        assert excerpt.padded.size == 50
        assert excerpt.length == 40


class TestRestore:
    """An 8-second excerpt with context goes through a system call intact."""

    def test_an_eight_second_input_with_context_round_trips(self, tmp_path: Path) -> None:
        samples = _tone(12.0)
        restored = cl._restore(_Identity(), tmp_path, cl.FrameIdentity("a" * 64, 5), samples, _RATE)
        assert restored.size == samples.size
        np.testing.assert_allclose(restored, samples.astype(np.float32), atol=1e-7)


class TestRenderPair:
    """Both conditions come out the same length, loudness matched, as 16-bit samples."""

    def test_lengths_loudness_and_format(self) -> None:
        original = _tone(8.0)
        restored = 0.5 * _tone(8.0, seed=1)[:-100]
        rule = load_listening_protocol().loudness
        pcm, gains = cl.render_pair(original, restored, _RATE, rule)
        assert pcm["original"].dtype == np.int16
        assert pcm["original"].size == pcm["opgan"].size == restored.size
        loudness = [integrated_loudness(pcm[name] / 2**15, _RATE) for name in cl.CONDITIONS]
        assert loudness[0] == pytest.approx(loudness[1], abs=0.05)
        assert set(gains) == set(cl.CONDITIONS)


class TestTrialPlan:
    """Every excerpt is one scored pair; two catches repeat one condition on both sides."""

    def test_structure(self) -> None:
        excerpts = [f"e{n}" for n in range(12)]
        trials = cl.trial_plan(excerpts, np.random.default_rng(4))
        assert [trial["trial"] for trial in trials] == list(range(1, 15))
        scored = [trial for trial in trials if trial["kind"] == "scored"]
        catches = [trial for trial in trials if trial["kind"] == "catch"]
        assert sorted(trial["excerpt"] for trial in scored) == sorted(excerpts)
        assert all({trial["A"], trial["B"]} == set(cl.CONDITIONS) for trial in scored)
        assert len(catches) == cl.CATCH_TRIALS
        assert all(trial["A"] == trial["B"] for trial in catches)
        assert len({trial["excerpt"] for trial in catches}) == cl.CATCH_TRIALS

    def test_the_seed_fixes_the_plan(self) -> None:
        excerpts = [f"e{n}" for n in range(12)]
        first = cl.trial_plan(excerpts, np.random.default_rng(9))
        assert first == cl.trial_plan(excerpts, np.random.default_rng(9))


def _trials() -> list[dict[str, Any]]:
    """Twelve scored trials with the original on side A, and two catches."""
    trials: list[dict[str, Any]] = [
        {"trial": n, "kind": "scored", "excerpt": f"e{n}", "A": "original", "B": "opgan"}
        for n in range(1, 13)
    ]
    trials += [
        {"trial": 13, "kind": "catch", "excerpt": "e1", "A": "opgan", "B": "opgan"},
        {"trial": 14, "kind": "catch", "excerpt": "e2", "A": "original", "B": "original"},
    ]
    return trials


class TestScoreAnswers:
    """Answers are mapped through the key to conditions, and the threshold decides."""

    def test_ten_preferences_for_the_input_meet_the_rule(self) -> None:
        answers = {str(n): "A" for n in range(1, 11)} | {"11": "B", "12": "N"}
        answers |= {"13": "N", "14": "A"}
        result = cl.score_answers(_trials(), answers)
        assert result["scored_counts"] == {"original": 10, "opgan": 1, "no_preference": 1}
        assert result["decision"] == "input_preferred"
        assert result["catch_answers"] == [
            {"trial": 13, "answer": "N"},
            {"trial": 14, "answer": "A"},
        ]

    def test_nine_preferences_do_not(self) -> None:
        answers = {str(n): "A" for n in range(1, 10)} | {"10": "B", "11": "B", "12": "B"}
        answers |= {"13": "N", "14": "N"}
        assert cl.score_answers(_trials(), answers)["decision"] == "input_not_preferred"

    @pytest.mark.parametrize(
        "change",
        [{"14": None}, {"15": "A"}, {"3": "maybe"}],
        ids=["missing", "extra", "invalid"],
    )
    def test_malformed_answers_are_refused(self, change: dict[str, str | None]) -> None:
        answers: dict[str, str] = {str(n): "N" for n in range(1, 15)}
        for trial, answer in change.items():
            if answer is None:
                del answers[trial]
            else:
                answers[trial] = answer
        with pytest.raises(cl.ListenError, match="answers must"):
            cl.score_answers(_trials(), answers)


class TestPackage:
    """The listener's files name no condition; the sealed key holds the seed and the plan."""

    def test_files_and_key(self) -> None:
        pcm = {name: np.zeros(8 * _RATE, dtype=np.int16) for name in cl.CONDITIONS}
        rendered = [
            cl.RenderedExcerpt(f"r{n}_f0", f"recording-{n}", 0, pcm, dict.fromkeys(pcm, 1.0))
            for n in range(12)
        ]
        files, sealed = cl.package(rendered, 1234, _RATE)
        wavs = sorted(name for name in files if name.endswith(".wav"))
        assert len(wavs) == 28
        assert wavs[0] == "trial_01_A.wav"
        build_record = files["build.json"].decode()
        assert "original" not in build_record and "opgan" not in build_record
        # No digest in the listener's record: a catch trial's two identical files would show.
        assert re.search(r"[0-9a-f]{64}", build_record) is None
        samples, rate = sf.read(io.BytesIO(files["trial_01_A.wav"]), dtype="int16")
        assert rate == _RATE and samples.size == 8 * _RATE
        key = json.loads(sealed["key.json"])
        assert key["seed"] == 1234
        assert len(key["trials"]) == 14
        assert set(key["file_sha256"]) == set(files)


class TestMain:
    """Existing output directories are refused before anything runs."""

    def test_measurement_refuses_an_existing_out_dir(self, tmp_path: Path) -> None:
        args = ["--walk-dir", "w", "--uvr-model-dir", "u", "--out-dir", str(tmp_path)]
        assert ci.main(args) == 1

    def test_build_refuses_an_existing_sealed_dir(self, tmp_path: Path) -> None:
        args = ["build", "--walk-dir", "w", "--uvr-model-dir", "u"]
        args += ["--out-dir", str(tmp_path / "new"), "--sealed-dir", str(tmp_path)]
        assert cl.main(args) == 1
