"""Tests for the listening analysis: exclusions, ties, the interval threshold, and refusals."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from test_listening_test_page import _make_build

from benchmark.listening.analysis import (
    ANALYSIS_JSON,
    ANALYSIS_MD,
    AnalysisError,
    analyse,
    write_analysis,
)
from benchmark.listening.build import BUILD_NAME, KEY_NAME
from benchmark.listening.protocol import ListeningProtocol, load_listening_protocol
from benchmark.listening.test_page import ANSWERS_FORMAT, recordings_of
from benchmark.listening.trials import Trial, listener_trials

_SCORED = 30
_TOTAL = 32


@pytest.fixture(scope="module")
def protocol() -> ListeningProtocol:
    """The frozen listening protocol, with its digests checked."""
    return load_listening_protocol()


def _trials(build: Path, protocol: ListeningProtocol, index: int) -> tuple[Trial, ...]:
    """The listener's trials, rebuilt from the build's sealed key."""
    key = json.loads((build / "sealed" / KEY_NAME).read_text())
    return listener_trials(protocol, index, recordings_of(key))


def _answers_file(
    folder: Path,
    protocol: ListeningProtocol,
    index: int,
    trials: tuple[Trial, ...],
    choose: Callable[[Trial], str | None],
    *,
    protocol_sha: str | None = None,
) -> Path:
    """Write one listener's answers file, where `choose` gives each trial's answer or None."""
    entries = []
    for trial in trials:
        answer = choose(trial)
        entry: dict[str, Any] = {
            "trial_id": trial.trial_id,
            "comment": "",
            "answered_at": "2026-11-24",
        }
        if answer is not None:
            entry["answer"] = answer
        entries.append(entry)
    document = {
        "format": ANSWERS_FORMAT,
        "protocol_sha256": protocol_sha or protocol.sha256,
        "listener_id": f"L{index:03d}",
        "listener_index": index,
        "answers": entries,
    }
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"answers_L{index:03d}.json"
    path.write_text(json.dumps(document))
    return path


def _prefer_first(trial: Trial) -> str:
    """The answer that prefers the pair's first condition, or no preference for a catch trial."""
    if trial.kind == "catch":
        return "No preference"
    return "A" if trial.first_on_a else "B"


def _prefer_second(trial: Trial) -> str:
    """The answer that prefers the pair's second condition, or no preference for a catch trial."""
    if trial.kind == "catch":
        return "No preference"
    return "B" if trial.first_on_a else "A"


def _with_catches(rule: Callable[[Trial], str], catch: str) -> Callable[[Trial], str]:
    """A rule that answers scored trials by *rule* and catch trials with *catch*."""
    return lambda trial: catch if trial.kind == "catch" else rule(trial)


def _analyse(
    tmp_path: Path,
    protocol: ListeningProtocol,
    indices: range,
    rule: Callable[[Trial], str],
    catch: str = "No preference",
) -> dict[str, Any]:
    """Build the fixture, write answers for *indices*, and return the analysis document."""
    build = _make_build(tmp_path, protocol)
    answers = tmp_path / "answers"
    for index in indices:
        trials = _trials(build, protocol, index)
        _answers_file(answers, protocol, index, trials, _with_catches(rule, catch))
    return analyse(protocol, build, answers)


def test_both_catch_trials_answered_a_or_b_exclude_the_listener(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """Both catch trials answered A or B exclude the listener, whose answers are not counted."""
    build = _make_build(tmp_path, protocol)
    answers = tmp_path / "answers"
    _answers_file(
        answers,
        protocol,
        0,
        _trials(build, protocol, 0),
        lambda t: "A" if t.kind == "catch" else _prefer_first(t),
    )
    _answers_file(
        answers,
        protocol,
        1,
        _trials(build, protocol, 1),
        lambda t: "No preference" if t.kind == "catch" else _prefer_first(t),
    )

    document = analyse(protocol, build, answers)

    assert document["excluded_listeners"] == [{"listener_id": "L000", "reason": "catch_trials"}]
    assert document["included_listeners"] == 1
    assert all(row["judgments"] == 10 for row in document["pairs"])


def test_one_catch_trial_answered_no_preference_keeps_the_listener(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """A listener with one catch answer of no preference is not excluded by the catch rule."""
    build = _make_build(tmp_path, protocol)
    answers = tmp_path / "answers"
    trials = _trials(build, protocol, 0)
    catch_ids = [t.trial_id for t in trials if t.kind == "catch"]
    first_catch = catch_ids[0]
    _answers_file(
        answers,
        protocol,
        0,
        trials,
        lambda t: (
            ("No preference" if t.trial_id == first_catch else "A")
            if t.kind == "catch"
            else _prefer_first(t)
        ),
    )

    document = analyse(protocol, build, answers)

    assert document["excluded_listeners"] == []
    assert document["included_listeners"] == 1


def test_a_missing_answer_excludes_the_listener_as_incomplete(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """An unanswered trial excludes the listener as incomplete, even when the catch trials pass."""
    build = _make_build(tmp_path, protocol)
    answers = tmp_path / "answers"
    trials = _trials(build, protocol, 0)
    skipped = trials[5].trial_id
    _answers_file(
        answers,
        protocol,
        0,
        trials,
        lambda t: (
            None if t.trial_id == skipped else _with_catches(_prefer_first, "No preference")(t)
        ),
    )

    document = analyse(protocol, build, answers)

    assert document["excluded_listeners"] == [{"listener_id": "L000", "reason": "incomplete"}]


def test_equal_preferences_give_equal_shares(tmp_path: Path, protocol: ListeningProtocol) -> None:
    """One listener preferring the first condition and one the second give shares of 0.5."""
    build = _make_build(tmp_path, protocol)
    answers = tmp_path / "answers"
    _answers_file(
        answers,
        protocol,
        0,
        _trials(build, protocol, 0),
        _with_catches(_prefer_first, "No preference"),
    )
    _answers_file(
        answers,
        protocol,
        1,
        _trials(build, protocol, 1),
        _with_catches(_prefer_second, "No preference"),
    )

    document = analyse(protocol, build, answers)

    for row in document["pairs"]:
        assert row["judgments"] == 20
        assert row["first"]["share"] == pytest.approx(0.5)
        assert row["second"]["share"] == pytest.approx(0.5)
        assert row["no_preference"]["share"] == pytest.approx(0.0)


def test_the_interval_is_not_reported_below_the_minimum_listener_count(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """With nine included listeners no interval is reported, and the reason is named."""
    document = _analyse(tmp_path, protocol, range(9), _prefer_first)

    assert document["interval"]["reported"] is False
    assert document["interval"]["reason"] == "too few included listeners"


def test_the_interval_is_reported_at_ten_listeners_and_repeats_with_the_seed(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """With ten included listeners an interval is reported for each pair and category."""
    first = _analyse(
        tmp_path / "a",
        protocol,
        range(10),
        lambda t: _prefer_first(t) if t.recording.endswith("1") else _prefer_second(t),
    )
    second = _analyse(
        tmp_path / "b",
        protocol,
        range(10),
        lambda t: _prefer_first(t) if t.recording.endswith("1") else _prefer_second(t),
    )

    assert first["interval"]["reported"] is True
    assert len(first["interval"]["bounds"]) == 9
    assert first["interval"]["bounds"] == second["interval"]["bounds"]
    for bound in first["interval"]["bounds"]:
        assert bound["low"] <= bound["high"]


def test_each_included_listener_answers_all_32_trials(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """The A, B and no-preference counts per listener add up to the 32 trials completed."""
    document = _analyse(tmp_path, protocol, range(3), _prefer_first)

    for row in document["per_listener"]:
        assert row["A"] + row["B"] + row["No preference"] == _TOTAL


def test_a_tampered_key_refuses_the_analysis_and_writes_nothing(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """A changed sealed key stops the analysis before any result is produced."""
    build = _make_build(tmp_path, protocol)
    answers = tmp_path / "answers"
    _answers_file(
        answers,
        protocol,
        0,
        _trials(build, protocol, 0),
        _with_catches(_prefer_first, "No preference"),
    )
    key_path = build / "sealed" / KEY_NAME
    key_path.write_bytes(key_path.read_bytes().replace(b'"clips"', b'"clipz"', 1))

    with pytest.raises(AnalysisError, match="cannot be checked"):
        analyse(protocol, build, answers)


def test_an_answers_file_for_another_protocol_is_refused(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """A file saved for a different protocol digest stops the analysis."""
    build = _make_build(tmp_path, protocol)
    answers = tmp_path / "answers"
    _answers_file(
        answers, protocol, 0, _trials(build, protocol, 0), _prefer_first, protocol_sha="0" * 64
    )

    with pytest.raises(AnalysisError, match="different protocol"):
        analyse(protocol, build, answers)


def test_the_written_report_names_no_p_value_no_significance_and_no_winner(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """The Markdown and JSON carry no p-value, significance wording or winner statement."""
    document = _analyse(tmp_path, protocol, range(10), _prefer_first)
    out = write_analysis(document, tmp_path / "result")
    text = (out / ANALYSIS_MD).read_text().lower() + (out / ANALYSIS_JSON).read_text().lower()

    for banned in ("p-value", "p value", "significan", "winner", "wins"):
        assert banned not in text


def test_an_existing_output_directory_is_refused_and_kept(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """The analysis does not overwrite an existing result directory."""
    document = _analyse(tmp_path, protocol, range(1), _prefer_first)
    (tmp_path / "result").mkdir()
    (tmp_path / "result" / "keep.txt").write_text("earlier")

    with pytest.raises(AnalysisError, match="already exists"):
        write_analysis(document, tmp_path / "result")

    assert [p.name for p in (tmp_path / "result").iterdir()] == ["keep.txt"]


def test_build_manifest_names_the_key_digest_used(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """The analysis records the key digest the build manifest holds."""
    build = _make_build(tmp_path, protocol)
    answers = tmp_path / "answers"
    _answers_file(
        answers,
        protocol,
        0,
        _trials(build, protocol, 0),
        _with_catches(_prefer_first, "No preference"),
    )

    document = analyse(protocol, build, answers)

    assert document["key_sha256"] == json.loads((build / BUILD_NAME).read_text())["key_sha256"]
