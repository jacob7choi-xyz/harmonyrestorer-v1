"""The descriptive analysis of the listening answers: exclusions, counts, shares and an interval.

The inputs are the answers files the test pages saved and the sealed key. The key is checked
against the build manifest first. Each answers file must name the protocol's digest, and each
listener's trials are rebuilt from the listener index and the key, so an answers file cannot
choose its own trial design.

A listener is excluded when an answer is missing from the scored or catch trials, or when both
catch trials are answered A or B. Excluded listeners are counted with their reason, and their
answers are not analysed. The intervals are a percentile bootstrap over listeners, reported only
when at least the protocol's minimum number of listeners is included.

> **Note:** The analysis is descriptive. It reports no p-value, no significance statement and no
> overall winner. The outputs are `analysis.json` and `analysis.md`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from benchmark.atomic_write import write_directory_atomically
from benchmark.canonical_json import canonical_json
from benchmark.listening.build import BUILD_NAME
from benchmark.listening.protocol import ListeningProtocol
from benchmark.listening.test_page import ANSWERS_FORMAT, PageError, load_sealed_key, recordings_of
from benchmark.listening.trials import Trial, listener_trials

__all__ = [
    "ANALYSIS_JSON",
    "ANALYSIS_MD",
    "AnalysisError",
    "analyse",
    "write_analysis",
]

logger = logging.getLogger(__name__)

ANALYSIS_JSON = "analysis.json"
ANALYSIS_MD = "analysis.md"
NO_PREFERENCE = "No preference"
_CATCH_ANSWERS = ("A", "B")
_FIRST = "first"
_SECOND = "second"
_NONE = "no_preference"
REASON_INCOMPLETE = "incomplete"
REASON_CATCH = "catch_trials"


class AnalysisError(RuntimeError):
    """Raised when the answers or the key cannot be analysed, so no result is written."""


@dataclass(frozen=True)
class Submission:
    """One listener's saved answers, with the trial answers keyed by trial identifier."""

    listener_id: str
    listener_index: int
    answers: Mapping[str, str]


@dataclass(frozen=True)
class Verdict:
    """One listener's outcome: the reason they were excluded, or None when they are included."""

    submission: Submission
    reason: str | None


def _refuse(message: str) -> AnalysisError:
    """Log an analysis refusal and return the error to raise."""
    logger.error("Analysis refused: %s", message)
    return AnalysisError(message)


def _parse_answer(protocol: ListeningProtocol, entry: Any) -> tuple[str, str]:
    """One saved answer as (trial identifier, answer). Refuses an unnamed answer."""
    if not isinstance(entry, dict) or not isinstance(entry.get("trial_id"), str):
        raise _refuse("an answer entry has no trial identifier")
    answer = entry.get("answer")
    if answer is not None and answer not in protocol.trials.answers:
        raise _refuse(f"answer {answer!r} is not one the protocol names")
    return entry["trial_id"], answer if answer is not None else ""


def _parse_submission(protocol: ListeningProtocol, path: Path) -> Submission:
    """Read one answers file and check its format, its protocol digest and its listener fields.

    Raises:
        AnalysisError: If the file is unreadable, of another format, made for another protocol,
            or holds a duplicate trial or an answer the protocol does not name.
    """
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise _refuse(f"cannot read answers file {path.name}: {error}") from error
    if data.get("format") != ANSWERS_FORMAT:
        raise _refuse(f"{path.name} is not an answers file of format {ANSWERS_FORMAT}")
    if data.get("protocol_sha256") != protocol.sha256:
        raise _refuse(f"{path.name} was saved for a different protocol")
    answers: dict[str, str] = {}
    for entry in data.get("answers", []):
        trial_id, answer = _parse_answer(protocol, entry)
        if trial_id in answers:
            raise _refuse(f"{path.name} answers trial {trial_id} twice")
        if answer:
            answers[trial_id] = answer
    return Submission(str(data["listener_id"]), int(data["listener_index"]), answers)


def _catch_ids(trials: Sequence[Trial]) -> tuple[str, ...]:
    """The identifiers of a listener's catch trials."""
    return tuple(trial.trial_id for trial in trials if trial.kind == "catch")


def _verdict(submission: Submission, trials: Sequence[Trial]) -> str | None:
    """The exclusion reason for one listener, or None when the listener is included.

    A missing answer excludes a listener as incomplete, and that check comes first. Both catch
    trials answered A or B excludes a listener as a catch failure.
    """
    if any(trial.trial_id not in submission.answers for trial in trials):
        return REASON_INCOMPLETE
    catches = _catch_ids(trials)
    if all(submission.answers[trial_id] in _CATCH_ANSWERS for trial_id in catches):
        return REASON_CATCH
    return None


def _categorise(trial: Trial, answer: str) -> str:
    """The category of one scored judgment: the first or second condition, or no preference."""
    if answer == NO_PREFERENCE:
        return _NONE
    chosen = trial.side_a if answer == "A" else trial.side_b
    return _FIRST if chosen == trial.first else _SECOND


def _counts(
    protocol: ListeningProtocol, included: Sequence[tuple[Submission, Sequence[Trial]]]
) -> npt.NDArray[np.int64]:
    """Judgment counts per listener, pair and category, shaped (listeners, pairs, 3)."""
    pairs = protocol.trials.pairs
    counts = np.zeros((len(included), len(pairs), 3), dtype=np.int64)
    for row, (submission, trials) in enumerate(included):
        for trial in trials:
            if trial.kind != "scored":
                continue
            category = _categorise(trial, submission.answers[trial.trial_id])
            pair_index = pairs.index((trial.first, trial.second))
            counts[row, pair_index, (_FIRST, _SECOND, _NONE).index(category)] += 1
    return counts


def _bootstrap(
    counts: npt.NDArray[np.int64], protocol: ListeningProtocol
) -> npt.NDArray[np.float64] | None:
    """Percentile bootstrap shares over listeners, or None below the minimum listener count.

    One set of resampled listener indices is drawn for all pairs and categories, from a
    generator seeded with the protocol's interval seed. The statistic is the pooled share.
    """
    interval = protocol.analysis
    listeners = counts.shape[0]
    if listeners < interval.interval_minimum_included_listeners:
        return None
    generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence(interval.interval_seed)))
    draws = generator.integers(0, listeners, size=(interval.interval_iterations, listeners))
    shares = np.empty((interval.interval_iterations, counts.shape[1], 3), dtype=np.float64)
    for iteration, sample in enumerate(draws):
        pooled = counts[sample].sum(axis=0)
        totals = pooled.sum(axis=1, keepdims=True)
        shares[iteration] = pooled / totals
    tail = (1.0 - interval.interval_level) / 2.0 * 100.0
    low, high = np.percentile(shares, [tail, 100.0 - tail], axis=0)
    return np.stack([low, high], axis=-1)


def _pair_rows(protocol: ListeningProtocol, counts: npt.NDArray[np.int64]) -> list[dict[str, Any]]:
    """Per pair: the judgment total, and each category's count and share."""
    rows = []
    totals = counts.sum(axis=(0, 2))
    pooled = counts.sum(axis=0)
    for index, (first, second) in enumerate(protocol.trials.pairs):
        total = int(totals[index])
        row: dict[str, Any] = {"first": first, "second": second, "judgments": total}
        for label, category in ((_FIRST, first), (_SECOND, second), (_NONE, NO_PREFERENCE)):
            count = int(pooled[index, (_FIRST, _SECOND, _NONE).index(label)])
            row[label] = {
                "condition": category,
                "count": count,
                "share": count / total if total else None,
            }
        rows.append(row)
    return rows


def _per_listener(included: Sequence[tuple[Submission, Sequence[Trial]]]) -> list[dict[str, Any]]:
    """Each included listener's raw answer counts."""
    rows = []
    for submission, trials in included:
        scored = [submission.answers[t.trial_id] for t in trials]
        rows.append(
            {
                "listener_id": submission.listener_id,
                "listener_index": submission.listener_index,
                "A": scored.count("A"),
                "B": scored.count("B"),
                NO_PREFERENCE: scored.count(NO_PREFERENCE),
            }
        )
    return rows


def _per_clip(
    protocol: ListeningProtocol,
    included: Sequence[tuple[Submission, Sequence[Trial]]],
    recordings: Sequence[str],
) -> list[dict[str, Any]]:
    """Each recording's counts per pair, over the included listeners."""
    rows = []
    for recording in recordings:
        per_pair = []
        for first, second in protocol.trials.pairs:
            tally = {_FIRST: 0, _SECOND: 0, _NONE: 0}
            for submission, trials in included:
                for trial in trials:
                    if (
                        trial.kind == "scored"
                        and trial.recording == recording
                        and (
                            trial.first,
                            trial.second,
                        )
                        == (first, second)
                    ):
                        tally[_categorise(trial, submission.answers[trial.trial_id])] += 1
            per_pair.append({"first": first, "second": second, **tally})
        rows.append({"recording": recording, "pairs": per_pair})
    return rows


def analyse(protocol: ListeningProtocol, build_dir: Path, answers_dir: Path) -> dict[str, Any]:
    """The analysis document for the answers in *answers_dir*, against the build's sealed key.

    Args:
        protocol: The loaded protocol.
        build_dir: The published build directory, whose key is checked against the manifest.
        answers_dir: A directory of answers files. Every `.json` file in it is read.

    Returns:
        The analysis as a JSON-compatible mapping.

    Raises:
        AnalysisError: If the key does not match the manifest, an answers file is refused, or two
            files name the same listener.
    """
    try:
        key = load_sealed_key(build_dir)
    except PageError as error:
        raise _refuse(f"the sealed key cannot be checked: {error}") from error
    key_digest = _manifest_key_digest(build_dir)
    recordings = recordings_of(key)
    submissions = [_parse_submission(protocol, p) for p in sorted(answers_dir.glob("*.json"))]
    _check_unique(submissions)
    verdicts = []
    for submission in submissions:
        trials = listener_trials(protocol, submission.listener_index, recordings)
        verdicts.append((Verdict(submission, _verdict(submission, trials)), trials))
    included = [(v.submission, trials) for v, trials in verdicts if v.reason is None]
    counts = _counts(protocol, included)
    interval = _bootstrap(counts, protocol)
    return _document(protocol, key_digest, verdicts, included, counts, interval, recordings)


def _check_unique(submissions: Sequence[Submission]) -> None:
    """Refuse two files that name the same listener identifier or index.

    Raises:
        AnalysisError: On a duplicate identifier or index.
    """
    ids = [s.listener_id for s in submissions]
    indices = [s.listener_index for s in submissions]
    if len(set(ids)) != len(ids) or len(set(indices)) != len(indices):
        raise _refuse("two answers files name the same listener")


def _document(
    protocol: ListeningProtocol,
    key_digest: str,
    verdicts: Sequence[tuple[Verdict, Sequence[Trial]]],
    included: Sequence[tuple[Submission, Sequence[Trial]]],
    counts: npt.NDArray[np.int64],
    interval: npt.NDArray[np.float64] | None,
    recordings: Sequence[str],
) -> dict[str, Any]:
    """The analysis document, assembled from the checked parts."""
    excluded = [
        {"listener_id": v.submission.listener_id, "reason": v.reason}
        for v, _ in verdicts
        if v.reason is not None
    ]
    return {
        "label": "descriptive",
        "protocol_sha256": protocol.sha256,
        "key_sha256": key_digest,
        "submitted_listeners": len(verdicts),
        "included_listeners": len(included),
        "excluded_listeners": excluded,
        "pairs": _pair_rows(protocol, counts),
        "per_listener": _per_listener(included),
        "per_clip": _per_clip(protocol, included, recordings),
        "interval": _interval_block(protocol, interval),
    }


def _manifest_key_digest(build_dir: Path) -> str:
    """The sealed key's digest as the build manifest records it, read after the key is checked."""
    manifest = json.loads((build_dir / BUILD_NAME).read_text())
    digest = str(manifest["key_sha256"])
    return digest


def _interval_block(
    protocol: ListeningProtocol, interval: npt.NDArray[np.float64] | None
) -> dict[str, Any]:
    """The interval section: the bounds per pair and category, or why none is reported."""
    settings = protocol.analysis
    base = {
        "method": "percentile bootstrap, resampling listeners",
        "level": settings.interval_level,
        "iterations": settings.interval_iterations,
        "seed": settings.interval_seed,
        "label": "descriptive",
    }
    if interval is None:
        return {**base, "reported": False, "reason": "too few included listeners"}
    bounds = []
    for index, (first, second) in enumerate(protocol.trials.pairs):
        for position, category in enumerate((_FIRST, _SECOND, _NONE)):
            low, high = interval[index, position]
            bounds.append(
                {
                    "first": first,
                    "second": second,
                    "category": category,
                    "low": float(low),
                    "high": float(high),
                }
            )
    return {**base, "reported": True, "bounds": bounds}


def write_analysis(document: Mapping[str, Any], out_dir: Path) -> Path:
    """Write the JSON and the Markdown into a new directory, or nothing.

    Raises:
        AnalysisError: If the directory exists or cannot be written.
    """
    if out_dir.exists():
        raise _refuse(f"output directory {out_dir} already exists")
    markdown = _markdown(document)
    try:
        write_directory_atomically(
            out_dir,
            {
                ANALYSIS_JSON: canonical_json(document).encode("utf-8"),
                ANALYSIS_MD: markdown.encode("utf-8"),
            },
        )
    except OSError as error:
        raise _refuse(f"cannot write {out_dir}: {error}") from error
    return out_dir


def _markdown(document: Mapping[str, Any]) -> str:
    """The Markdown report: the same numbers as the JSON, with the exclusions named."""
    lines = [
        "# Listening study: descriptive results",
        "",
        f"Protocol digest: `{document['protocol_sha256']}`",
        f"Key digest: `{document['key_sha256']}`",
        "",
        f"Submitted listeners: {document['submitted_listeners']}",
        f"Included listeners: {document['included_listeners']}",
        "",
        "## Excluded listeners",
        "",
    ]
    if not document["excluded_listeners"]:
        lines.append("None.")
    for entry in document["excluded_listeners"]:
        lines.append(f"- {entry['listener_id']}: {entry['reason']}")
    lines += ["", "## Judgments by pair", ""]
    for row in document["pairs"]:
        lines.append(f"### {row['first']} against {row['second']}")
        lines.append(f"Judgments: {row['judgments']}")
        for label in (_FIRST, _SECOND, _NONE):
            cell = row[label]
            share = "n/a" if cell["share"] is None else f"{cell['share']:.3f}"
            lines.append(f"- {cell['condition']}: {cell['count']} ({share})")
        lines.append("")
    lines += _interval_lines(document["interval"])
    return "\n".join(lines) + "\n"


def _interval_lines(interval: Mapping[str, Any]) -> list[str]:
    """The interval section of the Markdown report, or the reason it is absent."""
    lines = ["## Descriptive interval", ""]
    if not interval["reported"]:
        lines.append(f"Not reported: {interval['reason']}.")
        return [*lines, ""]
    lines.append(
        f"{interval['level']:.0%} percentile bootstrap over listeners, "
        f"{interval['iterations']} iterations, seed {interval['seed']}."
    )
    for bound in interval["bounds"]:
        lines.append(
            f"- {bound['first']} against {bound['second']}, {bound['category']}: "
            f"{bound['low']:.3f} to {bound['high']:.3f}"
        )
    return [*lines, ""]
