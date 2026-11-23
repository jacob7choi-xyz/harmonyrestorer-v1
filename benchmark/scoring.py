"""Population checks and coverage across systems, before anything is aggregated (protocol v6).

Both systems must receive an identical population, checked by identifier at two levels:
the recording identifiers each system has results for must equal the frozen expected set,
and within each recording the frame indices must equal that recording's frozen eligible
set. Equal counts are not enough. The expected population is always the caller's frozen
one, never inferred from the results, because deriving it from results would make coverage
complete by construction.

Coverage is computed per system: frames with a valid output (a `FrameScore`) over the
expected eligible frames. A `SystemFailure` stays in the population and lowers coverage.
The headline comparison needs coverage of exactly the protocol's `coverage_required` for
both systems; `recording_scores` refuses to build aggregation input otherwise.

The four gates computed here are named as `publication_gates.required` names them. The
others (the protocol and candidate universe digests, and the training screen) are decided
elsewhere.

The degraded input (v13's diagnostic baseline) is scored like a system output and passed under
`BASELINE`. It must cover the expected frames exactly, but it has no coverage gate, because it is
not a system and a gap in it is a harness fault rather than a system's failure.

A recording whose frames were all ineligible is not part of the expected population here.
v6 to v10 kept it in its partition and left it out of the across-recording count; from v11
the selection walk rejects it before admission, so no admitted recording has none.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from operator import attrgetter

from benchmark.aggregation import RecordingScores, SystemValues
from benchmark.frame_scoring import FrameOutcome, FrameScore
from benchmark.metrics import MetricResult
from benchmark.protocol import Protocol

__all__ = [
    "BASELINE",
    "OPGAN",
    "SYSTEMS",
    "UVR",
    "PopulationError",
    "SystemFrameResult",
    "coverage",
    "population_gates",
    "recording_scores",
]

OPGAN = "opgan"
UVR = "uvr"
SYSTEMS = (OPGAN, UVR)
# The degraded input, scored like a system output (v13). It is not a system, so it has no
# coverage gate and is not in SYSTEMS. Its results are keyed by this name in `results`.
BASELINE = "degraded_input"

_RECORDINGS_GATE = "population_ids_equal_across_systems"
_FRAMES_GATE = "frame_population_ids_equal_across_systems"
_COVERAGE_GATES = {OPGAN: "coverage_opgan_complete", UVR: "coverage_uvr_complete"}

Expected = Mapping[str, Sequence[int]]


class PopulationError(Exception):
    """Raised when results cannot be aggregated because a population gate does not hold."""


@dataclass(frozen=True)
class SystemFrameResult:
    """One system's outcome for one frame, identified by recording and frame index."""

    recording_id: str
    frame_index: int
    outcome: FrameOutcome


def _frames_by_recording(results: Sequence[SystemFrameResult]) -> dict[str, list[int]]:
    """Each recording's frame indices, in the order the results list them."""
    frames: dict[str, list[int]] = {}
    for result in results:
        frames.setdefault(result.recording_id, []).append(result.frame_index)
    return frames


def _matches(expected: Expected, results: Sequence[SystemFrameResult]) -> tuple[bool, bool]:
    """Whether one system's recordings, and each recording's frames, equal the expected set.

    Frames are compared as sorted lists, so a repeated frame is a mismatch rather than
    being absorbed by a set.
    """
    frames = _frames_by_recording(results)
    recordings_equal = set(frames) == set(expected)
    frames_equal = recordings_equal and all(
        sorted(frames[recording]) == sorted(indices) for recording, indices in expected.items()
    )
    return recordings_equal, frames_equal


def coverage(expected: Expected, results: Sequence[SystemFrameResult]) -> float:
    """Frames with a valid output over the expected eligible frames, for one system.

    Only results for expected frames count, each at most once, so a result for a frame
    outside the population cannot raise coverage.
    """
    eligible = {(recording, index) for recording, indices in expected.items() for index in indices}
    if not eligible:
        raise PopulationError("the expected population has no eligible frames")
    valid = {
        (result.recording_id, result.frame_index)
        for result in results
        if isinstance(result.outcome, FrameScore)
    }
    return len(valid & eligible) / len(eligible)


def population_gates(
    expected: Expected, results: Mapping[str, Sequence[SystemFrameResult]], protocol: Protocol
) -> dict[str, bool]:
    """The two population gates and the two coverage gates, by their protocol names."""
    matches = {system: _matches(expected, results.get(system, ())) for system in SYSTEMS}
    gates = {
        _RECORDINGS_GATE: all(recordings for recordings, _ in matches.values()),
        _FRAMES_GATE: all(frames for _, frames in matches.values()),
    }
    required = protocol.scoring.coverage_required
    for system in SYSTEMS:
        gates[_COVERAGE_GATES[system]] = coverage(expected, results.get(system, ())) == required
    return gates


_primary_table = attrgetter("results")
_shift_table = attrgetter("shift_tolerant")


def _lookup(results: Sequence[SystemFrameResult]) -> dict[tuple[str, int], FrameScore]:
    """One system's scored frames by recording and frame index, built once per system.

    Raises:
        PopulationError: If a frame is not scored. The coverage gate rules that out before
            this runs; raising rather than asserting keeps the guarantee under `python -O`.
    """
    lookup = {}
    for result in results:
        if not isinstance(result.outcome, FrameScore):
            raise PopulationError(
                f"frame {result.frame_index} of {result.recording_id} has no score"
            )
        lookup[(result.recording_id, result.frame_index)] = result.outcome
    return lookup


def _column(
    lookup: Mapping[tuple[str, int], FrameScore],
    recording: str,
    indices: Sequence[int],
    table: Callable[[FrameScore], Mapping[str, MetricResult]],
    metric: str,
) -> list[float]:
    """One metric's value on each expected frame of one recording, from one table of a score.

    Raises:
        PopulationError: If a frame has no score for the metric in that table.
    """
    values = []
    for index in indices:
        scores = table(lookup[(recording, index)])
        if metric not in scores:
            raise PopulationError(f"frame {index} of {recording} has no {metric} score")
        values.append(scores[metric].value)
    return values


def _system_values(
    lookup: Mapping[tuple[str, int], FrameScore],
    recording: str,
    indices: Sequence[int],
    protocol: Protocol,
) -> SystemValues:
    """One source's values for one recording: every primary and shift-tolerant metric."""
    primary = {
        metric: _column(lookup, recording, indices, _primary_table, metric)
        for metric in protocol.metrics.primary
    }
    shift = {
        metric: _column(lookup, recording, indices, _shift_table, metric)
        for metric in protocol.metrics.shift_tolerant.metrics
    }
    return SystemValues(primary, shift)


def _check_baseline(expected: Expected, results: Sequence[SystemFrameResult]) -> None:
    """The baseline must score every expected frame, once, with no failure.

    Raises:
        PopulationError: If the baseline's population differs from the expected one, or any
            baseline frame has no score. The baseline is not a system, so it has no coverage
            gate; a gap in it is a harness fault, not a system's failure.
    """
    recordings_equal, frames_equal = _matches(expected, results)
    if not (recordings_equal and frames_equal):
        raise PopulationError(f"the {BASELINE} results do not match the expected frames")
    _lookup(results)


def recording_scores(
    expected: Expected,
    results: Mapping[str, Sequence[SystemFrameResult]],
    protocol: Protocol,
    clusters: Mapping[str, str],
) -> list[RecordingScores]:
    """Aggregation input, one entry per expected recording in the expected order.

    Args:
        expected: The frozen eligible frames of each recording.
        results: Each system's results under its name, and the degraded input's under BASELINE.
        protocol: The loaded protocol.
        clusters: Each expected recording's performer cluster.

    Raises:
        PopulationError: If any of the four gates does not hold, if the baseline does not cover
            the expected frames, or if a recording has no cluster; the gates that failed are named.
    """
    gates = population_gates(expected, results, protocol)
    failed = [name for name, held in gates.items() if not held]
    if failed:
        raise PopulationError(f"gates do not hold: {failed}")
    _check_baseline(expected, results.get(BASELINE, ()))
    lookups = {system: _lookup(results[system]) for system in SYSTEMS}
    baseline = _lookup(results[BASELINE])
    scores = []
    for recording, indices in expected.items():
        if recording not in clusters:
            raise PopulationError(f"recording {recording} has no performer cluster")
        scores.append(
            RecordingScores(
                recording,
                clusters[recording],
                _system_values(lookups[OPGAN], recording, indices, protocol),
                _system_values(lookups[UVR], recording, indices, protocol),
                _system_values(baseline, recording, indices, protocol),
            )
        )
    return scores
