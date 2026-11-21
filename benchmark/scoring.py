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

A recording whose frames were all ineligible is not part of the expected population here:
v6 leaves it out of the across-recording count, so the caller does not pass it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from benchmark.aggregation import RecordingScores
from benchmark.frame_scoring import FrameOutcome, FrameScore
from benchmark.protocol import Protocol

__all__ = [
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


def recording_scores(
    expected: Expected, results: Mapping[str, Sequence[SystemFrameResult]], protocol: Protocol
) -> list[RecordingScores]:
    """Aggregation input, one entry per expected recording in the expected order.

    Raises:
        PopulationError: If any of the four gates does not hold; the gates that failed are
            named.
    """
    gates = population_gates(expected, results, protocol)
    failed = [name for name, held in gates.items() if not held]
    if failed:
        raise PopulationError(f"gates do not hold: {failed}")
    lookups = {system: _lookup(results[system]) for system in SYSTEMS}
    scores = []
    for recording, indices in expected.items():
        per_system = {
            system: {
                metric: [lookups[system][(recording, i)].results[metric].value for i in indices]
                for metric in protocol.metrics.primary
            }
            for system in SYSTEMS
        }
        scores.append(RecordingScores(recording, per_system[OPGAN], per_system[UVR]))
    return scores
