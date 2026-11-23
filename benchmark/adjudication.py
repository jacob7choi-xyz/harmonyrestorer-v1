"""The adjudication record file: a person's recorded answers for flagged references.

Protocol v10 to v13 admit a flagged candidate only on a recorded answer, given for the exact
bytes it names (`docs/benchmark-protocol.md`, Identity and disjointness). A record names the
candidate, and every reference the screen flagged it against, by source identifier or training
file name, raw SHA-256 and decoded-PCM SHA-256. It gives an answer and its evidence for each
reference. This module reads the file. Whether a record applies to a screen result is decided by
`benchmark.duplicate_screen.adjudicate`.

The file is a JSON list of records. Each record has exactly the keys `candidate` and `answers`,
and each answer has exactly the keys `reference`, `answer` and `evidence`. A missing file holds
no records. Every other departure from this shape is refused rather than repaired, so a record
cannot quietly apply to bytes it was not given for.

Two choices beyond the protocol text: an empty `answers` list is refused, since such a record
answers nothing, and a JSON object that repeats a key is refused, since the last value would
otherwise silently win.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from benchmark._source_validation import SourceOutcomeError, checked_digest, checked_identifier
from benchmark.duplicate_screen import Answer, AnswerSet, RecordingIdentity

__all__ = [
    "AdjudicationError",
    "AdjudicationRecord",
    "answer_sets",
    "read_adjudications",
]

logger = logging.getLogger(__name__)

_RECORD_KEYS = frozenset({"candidate", "answers"})
_IDENTITY_KEYS = frozenset({"identifier", "raw_sha256", "decoded_pcm_sha256"})
_ANSWER_KEYS = frozenset({"reference", "answer", "evidence"})


class AdjudicationError(ValueError):
    """Raised when the adjudication record file is malformed."""


@dataclass(frozen=True)
class AdjudicationRecord:
    """One candidate's recorded answers, each reference's answer with its evidence.

    `answers` and `evidence` are keyed by the same reference identities and hold the same keys.
    """

    candidate: RecordingIdentity
    answers: Mapping[RecordingIdentity, Answer]
    evidence: Mapping[RecordingIdentity, str]

    def answer_set(self) -> AnswerSet:
        """The record as the duplicate screen reads it, without the evidence."""
        return AnswerSet(self.candidate, dict(self.answers))


def _refuse(message: str) -> NoReturn:
    """Log and raise *message* as an AdjudicationError."""
    logger.error("Adjudication records: %s", message)
    raise AdjudicationError(message)


def _no_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """The JSON object's members, refusing a key that appears twice."""
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            _refuse(f"JSON object repeats the key {key!r}")
        seen[key] = value
    return seen


def _exact_keys(data: object, keys: frozenset[str], where: str) -> dict[str, Any]:
    """*data* as a dict holding exactly *keys*, or a refusal naming *where*."""
    if not isinstance(data, dict) or set(data) != keys:
        _refuse(f"{where} must have exactly the keys {sorted(keys)}")
    return data


def _digest(value: object, where: str) -> str:
    """A 64-character lowercase SHA-256 digest, or a refusal."""
    if not isinstance(value, str):
        _refuse(f"{where} must be a string")
    try:
        return checked_digest(value, where)
    except SourceOutcomeError as error:
        logger.error("Adjudication records: %s", error)
        raise AdjudicationError(str(error)) from error


def _identity(data: object, where: str) -> RecordingIdentity:
    """A recording named by identifier and both digests."""
    fields = _exact_keys(data, _IDENTITY_KEYS, where)
    identifier = fields["identifier"]
    if not isinstance(identifier, str):
        _refuse(f"{where}.identifier must be a string")
    try:
        checked_identifier(identifier, f"{where}.identifier")
    except SourceOutcomeError as error:
        logger.error("Adjudication records: %s", error)
        raise AdjudicationError(str(error)) from error
    raw = _digest(fields["raw_sha256"], f"{where}.raw_sha256")
    decoded = _digest(fields["decoded_pcm_sha256"], f"{where}.decoded_pcm_sha256")
    return RecordingIdentity(identifier, raw, decoded)


def _answer(data: object, where: str) -> tuple[RecordingIdentity, Answer, str]:
    """One answer: the reference it is for, the answer, and the evidence for it."""
    fields = _exact_keys(data, _ANSWER_KEYS, where)
    reference = _identity(fields["reference"], f"{where}.reference")
    try:
        answer = Answer(fields["answer"])
    except ValueError as error:
        message = f"{where}.answer {fields['answer']!r} is not one of {[a.value for a in Answer]}"
        logger.error("Adjudication records: %s", message)
        raise AdjudicationError(message) from error
    evidence = fields["evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        _refuse(f"{where}.evidence must be a non-empty string")
    return reference, answer, evidence


def _record(data: object, position: int) -> AdjudicationRecord:
    """One record, with every reference named once."""
    where = f"record {position}"
    fields = _exact_keys(data, _RECORD_KEYS, where)
    candidate = _identity(fields["candidate"], f"{where}.candidate")
    entries = fields["answers"]
    if not isinstance(entries, list) or not entries:
        _refuse(f"{where}.answers must be a non-empty list")
    answers: dict[RecordingIdentity, Answer] = {}
    evidence: dict[RecordingIdentity, str] = {}
    for index, entry in enumerate(entries):
        reference, answer, text = _answer(entry, f"{where}.answers[{index}]")
        if reference in answers:
            _refuse(f"{where} names the reference {reference.identifier!r} twice")
        answers[reference] = answer
        evidence[reference] = text
    return AdjudicationRecord(candidate, answers, evidence)


def read_adjudications(path: Path) -> tuple[AdjudicationRecord, ...]:
    """The records in the file at *path*, in file order; none if the file does not exist.

    Args:
        path: The adjudication record file. An absolute or relative path, read as given.

    Returns:
        Every record, validated.

    Raises:
        AdjudicationError: If the file is not JSON, repeats a key, is not a list of records of
            the exact shape described above, names a reference twice in one record, gives an
            answer outside `Answer`, or gives no evidence.
    """
    if not path.exists():
        logger.info("No adjudication records at %s", path)
        return ()
    try:
        data = json.loads(path.read_bytes(), object_pairs_hook=_no_repeated_keys)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        logger.error("Adjudication records: %s is not valid JSON: %s", path, error)
        raise AdjudicationError(f"{path} is not valid JSON") from error
    if not isinstance(data, list):
        _refuse(f"{path} must hold a JSON list of records")
    records = tuple(_record(entry, position) for position, entry in enumerate(data))
    logger.info("Read %d adjudication record(s) from %s", len(records), path)
    return records


def answer_sets(records: Sequence[AdjudicationRecord]) -> tuple[AnswerSet, ...]:
    """The records as the duplicate screen's answer sets, in the same order."""
    return tuple(record.answer_set() for record in records)
