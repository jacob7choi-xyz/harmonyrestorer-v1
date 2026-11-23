"""Tests for reading the adjudication record file: its shape, its refusals and its answers."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.adjudication import AdjudicationError, answer_sets, read_adjudications
from benchmark.duplicate_screen import Answer, AnswerSet, RecordingIdentity

RAW = "a" * 64
DECODED = "b" * 64
OTHER_RAW = "c" * 64
OTHER_DECODED = "d" * 64


def identity(identifier: str, raw: str, decoded: str) -> dict[str, str]:
    """A recording as the file names it."""
    return {"identifier": identifier, "raw_sha256": raw, "decoded_pcm_sha256": decoded}


def base_record() -> dict[str, Any]:
    """A record that the reader accepts: one candidate, two answers."""
    return {
        "candidate": identity(
            "musopen-chopin-complete-works-flac/Chopin/Nocturne.flac", RAW, DECODED
        ),
        "answers": [
            {
                "reference": identity(
                    "MusopenCollectionAsFlac__Bach.flac", OTHER_RAW, OTHER_DECODED
                ),
                "answer": "different_performance",
                "evidence": "Different movement, different pianist, checked from file names.",
            },
            {
                "reference": identity("musicnet/1727", "e" * 64, "f" * 64),
                "answer": "same_performance",
                "evidence": "Same session take named in the collection metadata.",
            },
        ],
    }


def write_records(path: Path, records: list[dict[str, Any]]) -> Path:
    """Write a record list as the file the reader reads."""
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def test_a_missing_file_holds_no_records(tmp_path: Path) -> None:
    """With no file there are no recorded answers, so every flag awaits adjudication."""
    assert read_adjudications(tmp_path / "absent.json") == ()


def test_a_record_reads_back_with_its_answers_and_evidence(tmp_path: Path) -> None:
    """Each answer is keyed by its reference's three values and keeps its evidence."""
    (record,) = read_adjudications(write_records(tmp_path / "a.json", [base_record()]))
    bach = RecordingIdentity("MusopenCollectionAsFlac__Bach.flac", OTHER_RAW, OTHER_DECODED)
    musicnet = RecordingIdentity("musicnet/1727", "e" * 64, "f" * 64)
    assert record.candidate == RecordingIdentity(
        "musopen-chopin-complete-works-flac/Chopin/Nocturne.flac", RAW, DECODED
    )
    assert record.answers == {bach: Answer.DIFFERENT_PERFORMANCE, musicnet: Answer.SAME_PERFORMANCE}
    assert record.evidence[musicnet] == "Same session take named in the collection metadata."


def test_answer_sets_keep_file_order_and_drop_the_evidence(tmp_path: Path) -> None:
    """The duplicate screen receives the answers in file order, without the evidence text."""
    second = copy.deepcopy(base_record())
    second["candidate"] = identity("musicnet/2", OTHER_RAW, OTHER_DECODED)
    records = read_adjudications(write_records(tmp_path / "a.json", [base_record(), second]))
    sets = answer_sets(records)
    assert [s.candidate.identifier for s in sets] == [
        "musopen-chopin-complete-works-flac/Chopin/Nocturne.flac",
        "musicnet/2",
    ]
    assert all(isinstance(s, AnswerSet) for s in sets)
    assert sets[0].answers == records[0].answers


def test_an_empty_list_holds_no_records(tmp_path: Path) -> None:
    """A file with an empty list is a valid file that records nothing."""
    assert read_adjudications(write_records(tmp_path / "a.json", [])) == ()


def mutated(change: Any) -> str:
    """The text of a one-record file with *change* applied to a fresh valid record."""
    record = base_record()
    change(record)
    return json.dumps([record])


def without_key(path: tuple[str, ...]) -> Any:
    """A change that removes the key at *path* from the record."""

    def change(record: dict[str, Any]) -> None:
        target: Any = record
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return change


def set_at(path: tuple[str, ...], value: object) -> Any:
    """A change that sets the value at *path* in the record."""

    def change(record: dict[str, Any]) -> None:
        target: Any = record
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


def twice(record: dict[str, Any]) -> None:
    """A change that names the first reference a second time in the same record."""
    record["answers"].append(copy.deepcopy(record["answers"][0]))


def extra_record_key(record: dict[str, Any]) -> None:
    """A change that adds a key the record shape does not have."""
    record["note"] = "unreviewed"


def extra_candidate_key(record: dict[str, Any]) -> None:
    """A change that adds a key to the candidate identity."""
    record["candidate"]["source_url"] = "https://example.invalid/x"


def repeated_key_text() -> str:
    """A record whose candidate object repeats a key, which JSON would otherwise accept."""
    text = json.dumps([base_record()])
    return text.replace('"raw_sha256"', '"raw_sha256": "' + OTHER_RAW + '", "raw_sha256"', 1)


MALFORMED: list[tuple[str, Any]] = [
    ("unknown answer", set_at(("answers", 0, "answer"), "probably_same")),
    ("blank evidence", set_at(("answers", 0, "evidence"), "   ")),
    ("answer without evidence", without_key(("answers", 0, "evidence"))),
    ("reference named twice", twice),
    ("extra record key", extra_record_key),
    ("extra candidate key", extra_candidate_key),
    ("empty answers", set_at(("answers",), [])),
    ("uppercase candidate digest", set_at(("candidate", "raw_sha256"), RAW.upper())),
    ("short reference digest", set_at(("answers", 0, "reference", "decoded_pcm_sha256"), "b" * 63)),
    ("padded identifier", set_at(("candidate", "identifier"), " padded ")),
    ("non-string digest", set_at(("candidate", "raw_sha256"), 7)),
]


@pytest.mark.parametrize("change", [row[1] for row in MALFORMED], ids=[row[0] for row in MALFORMED])
def test_a_malformed_record_is_refused(tmp_path: Path, change: Any) -> None:
    """A record outside the exact shape is refused, so no answer is ever read from it."""
    path = tmp_path / "a.json"
    path.write_text(mutated(change), encoding="utf-8")
    with pytest.raises(AdjudicationError):
        read_adjudications(path)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (json.dumps({"records": [base_record()]}), "must hold a JSON list"),
        ("[{", "is not valid JSON"),
        (repeated_key_text(), "repeats the key 'raw_sha256'"),
    ],
    ids=["top level is an object", "not JSON", "repeated JSON key"],
)
def test_a_file_that_is_not_a_record_list_is_refused(
    tmp_path: Path, text: str, reason: str
) -> None:
    """A file that is not a JSON list of records is refused, and the refusal names the reason."""
    path = tmp_path / "a.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(AdjudicationError, match=reason):
        read_adjudications(path)
