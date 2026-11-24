"""Tests for the canonical JSON serialisation of digested result files."""

from __future__ import annotations

import json
from typing import Any

import pytest

from benchmark.canonical_json import canonical_json, canonical_json_bytes


def _previous_encoder(document: dict[str, Any]) -> str:
    """The expression run_report.py and run_posthoc.py used before the shared encoder existed."""
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _representative_document() -> dict[str, Any]:
    """A document with nested, unsorted keys, non-ASCII text, and the number shapes results use."""
    return {
        "zeta": [{"b": 2, "a": 1}, {"nested": {"y": None, "x": True}}],
        "alpha": {"Ångström": "café", "emoji_free": "naïve", "tab\tkey": 'quote"back\\'},
        "numbers": [0.1, 1e-9, -0.0, 12345678901234567890, 3.0, -2.5e300],
        "empty_list": [],
        "empty_dict": {},
        "flags": [False, True],
    }


def test_matches_the_previous_encoder_byte_for_byte() -> None:
    """The shared encoder writes the same text and bytes as the expression it replaced."""
    document = _representative_document()

    assert canonical_json(document) == _previous_encoder(document)
    assert canonical_json_bytes(document) == _previous_encoder(document).encode("utf-8")


def test_ends_with_one_newline_and_sorts_keys() -> None:
    """Output ends with exactly one newline and lists keys in sorted order at every depth."""
    text = canonical_json({"b": {"d": 1, "c": 2}, "a": 3})

    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert text.index('"a"') < text.index('"b"') < text.index('"c"') < text.index('"d"')


def test_refuses_non_finite_numbers_without_output() -> None:
    """NaN and infinity raise ValueError, so no partial document is produced."""
    for value in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            canonical_json({"value": value})
        with pytest.raises(ValueError):
            canonical_json_bytes({"value": value})
