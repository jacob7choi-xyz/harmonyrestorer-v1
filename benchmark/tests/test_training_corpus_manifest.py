"""Tests for the frozen training corpus identity.

Disjointness between training and evaluation has to be checkable against a fixed
identity rather than asserted. These tests exercise the manifest itself; screening
candidates against it arrives with the selection walk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[1] / "manifests" / "training_corpus.json"
EXPECTED_RECORDINGS = 145


def manifest() -> dict:
    """Load the frozen training corpus manifest."""
    return json.loads(MANIFEST_PATH.read_text())


def test_manifest_covers_every_training_recording() -> None:
    data = manifest()
    assert data["count"] == len(data["recordings"]) == EXPECTED_RECORDINGS


def test_hashes_are_full_length_and_unique() -> None:
    hashes = [r["sha256"] for r in manifest()["recordings"]]
    assert all(len(h) == 64 for h in hashes)
    assert len(set(hashes)) == len(hashes)


def test_filenames_are_unique() -> None:
    """Two entries sharing a filename would make the corpus ambiguous to screen against."""
    names = [r["filename"] for r in manifest()["recordings"]]
    assert len(set(names)) == len(names)


def test_corpus_digest_is_reproducible_from_its_parts() -> None:
    data = manifest()
    recomputed = hashlib.sha256(
        "\n".join(r["sha256"] for r in data["recordings"]).encode()
    ).hexdigest()
    assert recomputed == data["corpus_sha256"]
