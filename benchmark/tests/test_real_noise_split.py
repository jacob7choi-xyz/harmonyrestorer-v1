"""Tests for the model v2 Phase 0 disc split of the Aalto noise set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.listening.manifest import universe_path
from benchmark.listening.protocol import load_listening_protocol
from benchmark.measurements import real_noise_split as split


class TestDiscNumber:
    """A Great 78 identifier's disc is its seven-digit gbia number, whatever the side suffix."""

    def test_both_sides_of_a_disc_share_the_number(self) -> None:
        side_a = "78_1-minuet-in-g-2-gavotte-in-d_efrem-zimbalist_gbia0520150a"
        side_b = "78_1-le-cygne-the-swan-2-waltz_efrem-zimbalist_gbia0520150b"
        assert split.disc_number(side_a) == split.disc_number(side_b) == "0520150"

    def test_odd_suffixes_still_give_the_disc(self) -> None:
        assert split.disc_number("78__moran-and-mack-mack_gbia0007600-2") == "0007600"
        assert split.disc_number("78_2-easter-chimes_gbia8000356aa") == "8000356"

    def test_an_identifier_without_a_disc_is_refused(self) -> None:
        with pytest.raises(split.NoiseSplitError, match="no gbia disc number"):
            split.disc_number("musopen-chopin-complete-works-flac/Op. 31.flac")


class TestSplitDiscs:
    """The split is seeded, disjoint, and never holds an excluded disc."""

    DISCS = tuple(f"{number:07d}" for number in range(200))

    def test_the_same_seed_gives_the_same_split(self) -> None:
        first = split.split_discs(self.DISCS, frozenset(), 7)
        assert split.split_discs(self.DISCS, frozenset(), 7) == first
        assert split.split_discs(self.DISCS, frozenset(), 8) != first

    def test_parts_are_disjoint_cover_the_rest_and_have_the_fixed_shares(self) -> None:
        excluded = frozenset({"0000003", "0000150"})
        parts = split.split_discs(self.DISCS, excluded, 7)
        sets = [set(parts.evaluation), set(parts.validation), set(parts.training)]
        assert sum(len(part) for part in sets) == len(set().union(*sets)) == 198
        assert set().union(*sets) == set(self.DISCS) - excluded
        assert (len(parts.evaluation), len(parts.validation)) == (
            round(198 * 0.15),
            round(198 * 0.05),
        )

    def test_too_few_discs_are_refused(self) -> None:
        with pytest.raises(split.NoiseSplitError, match="too few discs"):
            split.split_discs(["0000001", "0000002"], frozenset(), 7)


class TestStudyDiscs:
    """Every disc of the study listing is excluded, kept or not, both sides."""

    def test_the_pinned_listing_gives_both_sides_of_a_selected_disc(self) -> None:
        raw = universe_path(load_listening_protocol()).read_bytes()
        discs = split.study_discs(raw)
        # Zimbalist's Swan (a selected study item) and its other side share disc 0520150.
        assert "0520150" in discs
        document = json.loads(raw)
        assert len(discs) <= len(document["items"]) + len(document["excluded"])

    def test_a_changed_listing_is_refused(self) -> None:
        raw = universe_path(load_listening_protocol()).read_bytes()
        with pytest.raises(split.NoiseSplitError, match="study universe refused"):
            split.study_discs(raw + b" ")


class TestFrozenSplit:
    """The tracked manifest loads only with its pinned digest and keeps its parts apart."""

    def test_the_tracked_manifest_loads_and_has_no_study_disc(self) -> None:
        frozen = split.load_split()
        raw = universe_path(load_listening_protocol()).read_bytes()
        held = set(frozen.evaluation) | set(frozen.validation) | set(frozen.training)
        assert not held & split.study_discs(raw)
        assert (len(frozen.evaluation), len(frozen.validation)) == (174, 58)

    def test_a_manifest_with_another_digest_is_refused(self, tmp_path: Path) -> None:
        changed = tmp_path / "split.json"
        changed.write_bytes(split.SPLIT_PATH.read_bytes() + b"\n")
        with pytest.raises(split.NoiseSplitError, match="not the pinned one"):
            split.load_split(changed)

    def test_parts_that_share_a_disc_are_refused(self) -> None:
        document = json.loads(split.SPLIT_PATH.read_bytes())
        document["validation"] = sorted([*document["validation"], document["evaluation"][0]])
        with pytest.raises(split.NoiseSplitError, match="share a disc"):
            split.parse_split(json.dumps(document).encode())
