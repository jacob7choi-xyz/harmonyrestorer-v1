"""Tests for the selection walk's order and the automatic exclusions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmark.protocol import load_protocol
from benchmark.selection import (
    ExclusionRule,
    SelectionError,
    TrainingIdentities,
    exclusions,
    is_candidate,
    training_filename,
    training_identities,
    walk_order,
)

ITEM = "MusopenCollectionAsFlac"
MANIFEST = Path(__file__).resolve().parents[1] / "manifests" / "training_corpus.json"
TRAINING = TrainingIdentities(
    filenames=frozenset({f"{ITEM}__Bach_Goldberg_01-Aria.flac"}),
    raw_sha256=frozenset({"a" * 64}),
    decoded_sha256=frozenset({"b" * 64}),
)


class TestWalkOrder:
    """Code-point sort, then the seeded permutation over positions."""

    def test_the_order_is_the_sorted_list_indexed_by_the_permutation(self) -> None:
        identifiers = [f"{ITEM}/{name}" for name in ("b.flac", "a.flac", "d.flac", "c.flac")]
        ordered = sorted(identifiers)
        permutation = np.random.Generator(np.random.PCG64(7)).permutation(4)
        assert walk_order(identifiers, 7) == tuple(ordered[i] for i in permutation)

    def test_the_sort_is_by_code_point(self) -> None:
        """Uppercase before lowercase, and U+00E9 after z, where a locale sort would put it
        beside e."""
        identifiers = ["\u00e9.flac", "z.flac", "a.flac", "Z.flac", "f.flac"]
        ordered = ["Z.flac", "a.flac", "f.flac", "z.flac", "\u00e9.flac"]
        permutation = np.random.Generator(np.random.PCG64(1)).permutation(5)
        assert walk_order(identifiers, 1) == tuple(ordered[i] for i in permutation)

    def test_the_order_does_not_depend_on_the_input_order(self) -> None:
        identifiers = [f"{ITEM}/{i:03d}.flac" for i in range(50)]
        seed = load_protocol().selection.seed
        assert walk_order(identifiers, seed) == walk_order(reversed(identifiers), seed)
        assert sorted(walk_order(identifiers, seed)) == identifiers

    def test_a_repeated_identifier_is_refused(self) -> None:
        with pytest.raises(SelectionError, match="repeats"):
            walk_order(["a.flac", "b.flac", "a.flac"], 1)


class TestTrainingFilename:
    """The flattened form training files were named in."""

    def test_a_nested_path_is_flattened(self) -> None:
        assert (
            training_filename(f"{ITEM}/Bach/Goldberg/01.flac", ITEM)
            == f"{ITEM}__Bach_Goldberg_01.flac"
        )

    def test_every_training_file_name_is_in_the_form_rule_1_builds(self) -> None:
        """Checks the item prefix and the double underscore against the 145 names
        `dataset/acquire.py` wrote. The manifest does not record the original paths, so the
        slash replacement itself is checked by the nested-path test above, not here."""
        prefix = f"{ITEM}__"
        for entry in json.loads(MANIFEST.read_text())["recordings"]:
            name = entry["filename"]
            assert name.startswith(prefix), name
            assert training_filename(f"{ITEM}/{name.removeprefix(prefix)}", ITEM) == name

    def test_two_paths_can_flatten_alike(self) -> None:
        """Not one-to-one, which can only exclude more under rule 1."""
        assert training_filename(f"{ITEM}/a/b_c.flac", ITEM) == training_filename(
            f"{ITEM}/a_b/c.flac", ITEM
        )

    @pytest.mark.parametrize(
        "identifier",
        ["OtherItem/a.flac", ITEM, f"{ITEM}/"],
        ids=["another_item", "no_path", "empty_path"],
    )
    def test_an_identifier_outside_the_item_is_refused(self, identifier: str) -> None:
        with pytest.raises(SelectionError):
            training_filename(identifier, ITEM)


class TestExclusions:
    """Every rule is evaluated and every match returned, in rule order."""

    def test_an_unrelated_candidate_meets_no_rule(self) -> None:
        met = exclusions(
            f"{ITEM}/Other/x.flac", raw_sha256="c" * 64, decoded_sha256="d" * 64,
            collection=ITEM, training=TRAINING,
        )  # fmt: skip
        assert met == ()

    @pytest.mark.parametrize(
        ("identifier", "raw", "decoded", "expected"),
        [
            (
                f"{ITEM}/Bach/Goldberg_01-Aria.flac",
                "c" * 64,
                "d" * 64,
                (ExclusionRule.SOURCE_IDENTITY,),
            ),
            (f"{ITEM}/Other/x.flac", "a" * 64, "d" * 64, (ExclusionRule.RAW_SHA256,)),
            (f"{ITEM}/Other/x.flac", "c" * 64, "b" * 64, (ExclusionRule.DECODED_PCM_SHA256,)),
            (f"{ITEM}/Bach_Goldberg_01-Aria.flac", "a" * 64, "b" * 64, tuple(ExclusionRule)),
        ],
        ids=["identity", "raw_digest", "decoded_digest", "all_three"],
    )
    def test_each_rule_is_met_on_its_own_evidence(
        self, identifier: str, raw: str, decoded: str, expected: tuple[ExclusionRule, ...]
    ) -> None:
        met = exclusions(
            identifier, raw_sha256=raw, decoded_sha256=decoded, collection=ITEM, training=TRAINING
        )
        assert met == expected


class TestTrainingIdentities:
    """Built from the frozen training manifest."""

    def test_the_manifest_gives_every_name_and_digest(self) -> None:
        recordings = json.loads(MANIFEST.read_text())["recordings"]
        decoded = {entry["filename"]: f"{index:064x}" for index, entry in enumerate(recordings)}
        identities = training_identities(recordings, decoded)
        assert len(identities.filenames) == len(identities.raw_sha256) == 145
        assert identities.decoded_sha256 == frozenset(decoded.values())
        first = recordings[0]
        assert first["filename"] in identities.filenames
        assert first["sha256"] in identities.raw_sha256

    @pytest.mark.parametrize(
        ("manifest", "decoded"),
        [
            ([{"filename": "a", "sha256": "a" * 64}, {"filename": "b", "sha256": "b" * 64}],
             {"a": "c" * 64}),
            ([{"filename": "a", "sha256": "a" * 64}], {"a": "c" * 64, "b": "d" * 64}),
            ([{"filename": "a", "sha256": "a" * 64}, {"filename": "a", "sha256": "b" * 64}],
             {"a": "c" * 64}),
            ([{"filename": "a", "sha256": "A" * 64}], {"a": "c" * 64}),
            ([{"filename": "a", "sha256": "a" * 64}], {"a": "c" * 63}),
        ],
        ids=["decoded_missing", "decoded_unknown", "name_repeated", "raw_uppercase",
             "decoded_short"],
    )  # fmt: skip
    def test_an_incomplete_or_malformed_corpus_is_refused(
        self, manifest: list[dict[str, object]], decoded: dict[str, str]
    ) -> None:
        with pytest.raises(SelectionError):
            training_identities(manifest, decoded)


class TestCandidateFilter:
    """v9's filter: the final path component's suffix, lowercased, is .flac."""

    def test_the_protocol_names_the_filter_this_module_implements(self) -> None:
        filter_token = load_protocol().selection.candidate_filter
        assert filter_token == "final_path_component_suffix_lowercased_is_flac"

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("Bach/Goldberg/01.flac", True),
            ("Bach/01.FLAC", True),
            ("Bach/01.flac.txt", False),
            ("Bach.flac/01.mp3", False),
            ("Bach/flac", False),
        ],
        ids=["nested", "upper_case", "other_final_suffix", "flac_directory", "no_suffix"],
    )
    def test_only_the_final_component_s_suffix_decides(self, path: str, expected: bool) -> None:
        assert is_candidate(path) is expected

    def test_every_training_recording_passes_the_filter(self) -> None:
        """The universe counts them; rule 1 removes them before any fetch."""
        prefix = f"{ITEM}__"
        for entry in json.loads(MANIFEST.read_text())["recordings"]:
            assert is_candidate(entry["filename"].removeprefix(prefix)), entry["filename"]


class TestRejectionCodes:
    """The rules against training carry v9's codes, in the protocol's order."""

    def test_the_codes_are_the_protocol_s_training_codes_in_rule_order(self) -> None:
        codes = load_protocol().selection.rejection_codes
        training = tuple(code for code in codes if code.startswith("TRAINING_"))
        assert tuple(ExclusionRule) == training
