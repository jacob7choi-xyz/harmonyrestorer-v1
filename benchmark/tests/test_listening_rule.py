"""Tests for the listening universe rule and the metadata parsing it reads."""

from __future__ import annotations

import pytest

from benchmark.listening.metadata import (
    ItemFacts,
    item_facts,
    main_flac,
    parse_description,
    performer_key,
    strip_html,
    year_of,
)
from benchmark.listening.protocol import load_protocol_for_listing
from benchmark.listening.rule import Reason, UniverseRule, decide


def _flac(name: str = "track.flac", source: str = "original") -> dict[str, object]:
    """A listed main-FLAC entry with the fields the rule and the selection read.

    Args:
        name: The file name.
        source: The archive's source field; only "original" qualifies.

    Returns:
        One file entry from an item's metadata.
    """
    return {
        "name": name,
        "source": source,
        "format": "24bit Flac",
        "size": "1000",
        "md5": "a" * 32,
        "sha1": "b" * 40,
        "length": "120.5",
    }


def _facts(
    description: str, title: str = "Item", files: list[dict[str, object]] | None = None
) -> ItemFacts:
    """Facts for a synthetic item with the given description, title and files.

    Args:
        description: The HTML description, with Performer and instrumentation lines.
        title: The title.
        files: The file entries; one main FLAC when None.

    Returns:
        The item facts.
    """
    metadata = {
        "identifier": "78_fixture",
        "title": title,
        "date": "1912",
        "description": description,
    }
    return item_facts("78_fixture", metadata, files if files is not None else [_flac()])


@pytest.fixture
def rule() -> UniverseRule:
    """The compiled rule the frozen protocol names."""
    return load_protocol_for_listing().universe.rule


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        ("Performer: Ada Singer; Ben Bass<br>Soprano and Tenor; sung duet; Piano", Reason.VOCAL),
        ("Performer: Joe Reed<br>Accordion solo.", Reason.ACCORDION),
        ("Performer: Sam Brass<br>Cornet Solo; Orch. Acc.", Reason.NOT_SOLO_OR_STRING_ENSEMBLE),
        ("Performer: Jan Fiddle<br>Violin solo; Pianoforte.", Reason.KEPT),
        ("Performer: Gerard Quartet<br>String Quartet; Op. 12", Reason.KEPT),
        ("Performer: Ida Missing", Reason.NO_INSTRUMENTATION_LINE),
    ],
)
def test_rule_decides_each_example_row(
    rule: UniverseRule, description: str, expected: Reason
) -> None:
    """Each example row gets the reason the frozen rule gives it."""
    assert decide(_facts(description), rule).reason is expected


def test_rule_drops_an_item_with_two_main_flacs(rule: UniverseRule) -> None:
    """An item with two original 24-bit FLAC files is excluded before the patterns are read."""
    files = [_flac("a.flac"), _flac("b.flac")]
    decision = decide(_facts("Performer: Jan Fiddle<br>Violin solo.", files=files), rule)
    assert decision.reason is Reason.MAIN_FLAC_COUNT
    assert decision.kept is False


def test_rule_checks_the_title_for_vocal_terms(rule: UniverseRule) -> None:
    """The vocal pattern reads the title, so an instrumental item with a vocal title is dropped."""
    decision = decide(
        _facts("Performer: Jan Fiddle<br>Violin solo.", title="Chorus of Dervishes"), rule
    )
    assert decision.reason is Reason.VOCAL
    assert decision.matched == ("chorus",)


def test_rule_gives_the_vocal_reason_before_the_accordion_reason(rule: UniverseRule) -> None:
    """An item that fails both drop checks carries the first reason in the fixed order."""
    decision = decide(_facts("Performer: Joe Reed<br>Accordion solo; Tenor."), rule)
    assert decision.reason is Reason.VOCAL


def test_keep_pattern_reads_the_instrumentation_line_only(rule: UniverseRule) -> None:
    """A solo phrase in the title does not keep an item whose instrumentation names no solo."""
    decision = decide(_facts("Performer: Sam Brass<br>Cornet; Orch.", title="Violin solo"), rule)
    assert decision.reason is Reason.NOT_SOLO_OR_STRING_ENSEMBLE


def test_strip_html_keeps_each_line_and_drops_tags() -> None:
    """Line breaks and tags become separate lines, and entities are decoded."""
    assert strip_html("One<br/>Two &amp; <b>three</b></p>Four") == ("One", "Two & three", "Four")


def test_parse_description_separates_labels_boilerplate_and_instrumentation() -> None:
    """Performer and Writer lines are labelled names, boilerplate is dropped, the rest is kept."""
    raw = (
        "Performer: Ada Singer; Ben Bass<br>Writer: C. Composer"
        "<br>Digitized at 96kHz 24bit<br>Violin solo; Pianoforte."
    )
    parsed = parse_description(raw)
    assert parsed.performers == ("Ada Singer", "Ben Bass")
    assert parsed.writers == ("C. Composer",)
    assert parsed.instrumentation_lines == ("Violin solo; Pianoforte.",)


def test_item_facts_falls_back_to_the_creator_when_no_performer_line_exists() -> None:
    """Without a Performer line the creator is the performer, and the source says so."""
    metadata = {"identifier": "x", "creator": "Jane Doe", "description": "Violin solo."}
    facts = item_facts("x", metadata, [_flac()])
    assert facts.performers == ("Jane Doe",)
    assert facts.performer_source == "creator_fallback"


def test_main_flac_ignores_files_in_a_directory_and_other_sources() -> None:
    """Only a top-level original 24-bit FLAC is a candidate."""
    files = [_flac("dir/a.flac"), _flac("b.flac", source="derivative"), _flac("c.flac")]
    entry, count = main_flac(files)
    assert count == 1
    assert entry is not None and entry["name"] == "c.flac"


def test_year_of_reads_the_first_four_digits() -> None:
    """The year is the first four-digit run in the date, and None when there is none."""
    assert year_of("1923-05-01") == 1923
    assert year_of("n.d.") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("  Fritz   KREISLER ", "fritz kreisler"), ("Dvořák", "dvořák")],
)
def test_performer_key_casefolds_and_collapses_whitespace(raw: str, expected: str) -> None:
    """The key folds case and whitespace and keeps accents, so accented names stay distinct."""
    assert performer_key(raw) == expected
    assert performer_key("Dvorak") != performer_key("Dvořák")
