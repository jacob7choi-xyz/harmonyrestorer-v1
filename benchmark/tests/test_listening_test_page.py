"""Tests for the listening test page: trial list, consent, references, and refusals."""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from benchmark.canonical_json import canonical_json_bytes
from benchmark.listening.build import BUILD_NAME, CONDITIONS, KEY_NAME
from benchmark.listening.protocol import ListeningProtocol, load_listening_protocol
from benchmark.listening.test_page import PageError, page_html, recordings_of, write_test_page
from benchmark.listening.trials import listener_trials

_RECORDINGS = tuple(f"rec-{index:02d}" for index in range(10))
_FORBIDDEN_SUBSTRINGS = ("http://", "https://", "fetch(", "XMLHttpRequest", "<script src")


@pytest.fixture(scope="module")
def protocol() -> ListeningProtocol:
    """The frozen listening protocol, with its digests checked."""
    return load_listening_protocol()


def _make_build(root: Path, protocol: ListeningProtocol) -> Path:
    """A build directory with a sealed key, a manifest, and one tiny clip per key entry."""
    build = root / "build"
    (build / "clips").mkdir(parents=True)
    (build / "sealed").mkdir()
    clips = []
    for number, (recording, condition) in enumerate(
        (r, c) for r in _RECORDINGS for c in CONDITIONS
    ):
        blinded = f"{(number * 7919 + 13) % (2**48):012x}"
        sf.write(str(build / "clips" / f"{blinded}.wav"), np.zeros(160), 16_000, subtype="PCM_16")
        clips.append({"blinded_id": blinded, "identifier": recording, "condition": condition})
    key = {
        "study": protocol.study,
        "protocol_sha256": protocol.sha256,
        "blind_seed": 1,
        "clips": clips,
    }
    key_bytes = canonical_json_bytes(key)
    (build / "sealed" / KEY_NAME).write_bytes(key_bytes)
    manifest = {"key_sha256": hashlib.sha256(key_bytes).hexdigest()}
    (build / BUILD_NAME).write_bytes(canonical_json_bytes(manifest))
    return build


def _config(page: str) -> dict[str, Any]:
    """The page's embedded configuration, parsed from its script element."""
    match = re.search(r"const CONFIG = (\{.*?\});\n", page, re.S)
    assert match is not None
    data: dict[str, Any] = json.loads(match.group(1))
    return data


def _expected_trials(
    build: Path, pages: Path, protocol: ListeningProtocol, index: int
) -> list[dict[str, str]]:
    """The trial list the page should hold, derived from the sealed key, not from the page."""
    key = json.loads((build / "sealed" / KEY_NAME).read_text())
    blinded = {(clip["identifier"], clip["condition"]): clip["blinded_id"] for clip in key["clips"]}
    expected = []
    for trial in listener_trials(protocol, index, recordings_of(key)):
        a = blinded[(trial.recording, trial.side_a)]
        b = blinded[(trial.recording, trial.side_b)]
        expected.append(
            {
                "id": trial.trial_id,
                "a": Path(os.path.relpath(build / "clips" / f"{a}.wav", pages)).as_posix(),
                "b": Path(os.path.relpath(build / "clips" / f"{b}.wav", pages)).as_posix(),
            }
        )
    return expected


def test_the_page_trial_list_matches_the_protocol_rule_for_several_listeners(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """The page's trials, with their clip references, match the trials derived from the key."""
    build = _make_build(tmp_path, protocol)
    pages = tmp_path / "pages"
    for index in (0, 1, 7):
        page = write_test_page(build, pages, protocol, index, f"L{index:03d}").read_text()

        assert _config(page)["trials"] == _expected_trials(build, pages, protocol, index)


def test_the_page_has_no_network_reference(tmp_path: Path, protocol: ListeningProtocol) -> None:
    """The page makes no request: no external URL, fetch, XMLHttpRequest or remote script."""
    page = write_test_page(_make_build(tmp_path, protocol), tmp_path / "pages", protocol, 0, "L000")

    text = page.read_text()
    assert not any(token in text for token in _FORBIDDEN_SUBSTRINGS)
    assert "<script" in text and "src=" not in text.split("<script", 1)[1].split(">", 1)[0]


def test_the_consent_sentence_is_the_protocol_text_and_comes_before_the_trials(
    protocol: ListeningProtocol,
) -> None:
    """The consent sentence on the page is the protocol's, shown before the trial screens."""
    page = page_html(protocol, "L000", 0, [])

    assert html.escape(protocol.listeners.consent_text) in page
    assert page.index('id="consent-text"') < page.index('id="trials"')


def test_the_page_asks_the_protocol_question_with_its_answers(protocol: ListeningProtocol) -> None:
    """The question and the three answers are the protocol's."""
    page = page_html(protocol, "L000", 0, [])

    assert html.escape(protocol.trials.question) in page
    assert _config(page)["answers"] == list(protocol.trials.answers)


def test_the_page_states_the_playback_and_names_no_condition(protocol: ListeningProtocol) -> None:
    """The page asks for the protocol's playback and names no condition, in any letter case."""
    page = page_html(protocol, "L000", 0, [])
    lowered = page.lower()

    assert f"use {protocol.listeners.playback} for every trial." in lowered
    assert not any(name in lowered for name in CONDITIONS)


def test_both_players_are_labelled_and_no_condition_is_named(protocol: ListeningProtocol) -> None:
    """Each player carries a visible and an accessible label, and the page names no condition."""
    page = page_html(protocol, "L000", 0, [{"id": "t01", "a": "a.wav", "b": "b.wav"}])

    assert "<span>Clip A</span>" in page and "<span>Clip B</span>" in page
    assert 'id="clip-a" controls preload="auto" aria-label="Clip A"' in page
    assert 'id="clip-b" controls preload="auto" aria-label="Clip B"' in page
    lowered = page.lower()
    assert not any(name in lowered for name in CONDITIONS)


def test_a_listener_identifier_is_escaped_in_the_page(protocol: ListeningProtocol) -> None:
    """A listener identifier with markup is escaped, so it cannot inject elements."""
    page = page_html(protocol, "<b>x</b>", 0, [])

    assert "&lt;b&gt;x&lt;/b&gt;" in page
    assert "<b>x</b>" not in page


def test_a_key_whose_digest_differs_from_the_manifest_is_refused(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """A changed sealed key stops the page, and no page is written."""
    build = _make_build(tmp_path, protocol)
    key_path = build / "sealed" / KEY_NAME
    key_path.write_bytes(key_path.read_bytes().replace(b'blind_seed": 1', b'blind_seed": 2'))

    with pytest.raises(PageError, match="digest"):
        write_test_page(build, tmp_path / "pages", protocol, 0, "L000")

    assert not (tmp_path / "pages" / "listener_000.html").exists()


def test_a_page_is_not_overwritten_by_a_second_attempt(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """The page for an index is written once, and the second attempt leaves it unchanged."""
    build = _make_build(tmp_path, protocol)
    page = write_test_page(build, tmp_path / "pages", protocol, 2, "L002")
    before = page.read_bytes()

    with pytest.raises(PageError, match="already exists"):
        write_test_page(build, tmp_path / "pages", protocol, 2, "L002")

    assert page.read_bytes() == before


def test_a_missing_clip_refuses_the_page_and_writes_nothing(
    tmp_path: Path, protocol: ListeningProtocol
) -> None:
    """A clip the key names but the directory lacks stops the page before anything is written."""
    build = _make_build(tmp_path, protocol)
    first_clip = next((build / "clips").iterdir())
    first_clip.unlink()

    with pytest.raises(PageError, match="is missing"):
        write_test_page(build, tmp_path / "pages", protocol, 0, "L000")

    assert not (tmp_path / "pages" / "listener_000.html").exists()
