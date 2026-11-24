"""Tests for the listening-study protocol loader: exact keys, semantic checks and pinned digests."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from benchmark.listening import protocol as listening_protocol
from benchmark.listening.protocol import (
    PINNED_PROTOCOL_SHA256,
    ListeningProtocol,
    ListeningProtocolError,
    load_protocol_for_listing,
)

_PROTOCOL_BYTES: bytes = listening_protocol._PROTOCOL_PATH.read_bytes()


def _mutated(mutate: Any) -> bytes:
    """The protocol file's JSON after *mutate* has changed the parsed object.

    Args:
        mutate: A function that edits the parsed object in place.

    Returns:
        The re-serialised bytes.
    """
    data = copy.deepcopy(json.loads(_PROTOCOL_BYTES))
    mutate(data)
    return json.dumps(data).encode("utf-8")


def _parse_refused(raw: bytes, fragment: str) -> None:
    """Assert that parsing *raw* is refused with a message containing *fragment*.

    Args:
        raw: The bytes to parse.
        fragment: Text the refusal message must contain.

    Raises:
        AssertionError: If parsing succeeds or the message does not match.
    """
    with pytest.raises(ListeningProtocolError, match=fragment):
        listening_protocol._parse_bytes(raw)


def test_the_real_file_parses_and_reports_its_own_digest() -> None:
    """The shipped file parses, and the digest it reports is the digest of its bytes."""
    parsed = listening_protocol._parse_bytes(_PROTOCOL_BYTES)
    assert parsed.sha256 == hashlib.sha256(_PROTOCOL_BYTES).hexdigest()
    assert parsed.selection.sample_size == 10
    assert parsed.trials.scored_trials_per_listener == 30


def test_the_shipped_file_matches_its_pinned_digest() -> None:
    """The file on disk is the one the code pins, so the listing loader accepts it."""
    assert hashlib.sha256(_PROTOCOL_BYTES).hexdigest() == PINNED_PROTOCOL_SHA256
    assert isinstance(load_protocol_for_listing(), ListeningProtocol)


def test_an_unknown_top_level_key_is_refused() -> None:
    """A key the schema does not name is a load failure, not a silently ignored setting."""
    _parse_refused(_mutated(lambda d: d.update(unexpected=1)), "unknown keys")


def test_a_missing_required_key_is_refused() -> None:
    """A section that lost a required key is refused and the key is named."""
    _parse_refused(_mutated(lambda d: d["selection"].pop("seed")), "missing keys")


def test_a_repeated_key_is_refused_rather_than_keeping_the_last() -> None:
    """JSON with a duplicated key is refused, since the parser would otherwise keep one silently."""
    text = _PROTOCOL_BYTES.decode("utf-8").replace(
        '"seed": 20261124', '"seed": 1, "seed": 20261124', 1
    )
    _parse_refused(text.encode("utf-8"), "duplicate key")


def test_another_protocol_version_is_refused() -> None:
    """Only version 1 is supported, whatever the file says."""
    _parse_refused(_mutated(lambda d: d.update(protocol_version=2)), "only protocol version 1")


def test_a_wrong_scored_trial_count_is_refused() -> None:
    """Scored trials must equal pairs times clips, so an edited count cannot pass as the design."""
    _parse_refused(
        _mutated(lambda d: d["trials"].update(scored_trials_per_listener=29)),
        "scored trials must equal",
    )


def test_a_pair_naming_one_condition_twice_is_refused() -> None:
    """A pair must hold two different conditions."""
    _parse_refused(
        _mutated(lambda d: d["trials"].update(pairs=[["original", "original"]])),
        "invalid pair",
    )


def test_a_pattern_that_does_not_compile_is_refused() -> None:
    """An invalid regex source is a load failure, not a rule that silently matches nothing."""
    bad = _mutated(
        lambda d: d["universe"]["rule"].update(keep_if_instrumentation_matches=["(unclosed"])
    )
    _parse_refused(bad, "does not compile")


def test_a_benchmark_reference_to_another_file_is_refused() -> None:
    """The benchmark reference names one file, so the chain cannot be redirected."""
    _parse_refused(
        _mutated(lambda d: d["benchmark_protocol"].update(path="benchmark/protocols/v13.json")),
        r"benchmark_protocol\.path",
    )


def test_a_boolean_is_not_accepted_as_an_integer() -> None:
    """JSON true is a bool in Python, which is an int subclass, so the loader must reject it."""
    _parse_refused(_mutated(lambda d: d["selection"].update(seed=True)), "must be of type int")


def test_a_protocol_digest_mismatch_is_refused() -> None:
    """A parsed protocol whose bytes are not the pinned file is refused by the pin check."""
    parsed = listening_protocol._parse_bytes(_PROTOCOL_BYTES)
    altered = dataclasses.replace(parsed, sha256="0" * 64)
    with pytest.raises(ListeningProtocolError, match="protocol file's digest"):
        listening_protocol._check_pins(altered)


def test_a_universe_digest_that_differs_from_the_code_pin_is_refused() -> None:
    """The universe digest in the file must equal the digest pinned in code."""
    parsed = listening_protocol._parse_bytes(_PROTOCOL_BYTES)
    altered_universe = dataclasses.replace(parsed.universe, sha256="1" * 64)
    altered = dataclasses.replace(parsed, universe=altered_universe)
    with pytest.raises(ListeningProtocolError, match=r"universe\.sha256"):
        listening_protocol._check_pins(dataclasses.replace(altered, sha256=PINNED_PROTOCOL_SHA256))


def test_the_benchmark_file_on_disk_must_match_its_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A benchmark protocol file whose bytes differ from the pinned digest is refused."""
    fake_root = tmp_path
    (fake_root / "benchmark" / "protocols").mkdir(parents=True)
    (fake_root / "benchmark" / "protocols" / "v14.json").write_bytes(b"{}")
    monkeypatch.setattr(listening_protocol, "_REPO_ROOT", fake_root)
    parsed = listening_protocol._parse_bytes(_PROTOCOL_BYTES)
    with pytest.raises(ListeningProtocolError, match="does not match the digest"):
        listening_protocol._check_benchmark_file(parsed.benchmark)
