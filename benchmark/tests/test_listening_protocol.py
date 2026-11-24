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
    load_listening_protocol,
    load_protocol_for_listing,
)

_PROTOCOL_BYTES: bytes = listening_protocol._PROTOCOL_PATH.read_bytes()
# The archived predecessor: protocol v1 and its document, pinned by their digests.
_V1_PATH = "benchmark/listening/protocols/v1.json"
_V1_DOCUMENT_PATH = "docs/listening-study-protocol-v1.md"
_V1_PROTOCOL_SHA256 = (
    "c09ea242d9b98544d1b10c661fdd3dada6391066cb1dbc2b39ebe6a454175ad4"  # pragma: allowlist secret
)
_V1_DOCUMENT_SHA256 = (
    "8f7c5fe84aa3fa79201bcfc1d0e63753a1adf105e667fc502418f711b375d209"  # pragma: allowlist secret
)
_UVR_TEXT = (
    "The whole side is downmixed to mono and resampled to 44100 Hz with librosa soxr_hq, as "
    "protocol v14 resamples. The backend UVR service runs on that input. The service then "
    "converts its denoised stem itself. If the stem has more than one channel, the channels are "
    "averaged in float32. If the stem's sample rate is not 16000 Hz, the samples are resampled "
    "to 16000 Hz with librosa.resample, whose default res_type is soxr_hq in librosa 0.11.0. The "
    "16 kHz samples are written as a WAV with subtype PCM_16, which is a 16-bit file. This is "
    "the conversion the backend applies to the UVR stem before it returns the file. The build "
    "reads that 16 kHz output and does not resample it again."
)
_PRESENTATION_TEXT = (
    "All three conditions are presented at 16 kHz mono. Listeners never hear the full-band "
    "transfer. The UVR input is processed at 44.1 kHz. Listeners hear only the 16 kHz UVR "
    "output that the backend returns."
)
_REPLACEMENT_TEXT = (
    "An item is replaced only for one of four reasons, and the reason is recorded with the "
    "replacement. (1) Permanent unavailability: a fetch ends in an HTTP status that benchmark "
    "protocol v14 lists in selection.permanent_failure_codes, which are 401, 403, 404, 410 and "
    "451 in that file. (2) Checksum mismatch: every attempt on the item ends in a size, md5 or "
    "sha1 mismatch with the pinned universe. (3) Canonicalisation refusal: the canonicalisation "
    "of protocol v14 refuses the downloaded bytes. (4) Too short: the item has no valid window, "
    "as item_with_no_valid_window_rule states. Any other fetch failure is transient. A transient "
    "failure is retried on the same item up to selection.acquisition_retries times in benchmark "
    "protocol v14, which is 3 in that file, so at most 4 attempts are made. A size, md5 or sha1 "
    "mismatch on one attempt is a transient failure of that attempt. When every attempt ends in "
    "a mismatch, the item is replaced under reason (2). When the attempts end without success "
    "and any of them is a transient failure other than a mismatch, the acquisition run aborts "
    "with no output and is run again later, and no replacement is made. A network condition "
    "must not decide which items are tested. When an item is replaced, its selection is removed "
    "and the walk continues with the next item after the last item already considered. The walk "
    "does not return to the start of the order, so the names of a removed item do not reopen an "
    "earlier skipped item."
)


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


def test_the_predecessor_version_is_refused_as_current() -> None:
    """Only version 2 is loadable as current, so version 1 is refused whatever the file says."""
    _parse_refused(_mutated(lambda d: d.update(protocol_version=1)), "only protocol version 2")


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


def test_the_shipped_version_two_loads_with_every_check() -> None:
    """Version 2 loads through the full loader, which checks every pin and the amendment chain."""
    loaded = load_listening_protocol()
    assert loaded.protocol_version == 2
    assert loaded.sha256 == PINNED_PROTOCOL_SHA256
    assert loaded.amends.protocol_path == _V1_PATH
    assert loaded.amends.document_path == _V1_DOCUMENT_PATH


def test_the_predecessor_file_and_its_archived_document_match_their_digests() -> None:
    """Protocol v1 and its archived document are byte-identical to the digests they were frozen at."""
    root = listening_protocol._REPO_ROOT
    v1 = hashlib.sha256((root / _V1_PATH).read_bytes()).hexdigest()
    document = hashlib.sha256((root / _V1_DOCUMENT_PATH).read_bytes()).hexdigest()
    assert v1 == _V1_PROTOCOL_SHA256
    assert document == _V1_DOCUMENT_SHA256


def test_a_changed_amends_protocol_digest_is_refused() -> None:
    """The recorded digest of the predecessor file must match the file on disk."""
    parsed = listening_protocol._parse_bytes(
        _mutated(lambda d: d["amends"].update(protocol_sha256="0" * 64))
    )
    with pytest.raises(ListeningProtocolError, match="protocol_sha256 records"):
        listening_protocol._check_amends(parsed.amends)


def test_a_changed_amends_document_digest_is_refused() -> None:
    """The recorded digest of the archived document must match the file on disk."""
    parsed = listening_protocol._parse_bytes(
        _mutated(lambda d: d["amends"].update(document_sha256="0" * 64))
    )
    with pytest.raises(ListeningProtocolError, match="document_sha256 records"):
        listening_protocol._check_amends(parsed.amends)


def test_a_redirected_amends_path_is_refused() -> None:
    """The predecessor paths are frozen in code, so a configured path cannot redirect the chain."""
    _parse_refused(
        _mutated(lambda d: d["amends"].update(protocol_path="benchmark/listening/x.json")),
        r"amends\.protocol_path",
    )


def test_an_empty_amendment_reason_is_refused() -> None:
    """An amendment without a recorded reason is refused, since it would read as an edit."""
    _parse_refused(_mutated(lambda d: d["amends"].update(reason="  ")), "reason must not be empty")


def test_a_missing_predecessor_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A predecessor that is not in the tree is refused, so an older version stays interpretable."""
    monkeypatch.setattr(listening_protocol, "_REPO_ROOT", tmp_path)
    parsed = listening_protocol._parse_bytes(_PROTOCOL_BYTES)
    with pytest.raises(ListeningProtocolError, match="cannot read"):
        listening_protocol._check_amends(parsed.amends)


def test_the_changed_fields_hold_the_amended_text() -> None:
    """The three changed fields hold the v2 wording, exactly as the amendment record gives it."""
    conditions = json.loads(_PROTOCOL_BYTES)["conditions"]
    selection = json.loads(_PROTOCOL_BYTES)["selection"]
    assert conditions["uvr"] == _UVR_TEXT
    assert conditions["presentation"] == _PRESENTATION_TEXT
    assert selection["replacement_rule"] == _REPLACEMENT_TEXT


def test_every_other_section_equals_the_predecessor() -> None:
    """Apart from the amendment record and the three changed fields, v2 is v1 section for section."""
    v1 = json.loads((listening_protocol._REPO_ROOT / _V1_PATH).read_bytes())
    v2 = json.loads(_PROTOCOL_BYTES)
    assert set(v2) - set(v1) == {"amends"}
    assert {key for key in v1 if v1[key] != v2[key]} == {
        "protocol_version",
        "conditions",
        "selection",
    }
    assert set(v1["conditions"]) == set(v2["conditions"])
    assert {k for k in v1["conditions"] if v1["conditions"][k] != v2["conditions"][k]} == {
        "uvr",
        "presentation",
    }
    assert set(v1["selection"]) == set(v2["selection"])
    assert {k for k in v1["selection"] if v1["selection"][k] != v2["selection"][k]} == {
        "replacement_rule"
    }
