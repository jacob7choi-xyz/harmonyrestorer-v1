"""Tests for the listening acquisition: download, decode, training refusal and windows.

The network is replaced by a fake getter that serves synthetic 96 kHz stereo 24-bit FLACs. The
decoder is a fake that reports the file's own length, except in the tests that run the real
protocol v14 worker, which are marked in their names.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pytest
import soundfile as sf

from benchmark.fetch import download_url
from benchmark.https_get import Response
from benchmark.listening import acquire as acquire_module
from benchmark.listening.acquire import AcquisitionError, acquire
from benchmark.listening.manifest import UniverseItem
from benchmark.listening.protocol import ListeningProtocol, load_listening_protocol
from benchmark.listening.selection import (
    Walk,
    fail_choice,
    run_walk,
    window_bounds,
    window_rule_for,
    window_start,
)
from benchmark.protocol import Protocol, load_protocol
from benchmark.run_walk import WORKER_TIMEOUT_S
from benchmark.runner import Candidate, CanonicalFile, run_source
from benchmark.selection import TrainingIdentities

_FLAC_RATE = 96_000
_CANONICAL_RATE = 16_000
_AUDIO_S = 40.0
_LISTED_S = 40.0
_UNIVERSE_SIZE = 12
_NO_TRAINING = TrainingIdentities(frozenset(), frozenset(), frozenset())


def _flac(seconds: float, tone_hz: float) -> bytes:
    """A stereo 96 kHz, 24-bit FLAC of a sine, as the candidate files are."""
    t = np.arange(round(seconds * _FLAC_RATE)) / _FLAC_RATE
    tone = 0.5 * np.sin(2.0 * np.pi * tone_hz * t)
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([tone, tone], axis=1), _FLAC_RATE, format="FLAC", subtype="PCM_24")
    return buffer.getvalue()


def _digests(data: bytes) -> tuple[str, str, str]:
    """The sha256, md5 and sha1 of *data*, in hex."""
    return (
        hashlib.sha256(data).hexdigest(),
        hashlib.md5(data, usedforsecurity=False).hexdigest(),
        hashlib.sha1(data, usedforsecurity=False).hexdigest(),
    )


@dataclass
class _Fixture:
    """A synthetic universe, the bytes each URL serves, and the fake network's behaviour."""

    universe: tuple[UniverseItem, ...]
    served: dict[str, bytes]
    status: dict[str, int] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)


def _url(identifier: str, name: str) -> str:
    """The download URL the acquisition builds for a fixture item."""
    return download_url(identifier, name)


def _index(identifier: str) -> int:
    """The fixture index an identifier was built from."""
    return int(identifier.split("-")[1])


def _flac_name(universe: tuple[UniverseItem, ...], identifier: str) -> str:
    """The main FLAC's name for one fixture item."""
    return next(item.flac_name for item in universe if item.identifier == identifier)


def _build(count: int = _UNIVERSE_SIZE, *, audio: dict[int, bytes] | None = None) -> _Fixture:
    """A universe of *count* items. *audio* replaces the audio of the listed indices."""
    overrides = audio or {}
    items: list[UniverseItem] = []
    served: dict[str, bytes] = {}
    for index in range(count):
        identifier = f"fixture-{index:02d}"
        name = f"track {index:02d}; (main).flac"
        data = overrides.get(index, _flac(_AUDIO_S, 300.0 + 37.0 * index))
        _, md5, sha1 = _digests(data)
        items.append(
            UniverseItem(
                identifier=identifier,
                title=f"Fixture {index:02d}",
                year=1910,
                performers=(f"Performer {index:02d}",),
                instrumentation="Violin Solo",
                flac_name=name,
                flac_size=len(data),
                flac_md5=md5,
                flac_sha1=sha1,
                length_s=_LISTED_S,
            )
        )
        served[_url(identifier, name)] = data
    return _Fixture(tuple(items), served)


@pytest.fixture(scope="module")
def protocol() -> ListeningProtocol:
    """The frozen listening protocol, with its digests checked."""
    return load_listening_protocol()


@pytest.fixture(scope="module")
def benchmark() -> Protocol:
    """The benchmark protocol the worker and the retry settings come from."""
    return load_protocol()


class _FakeGet:
    """A getter that serves the fixture's bytes, or a status for a URL, and counts calls."""

    def __init__(self, fixture: _Fixture) -> None:
        self._fixture = fixture

    def __call__(self, url: str, sink: BinaryIO, *, max_bytes: int, timeout_s: float) -> Response:
        del timeout_s
        self._fixture.calls.append(url)
        status = self._fixture.status.get(url, 200)
        data = self._fixture.served.get(url, b"") if status == 200 else b""
        body = data[:max_bytes]
        sink.write(body)
        sha256, md5, sha1 = _digests(body)
        return Response(
            status=status,
            size=len(body),
            exceeded=len(data) > max_bytes,
            content_length=len(data),
            sha256=sha256,
            md5=md5,
            sha1=sha1,
        )


def _fake_decode(candidate: Candidate, out: Path) -> CanonicalFile:
    """Stand in for the worker: the canonical length is the file's length at 16 kHz."""
    out.mkdir()
    raw = candidate.path.read_bytes()
    info = sf.info(str(candidate.path))
    samples = round(info.frames * _CANONICAL_RATE / info.samplerate)
    canonical = np.zeros(samples, dtype="<f8").tobytes()
    path = out / "canonical.f64"
    path.write_bytes(canonical)
    raw_digest = hashlib.sha256(raw).hexdigest()
    return CanonicalFile(
        path=path,
        sha256=hashlib.sha256(canonical).hexdigest(),
        samples=samples,
        raw_sha256=raw_digest,
        decoded_pcm_sha256=decoded_digest_for(raw),
    )


def decoded_digest_for(raw: bytes) -> str:
    """The decoded-PCM digest the fake decoder reports for a raw file of these bytes."""
    return hashlib.sha256(b"decoded:" + hashlib.sha256(raw).digest()).hexdigest()


def _no_sleep(seconds: float) -> None:
    """A sleep that returns at once."""
    del seconds


def _walk(protocol: ListeningProtocol, universe: tuple[UniverseItem, ...]) -> Walk:
    """The walk the acquisition runs over the same universe, for the expectations."""
    return run_walk(universe, window_rule_for(protocol), protocol.selection.sample_size)


def _run(
    tmp_path: Path,
    fixture: _Fixture,
    protocol: ListeningProtocol,
    benchmark: Protocol,
    *,
    training: TrainingIdentities = _NO_TRAINING,
    decode: Callable[[Candidate, Path], Any] | None = None,
    sleeps: list[float] | None = None,
    name: str = "out",
) -> Path:
    """Run the acquisition over the fixture into a new directory under *tmp_path*."""
    recorded = sleeps if sleeps is not None else []
    return acquire(
        protocol,
        benchmark,
        fixture.universe,
        training,
        tmp_path / name,
        getter=_FakeGet(fixture),
        decode=decode or _fake_decode,
        sleep=lambda seconds: recorded.append(seconds),
        waits=(0.0, 0.0, 0.0),
    )


def _manifest(out: Path) -> dict[str, Any]:
    """The acquisition manifest, parsed."""
    data: dict[str, Any] = json.loads((out / "acquisition.json").read_text())
    return data


def _names(directory: Path) -> list[str]:
    """The entries of a directory, sorted, so a leftover partial directory shows up."""
    return sorted(entry.name for entry in directory.iterdir())


@pytest.fixture(scope="module")
def clean_fixture() -> _Fixture:
    """A universe whose every item downloads, decodes and has a window."""
    return _build()


def test_the_manifest_lists_the_walk_order_with_windows_inside_the_bounds(
    tmp_path: Path, clean_fixture: _Fixture, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """Accepted items follow the walk's order, and each window is inside its own bounds."""
    out = _run(tmp_path, clean_fixture, protocol, benchmark)
    manifest = _manifest(out)
    expected = _walk(protocol, clean_fixture.universe)
    rule = window_rule_for(protocol)

    assert [item["identifier"] for item in manifest["items"]] == [
        choice.identifier for choice in expected.selected
    ]
    assert manifest["replacements"] == []
    for item in manifest["items"]:
        low, high = item["window"]["low_s"], item["window"]["high_s"]
        assert low <= item["window"]["start_s"] <= high
        assert item["window"]["start_s"] == window_start(item["identifier"], rule, (low, high))
        assert item["canonical"]["samples"] == round(_AUDIO_S * _CANONICAL_RATE)


def test_the_manifest_records_the_digests_and_the_protocol_pins(
    tmp_path: Path, clean_fixture: _Fixture, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """The manifest names the protocol, universe and benchmark digests, and each file's digests."""
    out = _run(tmp_path, clean_fixture, protocol, benchmark)
    manifest = _manifest(out)
    first = manifest["items"][0]

    assert manifest["protocol_sha256"] == protocol.sha256
    assert manifest["universe_sha256"] == protocol.universe.sha256
    assert manifest["benchmark_protocol_sha256"] == protocol.benchmark.sha256
    assert (out / first["flac"]["path"]).read_bytes()
    assert (
        hashlib.sha256((out / first["flac"]["path"]).read_bytes()).hexdigest()
        == (first["flac"]["sha256"])
    )
    assert (
        hashlib.sha256((out / first["canonical"]["path"]).read_bytes()).hexdigest()
        == (first["canonical"]["sha256"])
    )


def test_a_download_refused_as_not_found_is_replaced_by_the_next_item(
    tmp_path: Path, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """A 404 is a permanent failure. The item is replaced and the walk moves on."""
    fixture = _build()
    first = _walk(protocol, fixture.universe).selected[0].identifier
    url = _url(first, _flac_name(fixture.universe, first))
    fixture.status[url] = 404

    out = _run(tmp_path, fixture, protocol, benchmark)
    manifest = _manifest(out)
    expected = fail_choice(
        _walk(protocol, fixture.universe),
        first,
        "permanent_unavailability:404",
        fixture.universe,
        window_rule_for(protocol),
    )

    assert manifest["replacements"] == [
        {"identifier": first, "reason": "permanent_unavailability:404"}
    ]
    assert first not in [item["identifier"] for item in manifest["items"]]
    assert [item["identifier"] for item in manifest["items"]] == [
        choice.identifier for choice in expected.selected
    ]


def test_a_listing_mismatch_is_replaced_with_its_failed_fields(
    tmp_path: Path, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """Bytes that differ from the listing fail the md5 and sha1 check, and the item is replaced."""
    fixture = _build()
    first = _walk(protocol, fixture.universe).selected[0]
    url = _url(first.identifier, _flac_name(fixture.universe, first.identifier))
    good = fixture.served[url]
    fixture.served[url] = good[:-1] + bytes([good[-1] ^ 0xFF])

    manifest = _manifest(_run(tmp_path, fixture, protocol, benchmark))

    assert [r["identifier"] for r in manifest["replacements"]] == [first.identifier]
    reason = manifest["replacements"][0]["reason"]
    assert reason.startswith("checksum_mismatch:") and "md5" in reason and "sha1" in reason


def test_exhausted_transient_retries_abort_the_run_with_no_output(
    tmp_path: Path, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """A server error on every attempt is not replaced: the run aborts and publishes nothing."""
    fixture = _build()
    first = _walk(protocol, fixture.universe).selected[0].identifier
    url = _url(first, _flac_name(fixture.universe, first))
    fixture.status[url] = 500

    with pytest.raises(AcquisitionError, match="retries exhausted"):
        _run(tmp_path, fixture, protocol, benchmark)

    assert not (tmp_path / "out").exists()
    assert _names(tmp_path) == []
    attempts = [call for call in fixture.calls if call == url]
    assert len(attempts) == 1 + benchmark.selection.acquisition_retries


def test_downloads_are_paced_by_the_protocol_pause(
    tmp_path: Path, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """Every download after the first waits the protocol's pause, and nothing else sleeps."""
    fixture = _build()
    sleeps: list[float] = []
    _run(tmp_path, fixture, protocol, benchmark, sleeps=sleeps)

    assert len(sleeps) == len(fixture.calls) - 1
    assert set(sleeps) == {protocol.fetch.pause_s_between_requests}


def test_a_decode_that_is_too_short_for_any_window_is_replaced(
    tmp_path: Path, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """Listed as 40 s but decoded as 30 s: no window fits, so the item is replaced as too_short."""
    first = _walk(protocol, _build().universe).selected[0].identifier
    fixture = _build(audio={_index(first): _flac(30.0, 333.0)})

    manifest = _manifest(_run(tmp_path, fixture, protocol, benchmark))

    assert manifest["replacements"] == [{"identifier": first, "reason": "too_short"}]


def test_a_raw_digest_matching_the_training_corpus_refuses_the_whole_run(
    tmp_path: Path, clean_fixture: _Fixture, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """A training raw match stops the run. Nothing is published and no partial directory stays."""
    first = _walk(protocol, clean_fixture.universe).selected[0].identifier
    url = _url(first, _flac_name(clean_fixture.universe, first))
    training = TrainingIdentities(
        frozenset(), frozenset({hashlib.sha256(clean_fixture.served[url]).hexdigest()}), frozenset()
    )

    with pytest.raises(AcquisitionError, match="raw digest"):
        _run(tmp_path, clean_fixture, protocol, benchmark, training=training)

    assert not (tmp_path / "out").exists()
    assert _names(tmp_path) == []


def test_a_decoded_pcm_digest_matching_the_training_corpus_refuses_the_run(
    tmp_path: Path, clean_fixture: _Fixture, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """A decoded-PCM training match stops the run, and the staged files are removed with it."""
    first = _walk(protocol, clean_fixture.universe).selected[0].identifier
    url = _url(first, _flac_name(clean_fixture.universe, first))
    training = TrainingIdentities(
        frozenset(), frozenset(), frozenset({decoded_digest_for(clean_fixture.served[url])})
    )

    with pytest.raises(AcquisitionError, match="decoded PCM digest"):
        _run(tmp_path, clean_fixture, protocol, benchmark, training=training)

    assert _names(tmp_path) == []


def test_an_existing_output_directory_is_refused_and_left_alone(
    tmp_path: Path, clean_fixture: _Fixture, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """The output directory must not exist, and an existing one is not touched."""
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("earlier run")

    with pytest.raises(AcquisitionError, match="already exists"):
        _run(tmp_path, clean_fixture, protocol, benchmark)

    assert _names(tmp_path / "out") == ["keep.txt"]


def test_a_universe_that_runs_out_before_the_sample_size_is_refused(
    tmp_path: Path, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """Three items cannot fill a sample of ten. The run is refused and nothing is left behind."""
    fixture = _build(count=3)

    with pytest.raises(AcquisitionError, match="ran out"):
        _run(tmp_path, fixture, protocol, benchmark)

    assert _names(tmp_path) == []


def test_the_real_worker_decodes_a_96_khz_flac_to_its_16_khz_length(
    tmp_path: Path, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """The protocol v14 worker decodes a staged 96 kHz FLAC to 16 kHz with the expected length."""
    source = tmp_path / "source.flac"
    source.write_bytes(_flac(_AUDIO_S, 440.0))

    outcome = run_source(
        Candidate("real-worker", source), tmp_path / "worker", benchmark, timeout_s=WORKER_TIMEOUT_S
    )

    assert isinstance(outcome.outcome, CanonicalFile)
    assert outcome.outcome.samples == round(_AUDIO_S * _CANONICAL_RATE)
    assert outcome.outcome.raw_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


def test_a_corrupt_file_refused_by_the_real_worker_is_replaced(
    tmp_path: Path, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """Bytes that are not a FLAC, with matching listed digests, are refused by the worker."""
    first = _walk(protocol, _build().universe).selected[0].identifier
    fixture = _build(audio={_index(first): b"this is not a flac stream" * 64})

    def route(candidate: Candidate, out: Path) -> Any:
        if candidate.logical_source_id == first:
            return run_source(candidate, out, benchmark, timeout_s=WORKER_TIMEOUT_S).outcome
        return _fake_decode(candidate, out)

    manifest = _manifest(_run(tmp_path, fixture, protocol, benchmark, decode=route))

    assert [r["identifier"] for r in manifest["replacements"]] == [first]
    assert manifest["replacements"][0]["reason"] != ""
    assert first not in [item["identifier"] for item in manifest["items"]]


def test_the_default_getter_sends_the_protocol_user_agent_and_timeout(
    monkeypatch: pytest.MonkeyPatch, protocol: ListeningProtocol
) -> None:
    """The getter used for downloads passes the protocol's user agent and timeout to the GET."""
    seen: dict[str, Any] = {}

    def fake_get(url: str, sink: BinaryIO, **kwargs: Any) -> Response:
        seen.update(kwargs, url=url)
        return Response(200, 0, False, 0, "", "", "")

    monkeypatch.setattr(acquire_module, "get", fake_get)
    getter = acquire_module._protocol_getter(protocol)
    getter("https://example.invalid/x", io.BytesIO(), max_bytes=1, timeout_s=999.0)

    assert seen["user_agent"] == protocol.fetch.user_agent
    assert seen["timeout_s"] == protocol.fetch.timeout_s
    assert seen["max_bytes"] == 1


def test_the_window_bounds_come_from_the_decoded_length(
    tmp_path: Path, clean_fixture: _Fixture, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """The recorded bounds are the frozen rule applied to the decoded duration, not the listing."""
    manifest = _manifest(_run(tmp_path, clean_fixture, protocol, benchmark))
    clips = protocol.clips
    expected = window_bounds(_AUDIO_S, clips.margin_s, int(clips.duration_s))

    assert expected is not None
    for item in manifest["items"]:
        assert (item["window"]["low_s"], item["window"]["high_s"]) == expected
