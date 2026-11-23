"""Tests for fetching and staging one candidate file, with a scripted stand-in for the GET.

No test reads the network. Each scripted step either writes a body, returns a status, or
raises, so every attempt rule can be exercised on its own.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from benchmark import fetch
from benchmark.fetch import (
    Expected,
    ListingMismatch,
    PermanentlyUnavailable,
    RetriesExhausted,
    Staged,
    download_url,
    fetch_file,
    staged_paths,
)
from benchmark.https_get import Response

ID = "musopen-chopin-complete-works-flac/a.flac"
BODY = b"fLaC" + bytes(range(256)) * 4
PERMANENT = (401, 403, 404, 410, 451)


def listed(body: bytes = BODY, **override: Any) -> Expected:
    """The listing values of *body*, under a bound far above it."""
    values = {
        "size": len(body),
        "md5": hashlib.md5(body).hexdigest(),
        "sha1": hashlib.sha1(body).hexdigest(),
        "oversize_bound": 10_000,
    }
    return Expected(**{**values, **override})


class Script:
    """A stand-in GET that plays one scripted step per call and records the calls."""

    def __init__(self, *steps: Any) -> None:
        self.steps = list(steps)
        self.calls = 0

    def __call__(self, url: str, sink: BinaryIO, *, max_bytes: int, timeout_s: float) -> Response:
        self.calls += 1
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, int):
            return Response(step, 0, False, None, "", "", "")
        if isinstance(step, tuple):
            return Response(step[0], 0, False, None, "", "", "", step[1])
        body, declared = step if isinstance(step, list) else (step, None)
        written = body[:max_bytes]
        sink.write(written)
        return Response(
            200,
            len(written),
            len(body) > max_bytes,
            len(body) if declared is None else declared,
            hashlib.sha256(written).hexdigest(),
            hashlib.md5(written).hexdigest(),
            hashlib.sha1(written).hexdigest(),
        )


def run(
    tmp_path: Path,
    script: Script,
    expected: Expected | None = None,
    sleeps: list[float] | None = None,
) -> Any:
    """`fetch_file` for ID into tmp_path, recording any waits in *sleeps*."""
    waits: list[float] = [] if sleeps is None else sleeps
    return fetch_file(
        ID,
        "https://archive.org/download/x/a.flac",
        listed() if expected is None else expected,
        staged_paths(tmp_path, ID, ".flac"),
        retries=3,
        permanent=PERMANENT,
        get_=script,
        sleep=waits.append,
    )


def leftovers(directory: Path) -> list[str]:
    """Temporary files left in *directory*."""
    return sorted(p.name for p in directory.iterdir() if p.name.startswith("."))


class TestDownloadUrl:
    """Real file names from the frozen piano universe, percent-encoded segment by segment."""

    @pytest.mark.parametrize(
        ("path", "encoded"),
        [
            (
                "E\u0301tude Op. 10, no. 5 in G flat major - 'Black Keys'.flac",
                "E%CC%81tude%20Op.%2010%2C%20no.%205%20in%20G%20flat%20major%20-%20%27Black%20Keys%27.flac",
            ),
            (
                "Nocturnes, Op. 32 - No. 2. Nocturne in A\u266d major.flac",
                "Nocturnes%2C%20Op.%2032%20-%20No.%202.%20Nocturne%20in%20A%E2%99%AD%20major.flac",
            ),
        ],
        ids=["decomposed_accent_comma_apostrophe", "flat_sign"],
    )
    def test_a_listed_name_is_encoded_from_its_utf8_bytes(self, path: str, encoded: str) -> None:
        """The listing gives the accent decomposed (E then U+0301), and the identifier is used
        exactly as listed, with no normalisation, so the URL encodes those code points."""
        universe = json.loads(Path("benchmark/manifests/universe_solo_piano.json").read_text())
        item = universe["source"]
        assert f"{item}/{path}" in {c["identifier"] for c in universe["candidates"]}
        assert download_url(item, path) == f"https://archive.org/download/{item}/{encoded}"

    def test_a_nested_path_keeps_its_slashes_and_encodes_each_segment(self) -> None:
        assert download_url("it em", "a b/c?d.flac") == (
            "https://archive.org/download/it%20em/a%20b/c%3Fd.flac"
        )


class TestOneAttempt:
    """What a single attempt records and stages."""

    def test_a_good_file_is_staged_with_its_sidecar(self, tmp_path: Path) -> None:
        sleeps: list[float] = []
        outcome = run(tmp_path, Script(BODY), sleeps=sleeps)
        staged, sidecar = staged_paths(tmp_path, ID, ".flac")
        assert isinstance(outcome, Staged)
        assert (outcome.path, outcome.size, outcome.oversize, outcome.reused) == (
            staged,
            len(BODY),
            False,
            False,
        )
        assert outcome.sha256 == hashlib.sha256(BODY).hexdigest()
        assert outcome.compared == ("size", "md5", "sha1")
        assert staged.read_bytes() == BODY
        assert json.loads(sidecar.read_text())["identifier"] == ID
        assert (sleeps, leftovers(tmp_path)) == ([], [])

    def test_the_staged_name_comes_from_the_identifier_not_the_listed_path(
        self, tmp_path: Path
    ) -> None:
        identifier = f"{ID[:-6]}../../x.flac"
        staged, sidecar = staged_paths(tmp_path, identifier, ".flac")
        stem = hashlib.sha256(identifier.encode()).hexdigest()
        assert (staged, sidecar) == (tmp_path / f"{stem}.flac", tmp_path / f"{stem}.flac.json")

    @pytest.mark.parametrize("status", PERMANENT)
    def test_a_permanent_status_excludes_at_once(self, tmp_path: Path, status: int) -> None:
        sleeps: list[float] = []
        outcome = run(tmp_path, Script(status), sleeps=sleeps)
        assert isinstance(outcome, PermanentlyUnavailable)
        assert (outcome.status, len(outcome.attempts), sleeps) == (status, 1, [])
        assert list(tmp_path.iterdir()) == []

    def test_absent_listing_values_are_not_compared(self, tmp_path: Path) -> None:
        outcome = run(tmp_path, Script(BODY), Expected(None, None, None, 10_000))
        assert isinstance(outcome, Staged)
        assert outcome.compared == ()

    def test_a_relative_staging_directory_is_refused(self) -> None:
        with pytest.raises(ValueError, match="absolute"):
            run(Path("relative"), Script(BODY))


class TestTransientFailures:
    """Every other failure retries, with the waits the harness sets."""

    @pytest.mark.parametrize(
        "first",
        [503, 429, 302, OSError("reset"), TimeoutError("slow"), [BODY, len(BODY) + 5]],
        ids=["server_error", "rate_limit", "other_status", "network", "timeout", "ended_short"],
    )
    def test_a_transient_failure_retries_after_the_first_wait(
        self, tmp_path: Path, first: Any
    ) -> None:
        sleeps: list[float] = []
        outcome = run(tmp_path, Script(first, BODY), sleeps=sleeps)
        assert isinstance(outcome, Staged)
        assert (len(outcome.attempts), sleeps) == (2, [30.0])
        assert leftovers(tmp_path) == []

    def test_the_error_class_of_a_network_failure_is_recorded(self, tmp_path: Path) -> None:
        outcome = run(tmp_path, Script(TimeoutError("slow"), BODY))
        assert outcome.attempts[0].error == "TimeoutError"

    @pytest.mark.parametrize(
        ("hint", "expected"),
        [(300.0, 300.0), (5000.0, 600.0), (10.0, 30.0)],
        ids=["honoured", "capped", "below_base"],
    )
    def test_a_retry_after_is_honoured_up_to_the_cap(
        self, tmp_path: Path, hint: float, expected: float
    ) -> None:
        sleeps: list[float] = []
        run(tmp_path, Script((429, hint), BODY), sleeps=sleeps)
        assert sleeps == [expected]

    def test_four_transient_failures_invalidate_the_run(self, tmp_path: Path) -> None:
        sleeps: list[float] = []
        with pytest.raises(RetriesExhausted) as raised:
            run(tmp_path, Script(503, OSError("x"), 503, 503), sleeps=sleeps)
        assert len(raised.value.attempts) == 4
        assert sleeps == [30.0, 120.0, 600.0]
        assert list(tmp_path.iterdir()) == []


class TestListingCheck:
    """Bytes the listing does not describe are a transient failure of that attempt."""

    @pytest.mark.parametrize("field", ["size", "md5", "sha1"])
    def test_every_attempt_mismatching_excludes_the_candidate(
        self, tmp_path: Path, field: str
    ) -> None:
        wrong = {"size": len(BODY) + 1, "md5": "0" * 32, "sha1": "0" * 40}[field]
        outcome = run(tmp_path, Script(BODY, BODY, BODY, BODY), listed(**{field: wrong}))
        assert isinstance(outcome, ListingMismatch)
        assert all(a.mismatched == (field,) for a in outcome.attempts)
        assert list(tmp_path.iterdir()) == []

    def test_mismatches_mixed_with_another_failure_invalidate_the_run(self, tmp_path: Path) -> None:
        with pytest.raises(RetriesExhausted):
            run(tmp_path, Script(BODY, 503, BODY, BODY), listed(md5="0" * 32))

    def test_a_mismatch_then_the_listed_bytes_is_staged(self, tmp_path: Path) -> None:
        outcome = run(tmp_path, Script(b"<html>error</html>", BODY))
        assert isinstance(outcome, Staged)
        assert outcome.attempts[0].mismatched == ("size", "md5", "sha1")


class TestOversize:
    """A body over the bound is staged as a prefix, with no raw digest."""

    def test_an_oversize_body_is_staged_one_byte_past_the_bound(self, tmp_path: Path) -> None:
        body = b"x" * 500
        outcome = run(tmp_path, Script(body), Expected(500, None, None, 100))
        assert isinstance(outcome, Staged)
        assert (outcome.oversize, outcome.sha256, outcome.size) == (True, None, 101)
        assert outcome.path.read_bytes() == body[:101]
        assert outcome.compared == ("size",)

    def test_an_oversize_body_listed_at_or_below_the_bound_mismatches(self, tmp_path: Path) -> None:
        body = b"x" * 500
        outcome = run(tmp_path, Script(body, body, body, body), Expected(90, None, None, 100))
        assert isinstance(outcome, ListingMismatch)

    def test_without_a_bound_a_longer_body_is_a_size_mismatch(self, tmp_path: Path) -> None:
        """An archive fetched whole: more bytes than listed fails the listing check."""
        body = BODY + b"extra"
        script = Script(body, body, body, body)
        outcome = run(tmp_path, script, listed(oversize_bound=None))
        assert isinstance(outcome, ListingMismatch)
        assert outcome.attempts[0].mismatched[0] == "size"


class TestReuse:
    """A staged file is reused on a rerun only if everything recorded still matches."""

    def test_a_staged_file_is_reused_without_a_request(self, tmp_path: Path) -> None:
        run(tmp_path, Script(BODY))
        again = Script()
        outcome = run(tmp_path, again)
        assert isinstance(outcome, Staged)
        assert (outcome.reused, again.calls) == (True, 0)

    def test_a_staged_file_changed_since_is_fetched_again(self, tmp_path: Path) -> None:
        first = run(tmp_path, Script(BODY))
        first.path.write_bytes(BODY[:-1] + b"\x00")
        again = Script(BODY)
        outcome = run(tmp_path, again)
        assert (outcome.reused, again.calls) == (False, 1)
        assert outcome.path.read_bytes() == BODY

    @pytest.mark.parametrize(
        "sidecar", ["other_id", "not json", "[]"], ids=["other_id", "garbled", "not_object"]
    )
    def test_a_sidecar_that_does_not_name_this_candidate_is_not_trusted(
        self, tmp_path: Path, sidecar: str
    ) -> None:
        run(tmp_path, Script(BODY))
        path = staged_paths(tmp_path, ID, ".flac")[1]
        if sidecar == "other_id":
            record = json.loads(path.read_text())
            sidecar = json.dumps({**record, "identifier": "other/b.flac"})
        path.write_text(sidecar)
        again = Script(BODY)
        assert run(tmp_path, again).reused is False
        assert again.calls == 1

    def test_a_changed_file_is_refetched_even_when_nothing_was_listed(self, tmp_path: Path) -> None:
        """With no listed values to compare, the sidecar's own digest still catches a change."""
        bare = Expected(None, None, None, 10_000)
        first = run(tmp_path, Script(BODY), bare)
        first.path.write_bytes(BODY[:-1] + b"\x00")
        again = Script(BODY)
        assert run(tmp_path, again, bare).reused is False
        assert again.calls == 1

    def test_a_staged_file_that_no_longer_matches_the_listing_is_fetched_again(
        self, tmp_path: Path
    ) -> None:
        run(tmp_path, Script(BODY))
        again = Script(BODY + b"!")
        outcome = run(tmp_path, again, listed(BODY + b"!"))
        assert (outcome.reused, again.calls) == (False, 1)


class TestInterruption:
    """An interrupt leaves no temporary file behind and is not swallowed."""

    def test_an_interrupt_during_a_get_removes_the_temporary_file(self, tmp_path: Path) -> None:
        with pytest.raises(KeyboardInterrupt):
            run(tmp_path, Script(KeyboardInterrupt()))
        assert list(tmp_path.iterdir()) == []

    def test_the_waits_are_the_harness_setting(self) -> None:
        assert (fetch.WAITS_S, fetch.MAX_WAIT_S) == ((30.0, 120.0, 600.0), 600.0)
