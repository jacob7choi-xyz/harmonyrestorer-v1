"""Acquire the listening study's recordings: download, decode, check, and cut a window for each.

For each item the walk selects, in walk order, the main FLAC is downloaded with its listed size,
md5 and sha1, then decoded by the protocol v14 worker. The raw and decoded-PCM digests are
compared with the training corpus, and a match refuses the run. The clip window is then drawn
from the decoded sample count. Under protocol v2 an item is replaced only for permanent
unavailability, a checksum mismatch on every attempt, a canonicalisation refusal, or no valid
window. Transient failures that exhaust their retries abort the run, with no output and no
replacement. A replacement removes the item, and the walk continues with the next item after the
last one it considered.

The output is a directory that did not exist before, holding `acquisition.json`, the staged
FLACs under `sources/`, and each worker's output under `work/`. It is built under a partial name
and renamed into place only when every item is accepted, so a failed run leaves nothing under the
output name.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from benchmark.atomic_write import write_atomically
from benchmark.canonical_json import canonical_json_bytes
from benchmark.fetch import (
    WAITS_S,
    Expected,
    Getter,
    PermanentlyUnavailable,
    RetriesExhausted,
    Staged,
    download_url,
    fetch_file,
    staged_paths,
)
from benchmark.https_get import Response, get
from benchmark.listening.manifest import UniverseItem
from benchmark.listening.protocol import ListeningProtocol
from benchmark.listening.selection import (
    Walk,
    WindowRule,
    fail_choice,
    run_walk,
    window_bounds,
    window_rule_for,
    window_start,
)
from benchmark.protocol import Protocol
from benchmark.run_walk import WORKER_TIMEOUT_S
from benchmark.runner import Candidate, CanonicalFile, SourceOutcome, run_source
from benchmark.selection import TrainingIdentities
from benchmark.source_outcome import RunAbortOutcome, SourceIneligibleOutcome

__all__ = ["ACQUISITION_NAME", "AcquisitionError", "Decoder", "acquire"]

logger = logging.getLogger(__name__)

ACQUISITION_NAME = "acquisition.json"
SOURCES_DIR = "sources"
WORK_DIR = "work"
SOURCE_SUFFIX = ".flac"

Decoder = Callable[[Candidate, Path], SourceOutcome]


class AcquisitionError(RuntimeError):
    """Raised when the run must stop: a training match, a harness abort, or too few items."""


@dataclass(frozen=True)
class _Window:
    """A clip window and the bounds it was drawn from, in whole seconds."""

    start_s: int
    low_s: int
    high_s: int
    duration_s: float


@dataclass(frozen=True)
class _Accepted:
    """One item that passed every stage, with what the build and the manifest need."""

    staged: Staged
    canonical: CanonicalFile
    window: _Window


@dataclass(frozen=True)
class _Context:
    """What one acquisition run reads and writes. Built once, then passed to each stage."""

    protocol: ListeningProtocol
    benchmark: Protocol
    rule: WindowRule
    training: TrainingIdentities
    getter: Getter
    decode: Decoder
    sleep: Callable[[float], None]
    waits: tuple[float, ...]
    sources: Path
    work: Path
    out_dir: Path


def _stem(identifier: str) -> str:
    """The file stem for an identifier. Identifiers are not used as file names."""
    return hashlib.sha256(identifier.encode("utf-8")).hexdigest()


def _protocol_getter(protocol: ListeningProtocol) -> Getter:
    """A getter that applies the listening protocol's user agent and timeout to each GET."""

    def getter(url: str, sink: BinaryIO, *, max_bytes: int, timeout_s: float) -> Response:
        del timeout_s  # The protocol's timeout replaces the one the fetch module passes.
        return get(
            url,
            sink,
            max_bytes=max_bytes,
            timeout_s=protocol.fetch.timeout_s,
            user_agent=protocol.fetch.user_agent,
        )

    return getter


def _download(context: _Context, item: UniverseItem, pacing: list[int]) -> Staged | str:
    """Download and check one main FLAC. Returns the staged file, or the replacement reason.

    `pacing` holds the number of downloads started so far, so the pause between requests is
    applied from the second download onward.
    """
    if pacing[0] > 0:
        context.sleep(context.protocol.fetch.pause_s_between_requests)
    pacing[0] += 1
    expected = Expected(item.flac_size, item.flac_md5, item.flac_sha1, None)
    try:
        outcome = fetch_file(
            item.identifier,
            download_url(item.identifier, item.flac_name),
            expected,
            staged_paths(context.sources, item.identifier, SOURCE_SUFFIX),
            retries=context.benchmark.selection.acquisition_retries,
            permanent=context.benchmark.selection.permanent_failure_codes,
            get_=context.getter,
            sleep=context.sleep,
            waits=context.waits,
        )
    except RetriesExhausted as error:
        # Protocol v2, selection.replacement_rule: transient failures that are not all mismatches
        # abort the run, so no replacement is made and nothing is published.
        logger.error(
            "Download of %s failed transiently on every attempt: %s", item.identifier, error
        )
        raise AcquisitionError(
            f"{item.identifier}: retries exhausted, the run is aborted"
        ) from error
    if isinstance(outcome, Staged):
        return outcome
    if isinstance(outcome, PermanentlyUnavailable):
        logger.warning("Download of %s is unavailable: status %d", item.identifier, outcome.status)
        return f"permanent_unavailability:{outcome.status}"
    mismatched = ",".join(outcome.attempts[-1].mismatched)
    logger.warning("Download of %s failed its listing check: %s", item.identifier, mismatched)
    return f"checksum_mismatch:{mismatched}"


def _refuse_training_match(identifier: str, kind: str) -> AcquisitionError:
    """Log a training match and return the error that stops the run."""
    logger.error("Acquisition refused: %s of %s matches a training recording", kind, identifier)
    return AcquisitionError(f"{identifier} matches a training recording by its {kind}")


def _decode(context: _Context, identifier: str, staged: Staged) -> CanonicalFile | str:
    """Decode one staged FLAC with the worker. Returns the canonical file, or the reason."""
    outcome = context.decode(Candidate(identifier, staged.path), context.work / _stem(identifier))
    if isinstance(outcome, RunAbortOutcome):
        logger.error("Decoding %s aborted the run: %s", identifier, outcome.detail)
        raise AcquisitionError(f"decoding {identifier} aborted: {outcome.detail}")
    if isinstance(outcome, SourceIneligibleOutcome):
        logger.warning("Decoding %s refused: %s", identifier, outcome.primary_reason_code)
        return f"canonicalisation_refusal:{outcome.primary_reason_code}"
    return outcome


def _window(context: _Context, identifier: str, canonical: CanonicalFile) -> _Window | str:
    """The clip window drawn from the decoded sample count, or 'too_short' when none fits."""
    clips = context.protocol.clips
    duration = canonical.samples / clips.sample_rate_hz
    bounds = window_bounds(duration, clips.margin_s, int(clips.duration_s))
    if bounds is None:
        logger.warning("Recording %s has no valid window at %.3f s", identifier, duration)
        return "too_short"
    return _Window(window_start(identifier, context.rule, bounds), bounds[0], bounds[1], duration)


def _check_decoded(
    context: _Context, item: UniverseItem, staged: Staged, canonical: CanonicalFile
) -> None:
    """Refuse the run when the decoder disagrees with the download or the decoded PCM is trained on.

    Raises:
        AcquisitionError: On a raw digest disagreement or a decoded-PCM training match.
    """
    if canonical.raw_sha256 != staged.sha256:
        raise AcquisitionError(f"{item.identifier}: decoder raw digest differs from the download")
    if canonical.decoded_pcm_sha256 in context.training.decoded_sha256:
        raise _refuse_training_match(item.identifier, "decoded PCM digest")


def _process(context: _Context, item: UniverseItem, pacing: list[int]) -> _Accepted | str:
    """Run every stage for one item. Returns the accepted record, or the replacement reason.

    Raises:
        AcquisitionError: On a training match, a decoder abort, or a digest inconsistency.
    """
    staged = _download(context, item, pacing)
    if isinstance(staged, str):
        return staged
    if staged.sha256 is None:
        raise AcquisitionError(f"{item.identifier} was staged without a raw digest")
    if staged.sha256 in context.training.raw_sha256:
        raise _refuse_training_match(item.identifier, "raw digest")
    canonical = _decode(context, item.identifier, staged)
    if isinstance(canonical, str):
        return canonical
    _check_decoded(context, item, staged, canonical)
    window = _window(context, item.identifier, canonical)
    if isinstance(window, str):
        return window
    logger.info(
        "Accepted %s: %d samples, window start %d s",
        item.identifier,
        canonical.samples,
        window.start_s,
    )
    return _Accepted(staged, canonical, window)


def _run_choices(
    walk: Walk,
    items: dict[str, UniverseItem],
    universe: Sequence[UniverseItem],
    context: _Context,
) -> tuple[Walk, dict[str, _Accepted]]:
    """Process selected items until every selection is accepted, replacing each failure."""
    accepted: dict[str, _Accepted] = {}
    pacing = [0]
    while True:
        pending = next((c.identifier for c in walk.selected if c.identifier not in accepted), None)
        if pending is None:
            return walk, accepted
        outcome = _process(context, items[pending], pacing)
        if isinstance(outcome, str):
            logger.warning("Replacing %s: %s", pending, outcome)
            walk = fail_choice(walk, pending, outcome, universe, context.rule)
        else:
            accepted[pending] = outcome


def _record(
    item: UniverseItem, accepted: _Accepted, out_dir: Path, clip_s: float, rate: int
) -> dict[str, object]:
    """The manifest entry for one accepted item. Paths are relative to the output directory."""
    staged, canonical, window = accepted.staged, accepted.canonical, accepted.window
    return {
        "identifier": item.identifier,
        "title": item.title,
        "year": item.year,
        "performers": list(item.performers),
        "flac": {
            "name": item.flac_name,
            "path": staged.path.relative_to(out_dir).as_posix(),
            "size": staged.size,
            "md5": item.flac_md5,
            "sha1": item.flac_sha1,
            "sha256": staged.sha256,
        },
        "canonical": {
            "path": canonical.path.relative_to(out_dir).as_posix(),
            "sha256": canonical.sha256,
            "samples": canonical.samples,
            "sample_rate_hz": rate,
        },
        "decoded_pcm_sha256": canonical.decoded_pcm_sha256,
        "duration_s": window.duration_s,
        "window": {
            "start_s": window.start_s,
            "duration_s": clip_s,
            "low_s": window.low_s,
            "high_s": window.high_s,
        },
    }


def _manifest(
    context: _Context,
    walk: Walk,
    items: dict[str, UniverseItem],
    accepted: dict[str, _Accepted],
) -> dict[str, object]:
    """The acquisition manifest: every accepted item in selection order, with the replacements."""
    protocol = context.protocol
    clips = protocol.clips
    entries = [
        _record(
            items[c.identifier],
            accepted[c.identifier],
            context.out_dir,
            clips.duration_s,
            clips.sample_rate_hz,
        )
        for c in walk.selected
    ]
    return {
        "study": protocol.study,
        "protocol_version": protocol.protocol_version,
        "protocol_sha256": protocol.sha256,
        "universe_sha256": protocol.universe.sha256,
        "benchmark_protocol_path": protocol.benchmark.path,
        "benchmark_protocol_sha256": protocol.benchmark.sha256,
        "seed": protocol.selection.seed,
        "sample_size": protocol.selection.sample_size,
        "sample_rate_hz": clips.sample_rate_hz,
        "clip_duration_s": clips.duration_s,
        "items": entries,
        "replacements": [{"identifier": r.identifier, "reason": r.reason} for r in walk.replaced],
        "skipped": [{"identifier": s.identifier, "reason": str(s.reason)} for s in walk.skipped],
    }


def _check_rates(protocol: ListeningProtocol, benchmark: Protocol) -> None:
    """Refuse when the benchmark's canonical rate is not the rate the study presents.

    Raises:
        AcquisitionError: If the two rates differ.
    """
    canonical_rate = benchmark.canonicalisation.resample.target_sample_rate
    if canonical_rate != protocol.clips.sample_rate_hz:
        logger.error("Canonical rate %d differs from the study rate", canonical_rate)
        raise AcquisitionError(
            f"benchmark canonical rate {canonical_rate} Hz differs from the study's "
            f"{protocol.clips.sample_rate_hz} Hz"
        )


def _partial_dir(out_dir: Path) -> Path:
    """A fresh sibling directory for the run's output, renamed into place only on success."""
    return out_dir.with_name(f".{out_dir.name}.{uuid.uuid4().hex}.partial")


def _new_context(
    protocol: ListeningProtocol,
    benchmark: Protocol,
    training: TrainingIdentities,
    partial: Path,
    getter: Getter | None,
    decode: Decoder | None,
    sleep: Callable[[float], None],
    waits: tuple[float, ...],
) -> _Context:
    """The run's context, with its staging directories under the partial output directory."""
    return _Context(
        protocol=protocol,
        benchmark=benchmark,
        rule=window_rule_for(protocol),
        training=training,
        getter=getter or _protocol_getter(protocol),
        decode=decode or _default_decoder(benchmark),
        sleep=sleep,
        waits=waits,
        sources=partial / SOURCES_DIR,
        work=partial / WORK_DIR,
        out_dir=partial,
    )


def acquire(
    protocol: ListeningProtocol,
    benchmark: Protocol,
    universe: Sequence[UniverseItem],
    training: TrainingIdentities,
    out_dir: Path,
    *,
    getter: Getter | None = None,
    decode: Decoder | None = None,
    sleep: Callable[[float], None] = time.sleep,
    waits: tuple[float, ...] = WAITS_S,
) -> Path:
    """Acquire every selected recording into a new output directory, or leave nothing behind.

    Args:
        protocol: The listening protocol.
        benchmark: The benchmark protocol, for the worker and the retry rules.
        universe: The kept items, in the order the universe file gives.
        training: The training corpus identities the matches are checked against.
        out_dir: A directory that does not exist yet.
        getter: The GET used for downloads. Defaults to the protocol's user agent and timeout.
        decode: Runs the worker on one staged FLAC. Defaults to the protocol v14 worker.
        sleep: Waits between downloads and retries; injected in tests.
        waits: The wait before each retry, in seconds.

    Returns:
        The output directory.

    Raises:
        AcquisitionError: If the directory exists, a training recording matches, a decode
            aborts, or the universe runs out before the sample size is accepted.
    """
    if out_dir.exists():
        logger.error("Acquisition output %s already exists", out_dir)
        raise AcquisitionError(f"output directory {out_dir} already exists")
    _check_rates(protocol, benchmark)
    partial = _partial_dir(out_dir.resolve())
    partial.mkdir(parents=True)
    try:
        context = _new_context(protocol, benchmark, training, partial, getter, decode, sleep, waits)
        context.sources.mkdir()
        context.work.mkdir()
        _publish(context, universe, partial, out_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return out_dir


def _default_decoder(benchmark: Protocol) -> Decoder:
    """The protocol v14 worker, run once per candidate with the benchmark's worker timeout."""

    def decode(candidate: Candidate, out: Path) -> SourceOutcome:
        return run_source(candidate, out, benchmark, timeout_s=WORKER_TIMEOUT_S).outcome

    return decode


def _publish(
    context: _Context, universe: Sequence[UniverseItem], partial: Path, out_dir: Path
) -> None:
    """Run the walk, write the manifest into the partial directory, and rename it into place.

    Raises:
        AcquisitionError: If the universe runs out before the sample size is accepted.
    """
    items = {item.identifier: item for item in universe}
    walk = run_walk(universe, context.rule, context.protocol.selection.sample_size)
    walk, accepted = _run_choices(walk, items, universe, context)
    if not walk.complete:
        logger.error("Universe ran out after %d accepted items", len(accepted))
        raise AcquisitionError(
            f"the universe ran out after {len(accepted)} of "
            f"{context.protocol.selection.sample_size} items"
        )
    manifest = _manifest(context, walk, items, accepted)
    write_atomically(partial / ACQUISITION_NAME, canonical_json_bytes(manifest))
    partial.replace(out_dir)
    logger.info("Acquisition written to %s with %d items", out_dir, len(accepted))
