"""The selection walk's command line: preconditions, a new run directory, and the walk.

Order of work, as protocol v13 and the walk's contract require:

1. The run preconditions (golden fixtures, degradation golden, model digests). A failure
   refuses the run before anything is created.
2. A new run directory under `data/benchmark_runs/`, named by the UTC start time. An existing
   directory is refused, so this command does not overwrite an earlier run.
3. The MusicNet archive, fetched whole and checked, then its 80 frozen members extracted.
4. The training references, computed again from the training files and their workers.
5. The walk of each stratum, then the manifest, or the halt or invalid record that ended it.

Exit codes: 0 for a completed run, 1 for an invalid run, 2 for a refused start, 3 for a walk
halted awaiting adjudication. A library failure without a protocol code (such as an unreadable
canonical file) exits 1 and writes `run_invalid.json` with the exception's type and message.

The worker timeout and the metadata size cap are harness settings. The protocol does not state
them, and they cannot change which candidates are admitted.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from benchmark.adjudication import read_adjudications
from benchmark.atomic_write import write_atomically
from benchmark.candidate_universe import ListedFile, load_universe, zenodo_file_url
from benchmark.duplicate_screen import unit_features
from benchmark.fetch import (
    WAITS_S,
    Expected,
    FetchOutcome,
    Getter,
    ListingMismatch,
    PermanentlyUnavailable,
    RetriesExhausted,
    Staged,
    download_url,
    fetch_file,
    staged_paths,
)
from benchmark.golden_fixture import platform_token
from benchmark.https_get import get
from benchmark.musicnet_archive import Member, extract_members, fetch_archive
from benchmark.protocol import (
    ArchiveItemStratum,
    Protocol,
    Selection,
    ZenodoArchiveStratum,
    load_protocol,
)
from benchmark.run_preconditions import verify_run_preconditions
from benchmark.runner import run_source
from benchmark.training_corpus import (
    RAW_DIR,
    load_training_digests,
    load_training_identities,
    load_training_manifest,
)
from benchmark.training_references import training_references
from benchmark.walk import (
    REPORTED_FAILURES,
    AwaitingAdjudication,
    ChamberPreparation,
    MetadataUnavailable,
    MusicNetMetadata,
    RunInvalid,
    RunLayout,
    WalkInputs,
    WalkRun,
    archive_retries_invalid,
    run_walk,
)

__all__ = ["main", "piano_fetch", "prepare_chamber"]

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "data" / "benchmark_runs"
STAGING_DIR = REPO_ROOT / "data" / "benchmark_staging"
ADJUDICATIONS_PATH = REPO_ROOT / "data" / "benchmark_adjudications.json"
OPGAN_CHECKPOINT = REPO_ROOT / "checkpoints" / "final.pt"
UVR_ENV = "AUDIO_SEPARATOR_MODEL_DIR"
WORKER_TIMEOUT_S = 1800.0
METADATA_MAX_BYTES = 64 * 1024 * 1024
HTTP_TIMEOUT_S = 120.0
EXIT_OK = 0
EXIT_INVALID = 1
EXIT_REFUSED = 2
EXIT_HALTED = 3


def piano_fetch(
    stratum: ArchiveItemStratum,
    selection: Selection,
    max_identity_stream_bytes: int,
    staging: Path,
    *,
    get_: Getter = get,
    sleep: Callable[[float], None] = time.sleep,
) -> Callable[[ListedFile], FetchOutcome]:
    """The fetch for an Internet Archive candidate, under the protocol's attempt rules."""

    def fetch(listed: ListedFile) -> FetchOutcome:
        path = listed.identifier.removeprefix(f"{stratum.source}/")
        return fetch_file(
            listed.identifier,
            download_url(stratum.source, path),
            Expected(listed.size, listed.md5, listed.sha1, max_identity_stream_bytes),
            staged_paths(staging, listed.identifier, ".flac"),
            retries=selection.acquisition_retries,
            permanent=selection.permanent_failure_codes,
            get_=get_,
            sleep=sleep,
        )

    return fetch


def _member_fetch(members: Mapping[int, Member]) -> Callable[[ListedFile], FetchOutcome]:
    """The chamber fetch once the archive is whole: each candidate is its extracted member."""

    def fetch(listed: ListedFile) -> FetchOutcome:
        member = members[int(listed.identifier.partition("/")[2])]
        return Staged(
            listed.identifier,
            member.path,
            member.size,
            member.sha256,
            member.oversize,
            (),
            (),
            False,
        )

    return fetch


def _archive_failure(
    outcome: PermanentlyUnavailable | ListingMismatch,
) -> Callable[[ListedFile], FetchOutcome]:
    """The chamber fetch when the archive itself was not obtained: every candidate fails so."""

    def fetch(listed: ListedFile) -> FetchOutcome:
        if isinstance(outcome, PermanentlyUnavailable):
            return PermanentlyUnavailable(listed.identifier, outcome.status, outcome.attempts)
        return ListingMismatch(listed.identifier, outcome.attempts)

    return fetch


def prepare_chamber(
    stratum: ZenodoArchiveStratum,
    selection: Selection,
    member_cap: int,
    staging: Path,
    members_dir: Path,
    *,
    get_: Getter = get,
    sleep: Callable[[float], None] = time.sleep,
    waits: tuple[float, ...] = WAITS_S,
) -> ChamberPreparation:
    """Fetch the MusicNet archive whole, then extract the frozen members.

    A permanent status or a listing mismatch on the archive is not a run failure. Each chamber
    candidate then fails under that code, so the stratum runs short.

    Raises:
        RetriesExhausted: If the archive's attempts all failed transiently.
        ArchiveError: If an id is missing from the archive or appears more than once.
    """
    outcome = fetch_archive(stratum, selection, staging, get_=get_, sleep=sleep, waits=waits)
    if not isinstance(outcome, Staged):
        return ChamberPreparation(None, _archive_failure(outcome))
    members = extract_members(
        outcome.path, stratum.candidate_ids, members_dir, member_cap=member_cap
    )
    return ChamberPreparation(outcome.sha256, _member_fetch(members))


def musicnet_metadata_provider(
    stratum: ZenodoArchiveStratum, *, get_: Getter = get
) -> MusicNetMetadata:
    """A provider of the MusicNet metadata file's bytes, fetched only when a halt names MusicNet."""

    def provide() -> bytes:
        url = zenodo_file_url(stratum.zenodo_record, stratum.metadata)
        sink = io.BytesIO()
        try:
            response = get_(url, sink, max_bytes=METADATA_MAX_BYTES, timeout_s=HTTP_TIMEOUT_S)
        except (OSError, ValueError) as error:
            logger.error("MusicNet metadata fetch failed: %s", error)
            raise MetadataUnavailable(f"{url}: {error}") from error
        if response.status != 200 or response.exceeded:
            message = f"{url} answered {response.status}, or was over {METADATA_MAX_BYTES} bytes"
            logger.error("MusicNet metadata unavailable: %s", message)
            raise MetadataUnavailable(message)
        return sink.getvalue()

    return provide


def _strata(protocol: Protocol) -> tuple[ArchiveItemStratum, ZenodoArchiveStratum]:
    """The protocol's Internet Archive stratum and its Zenodo archive stratum."""
    items = [s for s in protocol.selection.strata if isinstance(s, ArchiveItemStratum)]
    archives = [s for s in protocol.selection.strata if isinstance(s, ZenodoArchiveStratum)]
    if len(items) != 1 or len(archives) != 1:
        message = "the protocol must name one archive item and one Zenodo archive stratum"
        logger.error("%s", message)
        raise ValueError(message)
    return items[0], archives[0]


def new_run_layout(runs_root: Path, now: datetime) -> RunLayout:
    """Create the run directory named by *now*; refuse one that exists, so no run is overwritten.

    Raises:
        FileExistsError: If the run directory already exists.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    root = runs_root / now.strftime("%Y%m%dT%H%M%SZ")
    root.mkdir()
    logger.info("Created run directory %s", root)
    return RunLayout(root.resolve())


def _uvr_dir(argument: Path | None) -> Path | None:
    """The UVR model directory from the argument or the environment, if it is a directory."""
    value = os.environ.get(UVR_ENV)
    chosen = argument if argument is not None else (Path(value) if value else None)
    if chosen is None or not chosen.is_dir():
        return None
    return chosen


def _record_failure(layout: RunLayout, error: Exception) -> None:
    """Write `run_invalid.json` for a failure the protocol gives no code, naming its type."""
    payload = {
        "stratum": None,
        "position": None,
        "identifier": None,
        "code": type(error).__name__,
        "detail": str(error),
    }
    body = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    write_atomically(layout.invalid_path, (body + "\n").encode("utf-8"))
    logger.error("Run invalid: %s: %s", type(error).__name__, error)


def build_inputs(
    protocol: Protocol,
    layout: RunLayout,
    chamber: ChamberPreparation,
    adjudications_path: Path,
    staging: Path,
) -> WalkInputs:
    """The walk's inputs: universes, training references and records, and the real functions.

    Raises:
        CandidateUniverseError: If a frozen universe is missing or altered.
        TrainingCorpusError: If the training manifests are malformed.
        TrainingReferenceError: If a training file does not reproduce its committed digests.
        AdjudicationError: If the adjudication record file is malformed.
    """
    item, archive = _strata(protocol)
    universes = {s.name: load_universe(s) for s in protocol.selection.strata}
    training = load_training_manifest()
    references = training_references(
        protocol,
        training,
        load_training_digests(),
        RAW_DIR,
        layout.training_work_dir,
        timeout_s=WORKER_TIMEOUT_S,
    )
    member_cap = protocol.canonicalisation.decode.max_identity_stream_bytes
    return WalkInputs(
        training=load_training_identities(),
        training_references=references,
        adjudications=read_adjudications(adjudications_path),
        universes=universes,
        piano_fetch=piano_fetch(item, protocol.selection, member_cap, staging),
        chamber=chamber,
        canonicalise=lambda candidate, out_dir: run_source(
            candidate, out_dir, protocol, timeout_s=WORKER_TIMEOUT_S
        ),
        features=lambda samples: unit_features(samples, protocol.duplicate_detection.mfcc),
        musicnet_metadata=musicnet_metadata_provider(archive),
    )


def _report(outcome: WalkRun | RunInvalid | AwaitingAdjudication) -> int:
    """Log the outcome and return its exit code."""
    if isinstance(outcome, WalkRun):
        for walk in outcome.strata:
            logger.info("%s: %s with %d admissions", walk.name, walk.status, len(walk.placed))
        return EXIT_OK
    if isinstance(outcome, AwaitingAdjudication):
        logger.error("Walk halted: see adjudication_needed.json in the run directory")
        return EXIT_HALTED
    return EXIT_INVALID


def main(argv: Sequence[str] | None = None) -> int:
    """Run the walk and return the exit code.

    Args:
        argv: The arguments, or None for `sys.argv`.
    """
    parser = argparse.ArgumentParser(description="Walk the benchmark's candidate strata.")
    parser.add_argument("--uvr-model-dir", type=Path, default=None, help=f"defaults to ${UVR_ENV}")
    parser.add_argument("--adjudications", type=Path, default=ADJUDICATIONS_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvr_dir = _uvr_dir(args.uvr_model_dir)
    if uvr_dir is None:
        logger.error("No UVR model directory: pass --uvr-model-dir or set %s", UVR_ENV)
        return EXIT_REFUSED
    protocol = load_protocol()
    abort = verify_run_preconditions(
        protocol,
        runtime_platform=platform_token(),
        opgan_checkpoint=OPGAN_CHECKPOINT,
        uvr_model_dir=uvr_dir,
    )
    if abort is not None:
        logger.error("Run refused, preconditions failed: %s", abort.detail)
        return EXIT_REFUSED
    try:
        layout = new_run_layout(RUNS_ROOT, datetime.now(UTC))
    except FileExistsError as error:
        logger.error("Run refused: %s already exists", error.filename)
        return EXIT_REFUSED
    return _run_in(protocol, layout, args.adjudications)


def _run_in(protocol: Protocol, layout: RunLayout, adjudications: Path) -> int:
    """Everything after the run directory exists: archive, references, the walk, and the report."""
    layout.ensure_dirs()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    _, archive = _strata(protocol)
    try:
        chamber = prepare_chamber(
            archive,
            protocol.selection,
            protocol.canonicalisation.decode.max_identity_stream_bytes,
            STAGING_DIR,
            layout.chamber_members_dir,
        )
    except RetriesExhausted as error:
        archive_retries_invalid(protocol, layout, error)
        return EXIT_INVALID
    except REPORTED_FAILURES as error:
        _record_failure(layout, error)
        return EXIT_INVALID
    try:
        inputs = build_inputs(protocol, layout, chamber, adjudications, STAGING_DIR)
        return _report(run_walk(protocol, layout, inputs))
    except REPORTED_FAILURES as error:
        _record_failure(layout, error)
        return EXIT_INVALID


if __name__ == "__main__":
    raise SystemExit(main())
