"""The canonicalisation worker: one source, in a process of its own (protocol v8).

Protocol v8 runs each source in its own child process, one at a time, so that a worker that
dies cannot take a record with it and the parent decides what its death means. This module
is the child. It runs the four stages v8's capacity measurement ran (acquisition,
inspection, decode, and canonicalisation) on one source, and then reports through files in
its own output directory, which that measurement did not write:

- `canonical.f64`, written only when the source canonicalises: the canonical samples as
  contiguous little-endian float64 with no header, which are exactly the bytes the canonical
  digest covers, so the parent checks the file by hashing it.
- `record.json`, written last: the protocol digest and either the canonical digest, the
  sample count, the source's identifier, and its raw and decoded-PCM digests, or an
  ineligibility or abort record in the `outcome_wire` form. The parent checks the identifier
  against the candidate it asked about, and the raw digest against the candidate's
  expected digest when one is given. The decoded-PCM digest is what the third exclusion
  rule compares; the parent checks only its form.

Both are written atomically, so a file that exists is complete, and the record is written
after the samples, so a record that names a canonical waveform has its file in place.

**Exit status.** 0 after the record is written. A `MemoryError` is caught at the top level
only and reported as `EXIT_MEMORY_ERROR` without writing a record, so the report needs no
allocation; the exception is released before the worker exits. A canonical file written
before the error can remain, and the parent never reads one without a record from a worker
that exited 0. Nothing else is caught: an
uncaught exception exits with status 1 and the parent treats it as a death it cannot
attribute. Another native library calling `exit` with the same status would be reported as
a capacity failure; either way the run aborts and no source receives a verdict.

Run by the runner from the repository root as
`python -m benchmark.canonical_worker SOURCE LOGICAL_ID OUT_DIR [--expected-sha256 DIGEST]`,
without `-I`, as the measurement modules are: the `benchmark` package is not installed, so
`-I` would hide it.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from benchmark.atomic_write import write_atomically
from benchmark.canonicalisation import CanonicalWaveform, canonicalise
from benchmark.outcome_wire import outcome_to_wire
from benchmark.protocol import Canonicalisation, Protocol, load_protocol
from benchmark.source_acquisition import ArtifactBuffered, ArtifactTooLarge, acquire_local_artifact
from benchmark.source_decode import decode_source, decoded_pcm_sha256
from benchmark.source_inspection import InspectedSource, inspect_source
from benchmark.source_outcome import IneligibleReason, RunAbortOutcome, SourceIneligibleOutcome

__all__ = [
    "CANONICAL_FILE",
    "CANONICAL_KIND",
    "EXIT_MEMORY_ERROR",
    "RECORD_FILE",
    "CanonicalSource",
    "canonicalise_source",
    "main",
]

logger = logging.getLogger(__name__)

# Distinct from 1 (any uncaught exception) and 2 (an argument error).
EXIT_MEMORY_ERROR = 86
RECORD_FILE = "record.json"
CANONICAL_FILE = "canonical.f64"
CANONICAL_KIND = "canonical"
# The byte layout canonicalisation hashes, so the file's digest is the canonical digest.
_SAMPLE_FORMAT = "<f8"


@dataclass(frozen=True)
class CanonicalSource:
    """A source that canonicalised, with what identifies it to the parent."""

    logical_source_id: str
    waveform: CanonicalWaveform
    raw_sha256: str
    decoded_pcm_sha256: str


SourceResult = CanonicalSource | SourceIneligibleOutcome | RunAbortOutcome


def canonicalise_source(
    path: Path, *, logical_source_id: str, expected_sha256: str | None, protocol: Protocol
) -> SourceResult:
    """Acquire, inspect, decode, and canonicalise one source, stopping at the first outcome.

    Args:
        path: The staged source file.
        logical_source_id: Identifier of the candidate.
        expected_sha256: The digest the candidate is expected to have, when known.
        protocol: The loaded protocol.

    Returns:
        The canonical waveform, the ineligibility record, or the abort record.
    """
    config = protocol.canonicalisation
    digest = protocol.source_sha256
    artifact = acquire_local_artifact(
        path,
        logical_source_id=logical_source_id,
        config=config.decode,
        expected_sha256=expected_sha256,
    )
    if isinstance(artifact, RunAbortOutcome):
        return artifact
    if isinstance(artifact, ArtifactTooLarge):
        # Decided from fstat before buffering, so nothing was inspected: no metadata.
        return SourceIneligibleOutcome(
            logical_source_id=logical_source_id,
            primary_reason_code=IneligibleReason.SOURCE_EXCEEDS_MAX_BYTES,
            identity=artifact.identity,
            protocol_sha256=digest,
        )
    inspected = inspect_source(
        artifact, logical_source_id=logical_source_id, protocol_sha256=digest, config=config
    )
    if isinstance(inspected, SourceIneligibleOutcome):
        return inspected
    decoded = decode_source(
        artifact,
        inspected,
        logical_source_id=logical_source_id,
        protocol_sha256=digest,
        config=config,
    )
    if isinstance(decoded, SourceIneligibleOutcome):
        return decoded
    return _canonical_source(decoded, inspected, artifact, logical_source_id, config)


def _canonical_source(
    decoded: npt.NDArray[np.float64],
    inspected: InspectedSource,
    artifact: ArtifactBuffered,
    logical_source_id: str,
    config: Canonicalisation,
) -> CanonicalSource | RunAbortOutcome:
    """Canonicalise a decoded source and attach the digests that identify it."""
    # Hashed before canonicalisation, so the digest covers the decoder's output as returned.
    decoded_digest = decoded_pcm_sha256(decoded)
    canonical = canonicalise(decoded, inspected, logical_source_id=logical_source_id, config=config)
    if isinstance(canonical, RunAbortOutcome):
        return canonical
    raw_digest = artifact.identity.sha256
    if raw_digest is None:
        # A buffered artifact is hashed while it is read, so this is not expected; refusing
        # keeps a canonical result from ever reaching the parent without its raw digest.
        raise RuntimeError(f"{logical_source_id}: buffered artifact has no digest")
    return CanonicalSource(logical_source_id, canonical, raw_digest, decoded_digest)


def _record(result: SourceResult, protocol: Protocol) -> dict[str, Any]:
    """The record the parent reads: the protocol digest and what this source became."""
    if isinstance(result, CanonicalSource):
        body: dict[str, Any] = {
            "kind": CANONICAL_KIND,
            "sha256": result.waveform.sha256,
            "samples": int(result.waveform.samples.size),
            "logical_source_id": result.logical_source_id,
            "raw_sha256": result.raw_sha256,
            "decoded_pcm_sha256": result.decoded_pcm_sha256,
        }
    else:
        body = outcome_to_wire(result)
    return {"protocol_sha256": protocol.source_sha256, "result": body}


def _report(result: SourceResult, protocol: Protocol, out_dir: Path) -> None:
    """Write the canonical samples when there are any, then the record."""
    if isinstance(result, CanonicalSource):
        samples = np.ascontiguousarray(result.waveform.samples, dtype=_SAMPLE_FORMAT)
        write_atomically(out_dir / CANONICAL_FILE, samples.data)
    record = json.dumps(_record(result, protocol), sort_keys=True)
    write_atomically(out_dir / RECORD_FILE, record.encode())


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    """The worker's arguments."""
    parser = argparse.ArgumentParser(prog="python -m benchmark.canonical_worker")
    parser.add_argument("source", type=Path)
    parser.add_argument("logical_source_id")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--expected-sha256", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Canonicalise one source and report it; the return value is the exit status.

    Args:
        argv: The arguments, `sys.argv[1:]` when None.

    Returns:
        0 once the record is written, or `EXIT_MEMORY_ERROR` after a caught `MemoryError`.
    """
    args = _parse(argv)
    memory_exhausted = False
    try:
        protocol = load_protocol()
        result = canonicalise_source(
            args.source,
            logical_source_id=args.logical_source_id,
            expected_sha256=args.expected_sha256,
            protocol=protocol,
        )
        _report(result, protocol, args.out_dir)
    except MemoryError:
        # Only the flag leaves this block, so the exception and the frames it holds are
        # released before the exit status is returned.
        memory_exhausted = True
    if memory_exhausted:
        return EXIT_MEMORY_ERROR
    logger.info("Source %s reported", args.logical_source_id)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
