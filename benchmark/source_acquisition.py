"""Bounded, descriptor-safe access to a local candidate artifact.

The first module here that touches the filesystem, and the first that can hang, exhaust
memory, or be affected by another process. Everything before it was pure functions over
values.

**One descriptor, opened once, never reopened.** The classic time-of-check-to-time-of-use
shape is `stat(path)` then `open(path)`, where the thing measured and the thing read can be
different files. Every syscall here goes through the descriptor returned by a single
`open`, so the artifact inspected is the artifact read, whatever happens to the path
afterwards.

`O_NONBLOCK` is a liveness control, not decoration. Measured: `os.open` on a FIFO with
`O_RDONLY` alone blocks until a writer appears, which hangs the run before any bound,
budget, or timeout can apply. With `O_NONBLOCK` the descriptor returns immediately and
`fstat` rejects it as non-regular.

`O_NOFOLLOW` refuses a final-component symlink outright, which is what stops external path
state redirecting the benchmark at something other than the staged artifact. It protects
the final component only; intermediate directories still resolve through symlinks, and this
module does not pretend otherwise. Refusing symlinks is a code-owned local-artifact policy
derived from v4's `require_regular_file`, not a value v4 states.

**Adjudication order is frozen**, because a digest taken across an unstable read cannot
support a claim about which artifact it identifies:

    read and hash incrementally
    second fstat on the same descriptor
      fstat failed        -> UNEXPECTED_IO_FAILURE      not knowing is not detecting
      artifact moved      -> LOCAL_ARTIFACT_UNSTABLE    nothing escapes
      expected digest set and differs -> SOURCE_HASH_MISMATCH
    freeze the staging buffer to immutable bytes
    return

The expected-digest comparison happens before the freeze so an artifact destined for a
mismatch never allocates the immutable copy at all.

**Memory.** The staging buffer is a `bytearray` filled in bounded chunks, and the digest is
computed as the bytes arrive, so the artifact never needs to exist twice to be hashed. The
freeze is `bytes(memoryview(staging)[:filled])`: slicing the `bytearray` first would
materialise an intermediate copy and put three source-sized objects live at once, measured
at 2.00x additional allocation against 1.00x for the view form.

The freeze therefore holds two source buffers for an instant, and that is the peak of this
module. It does not add to v4's canonicalisation projection: the freeze is the last act
here, the decoded, downmix, and canonical arrays are the next slice's and do not exist yet,
and concurrency is pinned at one so no second artifact is live either. The peak is
`max(2S, S + 3D)`, which at the frozen bounds is `max(1.0, 5.0) = 5.0 GiB` against a
declared 6 GiB ceiling. Step 4a still has to measure resident memory rather than trust this
arithmetic.

**Close failure has two outcomes, and one of them is only observable in the log.** A close
that fails on an otherwise successful acquisition becomes `UNEXPECTED_IO_FAILURE`, because
failing to close a read-only descriptor means something is wrong with this module. A close
that fails when an abort is already established leaves the abort standing, since a cleanup
failure must not overwrite the finding that mattered, and the close failure is logged rather
than discarded. The result type carries no secondary failure and protocol v4 defines no
field for one, so an orchestration layer needing it structurally will have to add it.

A read-only `memoryview` over the staging buffer was considered and rejected as the result
type. `view.readonly` is True and `view.obj` is the mutable `bytearray`, reachable by
ordinary attribute access, so the bytes hashed here could differ from the bytes decoded
later. Immutability by contract is not immutability.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol as TypingProtocol

from benchmark.protocol import Decode
from benchmark.source_evidence import SourceIdentity
from benchmark.source_outcome import AbortReason, RunAbortOutcome

logger = logging.getLogger(__name__)

# Bounded per-read request. Peak is one staging buffer plus one of these, never two
# staging buffers, which is what keeps the read itself outside the freeze transition.
READ_CHUNK_BYTES = 4 * 1024 * 1024

# The fields compared before and after. Measured on the pinned platform: an in-place
# rewrite moves st_size, st_mtime_ns and st_ctime_ns while st_ino does not, so this detects
# that case. It cannot detect a same-size write inside timestamp granularity, which is why
# v4 claims detection and never proof.
_STABILITY_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")

# O_NOFOLLOW is required for an authoritative run rather than used when convenient: without
# it a symlink that resolves to a regular file is indistinguishable from the staged
# artifact, and the fstat check cannot tell them apart.
# O_NONBLOCK is required for the same reason: without it a FIFO hangs the run before any
# bound, budget, or timeout can apply, which is the worst failure this module has. It was
# briefly optional, which contradicted the docstring above calling it a liveness control.
_REQUIRED_FLAGS = ("O_NOFOLLOW", "O_NONBLOCK")
# O_CLOEXEC is genuinely defence in depth: nothing here spawns a subprocess.
_OPTIONAL_FLAGS = ("O_CLOEXEC",)


class ArtifactIoError(Exception):
    """Raised when this module is asked to do something it does not support."""


class ArtifactIo(TypingProtocol):
    """The syscall surface, narrow on purpose.

    Kept syscall-shaped so a fake scripts ordinary results and the production state machine
    is what the tests exercise. A fake that returned high-level states such as "the file
    grew" would test the fake's opinion instead.
    """

    def open(self, path: Path, flags: int) -> int:
        """Open `path` and return a descriptor."""

    def fstat(self, fd: int) -> os.stat_result:
        """Stat the open descriptor, never the path."""

    def read(self, fd: int, count: int) -> bytes:
        """Read at most `count` bytes, possibly fewer."""

    def close(self, fd: int) -> None:
        """Close the descriptor."""


class OsArtifactIo:
    """The production backend.

    `os.read` and `os.fstat` already retry on `EINTR` under PEP 475, so interruption is
    normalised here and callers never see it. `os.close` is deliberately not retried:
    after an interrupted close the descriptor state is ambiguous on some systems, and
    closing again can close a descriptor another thread has since been given.
    """

    def open(self, path: Path, flags: int) -> int:
        """Open `path` and return a descriptor."""
        return os.open(path, flags)

    def fstat(self, fd: int) -> os.stat_result:
        """Stat the open descriptor."""
        return os.fstat(fd)

    def read(self, fd: int, count: int) -> bytes:
        """Read at most `count` bytes."""
        return os.read(fd, count)

    def close(self, fd: int) -> None:
        """Close the descriptor."""
        os.close(fd)


@dataclass(frozen=True)
class ArtifactBuffered:
    """An artifact within the buffering bound, hashed and held as immutable bytes.

    `data` is the exact byte sequence the digest in `identity` was computed over, and the
    exact sequence the next slice inspects and decodes. That equality is the whole point of
    this module.
    """

    identity: SourceIdentity
    data: bytes


@dataclass(frozen=True)
class ArtifactTooLarge:
    """An artifact above the buffering bound, which will not be decoded.

    Carries whatever identity the frozen bounds permitted: a streamed digest below the
    identity bound, or none above it. Building the `SourceIneligibleOutcome` belongs to the
    caller, which holds the protocol digest and the stage context.
    """

    identity: SourceIdentity


ArtifactAcquisition = ArtifactBuffered | ArtifactTooLarge | RunAbortOutcome


def open_flags() -> int:
    """Assemble the open flags, requiring the ones an authoritative run depends on.

    Returns:
        The flag mask for `open`.

    Raises:
        ArtifactIoError: If a required flag is unavailable on this platform. Silently
            omitting `O_NOFOLLOW` would leave symlink refusal quietly best-effort.
    """
    flags = os.O_RDONLY
    for name in _REQUIRED_FLAGS:
        value = getattr(os, name, None)
        if value is None:
            raise ArtifactIoError(
                f"os.{name} is unavailable, and an authoritative run depends on it: without "
                "O_NOFOLLOW a symlink resolving to a regular file is indistinguishable from "
                "the staged artifact, and without O_NONBLOCK opening a FIFO hangs"
            )
        flags |= value
    for name in _OPTIONAL_FLAGS:
        flags |= getattr(os, name, 0)
    return flags


def _stability(status: os.stat_result) -> tuple[object, ...]:
    """The comparison tuple for one stat result."""
    return tuple(getattr(status, field) for field in _STABILITY_FIELDS)


def acquire_local_artifact(
    path: Path,
    *,
    logical_source_id: str,
    config: Decode,
    expected_sha256: str | None = None,
    io: ArtifactIo | None = None,
) -> ArtifactAcquisition:
    """Open, bound, hash, and adjudicate one local artifact.

    Args:
        path: The staged artifact.
        logical_source_id: Identifier of the candidate, so an abort can name it.
        config: The frozen `canonicalisation.decode` section, which owns the bounds.
        expected_sha256: A digest to compare against, when the caller has one.
        io: Syscall backend, defaulting to the real one. Injected in tests so short reads,
            growth, and stat failures are scripted rather than raced.

    Returns:
        `ArtifactBuffered` when the artifact is within the buffering bound,
        `ArtifactTooLarge` when it is above it, or a `RunAbortOutcome`.
    """
    backend = io if io is not None else OsArtifactIo()
    try:
        fd = backend.open(path, open_flags())
    except OSError as error:
        return _open_failure(error, logical_source_id)

    try:
        result = _with_descriptor(
            backend,
            fd,
            logical_source_id=logical_source_id,
            config=config,
            expected=expected_sha256,
        )
    finally:
        closed = _close(backend, fd)
    if not closed:
        if isinstance(result, RunAbortOutcome):
            # An established abort keeps its reason: a cleanup failure must not overwrite
            # the finding that mattered. The result type has no room for a secondary
            # failure, so rather than discard it silently the close failure goes to the log.
            # Carrying it structurally would need a field protocol v4 does not define.
            logger.warning(
                "Descriptor close failed for %s after %s; the abort stands and the close "
                "failure is recorded here only",
                logical_source_id,
                result.reason,
            )
        else:
            # A close that fails on a read-only descriptor means something is wrong with
            # this module's descriptor handling, so an otherwise successful acquisition does
            # not get to return as if nothing happened.
            return _io_abort(
                "the descriptor failed to close after an otherwise successful acquisition",
                logical_source_id,
            )
    return result


def _open_failure(error: OSError, logical_source_id: str) -> RunAbortOutcome:
    """Classify a failure to open.

    `ELOOP` is what `O_NOFOLLOW` raises for a final-component symlink, so it is the same
    finding as a FIFO or a device: the thing at that path is not the staged artifact.

    Args:
        error: The raised error.
        logical_source_id: The candidate being processed.

    Returns:
        The abort for this failure.
    """
    import errno

    if error.errno == errno.ELOOP:
        return RunAbortOutcome(
            reason=AbortReason.LOCAL_ARTIFACT_NOT_REGULAR_FILE,
            detail="the final path component is a symlink",
            logical_source_id=logical_source_id,
        )
    return RunAbortOutcome(
        reason=AbortReason.UNEXPECTED_IO_FAILURE,
        detail=f"open failed with {errno.errorcode.get(error.errno or 0, 'an unknown error')}",
        logical_source_id=logical_source_id,
    )


def _close(backend: ArtifactIo, fd: int) -> bool:
    """Close the descriptor and report whether it worked.

    A `finally` that raises would replace `LOCAL_ARTIFACT_UNSTABLE` or a hash mismatch with
    an I/O error, losing the finding that mattered, so the failure is returned rather than
    thrown. The caller surfaces it only when nothing else went wrong.

    Args:
        backend: The syscall backend.
        fd: The descriptor to close.

    Returns:
        True if the descriptor closed.
    """
    try:
        backend.close(fd)
    except OSError:
        return False
    return True


def _with_descriptor(
    backend: ArtifactIo,
    fd: int,
    *,
    logical_source_id: str,
    config: Decode,
    expected: str | None,
) -> ArtifactAcquisition:
    """Everything that happens while the descriptor is open.

    Args:
        backend: The syscall backend.
        fd: The open descriptor.
        logical_source_id: The candidate being processed.
        config: The frozen decode section.
        expected: A digest to compare against, if any.

    Returns:
        The acquisition result.
    """
    try:
        first = backend.fstat(fd)
    except OSError:
        return _io_abort("the first fstat failed", logical_source_id)

    if not stat.S_ISREG(first.st_mode):
        return RunAbortOutcome(
            reason=AbortReason.LOCAL_ARTIFACT_NOT_REGULAR_FILE,
            detail="the artifact is not a regular file",
            logical_source_id=logical_source_id,
        )

    size = first.st_size
    if size > config.max_identity_stream_bytes:
        return _classify_without_reading(
            backend, fd, first, size=size, logical_source_id=logical_source_id, config=config
        )
    if size > config.max_source_bytes:
        return _stream_identity(
            backend,
            fd,
            first,
            logical_source_id=logical_source_id,
            config=config,
            expected=expected,
        )
    return _buffer_artifact(
        backend, fd, first, logical_source_id=logical_source_id, config=config, expected=expected
    )


def _io_abort(detail: str, logical_source_id: str) -> RunAbortOutcome:
    """An abort for a syscall that failed rather than a fact that was established."""
    return RunAbortOutcome(
        reason=AbortReason.UNEXPECTED_IO_FAILURE,
        detail=detail,
        logical_source_id=logical_source_id,
    )


def _unstable(logical_source_id: str, detail: str) -> RunAbortOutcome:
    """An abort for an artifact that moved under the read."""
    return RunAbortOutcome(
        reason=AbortReason.LOCAL_ARTIFACT_UNSTABLE,
        detail=detail,
        logical_source_id=logical_source_id,
    )


def _content_disagreed_with_metadata(
    observed: int, routed: int, logical_source_id: str
) -> RunAbortOutcome | None:
    """Reject an artifact whose content contradicted the size that routed it.

    Direct evidence, independent of the metadata comparison that follows. An artifact that
    grew and returned to its original size and timestamps would pass the stat comparison
    and fail here. It also keeps the recorded identity honest by construction: the bytes
    hashed are the bytes the routing size promised, so the method the identity derives is
    the method that actually produced the digest.

    Args:
        observed: Bytes actually read and hashed.
        routed: The size the first stat reported.
        logical_source_id: The candidate being processed.

    Returns:
        An abort if they disagree, otherwise None.
    """
    if observed == routed:
        return None
    direction = "grew" if observed > routed else "shrank"
    return _unstable(
        logical_source_id,
        f"the artifact yielded {observed} bytes against a routing size of {routed}, so it "
        f"{direction} during the read",
    )


def _restat(
    backend: ArtifactIo, fd: int, first: os.stat_result, logical_source_id: str
) -> RunAbortOutcome | None:
    """Compare the artifact against itself across the operation.

    Args:
        backend: The syscall backend.
        fd: The open descriptor.
        first: The stat taken before any read.
        logical_source_id: The candidate being processed.

    Returns:
        An abort if the stat failed or the artifact moved, otherwise None.
    """
    try:
        second = backend.fstat(fd)
    except OSError:
        return _io_abort("the second fstat failed, so stability is unknown", logical_source_id)
    if _stability(second) != _stability(first):
        return _unstable(logical_source_id, "artifact metadata changed across the operation")
    return None


def _classify_without_reading(
    backend: ArtifactIo,
    fd: int,
    first: os.stat_result,
    *,
    size: int,
    logical_source_id: str,
    config: Decode,
) -> ArtifactAcquisition:
    """Handle an artifact above the identity stream bound.

    No content is read: the point of the bound is that a validly excluded source cannot buy
    unbounded work. The second stat still runs, because the decision *not* to hash is
    itself metadata-dependent, and if the size moved during classification this module
    cannot say the decision still describes what it saw.

    Args:
        backend: The syscall backend.
        fd: The open descriptor.
        first: The stat taken before classification.
        size: The observed size.
        logical_source_id: The candidate being processed.
        config: The frozen decode section.

    Returns:
        `ArtifactTooLarge` with no digest, or an abort.
    """
    moved = _restat(backend, fd, first, logical_source_id)
    if moved is not None:
        return moved
    return ArtifactTooLarge(identity=SourceIdentity(size, None, config))


def _read_bounded(
    backend: ArtifactIo, fd: int, *, initial: int, ceiling: int
) -> tuple[bytearray, int, str]:
    """Fill a bounded staging buffer, hashing as the bytes arrive.

    `os.read` may return fewer bytes than asked for, so the loop continues until it returns
    nothing. The initial capacity comes from the first stat, which routes but never
    authorises: capacity grows geometrically, capped at the ceiling, so an artifact that
    was reported as empty and then grew is still read up to the bound rather than truncated
    at a stale size.

    Args:
        backend: The syscall backend.
        fd: The open descriptor.
        initial: The size the first stat reported.
        ceiling: One byte past the largest admissible size, so growth past the bound is
            observed directly rather than inferred.

    Returns:
        The staging buffer, the number of bytes filled, and the digest of those bytes.
    """
    digest = hashlib.sha256()
    capacity = max(1, min(initial + 1, ceiling))
    staging = bytearray(capacity)
    filled = 0
    while filled < ceiling:
        if filled == len(staging):
            grown = min(max(len(staging) * 2, 1), ceiling)
            if grown == len(staging):
                break
            staging.extend(bytearray(grown - len(staging)))
        chunk = backend.read(fd, min(READ_CHUNK_BYTES, len(staging) - filled, ceiling - filled))
        if not chunk:
            break
        staging[filled : filled + len(chunk)] = chunk
        digest.update(chunk)
        filled += len(chunk)
    return staging, filled, digest.hexdigest()


def _stream_identity(
    backend: ArtifactIo,
    fd: int,
    first: os.stat_result,
    *,
    logical_source_id: str,
    config: Decode,
    expected: str | None,
) -> ArtifactAcquisition:
    """Hash an artifact above the buffering bound without retaining it.

    Args:
        backend: The syscall backend.
        fd: The open descriptor.
        first: The stat taken before reading.
        logical_source_id: The candidate being processed.
        config: The frozen decode section.
        expected: A digest to compare against, if any.

    Returns:
        `ArtifactTooLarge` carrying the streamed digest, or an abort.
    """
    ceiling = config.max_identity_stream_bytes + 1
    digest = hashlib.sha256()
    read_bytes = 0
    while read_bytes < ceiling:
        try:
            chunk = backend.read(fd, min(READ_CHUNK_BYTES, ceiling - read_bytes))
        except OSError:
            return _io_abort("a read failed during streamed identity", logical_source_id)
        if not chunk:
            break
        digest.update(chunk)
        read_bytes += len(chunk)

    disagreed = _content_disagreed_with_metadata(read_bytes, first.st_size, logical_source_id)
    if disagreed is not None:
        return disagreed
    moved = _restat(backend, fd, first, logical_source_id)
    if moved is not None:
        return moved
    computed = digest.hexdigest()
    if expected is not None and not hmac.compare_digest(computed, expected):
        return RunAbortOutcome(
            reason=AbortReason.SOURCE_HASH_MISMATCH,
            detail="the streamed digest differs from the expected digest",
            logical_source_id=logical_source_id,
        )
    return ArtifactTooLarge(identity=SourceIdentity(read_bytes, computed, config))


def _buffer_artifact(
    backend: ArtifactIo,
    fd: int,
    first: os.stat_result,
    *,
    logical_source_id: str,
    config: Decode,
    expected: str | None,
) -> ArtifactAcquisition:
    """Read an admissible artifact into one bounded buffer and freeze it.

    Args:
        backend: The syscall backend.
        fd: The open descriptor.
        first: The stat taken before reading.
        logical_source_id: The candidate being processed.
        config: The frozen decode section.
        expected: A digest to compare against, if any.

    Returns:
        `ArtifactBuffered` holding the exact hashed bytes, or an abort.
    """
    ceiling = config.max_source_bytes + 1
    try:
        staging, filled, computed = _read_bounded(
            backend, fd, initial=first.st_size, ceiling=ceiling
        )
    except OSError:
        return _io_abort("a read failed while buffering the artifact", logical_source_id)

    disagreed = _content_disagreed_with_metadata(filled, first.st_size, logical_source_id)
    if disagreed is not None:
        return disagreed
    moved = _restat(backend, fd, first, logical_source_id)
    if moved is not None:
        return moved
    if expected is not None and not hmac.compare_digest(computed, expected):
        # Before the freeze on purpose: an artifact destined for a mismatch never allocates
        # the immutable copy.
        return RunAbortOutcome(
            reason=AbortReason.SOURCE_HASH_MISMATCH,
            detail="the buffered digest differs from the expected digest",
            logical_source_id=logical_source_id,
        )
    # Slicing the bytearray first would materialise an intermediate copy and put three
    # source-sized objects live at once. The view form copies straight into the result.
    data = bytes(memoryview(staging)[:filled])
    del staging
    return ArtifactBuffered(identity=SourceIdentity(filled, computed, config), data=data)
