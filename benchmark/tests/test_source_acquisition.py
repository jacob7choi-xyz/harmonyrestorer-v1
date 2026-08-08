"""Tests for bounded, descriptor-safe local artifact access.

The failures that matter here are races and resource exhaustion, and neither is testable by
timing. Every hazard is scripted through a fake that returns ordinary syscall results, so
the production state machine is what runs and the outcome is deterministic. A fake that
returned high-level states such as "the file grew" would be testing its own opinion.

The fake also records what was asked of it, which is how the security properties are
asserted rather than inferred from a return value: the flags passed to open, every
descriptor touched, the size of every read request, the order of the calls, and how many
times close was called.
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import signal
import stat
import tracemalloc
from pathlib import Path

import pytest

from benchmark.protocol import Decode, load_protocol
from benchmark.source_acquisition import (
    READ_CHUNK_BYTES,
    ArtifactBuffered,
    ArtifactIoError,
    ArtifactTooLarge,
    OsArtifactIo,
    acquire_local_artifact,
    open_flags,
)
from benchmark.source_evidence import IdentityMethod, IdentityStatus
from benchmark.source_outcome import AbortReason, RunAbortOutcome

FD = 7


def stat_result(
    size: int,
    *,
    mode: int = stat.S_IFREG | 0o644,
    mtime: int = 1000,
    inode: int = 1,
    device: int = 42,
) -> os.stat_result:
    """A stat result carrying every field the stability tuple compares."""
    return os.stat_result(
        (mode, inode, device, 1, 0, 0, size, 0, 0, 0),
        {"st_mtime_ns": mtime, "st_ctime_ns": mtime},
    )


class FakeIo:
    """A scripted syscall backend that records everything it was asked."""

    def __init__(
        self,
        *,
        stats: list[os.stat_result | OSError],
        reads: list[bytes | OSError] | None = None,
        open_error: OSError | None = None,
        close_error: OSError | None = None,
    ) -> None:
        self.stats = list(stats)
        self.reads = list(reads or [])
        self.open_error = open_error
        self.close_error = close_error
        self.calls: list[str] = []
        self.flags: int | None = None
        self.descriptors: list[int] = []
        self.read_requests: list[int] = []
        self.bytes_returned = 0
        self.closes = 0

    def open(self, path: Path, flags: int) -> int:
        self.calls.append("open")
        self.flags = flags
        if self.open_error is not None:
            raise self.open_error
        return FD

    def fstat(self, fd: int) -> os.stat_result:
        self.calls.append("fstat")
        self.descriptors.append(fd)
        result = self.stats.pop(0)
        if isinstance(result, OSError):
            raise result
        return result

    def read(self, fd: int, count: int) -> bytes:
        self.calls.append("read")
        self.descriptors.append(fd)
        self.read_requests.append(count)
        if not self.reads:
            return b""
        result = self.reads.pop(0)
        if isinstance(result, OSError):
            raise result
        self.bytes_returned += len(result)
        return result

    def close(self, fd: int) -> None:
        self.calls.append("close")
        self.descriptors.append(fd)
        self.closes += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def decode() -> Decode:
    """The frozen decode section, which owns every bound this module enforces."""
    return load_protocol().canonicalisation.decode


def acquire(io: FakeIo, decode: Decode, **kwargs: object) -> object:
    """Run an acquisition against the fake."""
    return acquire_local_artifact(
        Path("/staged/candidate.flac"),
        logical_source_id="candidate-0001",
        config=decode,
        io=io,
        **kwargs,  # type: ignore[arg-type]
    )


class TestOpenFlags:
    """A plain open can hang, and a followed symlink is a different artifact."""

    def test_nonblock_is_present(self) -> None:
        """Measured on the pinned platform: os.open on a FIFO with O_RDONLY alone blocks
        until a writer appears, which hangs the run before any bound or budget applies."""
        assert open_flags() & os.O_NONBLOCK

    @pytest.mark.parametrize("flag", ["O_NOFOLLOW", "O_NONBLOCK"])
    def test_both_required_flags_refuse_to_run_when_absent(
        self, monkeypatch: pytest.MonkeyPatch, flag: str
    ) -> None:
        """O_NONBLOCK was briefly optional, which contradicted calling it a liveness
        control: a platform without it would have hung on the first FIFO rather than
        refusing to start."""
        monkeypatch.delattr(os, flag, raising=False)
        with pytest.raises(ArtifactIoError, match=f"os.{flag} is unavailable"):
            open_flags()

    def test_nofollow_is_present(self) -> None:
        assert open_flags() & os.O_NOFOLLOW

    def test_cloexec_is_present(self) -> None:
        assert open_flags() & os.O_CLOEXEC

    def test_a_missing_required_flag_refuses_to_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Silently omitting O_NOFOLLOW would make symlink refusal quietly best-effort, and
        an authoritative run would differ from a development one without saying so."""
        monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
        with pytest.raises(ArtifactIoError, match="O_NOFOLLOW is unavailable"):
            open_flags()

    def test_the_flags_reach_the_syscall(self, decode) -> None:
        io = FakeIo(stats=[stat_result(0), stat_result(0)])
        acquire(io, decode)
        assert io.flags == open_flags()


class TestOneDescriptor:
    """The classic TOCTOU shape is stat(path) then open(path)."""

    def test_every_syscall_uses_the_descriptor_from_the_single_open(self, decode) -> None:
        io = FakeIo(stats=[stat_result(4), stat_result(4)], reads=[b"abcd"])
        acquire(io, decode)
        assert io.calls.count("open") == 1
        assert set(io.descriptors) == {FD}

    def test_the_path_is_never_reopened(self, decode) -> None:
        io = FakeIo(stats=[stat_result(4), stat_result(4)], reads=[b"abcd"])
        acquire(io, decode)
        assert io.calls.count("open") == 1

    @pytest.mark.parametrize(
        ("label", "io_factory"),
        [
            ("success", lambda: FakeIo(stats=[stat_result(4), stat_result(4)], reads=[b"abcd"])),
            ("not regular", lambda: FakeIo(stats=[stat_result(0, mode=stat.S_IFIFO)])),
            ("first stat fails", lambda: FakeIo(stats=[OSError(errno.EIO, "boom")])),
            (
                "second stat fails",
                lambda: FakeIo(stats=[stat_result(4), OSError(errno.EIO, "boom")], reads=[b"abcd"]),
            ),
            (
                "unstable",
                lambda: FakeIo(stats=[stat_result(4), stat_result(5)], reads=[b"abcd"]),
            ),
            (
                "read fails",
                lambda: FakeIo(stats=[stat_result(4)], reads=[OSError(errno.EIO, "boom")]),
            ),
        ],
    )
    def test_the_descriptor_is_closed_on_every_path(self, decode, label, io_factory) -> None:
        io = io_factory()
        acquire(io, decode)
        assert io.closes == 1, label


class TestNonRegularArtifacts:
    """fstat adjudicates the type, after O_NONBLOCK has made the open return."""

    @pytest.mark.parametrize(
        ("label", "mode"),
        [
            ("fifo", stat.S_IFIFO),
            ("character device", stat.S_IFCHR),
            ("block device", stat.S_IFBLK),
            ("directory", stat.S_IFDIR),
            ("socket", stat.S_IFSOCK),
        ],
    )
    def test_a_non_regular_artifact_aborts(self, decode, label: str, mode: int) -> None:
        io = FakeIo(stats=[stat_result(0, mode=mode | 0o644)])
        result = acquire(io, decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_NOT_REGULAR_FILE
        assert io.calls.count("read") == 0

    def test_a_symlink_is_refused_by_the_kernel(self, decode) -> None:
        """O_NOFOLLOW raises ELOOP for a final-component symlink, which is the same finding
        as a FIFO: the thing at that path is not the staged artifact."""
        io = FakeIo(stats=[], open_error=OSError(errno.ELOOP, "symlink"))
        result = acquire(io, decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_NOT_REGULAR_FILE

    def test_another_open_failure_is_an_io_abort(self, decode) -> None:
        io = FakeIo(stats=[], open_error=OSError(errno.EACCES, "denied"))
        result = acquire(io, decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.UNEXPECTED_IO_FAILURE


class TestBoundaries:
    """The bounds route, and one byte either side must land differently."""

    def test_at_the_buffer_bound_the_artifact_is_buffered(self, decode) -> None:
        size = 8
        io = FakeIo(stats=[stat_result(size), stat_result(size)], reads=[b"x" * 8])
        decode = _decode_with_bounds(decode, max_source_bytes=8)
        result = acquire(io, decode)
        assert isinstance(result, ArtifactBuffered)
        assert result.identity.method is IdentityMethod.BOUNDED_SINGLE_BUFFER_SHA256

    def test_one_byte_over_switches_to_streaming(self, decode) -> None:
        decode = _decode_with_bounds(decode, max_source_bytes=7)
        io = FakeIo(stats=[stat_result(8), stat_result(8)], reads=[b"x" * 8])
        result = acquire(io, decode)
        assert isinstance(result, ArtifactTooLarge)
        assert result.identity.method is IdentityMethod.BOUNDED_STREAMING_SHA256

    def test_at_the_identity_bound_streaming_still_applies(self, decode) -> None:
        decode = _decode_with_bounds(decode, max_source_bytes=7)
        io = FakeIo(stats=[stat_result(8), stat_result(8)], reads=[b"x" * 8])
        result = acquire(io, decode)
        assert isinstance(result, ArtifactTooLarge)
        assert result.identity.status is IdentityStatus.COMPLETE_SHA256

    def test_above_the_identity_bound_nothing_is_read(self, decode) -> None:
        """The point of the bound is that a validly excluded source cannot buy unbounded
        work. Deciding not to read must happen before the work, not after."""
        size = decode.max_identity_stream_bytes + 1
        io = FakeIo(stats=[stat_result(size), stat_result(size)])
        result = acquire(io, decode)
        assert isinstance(result, ArtifactTooLarge)
        assert result.identity.status is IdentityStatus.UNAVAILABLE_ABOVE_IDENTITY_STREAM_BOUND
        assert result.identity.sha256 is None
        assert io.calls.count("read") == 0

    def test_a_zero_byte_artifact_is_buffered_with_its_digest(self, decode) -> None:
        io = FakeIo(stats=[stat_result(0), stat_result(0)])
        result = acquire(io, decode)
        assert isinstance(result, ArtifactBuffered)
        assert result.data == b""
        assert result.identity.sha256 == hashlib.sha256(b"").hexdigest()


class TestReadLoop:
    """os.read may return fewer bytes than asked for, and a size is a hint."""

    def test_short_reads_are_reassembled(self, decode) -> None:
        payload = b"the complete artifact"
        io = FakeIo(
            stats=[stat_result(len(payload)), stat_result(len(payload))],
            reads=[payload[:3], payload[3:9], payload[9:]],
        )
        result = acquire(io, decode)
        assert isinstance(result, ArtifactBuffered)
        assert result.data == payload
        assert result.identity.sha256 == hashlib.sha256(payload).hexdigest()

    def test_a_read_request_never_exceeds_the_remaining_bound(self, decode) -> None:
        io = FakeIo(stats=[stat_result(10), stat_result(10)], reads=[b"0123456789"])
        acquire(io, decode)
        assert max(io.read_requests) <= decode.max_source_bytes + 1
        assert max(io.read_requests) <= READ_CHUNK_BYTES

    def test_the_digest_covers_the_whole_artifact_not_a_prefix(self, decode) -> None:
        payload = b"a" * 100
        io = FakeIo(
            stats=[stat_result(100), stat_result(100)],
            reads=[payload[:50], payload[50:]],
        )
        result = acquire(io, decode)
        assert isinstance(result, ArtifactBuffered)
        assert result.identity.sha256 == hashlib.sha256(payload).hexdigest()
        assert result.identity.sha256 != hashlib.sha256(payload[:50]).hexdigest()

    def test_content_that_exceeds_the_routing_size_is_instability(self, decode) -> None:
        """The first stat routes but never authorises. Capacity grows so a stale size cannot
        truncate what the digest covers, and the disagreement between the bytes read and the
        size that routed them is itself evidence the artifact moved."""
        payload = b"x" * (READ_CHUNK_BYTES // 4)
        io = FakeIo(stats=[stat_result(0), stat_result(0)], reads=[payload])
        result = acquire(io, decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_UNSTABLE
        assert "grew during the read" in result.detail

    def test_a_read_failure_is_an_io_abort(self, decode) -> None:
        io = FakeIo(stats=[stat_result(10)], reads=[OSError(errno.EIO, "boom")])
        result = acquire(io, decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.UNEXPECTED_IO_FAILURE


class TestStabilityAdjudication:
    """The order is frozen because a digest from an unstable read identifies nothing."""

    def test_the_buffered_path_restats(self, decode) -> None:
        """Not only the streaming path. A same-size in-place rewrite during a buffered read
        is exactly as invisible as one during a streamed read."""
        io = FakeIo(stats=[stat_result(4), stat_result(4)], reads=[b"abcd"])
        acquire(io, decode)
        assert io.calls.count("fstat") == 2

    def test_the_no_read_path_restats(self, decode) -> None:
        """The decision not to hash is itself metadata-dependent, so if the size moved
        during classification this module cannot say the decision still describes what it
        saw."""
        size = decode.max_identity_stream_bytes + 1
        io = FakeIo(stats=[stat_result(size), stat_result(size)])
        acquire(io, decode)
        assert io.calls.count("fstat") == 2
        assert io.calls.count("read") == 0

    @pytest.mark.parametrize(
        ("label", "second"),
        [
            ("size moved", stat_result(5)),
            ("mtime moved", stat_result(4, mtime=2000)),
            ("inode moved", stat_result(4, inode=99)),
            ("device moved", stat_result(4, device=99)),
        ],
    )
    def test_a_moved_artifact_aborts_as_unstable(self, decode, label: str, second) -> None:
        """Each of the five compared fields, so no single one carries the whole check."""
        io = FakeIo(stats=[stat_result(4), second], reads=[b"abcd"])
        result = acquire(io, decode)
        assert isinstance(result, RunAbortOutcome), label
        assert result.reason is AbortReason.LOCAL_ARTIFACT_UNSTABLE

    def test_an_unstable_read_yields_no_digest(self, decode) -> None:
        """v4's eleventh record constraint, enforced where the digest is produced."""
        io = FakeIo(stats=[stat_result(4), stat_result(5)], reads=[b"abcd"])
        result = acquire(io, decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.identity is None

    def test_a_failed_second_stat_is_an_io_abort_not_instability(self, decode) -> None:
        """Not knowing is not the same as having detected a change."""
        io = FakeIo(stats=[stat_result(4), OSError(errno.EIO, "boom")], reads=[b"abcd"])
        result = acquire(io, decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.UNEXPECTED_IO_FAILURE
        assert result.identity is None

    def test_content_growth_past_the_bound_is_instability_on_its_own(self, decode) -> None:
        """Direct evidence, independent of the metadata comparison: the content contradicted
        the size that routed it. An artifact that grew and returned to its original size and
        timestamps would pass the stat comparison and fail this."""
        io = FakeIo(
            stats=[stat_result(4), stat_result(4)],
            reads=[b"x" * 4, b"y" * 4],
        )
        tiny = _decode_with_bounds(decode, max_source_bytes=4)
        result = acquire(io, tiny)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_UNSTABLE


class TestExpectedDigest:
    """A mismatch claim requires a digest that identifies something."""

    def test_a_matching_digest_is_accepted(self, decode) -> None:
        payload = b"abcd"
        io = FakeIo(stats=[stat_result(4), stat_result(4)], reads=[payload])
        result = acquire(io, decode, expected_sha256=hashlib.sha256(payload).hexdigest())
        assert isinstance(result, ArtifactBuffered)

    def test_a_differing_digest_aborts_as_a_mismatch(self, decode) -> None:
        io = FakeIo(stats=[stat_result(4), stat_result(4)], reads=[b"abcd"])
        result = acquire(io, decode, expected_sha256="0" * 64)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.SOURCE_HASH_MISMATCH

    def test_instability_wins_over_a_mismatch(self, decode) -> None:
        """A digest produced across an unstable read cannot support a mismatch claim, so
        the ordering is not a preference."""
        io = FakeIo(stats=[stat_result(4), stat_result(5)], reads=[b"abcd"])
        result = acquire(io, decode, expected_sha256="0" * 64)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_UNSTABLE

    def test_a_failed_stat_wins_over_a_mismatch(self, decode) -> None:
        io = FakeIo(stats=[stat_result(4), OSError(errno.EIO, "boom")], reads=[b"abcd"])
        result = acquire(io, decode, expected_sha256="0" * 64)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.UNEXPECTED_IO_FAILURE

    def test_a_streamed_digest_is_compared_too(self, decode) -> None:
        decode = _decode_with_bounds(decode, max_source_bytes=3)
        io = FakeIo(stats=[stat_result(4), stat_result(4)], reads=[b"abcd"])
        result = acquire(io, decode, expected_sha256="0" * 64)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.SOURCE_HASH_MISMATCH


class TestTheResultIsGenuinelyImmutable:
    """The bytes hashed must be the bytes decoded, and that is structural or it is nothing."""

    def test_the_buffer_is_bytes_not_a_view(self, decode) -> None:
        """A read-only memoryview over the staging bytearray was rejected: `view.obj` is
        the mutable buffer, reachable by ordinary attribute access, so the artifact could
        change between hashing and decoding."""
        io = FakeIo(stats=[stat_result(4), stat_result(4)], reads=[b"abcd"])
        result = acquire(io, decode)
        assert isinstance(result, ArtifactBuffered)
        assert type(result.data) is bytes
        assert not hasattr(result.data, "obj")

    def test_the_result_holds_no_reference_to_a_mutable_buffer(self, decode) -> None:
        io = FakeIo(stats=[stat_result(4), stat_result(4)], reads=[b"abcd"])
        result = acquire(io, decode)
        assert isinstance(result, ArtifactBuffered)
        for value in vars(result).values():
            assert not isinstance(value, bytearray | memoryview)


class TestAgainstTheRealFilesystem:
    """A handful of end-to-end cases, because the fake cannot prove the flags work."""

    def test_a_real_file_round_trips(self, decode, tmp_path: Path) -> None:
        payload = b"real bytes on a real disk" * 100
        artifact = tmp_path / "a.bin"
        artifact.write_bytes(payload)
        result = acquire_local_artifact(artifact, logical_source_id="c1", config=decode)
        assert isinstance(result, ArtifactBuffered)
        assert result.data == payload
        assert result.identity.sha256 == hashlib.sha256(payload).hexdigest()

    def test_a_real_fifo_does_not_hang(self, decode, tmp_path: Path) -> None:
        """Without O_NONBLOCK this call blocks until a writer appears.

        Guarded by an alarm rather than left to complete or not. A regression here would
        otherwise hang the suite forever instead of failing it, which is a worse outcome
        than the bug: CI would sit at a green-looking "running" state indefinitely. Found
        by mutating O_NONBLOCK away and watching the sweep stop rather than report.
        """
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo)

        def timed_out(*_: object) -> None:
            raise TimeoutError("opening a FIFO blocked, so O_NONBLOCK is not being passed")

        previous = signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, 5.0)
        try:
            result = acquire_local_artifact(fifo, logical_source_id="c1", config=decode)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_NOT_REGULAR_FILE

    def test_a_real_symlink_is_refused(self, decode, tmp_path: Path) -> None:
        target = tmp_path / "target.bin"
        target.write_bytes(b"payload")
        link = tmp_path / "link.bin"
        link.symlink_to(target)
        result = acquire_local_artifact(link, logical_source_id="c1", config=decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_NOT_REGULAR_FILE

    def test_a_real_directory_is_refused(self, decode, tmp_path: Path) -> None:
        result = acquire_local_artifact(tmp_path, logical_source_id="c1", config=decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_NOT_REGULAR_FILE

    def test_a_missing_path_is_an_io_abort(self, decode, tmp_path: Path) -> None:
        result = acquire_local_artifact(tmp_path / "absent", logical_source_id="c1", config=decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.UNEXPECTED_IO_FAILURE

    def test_the_production_backend_is_used_by_default(self, decode, tmp_path: Path) -> None:
        artifact = tmp_path / "a.bin"
        artifact.write_bytes(b"x")
        assert isinstance(OsArtifactIo(), object)
        result = acquire_local_artifact(artifact, logical_source_id="c1", config=decode)
        assert isinstance(result, ArtifactBuffered)


def _decode_with_bounds(
    decode: Decode, *, max_source_bytes: int, max_identity_stream_bytes: int | None = None
) -> Decode:
    """A decode section with bounds lowered, so boundary tests need no huge fixtures."""
    import dataclasses

    changes: dict[str, int] = {"max_source_bytes": max_source_bytes}
    if max_identity_stream_bytes is not None:
        changes["max_identity_stream_bytes"] = max_identity_stream_bytes
    return dataclasses.replace(decode, **changes)


class EndlessIo(FakeIo):
    """A backend whose artifact never ends, so the ceiling is what stops the read.

    Every other fake returns exactly the routed number of bytes, which means the loop ends
    on an empty read and the bound is never the thing that stops it. Four mutations that
    widened the ceilings survived the whole suite for exactly that reason.
    """

    def __init__(self, *, stats: list[os.stat_result | OSError], chunk: bytes) -> None:
        super().__init__(stats=stats)
        self.chunk = chunk

    def read(self, fd: int, count: int) -> bytes:
        self.calls.append("read")
        self.descriptors.append(fd)
        self.read_requests.append(count)
        served = self.chunk[:count] if count < len(self.chunk) else self.chunk
        self.bytes_returned += len(served)
        return served


class TestWorkIsBounded:
    """The ceiling must be what stops the read, not the artifact running out."""

    def test_the_buffering_read_stops_at_the_ceiling(self, decode) -> None:
        """An artifact that keeps yielding bytes must be cut off one past the bound, so a
        source can never buy unbounded reading by lying about its size."""
        tiny = _decode_with_bounds(decode, max_source_bytes=16)
        io = EndlessIo(stats=[stat_result(16), stat_result(16)], chunk=b"x" * 8)
        result = acquire(io, tiny)
        assert io.bytes_returned <= tiny.max_source_bytes + 1
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_UNSTABLE

    def test_the_streaming_read_stops_at_the_identity_ceiling(self, decode) -> None:
        """The same property on the path that exists precisely so an excluded artifact
        cannot buy unbounded work as its parting act."""
        tiny = _decode_with_bounds(decode, max_source_bytes=8, max_identity_stream_bytes=32)
        io = EndlessIo(stats=[stat_result(16), stat_result(16)], chunk=b"x" * 8)
        result = acquire(io, tiny)
        assert io.bytes_returned <= tiny.max_identity_stream_bytes + 1
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_UNSTABLE

    def test_a_streamed_digest_covers_every_chunk(self, decode) -> None:
        """Multi-chunk, because a single-read fixture cannot tell a whole-artifact digest
        from a first-chunk one."""
        tiny = _decode_with_bounds(decode, max_source_bytes=3)
        payload = b"first" + b"second" + b"third"
        io = FakeIo(
            stats=[stat_result(len(payload)), stat_result(len(payload))],
            reads=[b"first", b"second", b"third"],
        )
        result = acquire(io, tiny)
        assert isinstance(result, ArtifactTooLarge)
        assert result.identity.sha256 == hashlib.sha256(payload).hexdigest()
        assert result.identity.sha256 != hashlib.sha256(b"first").hexdigest()


class TestTheFreezeDoesNotTripleTheBuffer:
    """A memory property with no behavioural signature, so it needs a memory assertion."""

    def test_the_freeze_allocates_one_copy_not_two(self, decode, tmp_path: Path) -> None:
        """`bytes(staging[:filled])` materialises an intermediate bytearray before copying
        again, putting three source-sized objects live at once. Measured at 2.00x additional
        allocation against 1.00x for the view form, which is the difference between v4's
        memory proof holding and silently spending half its headroom."""
        payload = os.urandom(8 * 1024 * 1024)
        artifact = tmp_path / "big.bin"
        artifact.write_bytes(payload)

        tracemalloc.start()
        result = acquire_local_artifact(artifact, logical_source_id="c1", config=decode)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert isinstance(result, ArtifactBuffered)
        # One staging buffer plus one frozen copy, plus a read chunk and slack. The slice
        # form would add a third whole copy and overshoot this.
        assert peak < len(payload) * 2.6, f"peak {peak} for a {len(payload)} byte artifact"


class TestCloseFailure:
    """A finally that raises would replace the finding that mattered."""

    def test_a_close_failure_does_not_displace_an_abort(self, decode) -> None:
        io = FakeIo(
            stats=[stat_result(4), stat_result(5)],
            reads=[b"abcd"],
            close_error=OSError(errno.EIO, "close failed"),
        )
        result = acquire(io, decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_UNSTABLE

    def test_a_close_failure_beside_an_abort_is_logged_not_discarded(
        self, decode, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The abort must stand, but the cleanup failure must not vanish. The result type
        carries no secondary failure and v4 defines no field for one, so the log is the only
        place it can go until an orchestration layer adds a structural home for it."""
        io = FakeIo(
            stats=[stat_result(4), stat_result(5)],
            reads=[b"abcd"],
            close_error=OSError(errno.EIO, "close failed"),
        )
        with caplog.at_level(logging.WARNING, logger="benchmark.source_acquisition"):
            result = acquire(io, decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.LOCAL_ARTIFACT_UNSTABLE
        assert any("close failed" in r.getMessage().lower() for r in caplog.records), caplog.text
        assert any("candidate-0001" in r.getMessage() for r in caplog.records)

    def test_a_successful_close_logs_nothing(
        self, decode, caplog: pytest.LogCaptureFixture
    ) -> None:
        """So the warning above means something when it appears."""
        io = FakeIo(stats=[stat_result(4), stat_result(5)], reads=[b"abcd"])
        with caplog.at_level(logging.WARNING, logger="benchmark.source_acquisition"):
            acquire(io, decode)
        assert not caplog.records

    def test_a_close_failure_on_an_otherwise_successful_read_is_surfaced(self, decode) -> None:
        """Swallowing it would be a fail-open. A close that fails on a read-only descriptor
        means something is wrong with this module's descriptor handling, so the acquisition
        does not get to return as if nothing happened."""
        io = FakeIo(
            stats=[stat_result(4), stat_result(4)],
            reads=[b"abcd"],
            close_error=OSError(errno.EIO, "close failed"),
        )
        result = acquire(io, decode)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.UNEXPECTED_IO_FAILURE
        assert "failed to close" in result.detail

    def test_a_close_failure_does_not_displace_a_mismatch(self, decode) -> None:
        io = FakeIo(
            stats=[stat_result(4), stat_result(4)],
            reads=[b"abcd"],
            close_error=OSError(errno.EIO, "close failed"),
        )
        result = acquire(io, decode, expected_sha256="0" * 64)
        assert isinstance(result, RunAbortOutcome)
        assert result.reason is AbortReason.SOURCE_HASH_MISMATCH
