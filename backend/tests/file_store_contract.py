"""Behaviour every FileStore adapter that streams from local disk must show (ADR 010).

An adapter's test module subclasses FileStoreContract and provides two
fixtures: file_store, the adapter under test, and scratch_dir, where a source
file can be written before put_result moves it in. The class name does not
start with Test, so pytest collects these tests only through a subclass.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from app.services.file_store import FileStore

# A fixed ID with letters, so its upper-case form is a different string
_JOB = "3f2c9a4e-8b1d-4e6f-a5c7-9d0e1b2f3a4c"


def _source(scratch_dir: Path, content: bytes = b"RIFF result") -> Path:
    """Write a finished result where the denoiser would leave it."""
    source = scratch_dir / f"{uuid.uuid4()}_engine.wav"
    source.write_bytes(content)
    return source


class FileStoreContract:
    """Tests run against every FileStore adapter that streams from local disk."""

    @pytest.fixture()
    def file_store(self) -> FileStore:
        """Return the adapter under test."""
        raise NotImplementedError("Adapter test modules provide file_store")

    @pytest.fixture()
    def scratch_dir(self) -> Path:
        """Return a directory where put_result can take its source from."""
        raise NotImplementedError("Adapter test modules provide scratch_dir")

    def test_put_result_moves_the_file_in(self, file_store: FileStore, scratch_dir: Path) -> None:
        """After put_result, the result holds the source's bytes and the source is gone."""
        source = _source(scratch_dir, b"RIFF restored audio")

        file_store.put_result(_JOB, source)

        assert file_store.result_path(_JOB).read_bytes() == b"RIFF restored audio"
        assert not source.exists()

    def test_a_job_without_a_result_has_no_file(self, file_store: FileStore) -> None:
        """result_path for a job that has no result points at nothing."""
        assert not file_store.result_path(_JOB).exists()

    def test_delete_results_removes_the_result(
        self, file_store: FileStore, scratch_dir: Path
    ) -> None:
        """A deleted result is gone, and deleting again is safe."""
        file_store.put_result(_JOB, _source(scratch_dir))

        file_store.delete_results(_JOB)
        file_store.delete_results(_JOB)

        assert not file_store.result_path(_JOB).exists()

    def test_delete_results_leaves_other_jobs_alone(
        self, file_store: FileStore, scratch_dir: Path
    ) -> None:
        """Deleting one job's results keeps another job's result."""
        other = str(uuid.uuid4())
        file_store.put_result(_JOB, _source(scratch_dir))
        file_store.put_result(other, _source(scratch_dir, b"RIFF other"))

        file_store.delete_results(_JOB)

        assert file_store.result_path(other).read_bytes() == b"RIFF other"

    @pytest.mark.parametrize(
        "job_id",
        [
            "../escape",  # path traversal
            "not-a-uuid",  # arbitrary text
            _JOB.upper(),  # a UUID, but not the canonical lower-case form
            "{" + _JOB + "}",  # braces, accepted by uuid.UUID
            "urn:uuid:" + _JOB,  # URN prefix, accepted by uuid.UUID
        ],
    )
    def test_non_canonical_job_ids_are_refused(
        self, file_store: FileStore, scratch_dir: Path, job_id: str
    ) -> None:
        """Only the canonical UUID form the server generates can name a stored file."""
        source = _source(scratch_dir)

        with pytest.raises(ValueError):
            file_store.put_result(job_id, source)
        with pytest.raises(ValueError):
            file_store.result_path(job_id)
        with pytest.raises(ValueError):
            file_store.delete_results(job_id)

        assert source.exists()
