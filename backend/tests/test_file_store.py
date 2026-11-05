"""Tests for the local-disk FileStore adapter and its builder."""

import uuid
from pathlib import Path

import pytest
from app.config import FileStoreBackend, settings
from app.services.file_store import FileStore, LocalFileStore, build_file_store

from tests.file_store_contract import FileStoreContract


class TestLocalFileStore(FileStoreContract):
    """The shared contract, plus the file names other code relies on."""

    @pytest.fixture()
    def file_store(self, tmp_path: Path) -> FileStore:
        """A local store rooted in a temporary directory."""
        return LocalFileStore(tmp_path)

    @pytest.fixture()
    def scratch_dir(self, tmp_path: Path) -> Path:
        """The denoisers write into the root, so sources come from the same directory."""
        return tmp_path

    def test_result_is_named_for_the_job_in_the_root(self, tmp_path: Path) -> None:
        """The result is {job_id}_denoised.wav in the root, the name the transcoder
        derives cached conversions from."""
        job_id = str(uuid.uuid4())

        assert LocalFileStore(tmp_path).result_path(job_id) == tmp_path / f"{job_id}_denoised.wav"

    def test_delete_results_removes_cached_conversions(self, tmp_path: Path) -> None:
        """Conversions cached beside the result are removed with it."""
        job_id = str(uuid.uuid4())
        store = LocalFileStore(tmp_path)
        result = store.result_path(job_id)
        files = [result, result.with_suffix(".mp3"), result.with_suffix(".flac")]
        for f in files:
            f.write_bytes(b"x")

        store.delete_results(job_id)

        assert not any(f.exists() for f in files)


def test_build_file_store_roots_the_local_adapter_at_processed_dir() -> None:
    """The local backend stores results where the artifact budget looks."""
    store = build_file_store(FileStoreBackend.LOCAL)

    assert isinstance(store, LocalFileStore)
    job_id = str(uuid.uuid4())
    assert store.result_path(job_id).parent == settings.processed_dir
