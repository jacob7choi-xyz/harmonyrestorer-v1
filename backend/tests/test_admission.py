"""Tests for inference admission control, lifecycle-task slot ownership, and job status."""

import asyncio
import threading
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from app.config import settings
from app.routes import denoise
from app.schemas import JobStatusEnum
from app.services.admission import InferenceAdmission
from app.services.jobs import job_manager


@pytest.fixture()
def clean_jobs() -> Iterator[None]:
    """Start and end with no job records in the shared job manager."""
    job_manager._jobs.clear()
    yield
    job_manager._jobs.clear()


class TestInferenceAdmission:
    """Tests for the InferenceAdmission primitive."""

    def test_acquires_up_to_capacity(self) -> None:
        adm = InferenceAdmission(capacity=2)
        assert adm.try_acquire() is True
        assert adm.try_acquire() is True
        assert adm.try_acquire() is False

    def test_release_restores_capacity(self) -> None:
        adm = InferenceAdmission(capacity=1)
        assert adm.try_acquire() is True
        assert adm.try_acquire() is False
        adm.release()
        assert adm.try_acquire() is True

    def test_double_release_fails_loudly(self) -> None:
        adm = InferenceAdmission(capacity=1)
        with pytest.raises(asyncio.QueueFull):
            adm.release()


class _BlockingFakeInference:
    """Synchronous fake inference controllable from the test."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.finish = threading.Event()
        self.done = threading.Event()

    def __call__(self, job_id: str, input_path: Path) -> float:
        self.started.set()
        try:
            assert self.finish.wait(timeout=10), "test never released fake inference"
        finally:
            self.done.set()
        return 0.0


class TestLifecycleOwnership:
    """Slot ownership belongs to the lifecycle task, not the HTTP handler."""

    async def test_route_cancellation_does_not_release_slot(self, monkeypatch, clean_jobs) -> None:
        """Cancel A mid-inference: B stays rejected until A truly completes, then C admits.

        This is the invariant that makes orphaned work bounded: a cancelled
        request must not free inference capacity while the worker thread is
        still executing.
        """
        fake = _BlockingFakeInference()
        monkeypatch.setattr(job_manager, "run_denoiser", fake)
        job_manager.create_job("job-a")

        assert denoise.admission.try_acquire()
        task = asyncio.create_task(denoise._run_inference("job-a", Path("/tmp/none"), 0))
        denoise._inference_tasks.add(task)
        task.add_done_callback(denoise._observe_inference_task)

        async def route_wait() -> None:
            await asyncio.shield(task)

        route = asyncio.create_task(route_wait())
        try:
            await asyncio.to_thread(fake.started.wait, 5)

            route.cancel()
            with pytest.raises(asyncio.CancelledError):
                await route

            # Request B: capacity must still be occupied by the orphaned inference
            assert denoise.admission.try_acquire() is False
            # The registry still strongly references the surviving lifecycle task
            assert task in denoise._inference_tasks
        finally:
            fake.finish.set()
        await task

        # Registry releases its reference only after true completion
        assert task not in denoise._inference_tasks
        # Request C: capacity is available again
        assert denoise.admission.try_acquire() is True
        denoise.admission.release()

    async def test_inference_does_not_block_event_loop(self, monkeypatch, clean_jobs) -> None:
        """The event loop stays responsive while a fake inference occupies the thread.

        This proves loop liveness only; CPU-headroom behavior under real
        inference is verified in the live deployment characterization.
        """
        fake = _BlockingFakeInference()
        monkeypatch.setattr(job_manager, "run_denoiser", fake)
        job_manager.create_job("job-a")

        assert denoise.admission.try_acquire()
        task = asyncio.create_task(denoise._run_inference("job-a", Path("/tmp/none"), 0))
        denoise._inference_tasks.add(task)
        task.add_done_callback(denoise._observe_inference_task)
        try:
            await asyncio.to_thread(fake.started.wait, 5)

            # A loop-scheduled sleep completing promptly proves the loop is live
            await asyncio.wait_for(asyncio.sleep(0.01), timeout=1)
        finally:
            fake.finish.set()
        await task
        # The slot was released by the lifecycle task's own finally
        assert denoise.admission.try_acquire() is True
        denoise.admission.release()

    async def test_lifecycle_releases_slot_on_processing_exception(self, monkeypatch) -> None:
        """An exception escaping the lifecycle still releases the slot via finally.

        Worker errors become a FAILED job, so the escaping exception here comes
        from a status change, which the lifecycle does not catch.
        """

        def exploding(job_id: str) -> bool:
            raise RuntimeError("lifecycle escape")

        monkeypatch.setattr(job_manager, "start_processing", exploding)

        assert denoise.admission.try_acquire()
        task = asyncio.create_task(denoise._run_inference("job-x", Path("/tmp/none"), 0))
        denoise._inference_tasks.add(task)
        task.add_done_callback(denoise._observe_inference_task)

        with pytest.raises(RuntimeError, match="lifecycle escape"):
            await task

        assert denoise.admission.try_acquire() is True
        denoise.admission.release()
        assert task not in denoise._inference_tasks


class TestLifecycleStatus:
    """The lifecycle, not the worker thread, records each job's status."""

    async def test_worker_failure_marks_job_failed(self, tmp_path, monkeypatch, clean_jobs) -> None:
        """A denoiser error becomes a FAILED job, with no output file and no input left."""
        monkeypatch.setattr(settings, "processed_dir", tmp_path)
        denoiser = MagicMock()
        denoiser.denoise.side_effect = RuntimeError("model crashed")
        monkeypatch.setattr(job_manager, "_denoiser", denoiser)
        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"fake audio")
        job_manager.create_job("fail-job")

        await denoise._process_job("fail-job", input_file)

        job = job_manager.get_job("fail-job")
        assert job is not None
        assert job.status == JobStatusEnum.FAILED
        assert job.progress == -1
        assert job.download_url is None
        assert not (tmp_path / "fail-job_denoised.wav").exists()
        assert not input_file.exists()

    async def test_missing_job_removes_input_without_denoising(
        self, tmp_path, monkeypatch, clean_jobs
    ) -> None:
        """With no QUEUED record, the input is removed and the denoiser never runs."""
        denoiser = MagicMock()
        monkeypatch.setattr(job_manager, "_denoiser", denoiser)
        input_file = tmp_path / "input.wav"
        input_file.write_bytes(b"fake audio")

        await denoise._process_job("vanish-job", input_file)

        assert not input_file.exists()
        denoiser.denoise.assert_not_called()
        assert job_manager.get_job("vanish-job") is None

    async def test_cancelled_lifecycle_records_no_status(self, monkeypatch, clean_jobs) -> None:
        """Cancelling the lifecycle mid-worker leaves the job PROCESSING.

        The worker thread cannot be stopped and keeps going after the
        cancellation, so a status written at that point could contradict the
        files it has yet to write or remove.
        """
        fake = _BlockingFakeInference()
        monkeypatch.setattr(job_manager, "run_denoiser", fake)
        job_manager.create_job("job-c")
        task = asyncio.create_task(denoise._process_job("job-c", Path("/tmp/none")))
        try:
            await asyncio.to_thread(fake.started.wait, 5)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            job = job_manager.get_job("job-c")
            assert job is not None
            assert job.status == JobStatusEnum.PROCESSING
        finally:
            fake.finish.set()
            await asyncio.to_thread(fake.done.wait, 5)
