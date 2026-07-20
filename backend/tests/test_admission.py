"""Tests for inference admission control and lifecycle-task slot ownership."""

import asyncio
import threading
from pathlib import Path

import pytest
from app.routes import denoise
from app.services.admission import InferenceAdmission
from app.services.jobs import job_manager


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

    def __call__(self, job_id: str, input_path: Path) -> None:
        self.started.set()
        assert self.finish.wait(timeout=10), "test never released fake inference"


class TestLifecycleOwnership:
    """Slot ownership belongs to the lifecycle task, not the HTTP handler."""

    async def test_route_cancellation_does_not_release_slot(self, monkeypatch) -> None:
        """Cancel A mid-inference: B stays rejected until A truly completes, then C admits.

        This is the invariant that makes orphaned work bounded: a cancelled
        request must not free inference capacity while the worker thread is
        still executing.
        """
        fake = _BlockingFakeInference()
        monkeypatch.setattr(job_manager, "process", fake)

        assert denoise.admission.try_acquire()
        task = asyncio.create_task(denoise._run_inference("job-a", Path("/tmp/none")))
        denoise._inference_tasks.add(task)
        task.add_done_callback(denoise._observe_inference_task)

        async def route_wait() -> None:
            await asyncio.shield(task)

        route = asyncio.create_task(route_wait())
        await asyncio.to_thread(fake.started.wait, 5)

        route.cancel()
        with pytest.raises(asyncio.CancelledError):
            await route

        # Request B: capacity must still be occupied by the orphaned inference
        assert denoise.admission.try_acquire() is False
        # The registry still strongly references the surviving lifecycle task
        assert task in denoise._inference_tasks

        fake.finish.set()
        await task

        # Registry releases its reference only after true completion
        assert task not in denoise._inference_tasks
        # Request C: capacity is available again
        assert denoise.admission.try_acquire() is True
        denoise.admission.release()

    async def test_inference_does_not_block_event_loop(self, monkeypatch) -> None:
        """The event loop stays responsive while a fake inference occupies the thread.

        This proves loop liveness only; CPU-headroom behavior under real
        inference is verified in the live deployment characterization.
        """
        fake = _BlockingFakeInference()
        monkeypatch.setattr(job_manager, "process", fake)

        assert denoise.admission.try_acquire()
        task = asyncio.create_task(denoise._run_inference("job-a", Path("/tmp/none")))
        denoise._inference_tasks.add(task)
        task.add_done_callback(denoise._observe_inference_task)
        await asyncio.to_thread(fake.started.wait, 5)

        # A loop-scheduled sleep completing promptly proves the loop is live
        await asyncio.wait_for(asyncio.sleep(0.01), timeout=1)

        fake.finish.set()
        await task
        # The slot was released by the lifecycle task's own finally
        assert denoise.admission.try_acquire() is True
        denoise.admission.release()

    async def test_lifecycle_releases_slot_on_processing_exception(self, monkeypatch) -> None:
        """An exception escaping process() still releases the slot via finally."""

        def exploding(job_id: str, input_path: Path) -> None:
            raise RuntimeError("lifecycle escape")

        monkeypatch.setattr(job_manager, "process", exploding)

        assert denoise.admission.try_acquire()
        task = asyncio.create_task(denoise._run_inference("job-x", Path("/tmp/none")))
        denoise._inference_tasks.add(task)
        task.add_done_callback(denoise._observe_inference_task)

        with pytest.raises(RuntimeError, match="lifecycle escape"):
            await task

        assert denoise.admission.try_acquire() is True
        denoise.admission.release()
        assert task not in denoise._inference_tasks
