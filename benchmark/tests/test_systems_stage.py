"""Tests for the systems stage's engine boundary, with fake systems and tiny synthetic WAVs."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from benchmark.systems_stage import (
    FrameIdentity,
    HarnessFailure,
    SystemFailed,
    SystemOutput,
    call_system,
    write_frame_input,
    write_progress_marker,
)

RATE = 16000
IDENTITY = FrameIdentity(canonical_sha256="0" * 64, frame_index=3)
Behaviour = Callable[[Path], Path]


class FakeSystem:
    """A system whose `denoise` runs a behaviour, and which records each call and the marker."""

    def __init__(self, behaviour: Behaviour, name: str = "fake") -> None:
        """Store the behaviour and the system's name."""
        self.name = name
        self._behaviour = behaviour
        self.calls: list[Path] = []
        self.marker_seen_inside_call: dict[str, object] | None = None
        self.marker_path: Path | None = None

    def denoise(self, input_path: Path) -> Path:
        """Record the call, look at the marker from inside the call, then run the behaviour."""
        self.calls.append(input_path)
        if self.marker_path is not None and self.marker_path.exists():
            self.marker_seen_inside_call = json.loads(self.marker_path.read_text("utf-8"))
        return self._behaviour(input_path)


def write_wav(path: Path, samples: npt.NDArray[np.float32], rate: int = RATE) -> Path:
    """Write float samples as a float WAV, which is what the stage writes for its inputs."""
    sf.write(path, samples, rate, subtype="FLOAT")
    return path


def ramp(length: int = 320) -> npt.NDArray[np.float32]:
    """A short deterministic frame."""
    return np.linspace(-0.5, 0.5, length, dtype=np.float32)


def echo(directory: Path, *, rate: int = RATE, channels: int = 1) -> Behaviour:
    """A behaviour that writes a copy of its input as the output, at the given format."""

    def run(input_path: Path) -> Path:
        samples, _ = sf.read(input_path, dtype="float32", always_2d=False)
        output = directory / f"{input_path.stem}_out.wav"
        data = samples if channels == 1 else np.stack([samples] * channels, axis=1)
        sf.write(output, data, rate, subtype="FLOAT")
        return output

    return run


def failing(error: BaseException) -> Behaviour:
    """A behaviour that raises the given error inside the call."""

    def run(input_path: Path) -> Path:
        raise error

    return run


def boundary(system: FakeSystem, input_path: Path, marker: Path) -> SystemFailed | SystemOutput:
    """Call the system through the boundary with the test's identity and marker."""
    system.marker_path = marker
    return call_system(system, input_path, identity=IDENTITY, marker_path=marker, sample_rate=RATE)


class TestSuccessfulCall:
    """A readable mono output at the required rate is returned as float64 samples."""

    def test_the_output_is_returned_as_float64_samples(self, tmp_path: Path) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        system = FakeSystem(echo(tmp_path))

        result = boundary(system, frame, tmp_path / "progress.json")

        assert isinstance(result, SystemOutput)
        assert result.samples.dtype == np.float64
        np.testing.assert_array_equal(result.samples, ramp().astype(np.float64))
        assert system.calls == [frame]


class TestFailuresInsideTheCall:
    """Exceptions raised inside the call are that system's failure for the frame."""

    def test_an_exception_inside_the_call_is_that_systems_failure(self, tmp_path: Path) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        system = FakeSystem(failing(RuntimeError("engine broke")))

        result = boundary(system, frame, tmp_path / "progress.json")

        assert isinstance(result, SystemFailed)
        assert result.exception_type == "RuntimeError"
        assert "engine broke" in result.reason

    def test_memory_error_inside_the_call_is_that_systems_failure(self, tmp_path: Path) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        system = FakeSystem(failing(MemoryError()))

        result = boundary(system, frame, tmp_path / "progress.json")

        assert isinstance(result, SystemFailed)
        assert result.exception_type == "MemoryError"

    def test_a_wrong_sample_rate_is_that_systems_failure(self, tmp_path: Path) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        system = FakeSystem(echo(tmp_path, rate=22050))

        result = boundary(system, frame, tmp_path / "progress.json")

        assert isinstance(result, SystemFailed)
        assert result.exception_type is None
        assert "22050" in result.reason

    def test_a_stereo_output_is_that_systems_failure(self, tmp_path: Path) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        system = FakeSystem(echo(tmp_path, channels=2))

        result = boundary(system, frame, tmp_path / "progress.json")

        assert isinstance(result, SystemFailed)
        assert result.exception_type is None
        assert "2 channels" in result.reason


class TestPropagation:
    """Interrupts and exits are not system failures; they propagate and end the run."""

    def test_keyboard_interrupt_propagates(self, tmp_path: Path) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        system = FakeSystem(failing(KeyboardInterrupt()))

        with pytest.raises(KeyboardInterrupt):
            boundary(system, frame, tmp_path / "progress.json")

    def test_system_exit_propagates(self, tmp_path: Path) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        system = FakeSystem(failing(SystemExit(3)))

        with pytest.raises(SystemExit):
            boundary(system, frame, tmp_path / "progress.json")


class TestHarnessFailures:
    """Failures outside the call abort the run, and no system is charged for them."""

    def test_a_failed_input_write_raises_and_leaves_no_file(self, tmp_path: Path) -> None:
        target = tmp_path / "missing" / "frame.wav"

        with pytest.raises(HarnessFailure):
            write_frame_input(ramp(), target, RATE)

        assert not target.exists()

    def test_a_written_input_is_a_mono_float_wav_at_the_rate(self, tmp_path: Path) -> None:
        target = tmp_path / "frame.wav"

        write_frame_input(ramp(), target, RATE)

        info = sf.info(target)
        assert (info.samplerate, info.channels, info.subtype) == (RATE, 1, "FLOAT")
        samples, _ = sf.read(target, dtype="float32", always_2d=False)
        np.testing.assert_array_equal(samples, ramp())

    def test_a_failed_marker_write_raises_before_the_call(self, tmp_path: Path) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        system = FakeSystem(echo(tmp_path))

        with pytest.raises(HarnessFailure):
            call_system(
                system,
                frame,
                identity=IDENTITY,
                marker_path=tmp_path / "missing" / "progress.json",
                sample_rate=RATE,
            )

        assert system.calls == []

    def test_the_marker_is_written_before_the_call_and_names_the_frame(
        self, tmp_path: Path
    ) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        system = FakeSystem(echo(tmp_path), name="opgan")

        boundary(system, frame, tmp_path / "progress.json")

        assert system.marker_seen_inside_call == {
            "system": "opgan",
            "canonical_sha256": IDENTITY.canonical_sha256,
            "frame_index": IDENTITY.frame_index,
        }

    def test_an_output_path_that_is_missing_aborts_the_run(self, tmp_path: Path) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        system = FakeSystem(lambda _: tmp_path / "never_written.wav")

        with pytest.raises(HarnessFailure):
            boundary(system, frame, tmp_path / "progress.json")

        assert len(system.calls) == 1

    def test_an_unreadable_output_file_aborts_the_run(self, tmp_path: Path) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        not_audio = tmp_path / "not_audio.wav"
        not_audio.write_bytes(b"not a wav file at all")
        system = FakeSystem(lambda _: not_audio)

        with pytest.raises(HarnessFailure):
            boundary(system, frame, tmp_path / "progress.json")

        assert len(system.calls) == 1

    def test_a_read_that_fails_after_the_header_aborts_the_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frame = write_wav(tmp_path / "frame.wav", ramp())
        output = write_wav(tmp_path / "output.wav", ramp())
        system = FakeSystem(lambda _: output)

        def failing_read(*args: object, **kwargs: object) -> None:
            raise sf.SoundFileError("injected read failure")

        monkeypatch.setattr(sf, "read", failing_read)

        with pytest.raises(HarnessFailure, match="samples cannot be read"):
            boundary(system, frame, tmp_path / "progress.json")

        assert len(system.calls) == 1

    def test_a_marker_write_failure_is_not_a_system_failure(self, tmp_path: Path) -> None:
        with pytest.raises(HarnessFailure):
            write_progress_marker(tmp_path / "missing" / "m.json", "opgan", IDENTITY)
