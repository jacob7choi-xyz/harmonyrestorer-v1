"""Both shipped engines on degraded float WAV frames, as measured before v7 was frozen.

Frames: three synthetic tone frames and 20 training clean frames (`data/clean_frames`,
sorted, permuted by `default_rng(3)`). Each is degraded at training level with the
protocol's ranges and run seed and written as a 16 kHz mono float WAV. Each engine is
built as the backend builds it, and each call's success or exception, output format, and
output length minus input length are printed. The UVR model and its two parameter files
are then compared with the digests v7 froze. The first UVR call downloads them if they
are not already in audio-separator's model directory.

Run from the repository root with the backend importable:
`PYTHONPATH=backend uv run --frozen python -m benchmark.measurements.engine_smoke`.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol as Interface

import numpy as np
import numpy.typing as npt
import soundfile as sf
from app.config import OPGAN_CHECKPOINT_SHA256
from app.services.denoiser import DenoiserService
from app.services.opgan_denoiser import OpGANDenoiserService

from benchmark.degradation import degrade_frame, frame_seed
from benchmark.protocol import Protocol, load_protocol

_RATE, _FRAME = 16000, 32000
_CLEAN_FRAMES = Path("data/clean_frames")
_SAMPLE_SEED, _SAMPLED = 3, 20
_TONES = ((220.0, 440.0), (110.0, 330.0, 990.0), (523.25, 659.25, 783.99))
_UVR_DIR = Path("/tmp/audio-separator-models")


class Engine(Interface):
    """The engine contract the backend's job pipeline uses."""

    def denoise(self, input_path: Path) -> Path:
        """Denoise *input_path* and return the output WAV's path."""
        ...


@dataclass(frozen=True)
class Call:
    """One engine call: the frame, and its output format and length, or the exception."""

    frame: str
    error: str | None
    rate: int = 0
    subtype: str = ""
    channels: int = 0
    length_difference: int = 0


def frames() -> list[tuple[str, npt.NDArray[np.float64], float]]:
    """The measured frames, each with the recording peak that sets its training level.

    Training clean frames are already at training level, so their recording peak is the
    headroom; a synthetic frame is its own recording.
    """
    t = np.arange(_FRAME) / _RATE
    chosen: list[tuple[str, npt.NDArray[np.float64], float]] = []
    for index, tones in enumerate(_TONES):
        x = sum(np.sin(2 * np.pi * f * t) for f in tones)
        tone: npt.NDArray[np.float64] = 0.8 * x / np.max(np.abs(x))
        chosen.append((f"synthetic_{index}", tone, 0.8))
    names = sorted(path.name for path in _CLEAN_FRAMES.iterdir())
    for k in np.random.default_rng(_SAMPLE_SEED).permutation(len(names))[:_SAMPLED]:
        audio, rate = sf.read(_CLEAN_FRAMES / names[k], dtype="float64")
        assert rate == _RATE and audio.shape == (_FRAME,)
        chosen.append((names[k].removesuffix(".wav"), audio, 0.95))
    return chosen


def write_inputs(protocol: Protocol, directory: Path) -> list[Path]:
    """Degrade each frame at training level and write it as a 16 kHz mono float WAV."""
    paths = []
    for name, clean, peak in frames():
        digest = hashlib.sha256(clean.astype("<f8").tobytes()).hexdigest()
        result = degrade_frame(
            clean,
            recording_peak=peak,
            sample_rate=_RATE,
            ranges=protocol.degradation.ranges,
            seed=frame_seed(protocol.degradation.seed, digest, 0),
        )
        path = directory / f"{name}.wav"
        sf.write(path, result.degraded, _RATE, subtype="FLOAT")
        paths.append(path)
    return paths


def call(engine: Engine, path: Path) -> Call:
    """One `denoise` call, recorded rather than raised so every frame is measured."""
    try:
        info = sf.info(str(engine.denoise(path)))
    # The types the engines raise for a failed job (RuntimeError also covers soundfile and
    # torch errors). Anything else stops the measurement rather than being recorded.
    except (RuntimeError, OSError, ValueError) as e:
        return Call(path.stem, f"{type(e).__name__}: {e}")
    return Call(path.stem, None, info.samplerate, info.subtype, info.channels, info.frames - _FRAME)


def report(name: str, calls: list[Call]) -> None:
    """Print one engine's successes, formats, and length differences."""
    ok = [c for c in calls if c.error is None]
    formats = sorted({(c.rate, c.subtype, c.channels) for c in ok})
    differences = sorted({c.length_difference for c in ok})
    print(
        f"{name}: {len(ok)}/{len(calls)} succeeded; formats {formats}; length differences {differences}"
    )
    for c in calls:
        if c.error is not None:
            print(f"  FAILED {c.frame}: {c.error}")


def uvr_identity(protocol: Protocol) -> None:
    """Compare the UVR files audio-separator loaded with the digests v7 froze."""
    execution = protocol.system_execution
    for file, frozen in (
        (execution.uvr_model, execution.uvr_model_sha256),
        ("vr_model_data.json", execution.uvr_vr_model_data_sha256),
        ("mdx_model_data.json", execution.uvr_mdx_model_data_sha256),
    ):
        path = _UVR_DIR / file
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        print(f"{file}: sha256 {actual}, matches v7: {actual == frozen}")


def main() -> None:
    """Run both engines on every frame, then check the UVR files' identity."""
    protocol = load_protocol()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "inputs").mkdir()
        paths = write_inputs(protocol, root / "inputs")
        engines: dict[str, Engine] = {
            "opgan": OpGANDenoiserService(
                root / "opgan",
                Path("checkpoints/final.pt"),
                expected_sha256=OPGAN_CHECKPOINT_SHA256,
            ),
            "uvr": DenoiserService(root / "uvr", protocol.system_execution.uvr_model),
        }
        for name, engine in engines.items():
            report(name, [call(engine, path) for path in paths])
    uvr_identity(protocol)


if __name__ == "__main__":
    main()
