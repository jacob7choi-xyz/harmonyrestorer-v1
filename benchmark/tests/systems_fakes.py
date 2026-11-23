"""Fake systems, a passing precondition check, and the audit trail the walk writes.

An echo system writes its input back as its output, so scoring sees an exact copy of the frame,
and a system can be made to raise on one frame index. The systems stage reads the audit trail
beside the walk manifest and refuses it unless it hashes to the manifest's digest, so the
fixture trail is written with `write_audit_trail` and named by `AUDIT_DIGEST`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import soundfile as sf

from benchmark.protocol import Protocol
from benchmark.systems_pipeline import AUDIT_NAME

RATE = 16000
AUDIT_BYTES = b'{"fixture": "audit trail"}\n'
AUDIT_DIGEST = hashlib.sha256(AUDIT_BYTES).hexdigest()


def write_audit_trail(directory: Path, data: bytes = AUDIT_BYTES) -> Path:
    """Write the fixture audit trail into the walk directory and return its path."""
    path = directory / AUDIT_NAME
    path.write_bytes(data)
    return path


class EchoSystem:
    """A system that echoes its input, or raises for one frame index if asked to."""

    def __init__(self, name: str, output_dir: Path, fail_frame: int | None = None) -> None:
        """Store the name, the output folder, and the frame index that fails, if any."""
        self.name = name
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._fail_frame = fail_frame

    def denoise(self, input_path: Path) -> Path:
        """Raise for the failing frame, otherwise write the input back as the output."""
        if self._fail_frame is not None and input_path.stem.endswith(f"-{self._fail_frame:06d}"):
            raise RuntimeError("engine fault")
        samples, _ = sf.read(input_path, dtype="float32", always_2d=False)
        output = self._output_dir / f"{input_path.stem}_out.wav"
        sf.write(output, samples, RATE, subtype="FLOAT")
        return output


def passing(protocol: Protocol, **_: object) -> None:
    """A precondition check that every run passes."""
    return None


def echo_factory(fail_uvr_frame: int | None = 1) -> Any:
    """A systems factory: OpGAN echoes; UVR echoes except for one frame index."""

    def build(protocol: Protocol, run_dir: Path, checkpoint: Path, models: Path) -> list[Any]:
        outputs = run_dir / "outputs"
        return [
            EchoSystem("opgan", outputs / "opgan"),
            EchoSystem("uvr", outputs / "uvr", fail_frame=fail_uvr_frame),
        ]

    return build
