"""Tests for the listening build: blinding, the sealed key, loudness, windows and refusals.

The acquisition directory is written by hand from synthetic 96 kHz FLACs and 16 kHz canonical
waveforms. The engines are fakes. The UVR fake returns the service's 16 kHz output by default.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from benchmark.canonical_json import canonical_json_bytes
from benchmark.listening.acquire import ACQUISITION_NAME
from benchmark.listening.build import (
    BUILD_NAME,
    CONDITIONS,
    KEY_NAME,
    UVR_OUTPUT_RATE_HZ,
    BuildError,
    _assign,
    _cut,
    build,
    to_pcm16,
)
from benchmark.listening.loudness import integrated_loudness
from benchmark.listening.protocol import ListeningProtocol, load_listening_protocol
from benchmark.listening.selection import window_bounds, window_rule_for, window_start
from benchmark.protocol import Protocol, load_protocol

_RATE = 16_000
_FLAC_RATE = 96_000
_AUDIO_S = 40.0
_COUNT = 10
_CLIP_SAMPLES = 8 * _RATE
_FORBIDDEN_WORDS = (b"original", b"opgan", b"OpGAN", b"uvr", b"UVR", b"audio-separator")


def _flac(seconds: float, tone_hz: float) -> bytes:
    """A stereo 96 kHz, 24-bit FLAC of a sine, as the candidate files are."""
    t = np.arange(round(seconds * _FLAC_RATE)) / _FLAC_RATE
    tone = 0.5 * np.sin(2.0 * np.pi * tone_hz * t)
    buffer = io.BytesIO()
    sf.write(buffer, np.stack([tone, tone], axis=1), _FLAC_RATE, format="FLAC", subtype="PCM_24")
    return buffer.getvalue()


def _canonical(seconds: float, tone_hz: float, amplitude: float) -> bytes:
    """A 16 kHz float64 canonical side: a sine at *amplitude*."""
    t = np.arange(round(seconds * _RATE)) / _RATE
    return (amplitude * np.sin(2.0 * np.pi * tone_hz * t)).astype("<f8").tobytes()


def _write_acquisition(root: Path, protocol: ListeningProtocol, flac_tamper: bool = False) -> Path:
    """An acquisition directory of _COUNT recordings, written as `acquire` would lay it out."""
    acquisition = root / "acquisition"
    (acquisition / "sources").mkdir(parents=True)
    (acquisition / "work").mkdir()
    rule = window_rule_for(protocol)
    bounds = window_bounds(_AUDIO_S, protocol.clips.margin_s, int(protocol.clips.duration_s))
    assert bounds is not None
    items: list[dict[str, Any]] = []
    for index in range(_COUNT):
        identifier = f"rec-{index:02d}"
        flac = _flac(_AUDIO_S, 260.0 + 37.0 * index)
        flac_path = acquisition / "sources" / f"{index:02d}.flac"
        flac_path.write_bytes(flac)
        if flac_tamper and index == 0:
            flac_path.write_bytes(flac[:-1] + bytes([flac[-1] ^ 0xFF]))
        canonical = _canonical(_AUDIO_S, 300.0 + 23.0 * index, 0.2 + 0.03 * index)
        canonical_path = acquisition / "work" / f"{index:02d}" / "canonical.f64"
        canonical_path.parent.mkdir()
        canonical_path.write_bytes(canonical)
        items.append(
            {
                "identifier": identifier,
                "flac": {
                    "path": flac_path.relative_to(acquisition).as_posix(),
                    "sha256": hashlib.sha256(flac).hexdigest(),
                },
                "canonical": {
                    "path": canonical_path.relative_to(acquisition).as_posix(),
                    "sha256": hashlib.sha256(canonical).hexdigest(),
                    "samples": len(canonical) // 8,
                },
                "decoded_pcm_sha256": hashlib.sha256(canonical).hexdigest(),
                "window": {
                    "start_s": window_start(identifier, rule, bounds),
                    "duration_s": protocol.clips.duration_s,
                    "low_s": bounds[0],
                    "high_s": bounds[1],
                },
            }
        )
    manifest = {
        "protocol_sha256": protocol.sha256,
        "universe_sha256": protocol.universe.sha256,
        "items": items,
    }
    (acquisition / ACQUISITION_NAME).write_bytes(canonical_json_bytes(manifest))
    return acquisition


class _Engine:
    """A fake engine: scales its input, optionally resamples it, and writes the output WAV."""

    def __init__(self, out: Path, gain: float, rate_out: int | None = None) -> None:
        self._out = out
        self._gain = gain
        self._rate_out = rate_out

    def denoise(self, input_path: Path) -> Path:
        """Return the path of a WAV with the input scaled and, if asked, at another rate."""
        data, rate = sf.read(str(input_path), dtype="float64")
        target = self._rate_out or rate
        scaled = self._gain * data
        if target != rate:
            scaled = librosa.resample(scaled, orig_sr=rate, target_sr=target)
        path = self._out / f"{input_path.stem}_out.wav"
        sf.write(str(path), scaled, target, subtype="FLOAT")
        return path


def _run_build(
    root: Path,
    protocol: ListeningProtocol,
    benchmark: Protocol,
    acquisition: Path,
    *,
    seed: int = 7,
    uvr_rate: int = UVR_OUTPUT_RATE_HZ,
    out_name: str = "build",
) -> Path:
    """Build into a new directory under *root*, with fake engines and a scratch directory."""
    scratch = root / f"scratch-{out_name}"
    scratch.mkdir(exist_ok=True)
    engine_out = scratch / "engine_out"
    engine_out.mkdir(exist_ok=True)
    return build(
        protocol,
        benchmark,
        acquisition,
        root / out_name,
        seed=seed,
        opgan=_Engine(engine_out, gain=0.9),
        uvr=_Engine(engine_out, gain=0.8, rate_out=uvr_rate),
        scratch=scratch,
    )


@pytest.fixture(scope="module")
def protocol() -> ListeningProtocol:
    """The frozen listening protocol, with its digests checked."""
    return load_listening_protocol()


@pytest.fixture(scope="module")
def benchmark() -> Protocol:
    """The benchmark protocol, which the decoder and resampler arguments come from."""
    return load_protocol()


@pytest.fixture(scope="module")
def built(
    tmp_path_factory: pytest.TempPathFactory, protocol: ListeningProtocol, benchmark: Protocol
) -> tuple[Path, Path]:
    """One successful build, shared by the read-only tests. Returns (build dir, acquisition dir)."""
    root = tmp_path_factory.mktemp("built")
    acquisition = _write_acquisition(root, protocol)
    return _run_build(root, protocol, benchmark, acquisition), acquisition


def _key(out: Path) -> dict[str, Any]:
    """The sealed key of a build, parsed."""
    data: dict[str, Any] = json.loads((out / "sealed" / KEY_NAME).read_text())
    return data


def _clip(out: Path, blinded_id: str) -> npt.NDArray[np.int16]:
    """The 16-bit samples of one clip file."""
    samples, rate = sf.read(str(out / "clips" / f"{blinded_id}.wav"), dtype="int16")
    assert rate == _RATE
    return np.asarray(samples, dtype=np.int16)


def test_each_recording_gets_three_clips_under_blinded_names(built: tuple[Path, Path]) -> None:
    """Thirty clips for ten recordings, each named by twelve hexadecimal digits."""
    out, _ = built
    names = sorted(p.stem for p in (out / "clips").iterdir())

    assert len(names) == _COUNT * len(CONDITIONS)
    assert all(len(name) == 12 and set(name) <= set("0123456789abcdef") for name in names)


def test_no_condition_or_system_name_appears_in_the_listener_files(
    built: tuple[Path, Path],
) -> None:
    """Clip names, clip bytes and the public build manifest hold no condition or system name."""
    out, _ = built
    public = [p for p in out.rglob("*") if p.is_file() and "sealed" not in p.parts]

    for path in public:
        assert not any(word.decode() in path.name for word in _FORBIDDEN_WORDS), path.name
        data = path.read_bytes()
        assert not any(word in data for word in _FORBIDDEN_WORDS), path.name


def test_the_sealed_key_maps_each_blinded_clip_to_one_recording_and_condition(
    built: tuple[Path, Path],
) -> None:
    """Every (recording, condition) pair appears exactly once, under a clip file that exists."""
    out, acquisition = built
    clips = _key(out)["clips"]
    pairs = [(clip["identifier"], clip["condition"]) for clip in clips]

    assert len(set(pairs)) == _COUNT * len(CONDITIONS) == len(pairs)
    assert {clip["blinded_id"] for clip in clips} == {p.stem for p in (out / "clips").iterdir()}
    assert {clip["identifier"] for clip in clips} == {
        item["identifier"]
        for item in json.loads((acquisition / ACQUISITION_NAME).read_text())["items"]
    }


def test_the_public_manifest_carries_the_key_digest(built: tuple[Path, Path]) -> None:
    """The build manifest names the sealed key and carries its SHA-256."""
    out, _ = built
    manifest = json.loads((out / BUILD_NAME).read_text())

    assert (
        manifest["key_sha256"]
        == hashlib.sha256((out / "sealed" / KEY_NAME).read_bytes()).hexdigest()
    )
    assert manifest["key_path"] == "sealed/key.json"


def test_each_clip_measures_its_recordings_loudness_target(
    built: tuple[Path, Path], protocol: ListeningProtocol
) -> None:
    """Every clip of a recording measures at that recording's target, within 0.1 LU."""
    out, _ = built
    for clip in _key(out)["clips"]:
        samples = _clip(out, clip["blinded_id"]).astype(np.float64) / 2**15
        measured = integrated_loudness(samples, _RATE)

        assert measured == pytest.approx(clip["target_lkfs"], abs=0.1), clip["blinded_id"]


def test_the_original_clip_is_the_manifest_window_at_its_gain(
    built: tuple[Path, Path],
) -> None:
    """The original clip is the canonical window at the recorded start, times the gain."""
    out, acquisition = built
    manifest = json.loads((acquisition / ACQUISITION_NAME).read_text())
    canonical_by_id = {
        item["identifier"]: np.frombuffer(
            (acquisition / item["canonical"]["path"]).read_bytes(), dtype="<f8"
        )
        for item in manifest["items"]
    }
    starts = {item["identifier"]: item["window"]["start_s"] for item in manifest["items"]}
    for clip in _key(out)["clips"]:
        if clip["condition"] != "original":
            continue
        first = starts[clip["identifier"]] * _RATE
        window = canonical_by_id[clip["identifier"]][first : first + _CLIP_SAMPLES] * clip["gain"]
        expected = np.rint(window * 2**15) / 2**15

        assert np.array_equal(
            _clip(out, clip["blinded_id"]), np.rint(expected * 2**15).astype(np.int16)
        )


def test_the_engine_work_stays_outside_the_published_directory(built: tuple[Path, Path]) -> None:
    """The published directory holds only the clips, the sealed key and the build manifest."""
    out, _ = built

    assert sorted(p.name for p in out.iterdir()) == ["build.json", "clips", "sealed"]


def test_a_uvr_output_at_16_khz_is_accepted(
    tmp_path: Path, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """The UVR service's own 16 kHz mono output is read as it is and the build completes."""
    acquisition = _write_acquisition(tmp_path, protocol)

    out = _run_build(tmp_path, protocol, benchmark, acquisition, uvr_rate=UVR_OUTPUT_RATE_HZ)

    assert len(list((out / "clips").iterdir())) == _COUNT * len(CONDITIONS)


def test_a_uvr_output_at_another_rate_is_refused_and_nothing_is_written(
    tmp_path: Path, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """A UVR output at 22.05 kHz is not the service's 16 kHz output, so the build stops."""
    acquisition = _write_acquisition(tmp_path, protocol)

    with pytest.raises(BuildError, match=f"is not {UVR_OUTPUT_RATE_HZ} Hz mono"):
        _run_build(tmp_path, protocol, benchmark, acquisition, uvr_rate=22_050)

    assert not (tmp_path / "build").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["acquisition", "scratch-build"]


def test_a_staged_flac_that_fails_its_digest_is_refused(
    tmp_path: Path, protocol: ListeningProtocol, benchmark: Protocol
) -> None:
    """A staged FLAC that differs from its recorded digest stops the build before engines run."""
    acquisition = _write_acquisition(tmp_path, protocol, flac_tamper=True)

    with pytest.raises(BuildError, match="does not match its digest"):
        _run_build(tmp_path, protocol, benchmark, acquisition)

    assert not (tmp_path / "build").exists()


def test_an_existing_output_directory_is_refused(
    built: tuple[Path, Path], protocol: ListeningProtocol, benchmark: Protocol, tmp_path: Path
) -> None:
    """The build refuses an output directory that exists, and leaves its contents alone."""
    _, acquisition = built
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "keep.txt").write_text("earlier")

    with pytest.raises(BuildError, match="already exists"):
        build(
            protocol,
            benchmark,
            acquisition,
            existing,
            seed=1,
            opgan=_Engine(tmp_path, gain=1.0),
            uvr=_Engine(tmp_path, gain=1.0),
            scratch=tmp_path / "scratch",
        )

    assert [p.name for p in existing.iterdir()] == ["keep.txt"]


def test_a_clip_that_would_clip_at_16_bits_is_refused() -> None:
    """A sample that scales past the int16 range is refused, so nothing is clipped silently."""
    with pytest.raises(BuildError, match="would clip"):
        to_pcm16(np.array([0.0, 1.0]), "clip")
    assert to_pcm16(np.array([-1.0]), "clip")[0] == -(2**15)


def test_the_16_bit_rounding_takes_the_nearest_integer_and_ties_to_even() -> None:
    """Scaled values round to the nearest integer, with exact halves going to the even integer."""
    scaled = np.array([0.5, 1.5, 2.5, 2.4, -2.6]) / 2**15

    assert to_pcm16(scaled, "clip").tolist() == [0, 2, 2, 2, -3]


def test_a_condition_shorter_than_the_window_is_refused_and_nothing_is_cut() -> None:
    """A condition that ends before the window does refuses the cut, rather than shortening it."""
    sides = {"original": np.zeros(_CLIP_SAMPLES * 3), "opgan": np.zeros(_CLIP_SAMPLES - 1)}

    with pytest.raises(BuildError, match="too short"):
        _cut(sides, 0, _CLIP_SAMPLES, _RATE)


def test_the_blinding_is_fixed_by_the_seed() -> None:
    """The same seed gives the same assignment and a different seed gives a different one."""
    pairs = [(f"rec-{i}", condition) for i in range(3) for condition in CONDITIONS]

    assert _assign(5, pairs) == _assign(5, pairs)
    assert _assign(5, pairs) != _assign(6, pairs)
