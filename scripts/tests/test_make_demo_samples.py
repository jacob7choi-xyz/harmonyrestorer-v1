"""Tests for the demo sample generator.

The engines are fakes, so no model is loaded. Each fake writes its input back out through a
transform and records the samples it was given, so a test can check what the engines received.
The synthetic recording has its peak outside the clip, so a clip-peak gain cannot pass for the
whole-recording gain.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import soundfile as sf

from benchmark import atomic_write
from benchmark.canonical_worker import SAMPLE_FORMAT
from benchmark.degradation import DegradationParams, apply_degradation
from benchmark.protocol import load_protocol
from scripts import make_demo_samples as mds
from scripts.make_demo_samples import ClipOutput, ClipSpec, DemoSampleError

PROTOCOL = load_protocol()
RANGES = PROTOCOL.degradation.ranges
HEADROOM = RANGES.headroom
SR = mds.SAMPLE_RATE
RECORDING_SECONDS = 3.0
CLIP_START_S = 0.5
CLIP_DURATION_S = 0.5
CLIP_SAMPLES = round(CLIP_DURATION_S * SR)
SPIKE = 0.8
TONE = 0.2
UVR_SHORTFALL = 130
PCM16_LSB = 1 / mds.PCM16_SCALE

Transform = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def _identity(samples: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return samples


def scaled(factor: float) -> Transform:
    """A transform that multiplies the input by `factor`."""
    return lambda samples: samples * factor


def drop_tail(count: int) -> Transform:
    """A transform that returns the input without its last `count` samples."""
    return lambda samples: samples[: samples.size - count]


class FakeEngine:
    """Writes its input through `transform` and records the samples it was given."""

    def __init__(
        self,
        output_dir: Path,
        transform: Transform = _identity,
        *,
        rate: int = SR,
        subtype: str = "PCM_16",
    ) -> None:
        """Set where the output goes and how it is written."""
        self.output_dir = output_dir
        self.transform = transform
        self.rate = rate
        self.subtype = subtype
        self.inputs: list[npt.NDArray[np.float32]] = []

    def denoise(self, input_path: Path) -> Path:
        """Record the input, then write it back out through the transform."""
        data, _ = sf.read(str(input_path), dtype="float32")
        self.inputs.append(data.copy())
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / f"{input_path.stem}_fake.wav"
        sf.write(
            str(output),
            self.transform(data.astype(np.float64)),
            self.rate,
            subtype=self.subtype,
        )
        return output


def write_recording(path: Path) -> Path:
    """Write a 3 s tone at 0.2 with one 0.8 sample at the start, outside the test clip."""
    t = np.arange(int(RECORDING_SECONDS * SR)) / SR
    samples = TONE * np.sin(2 * np.pi * 440 * t)
    samples[0] = SPIKE
    path.write_bytes(samples.astype(SAMPLE_FORMAT).tobytes())
    return path


def make_spec(
    path: Path,
    *,
    slug: str = "bach",
    start_s: float = CLIP_START_S,
    duration_s: float = CLIP_DURATION_S,
    seed: int = 7,
) -> ClipSpec:
    """A clip spec with fixed metadata and the given recording, slug, window, and seed."""
    return ClipSpec(
        canonical_path=path,
        identifier="fixture",
        performer="Test Performer",
        slug=slug,
        start_s=start_s,
        duration_s=duration_s,
        seed=seed,
    )


def six_files(slug: str) -> list[str]:
    """The six relative paths one clip writes."""
    return [
        f"{directory}/{slug}_{kind}.wav"
        for directory in ("opgan", "uvr")
        for kind in ("clean", "noisy", "restored")
    ]


def run(
    tmp_path: Path,
    specs: list[ClipSpec],
    opgan: FakeEngine,
    uvr: FakeEngine,
) -> Path:
    """Generate the clips into tmp_path/samples and return the manifest path."""
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    return mds.generate(
        specs,
        opgan,
        uvr,
        RANGES,
        PROTOCOL.protocol_version,
        workdir,
        tmp_path / "samples",
    )


def engines(
    tmp_path: Path, opgan_transform: Transform, uvr_transform: Transform
) -> tuple[FakeEngine, FakeEngine]:
    """Fake OpGAN and UVR engines writing under tmp_path/engines."""
    return (
        FakeEngine(tmp_path / "engines" / "opgan", opgan_transform),
        FakeEngine(tmp_path / "engines" / "uvr", uvr_transform),
    )


def assert_nothing_written(tmp_path: Path) -> None:
    """A refused clip leaves no samples directory and no manifest."""
    assert not (tmp_path / "samples").exists()


def test_uvr_shortfall_sets_common_length_for_every_stream(tmp_path: Path) -> None:
    """All four streams are trimmed to UVR's shorter length, and the counts are recorded."""
    recording = write_recording(tmp_path / "rec.f8")
    opgan, uvr = engines(tmp_path, scaled(0.5), drop_tail(UVR_SHORTFALL))

    manifest = run(tmp_path, [make_spec(recording)], opgan, uvr)

    expected = CLIP_SAMPLES - UVR_SHORTFALL
    for relative in six_files("bach"):
        assert sf.info(tmp_path / "samples" / relative).frames == expected, relative
    record = json.loads(manifest.read_text())["clips"][0]
    assert record["samples_written"] == expected
    assert record["samples_trimmed"] == {
        "clean": UVR_SHORTFALL,
        "noisy": UVR_SHORTFALL,
        "opgan": UVR_SHORTFALL,
        "uvr": 0,
    }


def test_degradation_is_deterministic_for_a_seed_and_differs_for_another() -> None:
    """The same seed gives the same degraded clip, and another seed gives a different one."""
    t = np.arange(CLIP_SAMPLES) / SR
    frame = TONE * np.sin(2 * np.pi * 440 * t)

    first = mds.degrade_clip(frame, SPIKE, 7, RANGES)
    again = mds.degrade_clip(frame, SPIKE, 7, RANGES)
    other = mds.degrade_clip(frame, SPIKE, 8, RANGES)

    assert np.array_equal(first.degraded, again.degraded)
    assert not np.array_equal(first.degraded, other.degraded)
    assert first.params == mds.DEMO_PROFILE
    assert other.params == mds.DEMO_PROFILE


def test_clean_file_is_clip_times_gain_from_the_whole_recording_peak(tmp_path: Path) -> None:
    """The clean reference is the clip times headroom over the whole recording's peak."""
    recording = write_recording(tmp_path / "rec.f8")
    opgan, uvr = engines(tmp_path, _identity, _identity)

    manifest = run(tmp_path, [make_spec(recording)], opgan, uvr)

    whole = np.frombuffer(recording.read_bytes(), dtype=SAMPLE_FORMAT).astype(np.float64)
    start = round(CLIP_START_S * SR)
    frame = whole[start : start + CLIP_SAMPLES]
    expected_gain = HEADROOM / float(np.max(np.abs(whole)))
    assert expected_gain != pytest.approx(1.0)
    record = json.loads(manifest.read_text())["clips"][0]
    assert record["gain"] == pytest.approx(expected_gain, rel=1e-12)
    clean, _ = sf.read(tmp_path / "samples" / "opgan" / "bach_clean.wav", dtype="float64")
    np.testing.assert_allclose(clean, frame * expected_gain, rtol=0, atol=2 * PCM16_LSB)


def test_noisy_file_is_exactly_what_both_engines_received(tmp_path: Path) -> None:
    """The shipped noisy file holds the samples the engines read, bit for bit."""
    recording = write_recording(tmp_path / "rec.f8")
    opgan, uvr = engines(tmp_path, _identity, _identity)

    run(tmp_path, [make_spec(recording)], opgan, uvr)

    for directory, engine in (("opgan", opgan), ("uvr", uvr)):
        shipped, _ = sf.read(tmp_path / "samples" / directory / "bach_noisy.wav", dtype="float32")
        assert np.array_equal(engine.inputs[0], shipped), directory


@pytest.mark.parametrize(
    ("start_s", "duration_s"),
    [(-CLIP_DURATION_S, CLIP_DURATION_S), (2.8, CLIP_DURATION_S)],
    ids=["starts_before_recording", "runs_past_end"],
)
def test_refuses_clip_outside_its_recording(
    tmp_path: Path, start_s: float, duration_s: float
) -> None:
    """A clip outside the recording is refused before any engine runs or any file is written."""
    recording = write_recording(tmp_path / "rec.f8")
    opgan, uvr = engines(tmp_path, _identity, _identity)
    spec = make_spec(recording, start_s=start_s, duration_s=duration_s)

    with pytest.raises(DemoSampleError):
        run(tmp_path, [spec], opgan, uvr)

    assert opgan.inputs == []
    assert_nothing_written(tmp_path)


def test_refuses_invalid_clip_fields(tmp_path: Path) -> None:
    """Each bad slug, a negative seed, a zero duration, and a fractional start are refused."""
    recording = write_recording(tmp_path / "rec.f8")
    bad_specs = [
        make_spec(recording, slug=slug) for slug in ("Bach", "bach-1", "../x", "bach\n", "")
    ]
    bad_specs.append(make_spec(recording, seed=-1))
    bad_specs.append(make_spec(recording, duration_s=0.0))
    bad_specs.append(make_spec(recording, start_s=CLIP_START_S + 1e-7))
    for index, spec in enumerate(bad_specs):
        opgan, uvr = engines(tmp_path / str(index), _identity, _identity)
        with pytest.raises(DemoSampleError):
            run(tmp_path / str(index), [spec], opgan, uvr)
        assert_nothing_written(tmp_path / str(index))


def test_refuses_engine_output_at_the_wrong_rate(tmp_path: Path) -> None:
    """An engine output that is not 16 kHz is refused, and nothing is written."""
    recording = write_recording(tmp_path / "rec.f8")
    opgan, _ = engines(tmp_path, _identity, _identity)
    uvr = FakeEngine(tmp_path / "engines" / "uvr", rate=8_000)

    with pytest.raises(DemoSampleError):
        run(tmp_path, [make_spec(recording)], opgan, uvr)

    assert_nothing_written(tmp_path)


def test_refuses_a_sample_above_one_rather_than_clipping_it(tmp_path: Path) -> None:
    """An engine output sample of 1.2 is refused, even though a clipped file would look valid."""
    recording = write_recording(tmp_path / "rec.f8")
    opgan = FakeEngine(
        tmp_path / "engines" / "opgan",
        lambda s: np.full_like(s, 1.2),
        subtype="FLOAT",
    )
    _, uvr = engines(tmp_path, _identity, _identity)

    with pytest.raises(DemoSampleError):
        run(tmp_path, [make_spec(recording)], opgan, uvr)

    assert_nothing_written(tmp_path)


def test_refuses_a_non_finite_sample(tmp_path: Path) -> None:
    """A NaN in an engine output is refused, and nothing is written."""
    recording = write_recording(tmp_path / "rec.f8")
    opgan, _ = engines(tmp_path, _identity, _identity)
    uvr = FakeEngine(
        tmp_path / "engines" / "uvr",
        lambda s: np.full_like(s, np.nan),
        subtype="FLOAT",
    )

    with pytest.raises(DemoSampleError):
        run(tmp_path, [make_spec(recording)], opgan, uvr)

    assert_nothing_written(tmp_path)


def test_refuses_an_empty_engine_output(tmp_path: Path) -> None:
    """An engine that returns no samples is refused, because nothing could be written."""
    recording = write_recording(tmp_path / "rec.f8")
    opgan, uvr = engines(tmp_path, _identity, drop_tail(CLIP_SAMPLES))

    with pytest.raises(DemoSampleError):
        run(tmp_path, [make_spec(recording)], opgan, uvr)

    assert_nothing_written(tmp_path)


def test_manifest_digests_equal_the_files_on_disk(tmp_path: Path) -> None:
    """Every file the manifest lists has the digest of its bytes on disk, and no file is missing."""
    recording = write_recording(tmp_path / "rec.f8")
    opgan, uvr = engines(tmp_path, scaled(0.5), drop_tail(UVR_SHORTFALL))

    manifest = run(tmp_path, [make_spec(recording)], opgan, uvr)

    body = json.loads(manifest.read_text())
    record = body["clips"][0]
    listed = {entry["path"]: entry["sha256"] for entry in record["files"]}
    assert sorted(listed) == sorted(six_files("bach"))
    for path, digest in listed.items():
        on_disk = (tmp_path / "samples" / path).read_bytes()
        assert hashlib.sha256(on_disk).hexdigest() == digest, path
    assert body["protocol_version"] == PROTOCOL.protocol_version
    assert body["noise_policy"] == mds.NOISE_POLICY


def test_failed_write_leaves_the_old_file_and_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that fails before the replace leaves the old file and no temporary file."""
    recording = write_recording(tmp_path / "rec.f8")
    target_dir = tmp_path / "samples" / "opgan"
    target_dir.mkdir(parents=True)
    target = target_dir / "bach_clean.wav"
    target.write_bytes(b"old")
    clip = ClipOutput(
        spec=make_spec(recording),
        canonical_sha256="0" * 64,
        gain=1.0,
        samples_written=0,
        samples_trimmed={},
        files={"opgan/bach_clean.wav": b"new"},
    )

    def failing_fsync(_fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(atomic_write.os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="disk full"):
        mds.write_outputs(clip.files, tmp_path / "samples")

    assert target.read_bytes() == b"old"
    assert sorted(p.name for p in target_dir.iterdir()) == ["bach_clean.wav"]


def test_load_clip_specs_reads_every_field(tmp_path: Path) -> None:
    """A valid clips file loads into specs with every field as written."""
    clips_file = tmp_path / "clips.json"
    clips_file.write_text(
        json.dumps(
            [
                {
                    "canonical_path": str(tmp_path / "rec.f8"),
                    "identifier": "fixture",
                    "performer": "Test Performer",
                    "slug": "bach_1",
                    "start_s": 1.5,
                    "duration_s": 6,
                    "seed": 3,
                }
            ]
        )
    )

    (spec,) = mds.load_clip_specs(clips_file)

    assert spec == ClipSpec(
        canonical_path=tmp_path / "rec.f8",
        identifier="fixture",
        performer="Test Performer",
        slug="bach_1",
        start_s=1.5,
        duration_s=6.0,
        seed=3,
    )


def test_load_clip_specs_refuses_a_missing_key(tmp_path: Path) -> None:
    """A clip without its seed is refused with the expected key set named."""
    clips_file = tmp_path / "clips.json"
    clips_file.write_text(
        json.dumps(
            [
                {
                    "canonical_path": "rec.f8",
                    "identifier": "fixture",
                    "performer": "Test Performer",
                    "slug": "bach",
                    "start_s": 0.0,
                    "duration_s": 6.0,
                }
            ]
        )
    )

    with pytest.raises(DemoSampleError, match="exactly the keys"):
        mds.load_clip_specs(clips_file)


def test_read_recording_refuses_a_partial_float64_value(tmp_path: Path) -> None:
    """A canonical file that is not a whole number of float64 values is refused."""
    recording = tmp_path / "rec.f8"
    recording.write_bytes(b"\x00" * 12)

    with pytest.raises(DemoSampleError, match="whole number of float64"):
        mds.read_recording(recording)


def _write_clips_file(tmp_path: Path, recording: Path) -> Path:
    """A clips file with one short clip from the recording, for the command-line tests."""
    clips = tmp_path / "clips.json"
    clips.write_text(
        json.dumps(
            [
                {
                    "canonical_path": str(recording),
                    "identifier": "fixture",
                    "performer": "Test Performer",
                    "slug": "bach",
                    "start_s": 0.0,
                    "duration_s": CLIP_DURATION_S,
                    "seed": 7,
                }
            ]
        )
    )
    return clips


def _main_args(tmp_path: Path, clips: Path, out_dir: Path) -> list[str]:
    """Command-line arguments that point the generator at the given clips and output."""
    return [
        "--clips",
        str(clips),
        "--uvr-model-dir",
        str(tmp_path),
        "--opgan-checkpoint",
        str(tmp_path / "final.pt"),
        "--out-dir",
        str(out_dir),
    ]


def test_main_writes_every_file_and_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command line builds the engines, writes six WAVs and the manifest, and returns 0."""
    recording = write_recording(tmp_path / "rec.f8")
    clips = _write_clips_file(tmp_path, recording)
    built: list[Path] = []

    def fake_build(
        protocol: object, run_dir: Path, opgan_checkpoint: Path, uvr_model_dir: Path
    ) -> tuple[FakeEngine, FakeEngine]:
        built.append(run_dir)
        return FakeEngine(run_dir / "opgan"), FakeEngine(run_dir / "uvr")

    monkeypatch.setattr(mds, "build_production_systems", fake_build)
    monkeypatch.setattr(mds, "uvr_file_failures", lambda protocol, uvr_model_dir: [])
    out_dir = tmp_path / "samples"

    code = mds.main(_main_args(tmp_path, clips, out_dir))

    assert code == 0
    assert len(built) == 1
    assert (out_dir / "MANIFEST.json").is_file()
    for relative in six_files("bach"):
        assert (out_dir / relative).is_file(), relative


def test_main_refuses_a_uvr_model_dir_before_building_engines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A UVR model directory that fails its digests returns 1 before any engine is built."""
    recording = write_recording(tmp_path / "rec.f8")
    clips = _write_clips_file(tmp_path, recording)
    built: list[Path] = []

    def fake_build(
        protocol: object, run_dir: Path, opgan_checkpoint: Path, uvr_model_dir: Path
    ) -> tuple[FakeEngine, FakeEngine]:
        built.append(run_dir)
        return FakeEngine(run_dir / "opgan"), FakeEngine(run_dir / "uvr")

    monkeypatch.setattr(mds, "build_production_systems", fake_build)
    monkeypatch.setattr(
        mds, "uvr_file_failures", lambda protocol, uvr_model_dir: ["UVR-DeNoise.pth is missing"]
    )
    out_dir = tmp_path / "samples"

    code = mds.main(_main_args(tmp_path, clips, out_dir))

    assert code == 1
    assert built == []
    assert not out_dir.exists()


def test_demo_profile_lies_within_the_protocol_ranges() -> None:
    """The declared profile passes the range check against the protocol's training ranges."""
    mds.validate_profile(mds.DEMO_PROFILE, RANGES)


def test_profile_outside_the_ranges_is_refused() -> None:
    """A value outside its range, a colour not in training, or a hum frequency not in training
    is refused by the check."""
    bad_profiles = [
        replace(mds.DEMO_PROFILE, hiss_snr_db=3.0),
        replace(mds.DEMO_PROFILE, hiss_color="white"),
        replace(mds.DEMO_PROFILE, hum_freq=55.0),
        replace(mds.DEMO_PROFILE, saturation_drive=0.9),
    ]
    for profile in bad_profiles:
        with pytest.raises(DemoSampleError):
            mds.validate_profile(profile, RANGES)


def test_out_of_range_profile_refuses_generation_and_writes_nothing(tmp_path: Path) -> None:
    """A profile outside the ranges stops generation before any engine runs or file is written."""
    recording = write_recording(tmp_path / "rec.f8")
    opgan, uvr = engines(tmp_path, _identity, _identity)
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True)

    with pytest.raises(DemoSampleError):
        mds.generate(
            [make_spec(recording)],
            opgan,  # type: ignore[arg-type]
            uvr,  # type: ignore[arg-type]
            RANGES,
            PROTOCOL.protocol_version,
            workdir,
            tmp_path / "samples",
            replace(mds.DEMO_PROFILE, hiss_snr_db=3.0),
        )

    assert opgan.inputs == []
    assert_nothing_written(tmp_path)


def test_every_clip_is_degraded_with_the_same_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two clips with different seeds both pass the declared profile to the degradation."""
    recording = write_recording(tmp_path / "rec.f8")
    used: list[DegradationParams] = []
    real_apply = mds.apply_degradation

    def spy(signal: object, sample_rate: int, params: DegradationParams, rng: object) -> object:
        used.append(params)
        return real_apply(signal, sample_rate, params, rng)  # type: ignore[arg-type]

    monkeypatch.setattr(mds, "apply_degradation", spy)
    opgan, uvr = engines(tmp_path, _identity, _identity)
    specs = [make_spec(recording, slug="bach", seed=7), make_spec(recording, slug="waltz", seed=8)]

    run(tmp_path, specs, opgan, uvr)

    assert used == [mds.DEMO_PROFILE, mds.DEMO_PROFILE]


def test_degraded_clip_is_apply_degradation_with_the_demo_profile() -> None:
    """The noise comes from apply_degradation with the fixed profile and a generator seeded from
    the clip seed, so a drawn profile would fail this test."""
    t = np.arange(CLIP_SAMPLES) / SR
    frame = TONE * np.sin(2 * np.pi * 440 * t)
    gain = HEADROOM / SPIKE
    scaled = (frame * gain).astype(np.float32)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(7)))

    expected = apply_degradation(scaled, SR, mds.DEMO_PROFILE, rng)
    actual = mds.degrade_clip(frame, SPIKE, 7, RANGES)

    assert np.array_equal(actual.degraded, expected)
    assert actual.gain == pytest.approx(gain)


def test_manifest_records_the_profile_and_noise_policy(tmp_path: Path) -> None:
    """The manifest holds the exact profile values and the noise policy text."""
    recording = write_recording(tmp_path / "rec.f8")
    opgan, uvr = engines(tmp_path, _identity, _identity)

    manifest = run(tmp_path, [make_spec(recording)], opgan, uvr)

    body = json.loads(manifest.read_text())
    assert body["noise_profile"] == asdict(mds.DEMO_PROFILE)
    assert body["noise_policy"] == (
        "fixed demo profile within the training ranges; per-clip seed for noise samples only"
    )


def test_every_16_bit_value_survives_the_read_and_the_requantisation() -> None:
    """Over all 65,536 values, libsndfile's read and the writer's scale return every value, while
    the former scale of 32,767 changed 32,767 of them by one least significant bit each."""
    values = np.arange(-32768, 32768, dtype=np.int16)
    buffer = io.BytesIO()
    sf.write(buffer, values, SR, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    read, _ = sf.read(buffer, dtype="float32")
    assert np.array_equal(read, (values / 2**15).astype(np.float32))
    requantised = mds.to_pcm16(read.astype(np.float64), "every value")
    assert int(np.count_nonzero(requantised != values)) == 0
    former = np.round(read.astype(np.float64) * 32767).astype(np.int16)
    assert int(np.count_nonzero(former != values)) == 32767


def test_publish_replaces_the_run_files_and_leaves_siblings_alone(tmp_path: Path) -> None:
    """The run's files are replaced, an untracked sibling is untouched, and no staging remains."""
    out = tmp_path / "samples"
    (out / "opgan").mkdir(parents=True)
    (out / "opgan" / "bach_clean.wav").write_bytes(b"old")
    (out / "opgan_primrose").mkdir()
    (out / "opgan_primrose" / "keep.wav").write_bytes(b"kept")
    mds.write_outputs({"opgan/bach_clean.wav": b"new", "MANIFEST.json": b"manifest"}, out)
    assert (out / "opgan" / "bach_clean.wav").read_bytes() == b"new"
    assert (out / "MANIFEST.json").read_bytes() == b"manifest"
    assert (out / "opgan_primrose" / "keep.wav").read_bytes() == b"kept"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["samples"]


def test_a_run_directory_holding_another_file_is_refused_and_nothing_changes(
    tmp_path: Path,
) -> None:
    """A file the run would not write, inside a directory it writes to, refuses the whole run."""
    out = tmp_path / "samples"
    (out / "opgan").mkdir(parents=True)
    (out / "opgan" / "bach_clean.wav").write_bytes(b"old")
    (out / "opgan" / "extra.wav").write_bytes(b"extra")
    with pytest.raises(DemoSampleError, match="holds files the run does not write"):
        mds.write_outputs({"opgan/bach_clean.wav": b"new"}, out)
    assert (out / "opgan" / "bach_clean.wav").read_bytes() == b"old"
    assert sorted(p.name for p in (out / "opgan").iterdir()) == ["bach_clean.wav", "extra.wav"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["samples"]


def test_a_failed_move_restores_every_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A move that fails after one file was replaced puts back every previous file."""
    out = tmp_path / "samples"
    (out / "opgan").mkdir(parents=True)
    (out / "opgan" / "a.wav").write_bytes(b"old-a")
    (out / "opgan" / "b.wav").write_bytes(b"old-b")
    real_place = mds._place
    moves: list[Path] = []

    def fail_on_second_move(source: Path, target: Path) -> None:
        moves.append(target)
        if len(moves) == 2:
            raise OSError("simulated move failure")
        real_place(source, target)

    monkeypatch.setattr(mds, "_place", fail_on_second_move)
    with pytest.raises(OSError, match="simulated move failure"):
        mds.write_outputs({"opgan/a.wav": b"new-a", "opgan/b.wav": b"new-b"}, out)
    assert (out / "opgan" / "a.wav").read_bytes() == b"old-a"
    assert (out / "opgan" / "b.wav").read_bytes() == b"old-b"
    assert sorted(p.name for p in (out / "opgan").iterdir()) == ["a.wav", "b.wav"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["samples"]


def test_the_weights_only_override_refuses_the_run_before_anything_is_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD set, the refusal is printed, the command returns 1,
    and no directory is created. The clips file is absent, so a run that skipped the check would
    fail on it with a different message."""
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    out = tmp_path / "samples"
    assert mds.main(_main_args(tmp_path, tmp_path / "clips.json", out)) == 1
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD is set" in capsys.readouterr().err
    assert not out.exists()


def test_main_reports_a_file_system_error_and_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An OSError during the run is reported on stderr and gives exit status 1, not a traceback."""

    def fail(_path: Path) -> list[ClipSpec]:
        raise OSError("disk full")

    monkeypatch.setattr(mds, "load_clip_specs", fail)
    assert mds.main(_main_args(tmp_path, tmp_path / "clips.json", tmp_path / "samples")) == 1
    assert "disk full" in capsys.readouterr().err


def _write_engine_wav(path: Path, samples: np.ndarray, rate: int) -> Path:
    """Write float samples to *path* as a WAV at *rate*, for an engine output."""
    sf.write(str(path), samples, rate, subtype="FLOAT")
    return path


def test_read_engine_output_returns_mono_samples_at_the_canonical_rate(tmp_path: Path) -> None:
    """A mono output at the canonical rate is read back exactly."""
    samples = np.linspace(-0.5, 0.5, 1600)
    path = _write_engine_wav(tmp_path / "out.wav", samples, mds.SAMPLE_RATE)

    result = mds.read_engine_output(path, "clip OpGAN")

    assert result.dtype == np.float64
    assert np.allclose(result, samples, atol=1e-6)


def test_read_engine_output_refuses_wrong_rate_and_stereo(tmp_path: Path) -> None:
    """A wrong rate and a stereo output each raise DemoSampleError naming the clip."""
    wrong_rate = _write_engine_wav(tmp_path / "rate.wav", np.zeros(100), 8_000)
    stereo = tmp_path / "stereo.wav"
    sf.write(str(stereo), np.zeros((100, 2)), mds.SAMPLE_RATE, subtype="FLOAT")

    for path in (wrong_rate, stereo):
        with pytest.raises(DemoSampleError, match="clip UVR is not 16000 Hz mono"):
            mds.read_engine_output(path, "clip UVR")


def test_read_engine_output_refuses_an_unreadable_file(tmp_path: Path) -> None:
    """A file that is not audio raises DemoSampleError rather than a soundfile error."""
    path = tmp_path / "not_audio.wav"
    path.write_bytes(b"not a wav file at all")

    with pytest.raises(DemoSampleError, match="clip UVR cannot be read"):
        mds.read_engine_output(path, "clip UVR")
