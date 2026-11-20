"""Temporary measurement: do the golden fixture digests measured on macOS hold on Linux?

This file exists only on a measurement branch whose pull request is closed unmerged.
"""

from __future__ import annotations

import hashlib
import platform

import soundfile as sf
import soxr

from benchmark.canonicalisation import CanonicalWaveform
from benchmark.golden_fixture import FixtureSpec, canonical_fixture, fixture_integers
from benchmark.protocol import load_protocol

CONFIG = load_protocol().canonicalisation
ANALYTIC = FixtureSpec(seed=20261120, sample_rate=16000, channels=1, frames=64777, subtype="PCM_16")
RESAMPLER = FixtureSpec(
    seed=20261121, sample_rate=44100, channels=2, frames=132423, subtype="PCM_24"
)
MACOS_ANALYTIC = (
    "a092fd3b42e4c2142ce999ad11e7c6b8e05ff673408922d924c5367aec1d23b1"  # pragma: allowlist secret
)
MACOS_RESAMPLER = (
    "a82cd013caa67872a1eccc17c8b20a55afa785044c6952731398e739da7b3df5"  # pragma: allowlist secret
)


def environment() -> str:
    """The platform and native library versions, for the failure message."""
    return (
        f"{platform.platform()} {platform.machine()} soxr {soxr.__version__} "
        f"libsoxr {getattr(soxr, '__libsoxr_version__', '?')} "
        f"libsndfile {sf.__libsndfile_version__}"
    )


def test_analytic_fixture_matches_its_integer_oracle() -> None:
    """The analytic digest, through the pipeline and straight from the integers."""
    waveform = canonical_fixture(ANALYTIC, CONFIG)
    assert isinstance(waveform, CanonicalWaveform), waveform
    oracle = (fixture_integers(ANALYTIC)[:, 0] / 32768.0).astype("<f8")

    assert waveform.sha256 == hashlib.sha256(oracle.tobytes()).hexdigest(), environment()
    assert waveform.sha256 == MACOS_ANALYTIC, (waveform.sha256, environment())


def test_resampler_fixture_matches_macos() -> None:
    """The resampler digest measured on macOS arm64."""
    waveform = canonical_fixture(RESAMPLER, CONFIG)
    assert isinstance(waveform, CanonicalWaveform), waveform

    assert waveform.sha256 == MACOS_RESAMPLER, (waveform.sha256, environment())
