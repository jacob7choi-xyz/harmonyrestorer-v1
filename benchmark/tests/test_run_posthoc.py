"""Tests for the post-hoc command: what it writes, what it refuses, and the consistency check."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from posthoc_fixtures import build_bundle, copy_bundle, update_results
from report_fixtures import CHAMBER, SOLO, Bundle, pinned_universes

from benchmark import atomic_write
from benchmark import run_posthoc as run_posthoc_module
from benchmark.posthoc import BAND_METRIC, LABEL, PosthocError
from benchmark.protocol import Protocol, load_protocol
from benchmark.report import RESULTS_NAME, ReportError
from benchmark.run_posthoc import (
    INPUTS_DIR,
    JSON_NAME,
    MARKDOWN_NAME,
    OUTPUTS_DIR,
    _check_band_edge,
    main,
    run_posthoc,
)
from benchmark.run_systems import INPUTS_DIR as STAGE_INPUTS_DIR
from benchmark.run_systems import OUTPUTS_DIR as STAGE_OUTPUTS_DIR
from benchmark.scoring import BASELINE, OPGAN, UVR

FORBIDDEN_WORDS = ("significan", "p-value", "p value", "hypothesis", "winner", "wins ", "beats")
ANALYSIS_HEADINGS = 3


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol, loaded once for the module."""
    return load_protocol()


@pytest.fixture(scope="module")
def base(protocol: Protocol, tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    """A walk and systems directory with real solo output files, shared by the read-only tests."""
    return build_bundle(tmp_path_factory.mktemp("posthoc-base"), protocol)


def run_into(bundle: Bundle, output: Path) -> None:
    """Run the post-hoc command on a bundle, with the fixture's pinned universes in place."""
    with pinned_universes():
        run_posthoc(bundle.walk, bundle.systems, output)


def run_main(bundle: Bundle, output: Path) -> int:
    """The command line on a bundle, with the fixture's pinned universes in place."""
    with pinned_universes():
        return main(
            [
                "--walk-dir",
                str(bundle.walk),
                "--systems-dir",
                str(bundle.systems),
                "--output-dir",
                str(output),
            ]
        )


def rows_for(bundle: Bundle, stratum: str, system: str) -> list[dict[str, Any]]:
    """The results rows of one stratum and system, in file order."""
    lines = (bundle.systems / RESULTS_NAME).read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    return [row for row in rows if row["stratum"] == stratum and row["system"] == system]


def read_document(output: Path) -> dict[str, Any]:
    """The JSON document a run wrote."""
    return json.loads((output / JSON_NAME).read_text())


@pytest.fixture(scope="module")
def silent_base(protocol: Protocol, tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    """A run whose first OpGAN frame is all zeros and whose second holds one LSB."""
    return build_bundle(
        tmp_path_factory.mktemp("posthoc-silent"), protocol, zero_opgan=1, lsb_opgan=1
    )


@pytest.fixture
def copied(base: Bundle, tmp_path: Path) -> Bundle:
    """A private copy of the base run, which a test may change."""
    return copy_bundle(base, tmp_path / "copy")


@pytest.fixture
def fresh_output(tmp_path: Path) -> Iterator[Path]:
    """A directory path that does not exist yet, for the run to create."""
    yield tmp_path / "posthoc"


def test_the_comparison_stratum_gets_the_three_analyses_and_the_withheld_one_nothing(
    base: Bundle, fresh_output: Path
) -> None:
    """The solo stratum is a comparison, so it has all three analyses; the chamber one has none."""
    run_into(base, fresh_output)
    strata = read_document(fresh_output)["strata"]
    solo = strata[SOLO]
    assert solo["outcome"] == "comparison"
    assert set(solo["analysis_1"]) == {
        "reconstruction_snr",
        "si_snr",
        "log_spectral_distance",
        BAND_METRIC,
    }
    assert set(solo["analysis_2"]["sources"]) == {OPGAN, UVR, BASELINE}
    assert solo["analysis_3"][OPGAN]["eligible_frames"] == len(rows_for(base, SOLO, OPGAN))
    assert strata[CHAMBER] == {"outcome": "withheld"}


def test_a_tampered_engine_output_is_refused_and_no_directory_is_written(
    copied: Bundle, fresh_output: Path
) -> None:
    """Halving one UVR output changes its full-band distance, so the check aborts with no output."""
    target = sorted((copied.systems / OUTPUTS_DIR / UVR).glob("*.wav"))[0]
    samples, rate = sf.read(str(target), dtype="float64", always_2d=False)
    sf.write(str(target), samples * 0.5, rate, subtype="FLOAT")
    with pytest.raises(PosthocError, match="differs from the recorded value"):
        run_into(copied, fresh_output)
    assert not fresh_output.exists()


def test_a_tampered_degraded_input_is_refused_and_no_directory_is_written(
    copied: Bundle, fresh_output: Path
) -> None:
    """The baseline reads the saved input, so a changed input fails the same check."""
    target = sorted((copied.systems / INPUTS_DIR).glob("*.wav"))[0]
    samples, rate = sf.read(str(target), dtype="float64", always_2d=False)
    sf.write(str(target), samples * 0.5, rate, subtype="FLOAT")
    with pytest.raises(PosthocError, match="differs from the recorded value"):
        run_into(copied, fresh_output)
    assert not fresh_output.exists()


def test_a_missing_engine_output_is_refused_and_no_directory_is_written(
    copied: Bundle, fresh_output: Path
) -> None:
    """A recorded frame whose output file is gone cannot be checked, so the run stops."""
    sorted((copied.systems / OUTPUTS_DIR / OPGAN).glob("*.wav"))[0].unlink()
    with pytest.raises(PosthocError, match="cannot read"):
        run_into(copied, fresh_output)
    assert not fresh_output.exists()


def test_a_silent_output_is_counted_and_a_one_lsb_output_is_not(
    silent_base: Bundle, fresh_output: Path
) -> None:
    """The all-zero output counts as silent. The output holding one LSB does not, so exact zero holds."""
    run_into(silent_base, fresh_output)
    counts = read_document(fresh_output)["strata"][SOLO]["analysis_3"]
    assert counts[OPGAN]["silent"] == 1
    assert counts[UVR]["silent"] == 0
    assert counts[BASELINE]["silent"] == 0


def test_capped_frames_are_counted_from_the_recorded_flags(
    copied: Bundle, fresh_output: Path
) -> None:
    """Two recorded OpGAN frames flagged as capped are counted as two, and nothing else moves."""
    flagged: list[tuple[str, int]] = []

    def flag_two(row: dict[str, Any]) -> None:
        if row["stratum"] == SOLO and row["system"] == OPGAN and len(flagged) < 2:
            row["results"]["reconstruction_snr"]["capped"] = True
            flagged.append((row["canonical_sha256"], row["frame_index"]))

    update_results(copied, flag_two)
    run_into(copied, fresh_output)
    counts = read_document(fresh_output)["strata"][SOLO]["analysis_3"][OPGAN]
    assert counts["capped_reconstruction_snr"] == 2
    assert counts["capped_si_snr"] == 0


def test_an_existing_output_directory_is_refused_and_left_untouched(
    base: Bundle, tmp_path: Path
) -> None:
    """A published result is never overwritten, so an existing directory is refused and kept."""
    output = tmp_path / "existing"
    output.mkdir()
    (output / "sentinel.txt").write_text("keep")
    with pytest.raises(ReportError, match="already exists"):
        run_into(base, output)
    assert sorted(p.name for p in output.iterdir()) == ["sentinel.txt"]


def test_the_command_writes_both_files_and_returns_zero(base: Bundle, fresh_output: Path) -> None:
    """A clean run returns zero and leaves the JSON and the Markdown in the new directory."""
    assert run_main(base, fresh_output) == 0
    assert (fresh_output / JSON_NAME).is_file()
    assert (fresh_output / MARKDOWN_NAME).is_file()


def test_the_command_returns_one_and_writes_nothing_for_a_tampered_run(
    copied: Bundle, fresh_output: Path
) -> None:
    """The exit code is one for a refusal, and no directory is created."""
    target = sorted((copied.systems / OUTPUTS_DIR / OPGAN).glob("*.wav"))[0]
    samples, rate = sf.read(str(target), dtype="float64", always_2d=False)
    sf.write(str(target), samples * 0.5, rate, subtype="FLOAT")
    assert run_main(copied, fresh_output) == 1
    assert not fresh_output.exists()


def test_no_significance_or_winner_language_is_written(base: Bundle, fresh_output: Path) -> None:
    """Neither rendering uses significance language or names a winner."""
    run_into(base, fresh_output)
    for name in (JSON_NAME, MARKDOWN_NAME):
        text = (fresh_output / name).read_text().lower()
        for word in FORBIDDEN_WORDS:
            assert word not in text, f"{word!r} found in {name}"


def test_every_analysis_is_labelled_post_hoc(base: Bundle, fresh_output: Path) -> None:
    """The JSON carries the label on each analysis, and the Markdown on each analysis section."""
    run_into(base, fresh_output)
    document = read_document(fresh_output)
    solo = document["strata"][SOLO]
    assert document["label"] == LABEL
    assert solo["label"] == LABEL
    assert solo["analysis_2"]["label"] == LABEL
    assert all(entry["label"] == LABEL for entry in solo["analysis_1"].values())
    markdown = (fresh_output / MARKDOWN_NAME).read_text()
    assert markdown.count(f"All numbers in this section are {LABEL}.") == ANALYSIS_HEADINGS
    assert f"Every number in this document is {LABEL}." in markdown


def test_the_run_digests_name_the_systems_results_the_numbers_came_from(
    base: Bundle, fresh_output: Path
) -> None:
    """The systems results digest in the output is the SHA-256 of the results file's bytes."""
    run_into(base, fresh_output)
    expected = hashlib.sha256((base.systems / RESULTS_NAME).read_bytes()).hexdigest()
    assert read_document(fresh_output)["run_digests"]["systems_results_sha256"] == expected


def test_the_directory_names_match_the_systems_stage() -> None:
    """The post-hoc reader names the same two directories that the systems stage writes."""
    assert (INPUTS_DIR, OUTPUTS_DIR) == (STAGE_INPUTS_DIR, STAGE_OUTPUTS_DIR)


def test_the_band_metric_is_written_under_its_own_name(base: Bundle, fresh_output: Path) -> None:
    """Analysis 1 names the band metric, so the band is never read as a primary metric."""
    run_into(base, fresh_output)
    analysis_one = read_document(fresh_output)["strata"][SOLO]["analysis_1"]
    assert BAND_METRIC in analysis_one
    assert analysis_one[BAND_METRIC]["metric"] == BAND_METRIC


def test_a_recorded_distance_one_ulp_off_is_refused_and_no_directory_is_written(
    copied: Bundle, fresh_output: Path
) -> None:
    """The check is exact: one recorded distance moved by one unit in the last place is a mismatch."""
    changed = {"done": False}

    def nudge(row: dict[str, Any]) -> None:
        if not changed["done"] and row["stratum"] == SOLO and row["system"] == OPGAN:
            value = float(row["results"]["log_spectral_distance"]["value"])
            row["results"]["log_spectral_distance"]["value"] = float(np.nextafter(value, np.inf))
            changed["done"] = True

    update_results(copied, nudge)
    with pytest.raises(PosthocError, match="differs from the recorded value"):
        run_into(copied, fresh_output)
    assert changed["done"]
    assert not fresh_output.exists()


def test_a_reliability_finding_gets_no_post_hoc_number(copied: Bundle, fresh_output: Path) -> None:
    """A solo frame that OpGAN failed makes the stratum a reliability finding, with no numbers."""
    failed = {"done": False}

    def fail_one_frame(row: dict[str, Any]) -> None:
        if not failed["done"] and row["stratum"] == SOLO and row["system"] == OPGAN:
            for key in ("results", "shift_tolerant", "reference_samples_discarded"):
                row.pop(key)
            row.pop("estimate_samples_discarded")
            row["outcome"] = "system_failure"
            row["failure_kind"] = "exception"
            row["exception_type"] = "RuntimeError"
            row["failure_reason"] = "RuntimeError: engine fault"
            failed["done"] = True

    update_results(copied, fail_one_frame)
    assert run_main(copied, fresh_output) == 0
    strata = read_document(fresh_output)["strata"]
    assert strata[SOLO] == {"outcome": "reliability_finding"}
    assert strata[CHAMBER] == {"outcome": "withheld"}


def test_the_band_edge_agrees_with_the_loaded_protocol() -> None:
    """The committed protocol puts the band edge at 4 kHz, which is bin 256 at 16 kHz and 1024 points."""
    _check_band_edge(load_protocol())


def test_a_band_last_bin_that_disagrees_with_the_protocol_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A band whose last bin does not sit at the band edge is refused before any analysis runs."""
    monkeypatch.setattr(run_posthoc_module, "BAND_LAST_BIN", 255)
    with pytest.raises(PosthocError, match="BAND_LAST_BIN 255"):
        _check_band_edge(load_protocol())


def test_a_lowest_rolloff_other_than_the_band_edge_is_refused() -> None:
    """A degradation whose lowest cutoff is not the band edge is refused, since the band would be wrong."""
    protocol = load_protocol()
    ranges = dataclasses.replace(protocol.degradation.ranges, rolloff_hz=(4500.0, 8000.0))
    changed = dataclasses.replace(
        protocol, degradation=dataclasses.replace(protocol.degradation, ranges=ranges)
    )
    with pytest.raises(PosthocError, match=r"lowest rolloff drawn is 4500\.0 Hz"):
        _check_band_edge(changed)


def test_a_failed_file_write_leaves_no_post_hoc_directory(
    base: Bundle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure while the Markdown is written leaves neither the output nor a partial sibling."""
    real_write = atomic_write.write_atomically

    def fail_on_markdown(target: Path, data: bytes) -> None:
        if target.name == MARKDOWN_NAME:
            raise OSError("simulated disk full")
        real_write(target, data)

    monkeypatch.setattr(atomic_write, "write_atomically", fail_on_markdown)
    output = tmp_path / "posthoc"
    with pytest.raises(ReportError, match="cannot be written"):
        run_into(base, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".posthoc.*"))
