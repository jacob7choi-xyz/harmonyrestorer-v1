"""Tests for the report command: the layout rules, the digests, and the refusals it makes.

The layout rules are checked on the files the command writes. Rules about the Markdown read its
sections; rules about the JSON read its structure, because JSON has no reading order.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from report_fixtures import (
    CHAMBER,
    CLUSTERS,
    SOLO,
    Bundle,
    bind,
    copy_walk,
    pinned_universes,
    read_json,
    read_lines,
    write_base,
    write_json,
    write_lines,
)

from benchmark.candidate_universe import UNIVERSE_SHA256
from benchmark.protocol import Protocol, load_protocol
from benchmark.report import (
    AUDIT_NAME,
    MANIFEST_NAME,
    RESULTS_KEY,
    RESULTS_NAME,
    SUMMARY_NAME,
    ReportError,
)
from benchmark.run_report import JSON_NAME, MARKDOWN_NAME, main, run_report

PRIMARY_NAMES = ["reconstruction_snr", "si_snr", "log_spectral_distance"]
SHIFT_NAMES = ["reconstruction_snr", "si_snr"]
PRIMARY_LABELS = ["reconstruction SNR (dB)", "SI-SNR (dB)", "log-spectral distance (dB)"]
FORBIDDEN = ("significant", "significance", "p-value", "p <", "winner", "overall", "favour")


@pytest.fixture(scope="module", autouse=True)
def pinned() -> Iterator[None]:
    """The pinned universes are stand-ins, because the fixtures' identifiers are not the corpus."""
    with pinned_universes():
        yield


@pytest.fixture(scope="module")
def protocol() -> Protocol:
    """The frozen protocol."""
    return load_protocol()


@pytest.fixture(scope="module")
def base(protocol: Protocol, tmp_path_factory: pytest.TempPathFactory) -> Bundle:
    """The complete bundle, built once for the module."""
    return write_base(protocol, tmp_path_factory.mktemp("bundle"))


@pytest.fixture(scope="module")
def written(base: Bundle, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The report the command writes, into a new directory."""
    output = tmp_path_factory.mktemp("report") / "out"
    run_report(base.walk, base.systems, output)
    return output


@pytest.fixture(scope="module")
def document(written: Path) -> dict[str, Any]:
    """The JSON report, parsed."""
    return json.loads((written / JSON_NAME).read_text())


@pytest.fixture(scope="module")
def markdown(written: Path) -> str:
    """The Markdown report."""
    return (written / MARKDOWN_NAME).read_text()


def section(markdown: str, heading: str) -> str:
    """The text of one Markdown section, from its heading to the next heading of its own or a
    higher level."""
    start = markdown.index(heading)
    body = markdown[start + len(heading) :]
    ends = [body.find(marker) for marker in ("\n## ", "\n### ")]
    found = [end for end in ends if end >= 0]
    return body[: min(found)] if found else body


def file_sha(path: Path) -> str:
    """The SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_digests_name_the_protocol_universes_and_files(
    protocol: Protocol, base: Bundle, document: dict[str, Any]
) -> None:
    digests = document["digests"]
    assert digests["protocol_sha256"] == protocol.source_sha256
    assert digests["universe_sha256"] == {
        SOLO: UNIVERSE_SHA256[SOLO],
        CHAMBER: UNIVERSE_SHA256[CHAMBER],
    }
    assert digests["archive_sha256"] is None
    assert digests["walk_manifest_sha256"] == file_sha(base.walk / MANIFEST_NAME)
    assert digests["audit_trail_sha256"] == file_sha(base.walk / AUDIT_NAME)
    assert digests["systems_results_sha256"] == file_sha(base.systems / RESULTS_NAME)


def test_the_claim_scope_note_is_written_in_both_renderings(
    protocol: Protocol, document: dict[str, Any], markdown: str
) -> None:
    note = protocol.reporting.claim_scope_note
    assert document["claim_scope"] == note
    assert f"> **Note:** {note}" in markdown


@pytest.mark.parametrize("name", [JSON_NAME, MARKDOWN_NAME])
def test_no_significance_language_or_winner_is_written(name: str, written: Path) -> None:
    text = (written / name).read_text().lower()
    for word in FORBIDDEN:
        assert word not in text, word


def test_the_three_co_primary_metrics_appear_together_in_protocol_order_in_json(
    document: dict[str, Any],
) -> None:
    headline = document["strata"][SOLO]["headline"]
    assert [metric["metric"] for metric in headline] == PRIMARY_NAMES


def test_the_three_co_primary_metrics_appear_together_in_protocol_order_in_markdown(
    markdown: str,
) -> None:
    block = section(markdown, "### Co-primary metrics")
    positions = [block.index(label) for label in PRIMARY_LABELS]
    assert positions == sorted(positions)
    assert all(block.count(label) == 1 for label in PRIMARY_LABELS)


def test_the_diagnostic_and_the_baseline_come_after_the_headline_in_markdown(
    markdown: str,
) -> None:
    solo = markdown[markdown.index(f"## {SOLO}") :]
    headline_at = solo.index("### Co-primary metrics")
    diagnostic_at = solo.index("### Diagnostic")
    baseline_at = solo.index("### Input baseline")
    assert headline_at < diagnostic_at < baseline_at


def test_the_diagnostic_is_the_shift_tolerant_metrics_and_never_the_headline(
    document: dict[str, Any], markdown: str
) -> None:
    solo = document["strata"][SOLO]
    assert [metric["metric"] for metric in solo["diagnostic"]] == SHIFT_NAMES
    assert [metric["metric"] for metric in solo["headline"]] == PRIMARY_NAMES
    headline_block = section(markdown, "### Co-primary metrics")
    assert "Diagnostic" not in headline_block
    assert "Input baseline" not in headline_block


def test_the_baseline_has_no_paired_field_in_either_rendering(
    document: dict[str, Any], markdown: str
) -> None:
    baseline = document["strata"][SOLO]["baseline"]
    assert len(baseline) == 5
    for entry in baseline:
        assert "paired" not in entry
        assert set(entry) == {"series", "metric", "recordings", "mean", "median", "interval"}
    baseline_block = section(markdown, "### Input baseline")
    assert "Paired difference" not in baseline_block
    assert "scored higher" not in baseline_block


def test_a_stratum_without_a_comparison_has_no_metric_at_all(document: dict[str, Any]) -> None:
    chamber = document["strata"][CHAMBER]
    assert chamber["outcome"] == "withheld"
    assert chamber["headline"] == []
    assert chamber["diagnostic"] == []
    assert chamber["baseline"] == []


def test_counts_are_recordings_distinct_clusters_and_eligible_frames(
    document: dict[str, Any], markdown: str
) -> None:
    counts = document["strata"][SOLO]["counts"]
    assert counts == {
        "recordings": len(CLUSTERS),
        "clusters": len(set(CLUSTERS)),
        "eligible_frames": len(CLUSTERS) * 30 - 1,
    }
    assert "Recordings: 6. Performer clusters: 5. Eligible frames: 179." in markdown


def test_coverage_is_written_per_system(document: dict[str, Any], markdown: str) -> None:
    assert document["strata"][SOLO]["coverage"] == {"opgan": 1.0, "uvr": 1.0}
    assert "Coverage: OpGAN 1.000, UVR 1.000." in markdown


def test_a_withheld_stratum_writes_no_counts_coverage_or_metric_values(
    document: dict[str, Any], markdown: str
) -> None:
    chamber = document["strata"][CHAMBER]
    assert chamber["counts"] is None
    assert chamber["coverage"] is None
    text = section(markdown, f"## {CHAMBER}")
    assert "Recordings:" not in text
    assert "Coverage:" not in text
    assert "Reasons:" in text


def test_the_archive_digest_is_written_as_null_in_both_renderings(
    document: dict[str, Any], markdown: str
) -> None:
    assert document["digests"]["archive_sha256"] is None
    assert "- Archive SHA-256: null" in markdown


def test_the_markdown_names_every_digest(document: dict[str, Any], markdown: str) -> None:
    digests = document["digests"]
    values = [
        digests["protocol_sha256"],
        *digests["universe_sha256"].values(),
        digests["walk_manifest_sha256"],
        digests["audit_trail_sha256"],
        digests["systems_results_sha256"],
    ]
    for value in values:
        assert value in markdown


def copy_bundle(base: Bundle, root: Path) -> Bundle:
    """A private copy of a bundle, so a test may change it. The canonical files are shared."""
    walk = root / "walk"
    systems = root / "systems"
    copy_walk(base.walk, walk)
    shutil.copytree(base.systems, systems)
    bound = Bundle(walk, systems)
    bind(bound)
    return bound


def stratum_text(markdown: str, name: str) -> str:
    """The lines of one stratum's section: from its level-two heading to the next level-two one."""
    lines = markdown.splitlines()
    start = lines.index(f"## {name}")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def fail_one_opgan_frame(bundle: Bundle) -> None:
    """One OpGAN frame fails inside its call, so only the OpGAN coverage gate fails."""
    rows = read_lines(bundle.systems / RESULTS_NAME)
    for row in rows:
        if row["system"] == "opgan" and row["frame_index"] == 1:
            row.update(
                outcome="system_failure",
                failure_kind="exception",
                exception_type="RuntimeError",
                failure_reason="exception in denoise",
                results=None,
                shift_tolerant=None,
                reference_samples_discarded=None,
                estimate_samples_discarded=None,
            )
            break
    write_lines(bundle.systems / RESULTS_NAME, rows)
    bind(bundle)


def drop_audit_digest(bundle: Bundle) -> None:
    """The manifest names no audit digest, so the screen gate fails and the stratum is withheld."""
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    del manifest["audit_trail_sha256"]
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)


@pytest.mark.parametrize(
    ("outcome", "edit"),
    [("reliability_finding", fail_one_opgan_frame), ("withheld", drop_audit_digest)],
    ids=["reliability", "withheld"],
)
def test_a_stratum_that_is_not_a_comparison_publishes_no_metric_value(
    protocol: Protocol,
    base: Bundle,
    tmp_path: Path,
    outcome: str,
    edit: Callable[[Bundle], None],
) -> None:
    bundle = copy_bundle(base, tmp_path / "bundle")
    edit(bundle)
    output = tmp_path / "out"
    run_report(bundle.walk, bundle.systems, output)
    document = json.loads((output / JSON_NAME).read_text())
    markdown = (output / MARKDOWN_NAME).read_text()
    solo = document["strata"][SOLO]
    assert solo["outcome"] == outcome
    assert solo["headline"] == []
    assert solo["diagnostic"] == []
    assert solo["baseline"] == []
    text = stratum_text(markdown, SOLO)
    for heading in ("### Co-primary metrics", "### Diagnostic", "### Input baseline"):
        assert heading not in text
    assert "Paired difference" not in text
    if outcome == "reliability_finding":
        assert "Coverage: OpGAN" in text
        assert "### Failures by kind" in text
        assert solo["failure_counts"]["opgan"] == {"exception": 1}
    else:
        assert "- training_performance_screen_complete: failed" in text
        assert any("audit_trail_sha256" in reason for reason in solo["reasons"])
        assert solo["counts"] is None
    for name in (JSON_NAME, MARKDOWN_NAME):
        for word in FORBIDDEN:
            assert word not in (output / name).read_text().lower(), (name, word)


def test_the_command_writes_both_files_into_a_new_directory(written: Path) -> None:
    assert sorted(p.name for p in written.iterdir()) == [JSON_NAME, MARKDOWN_NAME]


def test_a_run_into_an_existing_directory_is_refused_and_changes_nothing(
    base: Bundle, written: Path
) -> None:
    before = {name: (written / name).read_bytes() for name in (JSON_NAME, MARKDOWN_NAME)}
    with pytest.raises(ReportError, match="already exists"):
        run_report(base.walk, base.systems, written)
    after = {name: (written / name).read_bytes() for name in (JSON_NAME, MARKDOWN_NAME)}
    assert after == before


def test_the_command_refuses_an_existing_directory_with_exit_code_one(
    base: Bundle, written: Path
) -> None:
    before = (written / JSON_NAME).read_bytes()
    code = main(
        [
            "--walk-dir",
            str(base.walk),
            "--systems-dir",
            str(base.systems),
            "--output-dir",
            str(written),
        ]
    )
    assert code == 1
    assert (written / JSON_NAME).read_bytes() == before


def test_a_missing_input_fails_with_exit_code_one_and_creates_no_directory(
    base: Bundle, tmp_path: Path
) -> None:
    output = tmp_path / "never"
    code = main(
        [
            "--walk-dir",
            str(tmp_path / "missing"),
            "--systems-dir",
            str(base.systems),
            "--output-dir",
            str(output),
        ]
    )
    assert code == 1
    assert not output.exists()


def test_a_successful_command_returns_zero_and_writes_both_files(
    base: Bundle, tmp_path: Path
) -> None:
    output = tmp_path / "published"
    code = main(
        [
            "--walk-dir",
            str(base.walk),
            "--systems-dir",
            str(base.systems),
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    assert sorted(p.name for p in output.iterdir()) == [JSON_NAME, MARKDOWN_NAME]


def corrupt_canonical(base: Bundle, root: Path) -> Bundle:
    """A private copy of the bundle whose second recording's canonical file no longer hashes to its
    digest. The manifest names the copies, and the summary is bound to the changed manifest."""
    bundle = copy_bundle(base, root)
    canonical = root / "canonical"
    canonical.mkdir()
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    for admission in manifest["admissions"]:
        source = Path(admission["canonical_path"])
        shutil.copy(source, canonical / source.name)
        admission["canonical_path"] = str(canonical / source.name)
    target = canonical / "recording-1.f64"
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)
    return bundle


def test_a_corrupt_canonical_file_fails_the_command_and_writes_nothing(
    base: Bundle, tmp_path: Path
) -> None:
    bundle = corrupt_canonical(base, tmp_path / "bundle")
    output = tmp_path / "out"
    code = main(
        [
            "--walk-dir",
            str(bundle.walk),
            "--systems-dir",
            str(bundle.systems),
            "--output-dir",
            str(output),
        ]
    )
    assert code == 1
    assert not output.exists()


def test_a_non_finite_value_is_refused_at_the_read_and_leaves_no_directory(
    base: Bundle, tmp_path: Path
) -> None:
    """A NaN written into the results is refused when the file is read, before any rendering, so
    no directory is created."""
    bundle = copy_bundle(base, tmp_path / "bundle")
    rows = read_lines(bundle.systems / RESULTS_NAME)
    for row in rows:
        if row["stratum"] == SOLO and row["system"] == "opgan" and row["frame_index"] == 1:
            row["results"]["reconstruction_snr"]["value"] = float("nan")
            break
    write_lines(bundle.systems / RESULTS_NAME, rows)
    bind(bundle)
    output = tmp_path / "out"
    with pytest.raises(ReportError, match="non-finite JSON value NaN"):
        run_report(bundle.walk, bundle.systems, output)
    assert not output.exists()


def test_a_tampered_systems_file_publishes_every_stratum_withheld(
    base: Bundle, tmp_path: Path
) -> None:
    """A results file changed after the systems stage wrote its summary withholds every stratum.
    The command still writes both files, and neither carries a number."""
    bundle = copy_bundle(base, tmp_path / "bundle")
    path = bundle.systems / RESULTS_NAME
    path.write_bytes(b" " + path.read_bytes())
    output = tmp_path / "out"
    run_report(bundle.walk, bundle.systems, output)
    document = json.loads((output / JSON_NAME).read_text())
    assert {s["outcome"] for s in document["strata"].values()} == {"withheld"}
    for stratum in document["strata"].values():
        assert stratum["headline"] == []
        assert stratum["counts"] is None
        assert any("systems_results.jsonl does not match" in r for r in stratum["reasons"])


def test_the_paired_difference_states_which_system_scored_higher(markdown: str) -> None:
    """A positive paired difference is OpGAN minus UVR, so it means OpGAN scored higher."""
    assert "a positive difference means OpGAN scored higher on this metric" in markdown


def test_a_lower_is_better_difference_is_read_as_distance_not_as_a_win(markdown: str) -> None:
    """UVR minus OpGAN on a distance is positive when UVR's distance is larger, which favours OpGAN.

    The note must say so. A note reading "UVR scored higher" would invert the result for a
    metric where lower is better.
    """
    assert (
        "a positive difference means UVR's distance is larger, so OpGAN's output is spectrally "
        "closer to the clean reference"
    ) in markdown
    assert "UVR scored higher" not in markdown


SENTINEL = 123456.5


def _systems_with_literal(base: Bundle, tmp_path: Path, literal: str) -> Path:
    """A copy of the systems outputs whose first scored metric value is written as `literal`.

    json.dumps writes a sentinel number, which is then swapped for the raw token, so the file
    holds exactly the literal under test. The summary's digest of the file is updated to match,
    so the report reads the value rather than withholding every stratum on a digest mismatch.
    """
    systems = tmp_path / "systems"
    shutil.copytree(base.systems, systems)
    path = systems / RESULTS_NAME
    rows = read_lines(path)
    row = next(r for r in rows if isinstance(r.get("results"), dict))
    row["results"][next(iter(row["results"]))]["value"] = SENTINEL
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    assert text.count(str(SENTINEL)) == 1
    path.write_text(text.replace(str(SENTINEL), literal, 1), encoding="utf-8")
    summary_path = systems / SUMMARY_NAME
    summary = read_json(summary_path)
    summary[RESULTS_KEY] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(summary_path, summary)
    return systems


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_a_non_finite_json_constant_refuses_the_report_and_writes_nothing(
    base: Bundle, tmp_path: Path, literal: str
) -> None:
    """A run file holding NaN, Infinity or -Infinity is refused, and no report directory exists."""
    systems = _systems_with_literal(base, tmp_path, literal)
    output = tmp_path / "out"
    with pytest.raises(ReportError, match="non-finite JSON value"):
        run_report(base.walk, systems, output)
    assert not output.exists()


def test_a_metric_literal_that_overflows_to_infinity_refuses_the_report(
    base: Bundle, tmp_path: Path
) -> None:
    """The literal 1e400 parses to infinity without a constant hook, so the finite check refuses it."""
    systems = _systems_with_literal(base, tmp_path, "1e400")
    output = tmp_path / "out"
    with pytest.raises(ReportError, match="is not a finite value"):
        run_report(base.walk, systems, output)
    assert not output.exists()


def test_a_canonical_path_outside_the_walk_directory_refuses_the_report(
    base: Bundle, tmp_path: Path
) -> None:
    """An admission whose canonical file lies outside the walk directory refuses the report."""
    walk = tmp_path / "walk"
    copy_walk(base.walk, walk)
    outside = tmp_path / "outside.f64"
    manifest = read_json(walk / MANIFEST_NAME)
    shutil.copy(Path(manifest["admissions"][0]["canonical_path"]), outside)
    manifest["admissions"][0]["canonical_path"] = str(outside)
    write_json(walk / MANIFEST_NAME, manifest)
    output = tmp_path / "out"
    with pytest.raises(ReportError, match="resolves outside the walk directory"):
        run_report(walk, base.systems, output)
    assert not output.exists()
