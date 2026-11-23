"""Tests for the per-stratum publication gates and the outcome each stratum publishes."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest
from report_fixtures import (
    CHAMBER,
    CLUSTERS,
    PRIMARY,
    SHIFT,
    SILENT_FRAME,
    SILENT_RECORDING,
    SOLO,
    Bundle,
    bind,
    frame_value,
    pinned_universes,
    read_json,
    read_lines,
    sha,
    write_base,
    write_json,
    write_lines,
)

from benchmark import report as report_module
from benchmark.aggregation import RecordingScores, SystemValues, summarise
from benchmark.candidate_universe import CandidateUniverseError
from benchmark.protocol import Protocol, load_protocol
from benchmark.report import (
    AUDIT_NAME,
    MANIFEST_KEY,
    MANIFEST_NAME,
    POPULATION_KEY,
    POPULATION_NAME,
    RESULTS_KEY,
    RESULTS_NAME,
    SUMMARY_NAME,
    Gate,
    Outcome,
    ReportError,
    RunReport,
    StratumReport,
    build_report,
)
from benchmark.scoring import BASELINE, OPGAN, UVR
from benchmark.systems_pipeline import Admission, Partition
from benchmark.walk import partition_plan


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
    """The complete bundle, built once. Tests copy it before they change anything."""
    return write_base(protocol, tmp_path_factory.mktemp("bundle"))


@pytest.fixture
def bundle(base: Bundle, tmp_path: Path) -> Bundle:
    """A private copy of the base bundle that a test may change. Canonical files are shared."""
    walk = tmp_path / "walk"
    systems = tmp_path / "systems"
    shutil.copytree(base.walk, walk)
    shutil.copytree(base.systems, systems)
    return Bundle(walk, systems)


def run(protocol: Protocol, bundle: Bundle) -> RunReport:
    """The report for a bundle."""
    return build_report(bundle.walk, bundle.systems, protocol)


def eligible_frames(recording: int) -> list[int]:
    """The eligible frame indices of a recording: all 30, less the silent one in recording 0."""
    frames = list(range(30))
    if recording == SILENT_RECORDING:
        frames.remove(SILENT_FRAME)
    return frames


def solo(report: RunReport) -> StratumReport:
    """The solo piano stratum's report."""
    return next(s for s in report.strata if s.stratum == SOLO)


def failing_gates(stratum: StratumReport) -> set[str]:
    """The names of the gates that do not hold."""
    return {name for name, held in stratum.gates.items() if not held}


def edit_results(bundle: Bundle, edit: Callable[[list[dict[str, Any]]], None]) -> None:
    """Apply an edit to the systems results rows, then write them back."""
    rows = read_lines(bundle.systems / RESULTS_NAME)
    edit(rows)
    write_lines(bundle.systems / RESULTS_NAME, rows)
    bind(bundle)


def test_the_complete_stratum_is_a_comparison(protocol: Protocol, bundle: Bundle) -> None:
    stratum = solo(run(protocol, bundle))
    assert stratum.outcome is Outcome.COMPARISON
    assert failing_gates(stratum) == set()
    assert set(stratum.gates) == {gate.value for gate in Gate}
    assert stratum.counts is not None
    assert stratum.counts.recordings == len(CLUSTERS)
    assert stratum.counts.clusters == 5
    assert stratum.counts.eligible_frames == len(CLUSTERS) * 30 - 1


def test_the_silent_frame_is_ineligible_by_recomputation(
    protocol: Protocol, bundle: Bundle
) -> None:
    """Recording 0 holds a silent frame, which the recomputation drops, so the population is 179."""
    stratum = solo(run(protocol, bundle))
    assert stratum.counts is not None
    assert stratum.counts.eligible_frames == 179
    assert stratum.coverage == {OPGAN: 1.0, UVR: 1.0}


def test_the_headline_is_the_three_primary_metrics_in_protocol_order(
    protocol: Protocol, bundle: Bundle
) -> None:
    stratum = solo(run(protocol, bundle))
    assert tuple(s.metric for s in stratum.headline) == PRIMARY
    assert tuple(s.metric for s in stratum.diagnostic) == SHIFT
    assert len(stratum.baseline) == 5


def recording_scores_in(order: Sequence[int]) -> list[RecordingScores]:
    """The three recordings' hand-built scores, in the order given."""
    scores = []
    for recording in order:
        frames = eligible_frames(recording)
        values = {
            system: SystemValues(
                {m: [frame_value(system, recording, i)[0][m] for i in frames] for m in PRIMARY},
                {m: [frame_value(system, recording, i)[1][m] for i in frames] for m in SHIFT},
            )
            for system in (OPGAN, UVR, BASELINE)
        }
        scores.append(
            RecordingScores(
                f"recording-{recording}",
                CLUSTERS[recording],
                values[OPGAN],
                values[UVR],
                values[BASELINE],
            )
        )
    return scores


def test_the_intervals_follow_the_first_admission_cluster_order(
    protocol: Protocol, bundle: Bundle
) -> None:
    """The clusters are c, a, b in admission order, so sorting by name would number them
    differently. The intervals are compared against those built from the recordings in admission
    order, and against an order that changes the first-admission cluster order."""
    stratum = solo(run(protocol, bundle))
    admission_order = list(range(len(CLUSTERS)))
    expected = summarise(recording_scores_in(admission_order), protocol).primary[0]
    # The first-admission cluster order changes the 97.5 percent bound for this fixture. The order
    # was found by checking the permutations of the six recordings against the aggregation, so the
    # difference is a fact about these values, not a general one.
    reordered_order = [1, 2, 0, 3, 4, 5]
    reordered = summarise(recording_scores_in(reordered_order), protocol).primary[0]
    assert expected.opgan_interval != reordered.opgan_interval
    assert stratum.headline[0].opgan_interval == expected.opgan_interval
    assert stratum.headline[0].paired_interval == expected.paired_interval


def test_a_short_stratum_publishes_its_reason_only(protocol: Protocol, bundle: Bundle) -> None:
    chamber = next(s for s in run(protocol, bundle).strata if s.stratum == CHAMBER)
    assert chamber.outcome is Outcome.WITHHELD
    assert chamber.counts is None
    assert chamber.coverage is None
    assert chamber.headline == ()
    assert chamber.diagnostic == ()
    assert chamber.baseline == ()
    assert len(chamber.reasons) == 1
    assert "short" in chamber.reasons[0]


def test_a_coverage_only_failure_publishes_a_reliability_finding(
    protocol: Protocol, bundle: Bundle
) -> None:
    """One OpGAN frame fails inside its call. Only the OpGAN coverage gate fails."""

    def fail_one_opgan_frame(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["system"] == OPGAN and row["frame_index"] == 1 and row["canonical_sha256"]:
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
                return

    edit_results(bundle, fail_one_opgan_frame)
    stratum = solo(run(protocol, bundle))
    assert stratum.outcome is Outcome.RELIABILITY_FINDING
    assert failing_gates(stratum) == {Gate.COVERAGE_OPGAN.value}
    assert stratum.coverage is not None
    assert stratum.coverage[OPGAN] < 1.0
    assert stratum.coverage[UVR] == 1.0
    assert stratum.failure_counts == {OPGAN: {"exception": 1}, UVR: {}}
    assert stratum.headline == ()
    assert stratum.diagnostic == ()
    assert stratum.baseline == ()


def test_a_uvr_coverage_only_failure_publishes_a_reliability_finding(
    protocol: Protocol, bundle: Bundle
) -> None:
    """The same path as OpGAN's, for UVR: the rule for coverage-only failures is not OpGAN's."""

    def fail_one_uvr_frame(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["system"] == UVR and row["frame_index"] == 2:
                row.update(
                    outcome="system_failure",
                    failure_kind="output_format",
                    failure_reason="output is not mono",
                    results=None,
                    shift_tolerant=None,
                    reference_samples_discarded=None,
                    estimate_samples_discarded=None,
                    exception_type=None,
                )
                return

    edit_results(bundle, fail_one_uvr_frame)
    stratum = solo(run(protocol, bundle))
    assert stratum.outcome is Outcome.RELIABILITY_FINDING
    assert failing_gates(stratum) == {Gate.COVERAGE_UVR.value}
    assert stratum.failure_counts == {OPGAN: {}, UVR: {"output_format": 1}}


def test_a_coverage_failure_with_a_population_failure_is_withheld(
    protocol: Protocol, bundle: Bundle
) -> None:
    """OpGAN loses a frame, and UVR also reports a frame outside the population. The frames gate
    fails with a non-coverage gate, so the stratum publishes reasons only."""

    def fail_and_add(rows: list[dict[str, Any]]) -> None:
        for row in rows:
            if row["system"] == OPGAN and row["frame_index"] == 1:
                row.update(
                    outcome="system_failure",
                    failure_kind="exception",
                    failure_reason="exception in denoise",
                    results=None,
                    shift_tolerant=None,
                    reference_samples_discarded=None,
                    estimate_samples_discarded=None,
                    exception_type="RuntimeError",
                )
                break
        extra = dict(rows[-1], frame_index=999, system=UVR)
        rows.append(extra)

    edit_results(bundle, fail_and_add)
    stratum = solo(run(protocol, bundle))
    assert stratum.outcome is Outcome.WITHHELD
    assert Gate.COVERAGE_OPGAN.value in failing_gates(stratum)
    assert Gate.FRAMES.value in failing_gates(stratum)
    assert stratum.counts is None
    assert stratum.coverage is None
    assert stratum.failure_counts is None
    assert stratum.headline == ()


def test_a_population_file_that_differs_fails_the_frames_gate(
    protocol: Protocol, bundle: Bundle
) -> None:
    """systems_population.jsonl is compared with the recomputation. One changed eligible list is
    a mismatch, so the frames gate fails and the stratum publishes reasons only."""
    path = bundle.systems / POPULATION_NAME
    rows = read_lines(path)
    rows[0]["eligible_frames"] = rows[0]["eligible_frames"][:-1]
    write_lines(path, rows)
    bind(bundle)
    stratum = solo(run(protocol, bundle))
    assert stratum.outcome is Outcome.WITHHELD
    assert failing_gates(stratum) == {Gate.FRAMES.value}
    assert any("systems_population.jsonl" in reason for reason in stratum.reasons)


def test_a_missing_audit_digest_key_fails_the_screen_gate(
    protocol: Protocol, bundle: Bundle
) -> None:
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    del manifest["audit_trail_sha256"]
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)
    stratum = solo(run(protocol, bundle))
    assert stratum.outcome is Outcome.WITHHELD
    assert failing_gates(stratum) == {Gate.SCREEN.value}
    assert any("has no audit_trail_sha256" in reason for reason in stratum.reasons)


def test_an_audit_trail_that_changed_after_the_walk_fails_the_screen_gate(
    protocol: Protocol, bundle: Bundle
) -> None:
    with (bundle.walk / AUDIT_NAME).open("a") as handle:
        handle.write("\n")
    stratum = solo(run(protocol, bundle))
    assert failing_gates(stratum) == {Gate.SCREEN.value}


def test_an_admission_without_an_audit_line_fails_the_screen_gate(
    protocol: Protocol, bundle: Bundle
) -> None:
    """The digest is updated to match the shortened file, so only the per-admission check can
    catch the missing line. Without it the gate would pass on a vacuous check."""
    lines = read_lines(bundle.walk / AUDIT_NAME)
    write_lines(bundle.walk / AUDIT_NAME, [line for line in lines if line["position"] != 2])
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    manifest["audit_trail_sha256"] = sha((bundle.walk / AUDIT_NAME).read_bytes())
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)
    stratum = solo(run(protocol, bundle))
    assert failing_gates(stratum) == {Gate.SCREEN.value}
    assert any("admissions differ" in reason for reason in stratum.reasons)


def test_another_stratums_pending_candidate_does_not_fail_this_stratum(
    protocol: Protocol, bundle: Bundle
) -> None:
    """The screen gate reads only the lines of the stratum it judges."""
    lines = read_lines(bundle.walk / AUDIT_NAME)
    lines.append(
        {
            "stratum": CHAMBER,
            "position": 1,
            "identifier": "musicnet/pending",
            "outcome": "awaiting_adjudication",
            "code": None,
            "partition": None,
            "admission_position": None,
            "details": {},
        }
    )
    write_lines(bundle.walk / AUDIT_NAME, lines)
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    manifest["audit_trail_sha256"] = sha((bundle.walk / AUDIT_NAME).read_bytes())
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)
    assert failing_gates(solo(run(protocol, bundle))) == set()


def test_an_admitted_candidate_without_a_screen_result_fails_the_screen_gate(
    protocol: Protocol, bundle: Bundle
) -> None:
    lines = read_lines(bundle.walk / AUDIT_NAME)
    lines[0]["details"]["screen"] = None
    write_lines(bundle.walk / AUDIT_NAME, lines)
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    manifest["audit_trail_sha256"] = sha((bundle.walk / AUDIT_NAME).read_bytes())
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)
    stratum = solo(run(protocol, bundle))
    assert failing_gates(stratum) == {Gate.SCREEN.value}
    assert any("no screen result" in reason for reason in stratum.reasons)


def test_a_candidate_awaiting_adjudication_fails_the_screen_gate(
    protocol: Protocol, bundle: Bundle
) -> None:
    lines = read_lines(bundle.walk / AUDIT_NAME)
    lines[-1]["outcome"] = "awaiting_adjudication"
    write_lines(bundle.walk / AUDIT_NAME, lines)
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    manifest["audit_trail_sha256"] = sha((bundle.walk / AUDIT_NAME).read_bytes())
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)
    stratum = solo(run(protocol, bundle))
    assert failing_gates(stratum) == {Gate.SCREEN.value}
    assert any("awaits adjudication" in reason for reason in stratum.reasons)


def test_a_manifest_naming_another_protocol_fails_the_protocol_gate(
    protocol: Protocol, bundle: Bundle
) -> None:
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    manifest["protocol_sha256"] = "0" * 64
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)
    stratum = solo(run(protocol, bundle))
    assert failing_gates(stratum) == {Gate.PROTOCOL.value}


def test_a_summary_naming_another_protocol_fails_the_protocol_gate(
    protocol: Protocol, bundle: Bundle
) -> None:
    summary = read_json(bundle.systems / SUMMARY_NAME)
    summary["protocol_sha256"] = "1" * 64
    write_json(bundle.systems / SUMMARY_NAME, summary)
    stratum = solo(run(protocol, bundle))
    assert failing_gates(stratum) == {Gate.PROTOCOL.value}


def test_a_universe_digest_that_is_not_pinned_fails_the_universe_gate(
    protocol: Protocol, bundle: Bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpinned universe (None) can never verify, even when the manifest names a digest."""
    monkeypatch.setitem(report_module.UNIVERSE_SHA256, SOLO, None)
    stratum = solo(run(protocol, bundle))
    assert failing_gates(stratum) == {Gate.UNIVERSE.value}


def test_a_manifest_universe_digest_that_differs_fails_the_universe_gate(
    protocol: Protocol, bundle: Bundle
) -> None:
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    manifest["universe_sha256"][SOLO] = "2" * 64
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)
    stratum = solo(run(protocol, bundle))
    assert failing_gates(stratum) == {Gate.UNIVERSE.value}


def test_a_changed_canonical_file_aborts_the_whole_report(
    protocol: Protocol, base: Bundle, tmp_path: Path
) -> None:
    """A canonical file that no longer hashes to its digest is a failed check on the evidence,
    so the report raises instead of withholding one stratum and publishing the rest."""
    copied = tmp_path / "walk"
    shutil.copytree(base.walk, copied)
    systems = tmp_path / "systems"
    shutil.copytree(base.systems, systems)
    canonical = tmp_path / "canonical"
    canonical.mkdir()
    manifest = read_json(copied / MANIFEST_NAME)
    for admission in manifest["admissions"]:
        source = Path(admission["canonical_path"])
        target = canonical / source.name
        shutil.copy(source, target)
        admission["canonical_path"] = str(target)
    target = canonical / "recording-1.f64"
    data = bytearray(target.read_bytes())
    data[0] ^= 0xFF
    target.write_bytes(bytes(data))
    write_json(copied / MANIFEST_NAME, manifest)
    bind(Bundle(copied, systems))
    with pytest.raises(ReportError, match="failed its check"):
        build_report(copied, systems, protocol)


def test_a_baseline_with_a_gap_stops_the_comparison(protocol: Protocol, bundle: Bundle) -> None:
    """The degraded input has no coverage gate, but a gap in it is a harness fault: the report
    stops rather than publish a comparison with a missing baseline frame."""

    def drop_one_baseline_frame(rows: list[dict[str, Any]]) -> None:
        index = next(i for i, row in enumerate(rows) if row["system"] == BASELINE)
        del rows[index]

    edit_results(bundle, drop_one_baseline_frame)
    with pytest.raises(ReportError, match="cannot be aggregated"):
        run(protocol, bundle)


def test_the_gate_names_are_the_protocol_names(protocol: Protocol) -> None:
    assert {gate.value for gate in Gate} == set(protocol.publication_gates.required)


def test_the_baseline_key_matches_the_systems_stage(protocol: Protocol) -> None:
    from benchmark.systems_pipeline import BASELINE as SYSTEMS_BASELINE

    assert BASELINE == SYSTEMS_BASELINE


def assert_withheld(stratum: StratumReport, failing: set[str], reason: str) -> None:
    """The stratum is withheld for these gates, with a reason containing `reason`, and publishes
    no counts, coverage or metric value."""
    assert stratum.outcome is Outcome.WITHHELD
    assert failing_gates(stratum) == failing
    assert any(reason in text for text in stratum.reasons), stratum.reasons
    assert stratum.counts is None
    assert stratum.coverage is None
    assert stratum.headline == ()
    assert stratum.diagnostic == ()
    assert stratum.baseline == ()


def assert_every_stratum_withheld(report: RunReport, reason: str) -> None:
    """Every stratum is withheld with no gate evaluated, and each reason contains `reason`."""
    assert [s.stratum for s in report.strata] == [SOLO, CHAMBER]
    for stratum in report.strata:
        assert stratum.gates == {}
        assert stratum.outcome is Outcome.WITHHELD
        assert any(reason in text for text in stratum.reasons), stratum.reasons
        assert stratum.counts is None
        assert stratum.coverage is None
        assert stratum.headline == ()
        assert stratum.diagnostic == ()
        assert stratum.baseline == ()


def rebind_audit(bundle: Bundle) -> None:
    """Record the audit trail's new digest in the manifest, and bind the summary to the manifest."""
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    manifest["audit_trail_sha256"] = sha((bundle.walk / AUDIT_NAME).read_bytes())
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)


def edit_admitted_line(bundle: Bundle, where: str, key: str, value: object) -> None:
    """Set one key of the first solo admitted audit line, or of its details, then rebind."""
    lines = read_lines(bundle.walk / AUDIT_NAME)
    line = next(i for i in lines if i["stratum"] == SOLO and i["admission_position"] == 1)
    target = line["details"] if where == "details" else line
    if value is _DELETE:
        del target[key]
    else:
        target[key] = value
    write_lines(bundle.walk / AUDIT_NAME, lines)
    rebind_audit(bundle)


_DELETE = object()


@pytest.mark.parametrize(
    ("name", "reason"),
    [
        (RESULTS_NAME, "systems_results.jsonl does not match"),
        (POPULATION_NAME, "systems_population.jsonl does not match"),
        (MANIFEST_NAME, "walk_manifest.json does not match"),
    ],
)
def test_a_changed_bound_file_withholds_every_stratum(
    protocol: Protocol, bundle: Bundle, name: str, reason: str
) -> None:
    """The summary records each file's digest. A file changed after the systems stage wrote its
    summary is not the file the stage produced, so no stratum publishes a number."""
    path = (bundle.walk if name == MANIFEST_NAME else bundle.systems) / name
    path.write_bytes(b" " + path.read_bytes())
    assert_every_stratum_withheld(run(protocol, bundle), reason)


@pytest.mark.parametrize(
    ("key", "reason"),
    [
        (RESULTS_KEY, "has no systems_results_sha256"),
        (POPULATION_KEY, "has no systems_population_sha256"),
        (MANIFEST_KEY, "has no walk_manifest_sha256"),
    ],
)
def test_a_summary_without_a_bound_digest_withholds_every_stratum(
    protocol: Protocol, bundle: Bundle, key: str, reason: str
) -> None:
    """A summary that does not record a digest does not bind the file, so the run is withheld."""
    summary = read_json(bundle.systems / SUMMARY_NAME)
    del summary[key]
    write_json(bundle.systems / SUMMARY_NAME, summary)
    assert_every_stratum_withheld(run(protocol, bundle), reason)


@pytest.mark.parametrize(
    ("where", "key", "value", "reason"),
    [
        ("top", "partition", "demo", "names another partition"),
        ("top", "admission_position", 9, "has 0 admitted audit lines at position 1"),
        ("details", "raw_sha256", "0" * 64, "details.raw_sha256 in the audit line"),
        ("details", "decoded_pcm_sha256", "0" * 64, "details.decoded_pcm_sha256 in the audit line"),
        ("details", "samples", 1, "details.samples in the audit line"),
        ("details", "canonical_sha256", "0" * 64, "details.canonical_sha256 in the audit line"),
        ("details", "canonical_sha256", _DELETE, "has no details.canonical_sha256"),
    ],
)
def test_an_admitted_audit_line_that_differs_from_its_admission_fails_the_screen_gate(
    protocol: Protocol, bundle: Bundle, where: str, key: str, value: object, reason: str
) -> None:
    """The audit trail and the manifest are rebound, so only the per-admission check can catch an
    audit line that differs from the admission it records."""
    edit_admitted_line(bundle, where, key, value)
    assert_withheld(solo(run(protocol, bundle)), {Gate.SCREEN.value}, reason)


def test_a_swapped_partition_label_fails_the_walk_plan_check(
    protocol: Protocol, bundle: Bundle
) -> None:
    """The walk's plan gives position 1 the test partition. A manifest that labels it demo is not
    what the walk admitted, and the stratum is withheld with that reason."""
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    manifest["admissions"][0]["partition"] = "demo"
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)
    stratum = solo(run(protocol, bundle))
    assert stratum.outcome is Outcome.WITHHELD
    assert Gate.SCREEN.value in failing_gates(stratum)
    assert any("plan gives test at position 1" in text for text in stratum.reasons)


def test_a_wrong_cluster_label_fails_the_pinned_universe_check(
    protocol: Protocol, bundle: Bundle
) -> None:
    """The pinned universe gives position 3 the cluster performer-d. A manifest that names another
    performer is withheld, and no other gate is affected by the label."""
    manifest = read_json(bundle.walk / MANIFEST_NAME)
    manifest["admissions"][2]["cluster"] = "performer-z"
    write_json(bundle.walk / MANIFEST_NAME, manifest)
    bind(bundle)
    stratum = solo(run(protocol, bundle))
    assert_withheld(
        stratum,
        {Gate.SCREEN.value},
        "is in cluster performer-z, but the pinned universe gives performer-d",
    )


def test_a_pinned_universe_that_fails_its_check_aborts_the_report(
    protocol: Protocol, bundle: Bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pinned universe that cannot be loaded is a fault in the repository, so the report raises
    and publishes nothing, rather than withholding strata for it."""

    def refuse(stratum: object) -> None:
        raise CandidateUniverseError("the universe file is altered")

    monkeypatch.setattr(report_module, "load_universe", refuse)
    with pytest.raises(ReportError, match="failed its check"):
        run(protocol, bundle)


def test_a_tampered_audit_trail_withholds_the_stratum_with_no_counts(
    protocol: Protocol, bundle: Bundle
) -> None:
    """A changed audit trail publishes no counts and no metric, not only a failed gate."""
    with (bundle.walk / AUDIT_NAME).open("a") as handle:
        handle.write("\n")
    stratum = solo(run(protocol, bundle))
    assert_withheld(stratum, {Gate.SCREEN.value}, "the audit trail's digest differs")


def test_a_stratum_with_no_pinned_digest_is_not_loaded(
    protocol: Protocol, bundle: Bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stratum with no pinned digest has no universe to load. Loading it would raise, so the
    report must skip the load and leave the universe gate as the only failure."""
    monkeypatch.setitem(report_module.UNIVERSE_SHA256, SOLO, None)

    def refuse(stratum: object) -> None:
        raise CandidateUniverseError("loaded a stratum that has no pinned digest")

    monkeypatch.setattr(report_module, "load_universe", refuse)
    stratum = solo(run(protocol, bundle))
    assert failing_gates(stratum) == {Gate.UNIVERSE.value}


def test_a_position_beyond_the_walk_plan_is_reported_not_indexed() -> None:
    """The plan has two positions, so position 3 has no partition. The check returns a reason
    instead of raising an IndexError. The manifest can reach this only through a stratum whose
    admitted count exceeds the plan, so the check is exercised on its own."""
    admission = Admission(
        stratum=SOLO,
        partition=Partition.TEST,
        position=3,
        identifier="musopen/track-2.flac",
        cluster="performer-d",
        canonical_path=Path("/unused.f64"),
        canonical_sha256="0" * 64,
        samples=1,
        raw_sha256="0" * 64,
        decoded_pcm_sha256="0" * 64,
    )
    reason = report_module._partition_reason(admission, ("test", "test"))
    assert reason is not None
    assert "outside the walk's plan of 2 positions" in reason


def test_the_walk_plans_have_the_v13_lengths(protocol: Protocol) -> None:
    """The solo plan is 30 test, 6 demo and 6 test_reserve positions, 42 in all. The chamber plan
    is 30 test and 6 test_reserve, 36 in all. The fixtures' plan cut is checked against these."""
    lengths = {s.name: len(partition_plan(s)) for s in protocol.selection.strata}
    assert lengths == {SOLO: 42, CHAMBER: 36}


@pytest.mark.parametrize(
    ("plan_length", "reason"),
    [
        (len(CLUSTERS) - 1, "with 6 admissions, but its partition plan has 5 positions"),
        (len(CLUSTERS) + 1, "with 6 admissions, but its partition plan has 7 positions"),
    ],
    ids=["plan-one-shorter", "plan-one-longer"],
)
def test_a_complete_stratum_off_by_one_from_its_plan_fails_the_screen_gate(
    protocol: Protocol,
    bundle: Bundle,
    monkeypatch: pytest.MonkeyPatch,
    plan_length: int,
    reason: str,
) -> None:
    """A complete stratum must admit one recording per plan position. The bundle admits six, and
    the plan is one position shorter or one longer. The stratum publishes its reason only."""
    monkeypatch.setattr(report_module, "partition_plan", lambda stratum: ("test",) * plan_length)
    stratum = solo(run(protocol, bundle))
    assert_withheld(stratum, {Gate.SCREEN.value}, f"the walk marks solo_piano complete {reason}")


def test_a_plan_shifted_by_one_position_is_named_at_the_position_that_differs(
    protocol: Protocol, bundle: Bundle, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The plan gives position 1 the demo partition and the rest test. Only position 1 may be
    named. A check that read the plan one position off would name position 2 as well, because
    the admission there is test and the plan would give it demo."""
    monkeypatch.setattr(
        report_module,
        "partition_plan",
        lambda stratum: ("demo",) + ("test",) * (len(CLUSTERS) - 1),
    )
    stratum = solo(run(protocol, bundle))
    assert_withheld(stratum, {Gate.SCREEN.value}, "the walk's plan gives demo at position 1")
    assert not any("at position 2" in text for text in stratum.reasons)
