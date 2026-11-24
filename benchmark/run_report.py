"""The report command: per-stratum gates and results, written as `report.json` and `report.md`.

Run it from the repository root:

    uv run python -m benchmark.run_report --walk-dir DIR --systems-dir DIR --output-dir NEW_DIR

The output directory must not exist, so a published report is never overwritten. The Markdown
and the JSON carry the same content. The layout rules are enforced by tests, not by this module:
the three co-primary metrics appear together in protocol order or not at all, the diagnostic and
the baseline follow them and are never headlines, and no significance language is written.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmark.aggregation import BaselineSummary, MetricSummary
from benchmark.atomic_write import write_directory_atomically
from benchmark.protocol import ProtocolError, load_protocol
from benchmark.report import Outcome, ReportError, RunReport, StratumReport, build_report
from benchmark.scoring import OPGAN, UVR

__all__ = ["main", "render_json", "render_markdown", "run_report"]

logger = logging.getLogger(__name__)

JSON_NAME = "report.json"
MARKDOWN_NAME = "report.md"

_METRIC_LABELS = {
    "reconstruction_snr": "reconstruction SNR",
    "si_snr": "SI-SNR",
    "log_spectral_distance": "log-spectral distance",
}
_EFFECT_LABELS = {"opgan_minus_uvr": "OpGAN minus UVR", "uvr_minus_opgan": "UVR minus OpGAN"}
_EFFECT_SIGN_NOTES = {
    "opgan_minus_uvr": "a positive difference means OpGAN scored higher on this metric",
    "uvr_minus_opgan": (
        "a positive difference means UVR's distance is larger, so OpGAN's output is spectrally "
        "closer to the clean reference"
    ),
}


def _interval_json(low: float, high: float) -> dict[str, float]:
    """A percentile interval as its two bounds."""
    return {"low": low, "high": high}


def _metric_json(summary: MetricSummary) -> dict[str, Any]:
    """One metric with its two systems and the paired difference, each with its interval."""
    return {
        "metric": summary.metric,
        "recordings": summary.recordings,
        "opgan": {
            "mean": summary.opgan_mean,
            "median": summary.opgan_median,
            "interval": _interval_json(summary.opgan_interval.low, summary.opgan_interval.high),
        },
        "uvr": {
            "mean": summary.uvr_mean,
            "median": summary.uvr_median,
            "interval": _interval_json(summary.uvr_interval.low, summary.uvr_interval.high),
        },
        "paired": {
            "effect": summary.paired_effect,
            "mean": summary.paired_mean,
            "median": summary.paired_median,
            "interval": _interval_json(summary.paired_interval.low, summary.paired_interval.high),
        },
    }


def _baseline_json(summary: BaselineSummary) -> dict[str, Any]:
    """The degraded input: its mean, median and interval. It has no paired field."""
    return {
        "series": summary.series.value,
        "metric": summary.metric,
        "recordings": summary.recordings,
        "mean": summary.mean,
        "median": summary.median,
        "interval": _interval_json(summary.interval.low, summary.interval.high),
    }


def _stratum_json(stratum: StratumReport) -> dict[str, Any]:
    """One stratum as the JSON document lists it. Absent numbers are null, never zero."""
    counts = stratum.counts
    failures = stratum.failure_counts
    return {
        "outcome": stratum.outcome.value,
        "gates": dict(stratum.gates),
        "reasons": list(stratum.reasons),
        "counts": None
        if counts is None
        else {
            "recordings": counts.recordings,
            "clusters": counts.clusters,
            "eligible_frames": counts.eligible_frames,
        },
        "coverage": None if stratum.coverage is None else dict(stratum.coverage),
        "failure_counts": None
        if failures is None
        else {system: dict(kinds) for system, kinds in failures.items()},
        "headline": [_metric_json(m) for m in stratum.headline],
        "diagnostic": [_metric_json(m) for m in stratum.diagnostic],
        "baseline": [_baseline_json(b) for b in stratum.baseline],
    }


def render_json(report: RunReport) -> str:
    """The JSON document: the claim scope, the digests, and one entry per stratum."""
    digests = report.digests
    document = {
        "protocol_version": report.protocol_version,
        "claim_scope": report.claim_scope_note,
        "digests": {
            "protocol_sha256": digests.protocol_sha256,
            "universe_sha256": dict(digests.universe_sha256),
            "archive_sha256": digests.archive_sha256,
            "walk_manifest_sha256": digests.walk_manifest_sha256,
            "audit_trail_sha256": digests.audit_trail_sha256,
            "systems_results_sha256": digests.systems_results_sha256,
        },
        "strata": {s.stratum: _stratum_json(s) for s in report.strata},
    }
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _format_interval(low: float, high: float) -> str:
    """A bootstrap interval in square brackets, to three decimals."""
    return f"[{low:.3f}, {high:.3f}]"


def _metric_lines(summary: MetricSummary) -> list[str]:
    """One co-primary or diagnostic metric: both systems, then the paired difference."""
    label = _METRIC_LABELS[summary.metric]
    effect = _EFFECT_LABELS[summary.paired_effect]
    return [
        f"- {label} (dB). OpGAN mean {summary.opgan_mean:.3f}, median {summary.opgan_median:.3f}, "
        f"95% CI {_format_interval(summary.opgan_interval.low, summary.opgan_interval.high)}. "
        f"UVR mean {summary.uvr_mean:.3f}, median {summary.uvr_median:.3f}, "
        f"95% CI {_format_interval(summary.uvr_interval.low, summary.uvr_interval.high)}.",
        f"  Paired difference ({effect}): {summary.paired_mean:+.3f} dB, "
        f"95% CI {_format_interval(summary.paired_interval.low, summary.paired_interval.high)}; "
        f"{_EFFECT_SIGN_NOTES[summary.paired_effect]}.",
    ]


def _gate_lines(stratum: StratumReport) -> list[str]:
    """Each evaluated gate, by its protocol name, as held or failed."""
    return [f"- {name}: {'held' if held else 'failed'}" for name, held in stratum.gates.items()]


def _count_line(stratum: StratumReport) -> str:
    """The recordings, performer clusters and eligible frames of a stratum that publishes them."""
    counts = stratum.counts
    if counts is None:
        return ""
    return (
        f"Recordings: {counts.recordings}. Performer clusters: {counts.clusters}. "
        f"Eligible frames: {counts.eligible_frames}."
    )


def _coverage_line(stratum: StratumReport) -> str:
    """Coverage per system, to three decimals."""
    coverage = stratum.coverage or {}
    return f"Coverage: OpGAN {coverage.get(OPGAN, 0.0):.3f}, UVR {coverage.get(UVR, 0.0):.3f}."


def _baseline_lines(summaries: Sequence[BaselineSummary]) -> list[str]:
    """The degraded input's mean, median and interval for each metric. It has no paired line."""
    lines = []
    for summary in summaries:
        label = _METRIC_LABELS[summary.metric]
        lines.append(
            f"- {label} (dB), {summary.series.value}: mean {summary.mean:.3f}, "
            f"median {summary.median:.3f}, 95% CI "
            f"{_format_interval(summary.interval.low, summary.interval.high)}."
        )
    return lines


def _comparison_lines(stratum: StratumReport) -> list[str]:
    """A comparison: the three co-primary metrics together, then the diagnostic and the baseline."""
    lines = [_count_line(stratum), _coverage_line(stratum), "", "### Co-primary metrics"]
    for summary in stratum.headline:
        lines.extend(_metric_lines(summary))
    lines.extend(["", "### Diagnostic: shift-tolerant, not a headline"])
    for summary in stratum.diagnostic:
        lines.extend(_metric_lines(summary))
    lines.extend(["", "### Input baseline: degraded input, not a system, no paired difference"])
    lines.extend(_baseline_lines(stratum.baseline))
    return lines


def _reliability_lines(stratum: StratumReport) -> list[str]:
    """A reliability finding: coverage and failure counts by kind, and no metric value."""
    lines = [_count_line(stratum), _coverage_line(stratum), "", "### Failures by kind"]
    for system, kinds in (stratum.failure_counts or {}).items():
        described = ", ".join(f"{kind} {count}" for kind, count in kinds.items()) or "none"
        lines.append(f"- {system}: {described}")
    lines.extend(
        [
            "",
            "The paired quality comparison is withheld: a comparison over the frames that survived "
            "would reward failure.",
        ]
    )
    return lines


def _stratum_lines(stratum: StratumReport) -> list[str]:
    """One stratum's section: its outcome, its gates and reasons, then what it publishes."""
    lines = [f"## {stratum.stratum}", "", f"Outcome: {stratum.outcome.value}.", ""]
    if stratum.gates:
        lines.extend(["Gates:", *_gate_lines(stratum), ""])
    if stratum.reasons:
        lines.extend(["Reasons:", *(f"- {reason}" for reason in stratum.reasons), ""])
    if stratum.outcome is Outcome.COMPARISON:
        lines.extend(_comparison_lines(stratum))
    elif stratum.outcome is Outcome.RELIABILITY_FINDING:
        lines.extend(_reliability_lines(stratum))
    lines.append("")
    return lines


def _digest_lines(report: RunReport) -> list[str]:
    """Every digest the report names. An absent archive digest is written as null."""
    digests = report.digests
    lines = [
        f"- Protocol SHA-256: `{digests.protocol_sha256}`",
        *(
            f"- Candidate universe SHA-256, {name}: `{digest}`"
            for name, digest in digests.universe_sha256.items()
        ),
        f"- Archive SHA-256: `{digests.archive_sha256}`"
        if digests.archive_sha256 is not None
        else "- Archive SHA-256: null",
        f"- Walk manifest SHA-256: `{digests.walk_manifest_sha256}`",
        f"- Audit trail SHA-256: `{digests.audit_trail_sha256}`",
        f"- Systems results SHA-256: `{digests.systems_results_sha256}`",
    ]
    return lines


def render_markdown(report: RunReport) -> str:
    """The Markdown report: the claim scope and digests, then one section per stratum."""
    lines = [
        "# Benchmark report",
        "",
        f"> **Note:** {report.claim_scope_note}",
        "",
        f"Protocol version {report.protocol_version}. Each stratum is reported on its own.",
        "",
        "## Digests",
        "",
        *_digest_lines(report),
        "",
    ]
    for stratum in report.strata:
        lines.extend(_stratum_lines(stratum))
    return "\n".join(lines) + "\n"


def _renderings(report: RunReport) -> tuple[bytes, bytes]:
    """The JSON and the Markdown as UTF-8 bytes, rendered before anything is written.

    Raises:
        ReportError: If a rendering fails. JSON refuses a non-finite number, for example.
    """
    try:
        return render_json(report).encode("utf-8"), render_markdown(report).encode("utf-8")
    except ValueError as error:
        logger.error("Cannot render the report: %s", error)
        raise ReportError(f"the report cannot be rendered: {error}") from error


def write_report(output_dir: Path, report: RunReport) -> None:
    """Write the JSON and the Markdown into a new directory, published together or not at all.

    Both renderings are built before the directory is created, so a rendering failure leaves no
    directory behind.

    Raises:
        ReportError: If the directory exists already, a rendering fails, or the directory cannot be
            written.
    """
    _refuse_existing(output_dir)
    json_bytes, markdown_bytes = _renderings(report)
    try:
        write_directory_atomically(
            output_dir, {JSON_NAME: json_bytes, MARKDOWN_NAME: markdown_bytes}
        )
    except OSError as error:
        logger.error("Cannot write report directory %s: %s", output_dir, error)
        raise ReportError(f"report directory {output_dir} cannot be written: {error}") from error


def _refuse_existing(output_dir: Path) -> None:
    """Refuse a directory that exists, since published reports are never overwritten.

    Raises:
        ReportError: If the output directory exists.
    """
    if output_dir.exists():
        logger.error("Output directory %s exists; reports are never overwritten", output_dir)
        raise ReportError(f"output directory {output_dir} already exists")


def run_report(walk_dir: Path, systems_dir: Path, output_dir: Path) -> RunReport:
    """Build the report from a walk and a systems directory, and write it to a new directory.

    Raises:
        ReportError: If the output exists, or an input cannot be read or is refused.
        ProtocolError: If the protocol does not load.
    """
    _refuse_existing(output_dir)
    report = build_report(walk_dir, systems_dir, load_protocol())
    write_report(output_dir, report)
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The command line: the walk, the systems outputs, and the new output directory."""
    parser = argparse.ArgumentParser(description="Write the benchmark's report.")
    parser.add_argument("--walk-dir", type=Path, required=True)
    parser.add_argument("--systems-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the report from the command line. Returns 0 on success and 1 on a refusal."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    try:
        run_report(args.walk_dir, args.systems_dir, args.output_dir)
    except (ReportError, ProtocolError) as error:
        logger.error("Report not written: %s", error)
        return 1
    print(f"Wrote {args.output_dir / JSON_NAME} and {args.output_dir / MARKDOWN_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
