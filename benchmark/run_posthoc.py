"""The post-hoc analyses command: reads a finished run and writes `posthoc.json` and `posthoc.md`.

Run it from the repository root:

    uv run python -m benchmark.run_posthoc --walk-dir DIR --systems-dir DIR --output-dir NEW_DIR

The output directory must not exist, so a published result is never overwritten. Every stratum
the report publishes as a comparison gets the three analyses in `benchmark.posthoc`. The other
strata get their outcome and nothing else.

Each frame's clean reference is rebuilt the way the systems stage built it, and each source's
output is read from the file the stage wrote. The conditioned pair is then scored by the same
functions the stage used. Full-band log-spectral distance is recomputed as a check, and it must
equal the value recorded in `systems_results.jsonl` exactly. Any mismatch aborts the command
before anything is written.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from benchmark.atomic_write import write_atomically
from benchmark.canonicalisation import CanonicalWaveform
from benchmark.conditioning import InvalidSystemOutput, condition_pair
from benchmark.degradation import degrade_frame, frame_seed
from benchmark.eligibility import frame_floors
from benchmark.eligibility_outcomes import frame_verdicts
from benchmark.frame_scoring import reference_frame
from benchmark.framing import frame
from benchmark.metrics import MetricInputError, log_spectral_distance
from benchmark.posthoc import (
    BAND_METRIC,
    CAPPED_METRICS,
    LABEL,
    LOG_SPECTRAL_DISTANCE,
    ClusterSummary,
    PairedMetric,
    PosthocError,
    RecordingFrames,
    SourceCounts,
    SourceFrames,
    StratumAnalysis,
    analyse_stratum,
    band_log_spectral_distance,
    is_silent,
)
from benchmark.protocol import Protocol, ProtocolError, load_protocol
from benchmark.report import (
    Inputs,
    Outcome,
    ReportError,
    RunReport,
    StratumReport,
    build_report,
    read_inputs,
)
from benchmark.runner import CanonicalFile, CanonicalFileError
from benchmark.scoring import BASELINE, OPGAN, UVR
from benchmark.systems_pipeline import Admission, Partition
from benchmark.systems_stage import HarnessFailure, SystemFailed, read_output

__all__ = ["JSON_NAME", "MARKDOWN_NAME", "main", "render_json", "render_markdown", "run_posthoc"]

logger = logging.getLogger(__name__)

JSON_NAME = "posthoc.json"
MARKDOWN_NAME = "posthoc.md"
# The layout run_systems writes: the degraded inputs, then each engine's outputs. These two names
# are the values of benchmark.run_systems.INPUTS_DIR and OUTPUTS_DIR. They are repeated here
# because importing run_systems loads both engines, and the post-hoc reader needs neither.
INPUTS_DIR = "inputs"
OUTPUTS_DIR = "outputs"
# The suffix each engine appends to its input's stem: backend/app/services/opgan_denoiser.py and
# backend/app/services/denoiser.py.
_ENGINE_SUFFIX = {OPGAN: "_opgan", UVR: "_uvr"}
_SOURCES = (OPGAN, UVR, BASELINE)
_BAND_LABEL = "log-spectral distance, bins 0 to 256 inclusive (0 to 4 kHz)"
_METRIC_LABELS = {
    "reconstruction_snr": "reconstruction SNR",
    "si_snr": "SI-SNR",
    LOG_SPECTRAL_DISTANCE: "log-spectral distance",
    BAND_METRIC: _BAND_LABEL,
}
_EFFECT_NOTES = {
    "opgan_minus_uvr": "OpGAN minus UVR; a positive value means OpGAN scored higher on this metric",
    "uvr_minus_opgan": (
        "UVR minus OpGAN; a positive value means UVR's distance is larger, so OpGAN's output is "
        "closer to the clean reference"
    ),
}
_CLAIM_NOTE = (
    "These analyses were fixed in docs/benchmark-posthoc-analyses.md before any score was read. "
    "They are secondary to the protocol's primary results and change none of them."
)
_RowKey = tuple[str, int, str]


@dataclass(frozen=True)
class _Cut:
    """A recording's complete frames, its recording peak, and its eligible frame indices."""

    frames: npt.NDArray[np.float64]
    peak: float
    eligible: tuple[int, ...]


@dataclass(frozen=True)
class _Measure:
    """One source's measurements on one eligible frame."""

    primary: Mapping[str, float]
    band: float
    capped: Mapping[str, bool]
    silent: bool


def _frame_stem(canonical_sha256: str, index: int) -> str:
    """The file stem of one frame, as the systems stage names its input file."""
    return f"{canonical_sha256}-{index:06d}"


def _source_path(systems_dir: Path, source: str, stem: str) -> Path:
    """Where a source's file for a frame lives: the degraded input, or the engine's output."""
    if source == BASELINE:
        return systems_dir / INPUTS_DIR / f"{stem}.wav"
    return systems_dir / OUTPUTS_DIR / source / f"{stem}{_ENGINE_SUFFIX[source]}.wav"


def _load_waveform(admission: Admission) -> CanonicalWaveform:
    """The canonical waveform of a recording, with its bytes checked against the digest.

    Raises:
        PosthocError: If the canonical file fails its check.
    """
    canonical = CanonicalFile(
        path=admission.canonical_path,
        sha256=admission.canonical_sha256,
        samples=admission.samples,
        raw_sha256=admission.raw_sha256,
        decoded_pcm_sha256=admission.decoded_pcm_sha256,
    )
    try:
        samples = canonical.load()
    except CanonicalFileError as error:
        logger.error("Canonical file for %s failed its check: %s", admission.identifier, error)
        raise PosthocError(
            f"canonical file for {admission.identifier} failed its check: {error}"
        ) from error
    return CanonicalWaveform(samples=samples, sha256=admission.canonical_sha256)


def _cut(admission: Admission, protocol: Protocol) -> _Cut:
    """Frame a recording and find its eligible frames, with the same functions the stage uses.

    Raises:
        PosthocError: If the recording has no eligible frame.
    """
    waveform = _load_waveform(admission)
    framed = frame(waveform, protocol.framing)
    verdicts = frame_verdicts(framed, frame_floors(framed, protocol.eligibility), protocol.metrics)
    eligible = tuple(verdict.index for verdict in verdicts if verdict.eligible)
    if not eligible:
        logger.error("Recording %s has no eligible frame", admission.identifier)
        raise PosthocError(f"recording {admission.identifier} has no eligible frame")
    return _Cut(framed.frames, float(np.max(np.abs(waveform.samples))), eligible)


def _rows_by_key(
    rows: Sequence[Mapping[str, Any]], stratum: str
) -> dict[_RowKey, Mapping[str, Any]]:
    """A stratum's result rows keyed by recording, frame and source.

    Raises:
        PosthocError: If two rows share a key.
    """
    indexed: dict[_RowKey, Mapping[str, Any]] = {}
    for row in rows:
        if row.get("stratum") != stratum:
            continue
        key = (str(row["canonical_sha256"]), int(row["frame_index"]), str(row["system"]))
        if key in indexed:
            logger.error("Systems results repeat %s frame %d for %s", key[0], key[1], key[2])
            raise PosthocError(f"systems results repeat {key[0]} frame {key[1]} for {key[2]}")
        indexed[key] = row
    return indexed


def _check_coverage(cuts: Mapping[str, _Cut], rows: Mapping[_RowKey, Mapping[str, Any]]) -> None:
    """The results must cover exactly the recomputed eligible frames of every source.

    Raises:
        PosthocError: If a frame is missing, or a result names a frame that is not eligible.
    """
    expected = {
        (sha, index, source)
        for sha, cut in cuts.items()
        for index in cut.eligible
        for source in _SOURCES
    }
    if set(rows) != expected:
        logger.error(
            "Systems results cover %d frame sources and %d were expected",
            len(rows),
            len(expected),
        )
        raise PosthocError("systems results do not cover exactly the recomputed eligible frames")


def _read_samples(path: Path, sample_rate: int) -> npt.NDArray[np.float64]:
    """A source file's samples, read as the systems stage reads an output.

    Raises:
        PosthocError: If the file cannot be read or is not mono at the conditioning rate.
    """
    try:
        output = read_output(path, sample_rate)
    except HarnessFailure as error:
        logger.error("Cannot read %s: %s", path, error)
        raise PosthocError(f"cannot read {path}: {error}") from error
    if isinstance(output, SystemFailed):
        logger.error("File %s is not the expected format: %s", path, output.reason)
        raise PosthocError(f"file {path} is not the expected format: {output.reason}")
    return output.samples


def _measure(
    source: str,
    path: Path,
    reference: npt.NDArray[np.float64],
    row: Mapping[str, Any],
    protocol: Protocol,
) -> _Measure:
    """One source's values on one frame, after checking its full-band distance against the record.

    Raises:
        PosthocError: If the file does not condition, the pair cannot be scored, or the recomputed
            full-band distance differs from the recorded value.
    """
    if row.get("outcome") != "scored":
        logger.error("Systems results hold no scored %s result for %s", source, path.name)
        raise PosthocError(f"{source} has no scored result for {path.name}")
    estimate = _read_samples(path, protocol.conditioning.sample_rate)
    conditioned = condition_pair(reference, estimate, protocol.conditioning)
    if isinstance(conditioned, InvalidSystemOutput):
        logger.error("File %s does not condition: %s", path, conditioned.reason)
        raise PosthocError(f"file {path} does not condition: {conditioned.reason}")
    config = protocol.metrics.log_spectral_distance
    try:
        full = log_spectral_distance(conditioned.reference, conditioned.estimate, config).value
        band = band_log_spectral_distance(conditioned.reference, conditioned.estimate, config)
    except MetricInputError as error:
        logger.error("Pair for %s cannot be scored: %s", path, error)
        raise PosthocError(f"pair for {path} cannot be scored: {error}") from error
    recorded = float(row["results"][LOG_SPECTRAL_DISTANCE]["value"])
    if full != recorded:
        logger.error("Full-band distance of %s is %r, recorded as %r", path, full, recorded)
        raise PosthocError(f"full-band distance of {path} differs from the recorded value")
    return _Measure(
        primary={name: float(row["results"][name]["value"]) for name in protocol.metrics.primary},
        band=band,
        capped={name: bool(row["results"][name]["capped"]) for name in CAPPED_METRICS},
        silent=is_silent(conditioned.estimate),
    )


def _frame_measures(
    canonical_sha256: str,
    index: int,
    cut: _Cut,
    rows: Mapping[_RowKey, Mapping[str, Any]],
    systems_dir: Path,
    protocol: Protocol,
) -> dict[str, _Measure]:
    """Every source's measurements on one eligible frame, against the rebuilt reference."""
    clean = cut.frames[index]
    degraded = degrade_frame(
        clean,
        recording_peak=cut.peak,
        sample_rate=protocol.canonicalisation.resample.target_sample_rate,
        ranges=protocol.degradation.ranges,
        seed=frame_seed(protocol.degradation.seed, canonical_sha256, index),
    )
    reference = reference_frame(clean, degraded.gain)
    stem = _frame_stem(canonical_sha256, index)
    return {
        source: _measure(
            source,
            _source_path(systems_dir, source, stem),
            reference,
            rows[(canonical_sha256, index, source)],
            protocol,
        )
        for source in _SOURCES
    }


def _source_frames(measures: Sequence[_Measure], protocol: Protocol) -> SourceFrames:
    """One source's per-frame values, arranged by metric, in frame order."""
    return SourceFrames(
        primary={name: [m.primary[name] for m in measures] for name in protocol.metrics.primary},
        band=[m.band for m in measures],
        capped={name: [m.capped[name] for m in measures] for name in CAPPED_METRICS},
        silent=[m.silent for m in measures],
    )


def _recording_frames(
    admission: Admission,
    cut: _Cut,
    rows: Mapping[_RowKey, Mapping[str, Any]],
    systems_dir: Path,
    protocol: Protocol,
) -> RecordingFrames:
    """One test recording's frames for every source."""
    per_frame = [
        _frame_measures(admission.canonical_sha256, index, cut, rows, systems_dir, protocol)
        for index in cut.eligible
    ]
    sources = {
        source: _source_frames([measures[source] for measures in per_frame], protocol)
        for source in _SOURCES
    }
    return RecordingFrames(admission.identifier, admission.cluster, sources)


def _stratum_frames(
    stratum: str, inputs: Inputs, systems_dir: Path, protocol: Protocol
) -> tuple[RecordingFrames, ...]:
    """Every test recording of one stratum, with its frames measured.

    Raises:
        PosthocError: If the systems results do not cover the recomputed frames, or a file fails.
    """
    admissions = [
        a
        for a in inputs.manifest.admissions
        if a.stratum == stratum and a.partition is Partition.TEST
    ]
    cuts = {a.canonical_sha256: _cut(a, protocol) for a in admissions}
    rows = _rows_by_key(inputs.results, stratum)
    _check_coverage(cuts, rows)
    return tuple(
        _recording_frames(a, cuts[a.canonical_sha256], rows, systems_dir, protocol)
        for a in admissions
    )


def _cluster_json(summary: ClusterSummary, label: str) -> dict[str, Any]:
    """A cluster-level summary as the JSON document lists it."""
    return {
        "label": LABEL,
        "metric": label,
        "clusters": summary.clusters,
        "mean": summary.mean,
        "sd": summary.sd,
        "interval": {"low": summary.low, "high": summary.high},
        "positive": summary.positive,
        "negative": summary.negative,
        "zero": summary.zero,
    }


def _paired_json(metric: PairedMetric) -> dict[str, Any]:
    """One paired metric: each source's mean and median, the difference, and its summary."""
    return {
        "label": LABEL,
        "metric": metric.metric,
        "effect": metric.effect,
        "recordings": metric.recordings,
        "sources": {
            source: {"mean": means.mean, "median": means.median}
            for source, means in metric.sources.items()
        },
        "paired": {
            "mean": metric.paired_mean,
            "median": metric.paired_median,
            "cluster_summary": _cluster_json(metric.clusters, metric.metric),
        },
    }


def _counts_json(counts: SourceCounts) -> dict[str, int]:
    """One source's frame counts."""
    return {
        "eligible_frames": counts.eligible_frames,
        "capped_reconstruction_snr": counts.capped_reconstruction_snr,
        "capped_si_snr": counts.capped_si_snr,
        "silent": counts.silent,
    }


def _analysis_json(analysis: StratumAnalysis) -> dict[str, Any]:
    """A comparison stratum's three analyses as the JSON document lists them."""
    return {
        "outcome": Outcome.COMPARISON.value,
        "label": LABEL,
        "recordings": analysis.recordings,
        "clusters": analysis.clusters,
        "analysis_1": {
            metric.metric: _cluster_json(metric.clusters, metric.metric)
            for metric in (*analysis.primary, analysis.band)
        },
        "analysis_2": _paired_json(analysis.band),
        "analysis_3": {source: _counts_json(counts) for source, counts in analysis.counts.items()},
    }


def _stratum_json(stratum: StratumReport, analysis: StratumAnalysis | None) -> dict[str, Any]:
    """One stratum's entry. A stratum without a comparison carries its outcome and no number."""
    if analysis is None:
        return {"outcome": stratum.outcome.value}
    return _analysis_json(analysis)


def _digests_json(report: RunReport) -> dict[str, Any]:
    """The digests of the run that the post-hoc numbers were read from."""
    digests = report.digests
    return {
        "protocol_sha256": digests.protocol_sha256,
        "walk_manifest_sha256": digests.walk_manifest_sha256,
        "audit_trail_sha256": digests.audit_trail_sha256,
        "systems_results_sha256": digests.systems_results_sha256,
    }


def render_json(report: RunReport, analyses: Mapping[str, StratumAnalysis]) -> str:
    """The JSON document: the label, the claim scope, the digests, and one entry per stratum.

    Raises:
        ValueError: If a number is not finite, since JSON cannot carry it.
    """
    document = {
        "label": LABEL,
        "protocol_version": report.protocol_version,
        "claim_scope": _CLAIM_NOTE,
        "run_digests": _digests_json(report),
        "strata": {s.stratum: _stratum_json(s, analyses.get(s.stratum)) for s in report.strata},
    }
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _interval_text(low: float, high: float) -> str:
    """A 95% interval in square brackets, to three decimals."""
    return f"[{low:.3f}, {high:.3f}]"


def _cluster_line(metric: PairedMetric) -> str:
    """One paired metric's cluster-level summary as a Markdown bullet."""
    summary = metric.clusters
    return (
        f"- {_METRIC_LABELS[metric.metric]}. Paired difference ({_EFFECT_NOTES[metric.effect]}). "
        f"Clusters k = {summary.clusters}. Mean of the cluster means {summary.mean:+.3f} dB, "
        f"95% t-interval {_interval_text(summary.low, summary.high)} with "
        f"{summary.clusters - 1} degrees of freedom. Cluster values: positive {summary.positive}, "
        f"negative {summary.negative}, zero {summary.zero}."
    )


def _analysis_one_lines(analysis: StratumAnalysis) -> list[str]:
    """Analysis 1: the cluster-level summary of the three primary metrics and the band."""
    lines = [
        "### Analysis 1: cluster-level paired summary",
        "",
        f"All numbers in this section are {LABEL}.",
        "",
    ]
    lines.extend(_cluster_line(metric) for metric in (*analysis.primary, analysis.band))
    return [*lines, ""]


def _analysis_two_lines(analysis: StratumAnalysis) -> list[str]:
    """Analysis 2: the band-limited distance per source, its paired difference, and that summary."""
    band = analysis.band
    lines = [
        "### Analysis 2: log-spectral distance below 4 kHz",
        "",
        f"All numbers in this section are {LABEL}.",
        "",
    ]
    for source, label in ((OPGAN, "OpGAN"), (UVR, "UVR"), (BASELINE, "degraded input")):
        means = band.sources[source]
        lines.append(
            f"- {label} (dB), over {band.recordings} recordings: mean {means.mean:.3f}, "
            f"median {means.median:.3f}."
        )
    lines.append(
        f"- Paired difference ({_EFFECT_NOTES[band.effect]}), over {band.recordings} recordings: "
        f"mean {band.paired_mean:+.3f}, median {band.paired_median:+.3f}."
    )
    lines.append(_cluster_line(band))
    return [*lines, ""]


def _analysis_three_lines(analysis: StratumAnalysis) -> list[str]:
    """Analysis 3: eligible frames, frames at each cap, and silent outputs, per source."""
    lines = [
        "### Analysis 3: capped and silent outputs",
        "",
        f"All numbers in this section are {LABEL}.",
        "",
    ]
    for source, label in ((OPGAN, "OpGAN"), (UVR, "UVR"), (BASELINE, "degraded input")):
        counts = analysis.counts[source]
        lines.append(
            f"- {label}: eligible frames {counts.eligible_frames}; frames at the reconstruction "
            f"SNR cap {counts.capped_reconstruction_snr}; frames at the SI-SNR cap "
            f"{counts.capped_si_snr}; silent outputs {counts.silent}."
        )
    return [*lines, ""]


def _stratum_markdown(stratum: StratumReport, analysis: StratumAnalysis | None) -> list[str]:
    """One stratum's section. Without a comparison, the section names the outcome and no number."""
    lines = [f"## {stratum.stratum}", ""]
    if analysis is None:
        lines.extend(
            [
                f"Outcome: {stratum.outcome.value}. The report does not publish this stratum as a "
                "comparison, so no post-hoc number is reported for it.",
                "",
            ]
        )
        return lines
    lines.extend(
        [
            f"Outcome: comparison. Recordings: {analysis.recordings}. Performer clusters: "
            f"{analysis.clusters}.",
            "",
        ]
    )
    lines.extend(_analysis_one_lines(analysis))
    lines.extend(_analysis_two_lines(analysis))
    lines.extend(_analysis_three_lines(analysis))
    return lines


def _digest_lines(report: RunReport) -> list[str]:
    """The run digests the post-hoc numbers were read from, as Markdown bullets."""
    digests = report.digests
    return [
        f"- Protocol SHA-256: `{digests.protocol_sha256}`",
        f"- Walk manifest SHA-256: `{digests.walk_manifest_sha256}`",
        f"- Audit trail SHA-256: `{digests.audit_trail_sha256}`",
        f"- Systems results SHA-256: `{digests.systems_results_sha256}`",
    ]


def render_markdown(report: RunReport, analyses: Mapping[str, StratumAnalysis]) -> str:
    """The Markdown document: the label and claim scope, the digests, then one section per stratum."""
    lines = [
        "# Post-hoc analyses",
        "",
        f"> **Note:** {_CLAIM_NOTE}",
        "",
        f"Every number in this document is {LABEL}.",
        "",
        "## Run digests",
        "",
        *_digest_lines(report),
        "",
    ]
    for stratum in report.strata:
        lines.extend(_stratum_markdown(stratum, analyses.get(stratum.stratum)))
    return "\n".join(lines) + "\n"


def _write(output_dir: Path, json_bytes: bytes, markdown_bytes: bytes) -> None:
    """Create the new directory and write both files, each through an atomic replace."""
    try:
        output_dir.mkdir(parents=True)
    except OSError as error:
        logger.error("Cannot create post-hoc directory %s: %s", output_dir, error)
        raise ReportError(f"post-hoc directory {output_dir} cannot be created: {error}") from error
    write_atomically(output_dir / JSON_NAME, json_bytes)
    write_atomically(output_dir / MARKDOWN_NAME, markdown_bytes)


def _refuse_existing(output_dir: Path) -> None:
    """Refuse a directory that exists, since published results are never overwritten.

    Raises:
        ReportError: If the output directory exists.
    """
    if output_dir.exists():
        logger.error(
            "Output directory %s exists; post-hoc results are never overwritten", output_dir
        )
        raise ReportError(f"output directory {output_dir} already exists")


def run_posthoc(walk_dir: Path, systems_dir: Path, output_dir: Path) -> RunReport:
    """Compute the post-hoc analyses for every comparison stratum and write them to a new directory.

    Both renderings are built before the directory is created, so a failure leaves none behind.

    Raises:
        ReportError: If the output exists, or the report's inputs cannot be read.
        ProtocolError: If the protocol does not load.
        PosthocError: If a file does not match the systems results, or an analysis cannot run.
    """
    _refuse_existing(output_dir)
    protocol = load_protocol()
    report = build_report(walk_dir, systems_dir, protocol)
    inputs = read_inputs(walk_dir, systems_dir, protocol)
    analyses = {
        s.stratum: analyse_stratum(
            _stratum_frames(s.stratum, inputs, systems_dir, protocol), protocol
        )
        for s in report.strata
        if s.outcome is Outcome.COMPARISON
    }
    try:
        json_bytes = render_json(report, analyses).encode("utf-8")
        markdown_bytes = render_markdown(report, analyses).encode("utf-8")
    except ValueError as error:
        logger.error("Cannot render the post-hoc results: %s", error)
        raise PosthocError(f"the post-hoc results cannot be rendered: {error}") from error
    _write(output_dir, json_bytes, markdown_bytes)
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """The command line: the walk, the systems outputs, and the new output directory."""
    parser = argparse.ArgumentParser(description="Write the benchmark's post-hoc analyses.")
    parser.add_argument("--walk-dir", type=Path, required=True)
    parser.add_argument("--systems-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the post-hoc analyses from the command line. Returns 0 on success and 1 on a refusal."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    try:
        run_posthoc(args.walk_dir, args.systems_dir, args.output_dir)
    except (ReportError, ProtocolError, PosthocError) as error:
        logger.error("Post-hoc analyses not written: %s", error)
        return 1
    print(f"Wrote {args.output_dir / JSON_NAME} and {args.output_dir / MARKDOWN_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
