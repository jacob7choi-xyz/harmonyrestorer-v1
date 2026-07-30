"""Assert that the dependency audit covered every dependency it was asked about.

pip-audit exiting zero does not establish coverage. It reports some packages as
skipped, and it omits others from its report entirely, leaving no entry to inspect.
Both were observed here: `torch 2.12.1+cpu` was skipped because a PEP 440 local
version identifier does not exist on PyPI, and `packaging 26.2` was dropped with no
entry at all, which is why the lock constrains packaging to a version the advisory
service actually has a record of.

So coverage is asserted by identity, in both directions. Every requested dependency
must appear in the report at the same version with no skip reason, and the report may
not contain identities that were never requested. Name presence is not identity: for a
vulnerability scanner the version is the security-relevant half.

The reverse direction is not decoration. It caught the audit resolving a different
distribution than the deployed one: a fabricated base-version torch requirement, fed to
a pip-audit run that still resolved despite --no-deps, pulled the CUDA closure of
generic PyPI torch on Linux and reported 19 packages that were neither installed nor
requested. The run now passes --disable-pip so nothing is resolved. Requested versions
are runtime identities except where the gate maps one deliberately for advisory lookup,
which it documents at the call site.

There is deliberately no exception mechanism. An earlier version declared omitted
identities and covered them through OSV and PyPI lookups, which made every green run
depend on two live external services in order to work around one package's newest
release being unindexed. Pinning that package to an auditable version removed the
problem instead of hardening the workaround. If another package is ever omitted, the
first thing to try is the same move, not a new exception path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from packaging.utils import canonicalize_name

PINNED_REQUIREMENT = re.compile(r"([A-Za-z0-9._-]+)==([^\s;]+)")


def parse_pinned(requirements_path: Path) -> dict[str, str]:
    """Read a fully pinned requirements file into canonical name to version.

    Args:
        requirements_path: File of `name==version` lines.

    Returns:
        Mapping of PEP 503 canonical project name to version string.

    Raises:
        ValueError: If a line is not an exact pinned requirement, or two lines
            canonicalise to the same project.
    """
    pinned: dict[str, str] = {}
    for raw in requirements_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = PINNED_REQUIREMENT.fullmatch(line)
        if not match:
            # Skipping a line the parser cannot read would shrink the expected
            # population before the comparison meant to prove that population was
            # covered, which is the defect this module exists to catch.
            raise ValueError(
                f"unsupported dependency representation in {requirements_path}: {line!r}"
            )
        name = canonicalize_name(match.group(1))
        if name in pinned:
            # foo_bar and foo-bar canonicalise together. Silently keeping the last one
            # would let line order decide the expected version.
            raise ValueError(
                f"duplicate canonical package {name!r} in {requirements_path}: "
                f"{pinned[name]} and {match.group(2)}"
            )
        pinned[name] = match.group(2)
    return pinned


def parse_report(report_path: Path) -> dict[str, dict]:
    """Read a pip-audit JSON report into canonical name to dependency entry.

    Raises:
        ValueError: If two entries canonicalise to the same project, since an ambiguous
            report cannot establish which identity was covered.
    """
    report = json.loads(report_path.read_text())
    reported: dict[str, dict] = {}
    for dep in report.get("dependencies", []):
        name = canonicalize_name(dep["name"])
        if name in reported:
            raise ValueError(
                f"duplicate canonical package {name!r} in {report_path}: an ambiguous "
                "report cannot establish which identity was covered"
            )
        reported[name] = dep
    return reported


def check_coverage(pinned: dict[str, str], reported: dict[str, dict]) -> list[str]:
    """Find every way the audited population differs from the requested one.

    Args:
        pinned: Requested packages, canonical name to version.
        reported: pip-audit entries, canonical name to entry.

    Returns:
        Human-readable discrepancies, empty when the two populations match exactly.
    """
    gaps: list[str] = []

    for name, version in sorted(pinned.items()):
        entry = reported.get(name)
        if entry is None:
            gaps.append(f"{name} {version}: requested but absent from the audit report")
            continue
        if entry.get("skip_reason"):
            gaps.append(f"{name} {entry.get('version')}: {entry['skip_reason']}")
            continue
        covered = entry.get("version")
        if covered is None:
            gaps.append(f"{name} {version}: report entry carries no version")
        elif covered != version:
            gaps.append(f"{name}: requested {version}, report covered {covered}")

    # The audit runs with --no-deps, so an identity nobody asked about means the tool's
    # behaviour has changed and this comparison no longer means what it claims.
    for name in sorted(set(reported) - set(pinned)):
        gaps.append(f"{name} {reported[name].get('version')}: reported but never requested")

    return gaps


def main() -> int:
    """Compare requested dependencies against what the audit actually covered."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    try:
        pinned = parse_pinned(args.requirements)
        reported = parse_report(args.report)
    except ValueError as e:
        print(f"ambiguous dependency identity: {e}", file=sys.stderr)
        return 1

    gaps = check_coverage(pinned, reported)
    if gaps:
        print("coverage gap, not a pass:", file=sys.stderr)
        for gap in gaps:
            print(f"  {gap}", file=sys.stderr)
        return 1

    print(f"all {len(pinned)} requested dependencies audited at the requested versions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
