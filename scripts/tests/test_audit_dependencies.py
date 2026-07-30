"""Tests for the dependency-audit coverage assertion.

This module decides whether CI may claim every dependency was accounted for, so its
failure paths matter more than its success path. There is nothing to mock: the earlier
network fallback was removed by pinning the one unindexed package to an auditable
version, so the whole check is now pure comparison over local inputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_dependencies as audit

CLEAN_ENTRY = {"name": "numpy", "version": "2.3.5", "vulns": []}


def write(path: Path, text: str) -> Path:
    """Write text to a path and return it."""
    path.write_text(text)
    return path


def report(path: Path, dependencies: list[dict]) -> Path:
    """Write a pip-audit style JSON report."""
    return write(path, json.dumps({"dependencies": dependencies, "fixes": []}))


class TestParsePinned:
    """Requested identities are the expected population, so ambiguity is fatal."""

    def test_canonicalises_separators(self, tmp_path: Path) -> None:
        path = write(tmp_path / "r.txt", "Foo_Bar==1.0\nbaz.qux==2.0\n")
        assert audit.parse_pinned(path) == {"foo-bar": "1.0", "baz-qux": "2.0"}

    def test_ignores_only_blank_lines_and_comments(self, tmp_path: Path) -> None:
        path = write(tmp_path / "r.txt", "numpy==2.3.5\n\n# a comment\n")
        assert audit.parse_pinned(path) == {"numpy": "2.3.5"}

    @pytest.mark.parametrize(
        "line",
        [
            "foo>=1.0",
            "foo @ https://example.invalid/foo.whl",
            "-e .",
            "foo==1.0 ; python_version < '3.13'",
            "garbage",
        ],
    )
    def test_every_unsupported_representation_is_rejected(self, tmp_path: Path, line: str) -> None:
        """Skipping a line it cannot parse would shrink the expected population before
        the comparison meant to prove that population was covered. An earlier version
        did exactly that, and a test asserted the behaviour, which is how a fail-open
        becomes policy."""
        path = write(tmp_path / "r.txt", f"numpy==2.3.5\n{line}\n")
        with pytest.raises(ValueError, match="unsupported dependency representation"):
            audit.parse_pinned(path)

    def test_duplicate_canonical_name_is_rejected(self, tmp_path: Path) -> None:
        """foo_bar and foo-bar are one package. Silently keeping the last one would let
        the expected version be decided by line order."""
        path = write(tmp_path / "r.txt", "foo_bar==1.0\nfoo-bar==2.0\n")
        with pytest.raises(ValueError, match="duplicate canonical package"):
            audit.parse_pinned(path)


class TestParseReport:
    """An ambiguous report cannot establish which identity was covered."""

    def test_canonicalises_names(self, tmp_path: Path) -> None:
        path = report(tmp_path / "p.json", [{"name": "Foo_Bar", "version": "1.0"}])
        assert set(audit.parse_report(path)) == {"foo-bar"}

    def test_duplicate_canonical_name_is_rejected(self, tmp_path: Path) -> None:
        path = report(
            tmp_path / "p.json",
            [{"name": "foo_bar", "version": "1.0"}, {"name": "foo-bar", "version": "2.0"}],
        )
        with pytest.raises(ValueError, match="duplicate canonical package"):
            audit.parse_report(path)

    def test_missing_dependencies_key_is_empty_not_an_error(self, tmp_path: Path) -> None:
        """An empty report is a real pip-audit response and must be treated as covering
        nothing, rather than crashing before the coverage comparison runs."""
        assert audit.parse_report(write(tmp_path / "p.json", "{}")) == {}


class TestCheckCoverage:
    """The audited population must equal the requested population, exactly."""

    def test_matching_populations_pass(self) -> None:
        assert audit.check_coverage({"numpy": "2.3.5"}, {"numpy": CLEAN_ENTRY}) == []

    def test_absent_from_the_report_fails(self) -> None:
        """The defect this module exists for: a package pip-audit omits entirely leaves
        no entry to inspect, so scanning for skip_reason alone would pass."""
        gaps = audit.check_coverage({"packaging": "26.1"}, {})
        assert len(gaps) == 1
        assert "absent from the audit report" in gaps[0]

    def test_skip_reason_fails(self) -> None:
        entry = {"name": "torch", "version": "2.12.1+cpu", "skip_reason": "not on PyPI"}
        gaps = audit.check_coverage({"torch": "2.12.1"}, {"torch": entry})
        assert len(gaps) == 1
        assert "not on PyPI" in gaps[0]

    def test_reported_at_a_different_version_fails(self) -> None:
        """Name presence is not identity. For a vulnerability scanner the version is the
        security-relevant half, so a report for 2.3.4 does not audit a request for
        2.3.5."""
        entry = {"name": "numpy", "version": "2.3.4", "vulns": []}
        gaps = audit.check_coverage({"numpy": "2.3.5"}, {"numpy": entry})
        assert len(gaps) == 1
        assert "requested 2.3.5, report covered 2.3.4" in gaps[0]

    def test_reported_without_a_version_fails(self) -> None:
        gaps = audit.check_coverage({"numpy": "2.3.5"}, {"numpy": {"name": "numpy"}})
        assert len(gaps) == 1
        assert "carries no version" in gaps[0]

    def test_reported_but_never_requested_fails(self) -> None:
        """The audit runs with --no-deps, so an unrequested identity means the tool's
        behaviour changed and the comparison no longer means what it claims."""
        gaps = audit.check_coverage({}, {"numpy": CLEAN_ENTRY})
        assert len(gaps) == 1
        assert "reported but never requested" in gaps[0]

    def test_every_discrepancy_is_reported_not_just_the_first(self) -> None:
        pinned = {"a": "1", "b": "2", "c": "3"}
        gaps = audit.check_coverage(pinned, {"a": {"name": "a", "version": "1"}})
        assert len(gaps) == 2


class TestMain:
    """Exit status is what the gate reads."""

    def test_matching_populations_exit_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requirements = write(tmp_path / "r.txt", "numpy==2.3.5\n")
        audit_report = report(tmp_path / "p.json", [CLEAN_ENTRY])
        monkeypatch.setattr(
            "sys.argv",
            ["x", "--requirements", str(requirements), "--report", str(audit_report)],
        )
        assert audit.main() == 0

    def test_ambiguous_identity_exits_one_without_a_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        requirements = write(tmp_path / "r.txt", "foo_bar==1.0\nfoo-bar==2.0\n")
        audit_report = report(tmp_path / "p.json", [])
        monkeypatch.setattr(
            "sys.argv",
            ["x", "--requirements", str(requirements), "--report", str(audit_report)],
        )
        assert audit.main() == 1

    def test_coverage_gap_exits_one(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        requirements = write(tmp_path / "r.txt", "numpy==2.3.5\npackaging==26.1\n")
        audit_report = report(tmp_path / "p.json", [CLEAN_ENTRY])
        monkeypatch.setattr(
            "sys.argv",
            ["x", "--requirements", str(requirements), "--report", str(audit_report)],
        )
        assert audit.main() == 1
