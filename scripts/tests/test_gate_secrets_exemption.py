"""Tests for the secrets-scan exemptions in scripts/gate_python.sh.

The gate passes one regular expression to detect-secrets as --exclude-lines. These tests
read that expression from the script itself, so they check the value the gate uses rather
than a copy of it.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
GATE = REPO / "scripts" / "gate_python.sh"
UNIVERSE = REPO / "benchmark" / "listening" / "manifests" / "universe_v1.json"
_DEFINITIONS = r'sed -n "/^DIGEST_FIELDS=/,/^SECRETS_EXEMPTION=/p" "$1"'


def _gate_exemption() -> str:
    """The SECRETS_EXEMPTION value, evaluated by bash from the gate's own definitions."""
    command = f'eval "$({_DEFINITIONS})"; printf %s "$SECRETS_EXEMPTION"'
    result = subprocess.run(
        ["bash", "-c", command, "bash", str(GATE)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout, "SECRETS_EXEMPTION is empty or missing from the gate"
    return result.stdout


@pytest.fixture(scope="module")
def exemption() -> re.Pattern[str]:
    """The compiled exemption, as detect-secrets applies it with re.search per line."""
    return re.compile(_gate_exemption())


def _identifier_lines() -> list[str]:
    """Every line of the frozen universe manifest that holds an "identifier" key."""
    lines = UNIVERSE.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line.lstrip().startswith('"identifier":')]


def test_every_universe_identifier_line_is_exempt(exemption: re.Pattern[str]) -> None:
    """Each identifier line in the manifest matches, so the scan does not flag it."""
    lines = _identifier_lines()
    count = json.loads(UNIVERSE.read_text(encoding="utf-8"))["listed"]
    assert len(lines) == count
    assert [line for line in lines if not exemption.search(line)] == []


@pytest.mark.parametrize(
    "line",
    [
        '      "identifier": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",',  # pragma: allowlist secret
        '      "identifier": "78_a-dream_gbia0045837a_ghp16C7e42F292c6912E7710c838347Ae178B4a",',
        '      "identifier": "78_A-Dream_John-McCormack_gbia0045837a",',
        '      "identifier": "78_a-dream_gbia0045837a", "token": "x"',
        '      "token": "78_a-dream_john-mccormack-chas-b-cory-j-c-bartlett_gbia0045837a",',
        '      "api_key": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",',  # pragma: allowlist secret
    ],
)
def test_other_values_and_keys_are_not_exempt(exemption: re.Pattern[str], line: str) -> None:
    """A different value shape, extra content, or another key is still scanned."""
    assert exemption.search(line) is None


def test_existing_digest_exemptions_are_unchanged(exemption: re.Pattern[str]) -> None:
    """The digest, md5 and sha1 exemptions still apply at their own lengths only."""
    digest = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"  # pragma: allowlist secret
    assert exemption.search(f'"sha256": "{digest}"')
    assert exemption.search(f'"md5": "{digest[:32]}"')
    assert exemption.search(f'"sha1": "{digest[:40]}"')
    assert exemption.search(f'"md5": "{digest}"') is None


def _scan(path: Path) -> int:
    """Run detect-secrets on one file with the gate's exemption; return its exit status."""
    hook = Path(sys.executable).with_name("detect-secrets-hook")
    command = [str(hook), "--exclude-lines", _gate_exemption(), str(path)]
    return subprocess.run(command, capture_output=True, text=True, check=False).returncode


def test_scanner_passes_a_real_identifier_and_flags_a_planted_secret(tmp_path: Path) -> None:
    """End to end: the real identifier passes, a high-entropy value under it still fails."""
    real = tmp_path / "real.json"
    real.write_text(_identifier_lines()[0] + "\n", encoding="utf-8")
    planted = tmp_path / "planted.json"
    secret = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"  # pragma: allowlist secret
    planted.write_text(f'      "identifier": "{secret}",\n', encoding="utf-8")
    assert _scan(real) == 0
    assert _scan(planted) == 1
