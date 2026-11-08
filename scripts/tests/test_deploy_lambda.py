"""Tests for the Lambda deploy script.

The script runs with stand-ins for aws, docker and curl on PATH. Each stand-in logs its
arguments and answers from the scenario's settings, so a test can check which calls were
made, in which order, without touching AWS. A failure path is checked both for the
failure and for the calls it must not make.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "deploy_lambda.sh"
DIGEST = "sha256:" + "a" * 64
OTHER_DIGEST = "sha256:" + "b" * 64

AWS_STUB = r"""#!/usr/bin/env bash
echo "aws $*" >> "$STUB_LOG"
case "$1 $2" in
  "sts get-caller-identity") echo 123456789012 ;;
  "ecr describe-images") [ "$STUB_ECR_HAS_DIGEST" = 1 ] || exit 254 ;;
  "ecr get-login-password") echo password ;;
  "lambda get-alias") echo 4 ;;
  "lambda update-function-code") echo Successful ;;
  "lambda publish-version") echo "$STUB_NEW_VERSION" ;;
  "lambda invoke")
    for arg in "$@"; do case "$arg" in *.out) out="$arg" ;; esac; done
    printf '{"statusCode":200}\0\0\0\0\0\0\0\0{"status":"%s"}' "$STUB_WARM_STATUS" > "$out"
    echo 200 ;;
  "lambda update-alias")
    for arg in "$@"; do [ "$prev" = --function-version ] && target="$arg"; prev="$arg"; done
    [ "$target" = "$STUB_FAILING_ALIAS_TARGET" ] && exit 254
    echo "$target" ;;
esac
"""

DOCKER_STUB = r"""#!/usr/bin/env bash
echo "docker $*" >> "$STUB_LOG"
# Like the real command, login reads the password from stdin
if [ "$1" = login ]; then cat > /dev/null; fi
if [ "$1" = push ]; then echo "abc: digest: $STUB_PUSHED_DIGEST size: 1"; fi
"""

CURL_STUB = r"""#!/usr/bin/env bash
echo "curl $*" >> "$STUB_LOG"
for arg in "$@"; do
  case "$arg" in
    */health) echo "{\"status\":\"$STUB_HEALTH_STATUS\"}" ;;
    */api/v1/denoise) echo "{\"job_id\":\"job-1\",\"status\":\"$STUB_JOB_STATUS\"}" ;;
    */api/v1/download/*) printf '%s' "$STUB_CONTENT_TYPE" ;;
  esac
done
"""


def run_deploy(
    tmp_path: Path,
    *,
    ecr_has_digest: bool = False,
    pushed_digest: str = DIGEST,
    warm_status: str = "healthy",
    health_status: str = "healthy",
    job_status: str = "completed",
    content_type: str = "audio/wav",
    new_version: str = "5",
    failing_alias_target: str = "",
    digest: str = DIGEST,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the script against the stand-ins and return its result and the logged calls."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("aws", AWS_STUB), ("docker", DOCKER_STUB), ("curl", CURL_STUB)):
        stub = bin_dir / name
        stub.write_text(body)
        stub.chmod(0o755)
    log = tmp_path / "calls.log"
    log.touch()
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "STUB_LOG": str(log),
        "STUB_ECR_HAS_DIGEST": "1" if ecr_has_digest else "0",
        "STUB_PUSHED_DIGEST": pushed_digest,
        "STUB_WARM_STATUS": warm_status,
        "STUB_HEALTH_STATUS": health_status,
        "STUB_JOB_STATUS": job_status,
        "STUB_CONTENT_TYPE": content_type,
        "STUB_NEW_VERSION": new_version,
        "STUB_FAILING_ALIAS_TARGET": failing_alias_target,
        "GHCR_IMAGE": "ghcr.io/example/backend",
        "DIGEST": digest,
        "COMMIT": "c0ffee",
        "FUNCTION": "backend",
        "REPO": "backend",
        "ALIAS": "live",
        "BACKEND_URL": "https://function.example",
        "AWS_REGION": "us-east-1",
    }
    result = subprocess.run(
        ["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60, check=False
    )
    return result, log.read_text().splitlines()


def alias_moves(calls: list[str]) -> list[str]:
    """The versions passed to update-alias, in order."""
    moves = []
    for call in calls:
        if call.startswith("aws lambda update-alias"):
            words = call.split()
            moves.append(words[words.index("--function-version") + 1])
    return moves


def test_deploy_copies_publishes_warms_and_moves_the_alias(tmp_path: Path) -> None:
    result, calls = run_deploy(tmp_path)

    assert result.returncode == 0, result.stderr
    assert any(
        c.startswith(f"docker pull --platform linux/amd64 ghcr.io/example/backend@{DIGEST}")
        for c in calls
    )
    assert any(
        c.startswith("docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/backend:c0ffee")
        for c in calls
    )
    update = next(c for c in calls if c.startswith("aws lambda update-function-code"))
    assert f"--image-uri 123456789012.dkr.ecr.us-east-1.amazonaws.com/backend@{DIGEST}" in update
    invoke_index = next(i for i, c in enumerate(calls) if c.startswith("aws lambda invoke"))
    move_index = next(i for i, c in enumerate(calls) if c.startswith("aws lambda update-alias"))
    assert "--qualifier 5" in calls[invoke_index]
    assert invoke_index < move_index
    assert alias_moves(calls) == ["5"]


def test_deploy_skips_the_copy_when_ecr_holds_the_digest(tmp_path: Path) -> None:
    result, calls = run_deploy(tmp_path, ecr_has_digest=True)

    assert result.returncode == 0, result.stderr
    assert not any(c.startswith("docker") for c in calls)
    assert alias_moves(calls) == ["5"]


def test_deploy_stops_before_any_lambda_change_when_the_digest_changes(tmp_path: Path) -> None:
    result, calls = run_deploy(tmp_path, pushed_digest=OTHER_DIGEST)

    assert result.returncode != 0
    assert f"ECR stored '{OTHER_DIGEST}' under tag c0ffee for {DIGEST}" in result.stderr
    assert "delete tag c0ffee before a re-run" in result.stderr
    assert not any(c.startswith("aws lambda update-function-code") for c in calls)
    assert alias_moves(calls) == []


def test_deploy_leaves_the_alias_when_the_new_version_is_not_healthy(tmp_path: Path) -> None:
    result, calls = run_deploy(tmp_path, warm_status="unhealthy")

    assert result.returncode != 0
    assert "version 5 did not report healthy when warmed" in result.stderr
    assert alias_moves(calls) == []


@pytest.mark.parametrize(
    ("health_status", "job_status", "content_type"),
    [
        ("unhealthy", "completed", "audio/wav"),
        ("healthy", "failed", "audio/wav"),
        ("healthy", "completed", "application/json"),
    ],
    ids=["health", "upload", "download"],
)
def test_deploy_moves_the_alias_back_when_the_smoke_test_fails(
    tmp_path: Path, health_status: str, job_status: str, content_type: str
) -> None:
    result, calls = run_deploy(
        tmp_path, health_status=health_status, job_status=job_status, content_type=content_type
    )

    assert result.returncode != 0
    assert f"deploy of {DIGEST} was rolled back to version 4" in result.stderr
    assert alias_moves(calls) == ["5", "4"]


def test_deploy_reports_no_rollback_target_when_the_version_was_already_live(
    tmp_path: Path,
) -> None:
    # publish-version returns the existing version when nothing changed since it
    result, calls = run_deploy(tmp_path, new_version="4", health_status="unhealthy")

    assert result.returncode != 0
    assert "version 4 was live before this deploy" in result.stderr
    assert alias_moves(calls) == ["4"]


def test_deploy_says_so_when_moving_the_alias_back_fails(tmp_path: Path) -> None:
    result, calls = run_deploy(tmp_path, health_status="unhealthy", failing_alias_target="4")

    assert result.returncode != 0
    assert "moving live back to version 4 failed; it may still be on 5" in result.stderr
    assert alias_moves(calls) == ["5", "4"]


def test_deploy_refuses_a_malformed_digest_before_any_call(tmp_path: Path) -> None:
    result, calls = run_deploy(tmp_path, digest="latest")

    assert result.returncode != 0
    assert calls == []
