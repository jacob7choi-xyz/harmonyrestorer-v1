#!/usr/bin/env bash
# The single definition of a green Python gate, for backend, dataset, and benchmark.
#
# GitHub Actions invokes this script rather than restating the checks, so local and
# remote agree by construction. Maintaining the list in two places produced a real
# failure: the secrets scan ran only in CI, so a scanner rejection was undetectable
# before pushing. Add a check here and both consumers get it.
#
# Usage: scripts/gate_python.sh
# Frontend has its own gate in package.json scripts and its own CI job.

set -euo pipefail

cd "$(dirname "$0")/.."

# Field names whose values are intentionally stored content digests. Closed on purpose:
# a new *_sha256 name does not inherit the exemption by matching a convention, it has to
# be added here and reviewed. Note what this is, and is not. It exempts a line whose key
# is one of these names and whose value is 64 lowercase hex characters. It does not
# verify that the value is a digest, so it is a trusted-field-name exemption rather than
# content verification. A high-entropy string under any other key still fails.
DIGEST_FIELDS='(sha256|checkpoint_sha256|corpus_sha256|document_sha256|protocol_sha256)'
DIGEST_EXEMPTION="\"${DIGEST_FIELDS}\": \"[0-9a-f]{64}\""

failed=()

run() {
  local label="$1"
  shift
  printf '\n=== %s ===\n' "$label"
  if "$@"; then
    return 0
  fi
  printf 'FAILED: %s\n' "$label"
  failed+=("$label")
  return 0
}

# The dependency population is proved before anything reads it, in two steps, because
# either alone is insufficient. `uv lock --check` proves the lock is current with
# pyproject.toml; `--frozen` alone would happily accept a stale lock. The sync check
# then proves the installed environment matches that lock.
run "lock is current with pyproject" uv lock --check
run "environment matches lock" uv sync --frozen --all-groups --all-extras --check

# Every subsequent tool runs with mutation forbidden. Bare `uv run` synchronises the
# environment by default, so the population proved above could otherwise be replaced
# part-way through the gate and the audit would describe a different set than the one
# that was checked.
UV_RUN=(uv run --frozen --no-sync)

# Recorded now, asserted at the end, so a mutation nothing else noticed still fails.
LOCK_DIGEST_BEFORE="$(shasum -a 256 uv.lock | cut -d" " -f1)"

run "format (ruff)" "${UV_RUN[@]}" ruff format --check backend/ dataset/ benchmark/ scripts/
run "lint (ruff)" "${UV_RUN[@]}" ruff check backend/ dataset/ benchmark/ scripts/
run "types (mypy)" "${UV_RUN[@]}" mypy backend/app/
run "security (bandit)" "${UV_RUN[@]}" bandit -r backend/app/ -q

# Dependency audit, with coverage asserted rather than assumed.
#
# GitHub Actions installs torch from the PyTorch CPU index, under a +cpu local version
# identifier (2.12.1+cpu when this check was written). A local version identifier does
# not exist on PyPI, so auditing the live environment made pip-audit report "could not
# be audited" for torch and exit zero anyway. CI had therefore never audited the largest
# runtime dependency while reporting a clean audit. That is the dependency-audit form
# of a silently shrinking denominator: coverage has to be explicit and fail closed.
#
# So the audit runs against a pinned list and then asserts nothing was skipped.
# Anything unauditable fails the gate instead of disappearing from the result. The
# population is the frozen lock rather than whatever happens to be installed: the
# "environment matches lock" check above is what makes this script audit the same
# dependency set locally and in CI.
#
# Advisory-version mapping, which is deliberately not called normalisation. PyTorch
# publishes its CPU build on a separate index under a local version identifier, so the
# deployed artifact is torch X.Y.Z+cpu while vulnerability databases key PyTorch
# advisories to upstream releases. No scanner indexes X.Y.Z+cpu, so the mapping has to
# happen somewhere; what matters is being clear about which identity each claim is about.
#
#   runtime artifact identity   torch==X.Y.Z+cpu
#   advisory lookup identity    torch==X.Y.Z
#
# The two are not interchangeable, and treating them as one caused a real defect: an
# earlier version rewrote the requirement and let pip-audit resolve it, which on Linux
# pulled the CUDA dependency closure of generic PyPI torch and audited a distribution
# that is not deployed. The audit now runs with --disable-pip so nothing is resolved and
# the mapping only affects which advisory record is consulted.
#
# Closed to torch alone. Any other package arriving with a local version identifier
# stays unmapped, is not represented in the report, and trips the coverage assertion,
# which is the intended outcome: a person decides whether upstream advisory data applies
# to that artifact.
ADVISORY_VERSION_MAPPED_PACKAGES="torch"
#
# No advisory is waived. The last two waivers were removed with the upgrade to torch
# 2.14.1, a release later than the 2.13.0 fix for PYSEC-2025-194. That torch resolves
# with setuptools 84.0.0, later than the 83.0.0 fix for PYSEC-2026-3447, which the
# setuptools<82 requirement of torch 2.12.1+cpu did not allow. Git history holds the
# waivers' reasoning and the version-binding and tripwire checks that guarded them. The
# audit step now fails on any advisory pip-audit reports.



AUDIT_REQ="$(mktemp)"
AUDIT_JSON="$(mktemp)"
# One script-level cleanup. A RETURN trap would not be function-local: it fires again
# when later functions return, where its temp-file variable is out of scope.
trap 'rm -f "$AUDIT_REQ" "$AUDIT_JSON"' EXIT

audit_dependencies() {
  local req="$AUDIT_REQ" skipped normalise
  # Only torch is mapped from its runtime identity to its advisory identity.
  normalise="s/^(${ADVISORY_VERSION_MAPPED_PACKAGES})==([^+]+)\\+[A-Za-z0-9._]+\$/\\1==\\2/"

  # Drop the local project, which exists on no index and can never be audited.
  uv pip freeze | grep -vE '^(-e |harmonyrestorer([=<>@ ]|$))' | sed -E "$normalise" >"$req"

  # --disable-pip alongside --no-deps is what actually prevents resolution. --no-deps
  # by itself still let the pip-backed collector expand the dependency closure, which is
  # how a fabricated base-version torch requirement dragged 19 CUDA packages into the
  # report on Linux while passing on macOS, where that closure happened to be a subset
  # of what was already installed.
  "${UV_RUN[@]}" pip-audit --requirement "$req" --no-deps --disable-pip --format json \
    >"$AUDIT_JSON" || return 1

  # Coverage is asserted by identity, not by absence of complaint. Checking only for
  # skip_reason would miss a package pip-audit drops from its report entirely, since a
  # dropped package leaves no entry to inspect. Extracted to Python because the check
  # canonicalises names, which is more than shell should carry, and because it is
  # unit-testable there.
  "${UV_RUN[@]}" python scripts/audit_dependencies.py \
    --requirements "$req" \
    --report "$AUDIT_JSON" || return 1
}

run "dependency audit (pip-audit, coverage asserted)" audit_dependencies

run "backend tests" \
  "${UV_RUN[@]}" pytest backend/tests/ -v --tb=short --cov=backend/app --cov-fail-under=80
run "dataset tests" "${UV_RUN[@]}" pytest dataset/tests/ -v --tb=short
run "benchmark tests" "${UV_RUN[@]}" pytest benchmark/tests/ -v --tb=short
run "gate policy tests" "${UV_RUN[@]}" pytest scripts/tests/ -v --tb=short

scan_secrets() {
  # Null-delimited so a tracked filename containing whitespace is scanned rather than
  # split into fragments that silently match nothing. xargs exits non-zero if any
  # invocation reports a finding.
  git ls-files -z | xargs -0 "${UV_RUN[@]}" detect-secrets-hook --exclude-lines "$DIGEST_EXEMPTION"
}

run "secrets scan (detect-secrets)" scan_secrets

# The population proved at the start must be the population that was audited.
if [ "$(shasum -a 256 uv.lock | cut -d" " -f1)" != "$LOCK_DIGEST_BEFORE" ]; then
  printf '\nuv.lock changed while the gate was running; the audited population is not the proved one\n'
  failed+=("dependency population immutability")
fi

printf '\n'
if [ ${#failed[@]} -gt 0 ]; then
  printf 'GATE FAILED (%d): %s\n' "${#failed[@]}" "${failed[*]}"
  exit 1
fi
printf 'GATE PASSED\n'
