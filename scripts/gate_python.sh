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

run "format (ruff)" "${UV_RUN[@]}" ruff format --check backend/ dataset/ benchmark/
run "lint (ruff)" "${UV_RUN[@]}" ruff check backend/ dataset/ benchmark/
run "types (mypy)" "${UV_RUN[@]}" mypy backend/app/
run "security (bandit)" "${UV_RUN[@]}" bandit -r backend/app/ -q

# Dependency audit, with coverage asserted rather than assumed.
#
# GitHub Actions installs torch as 2.12.1+cpu. A local version identifier does not
# exist on PyPI, so auditing the live environment made pip-audit report "could not be
# audited" for torch and exit zero anyway. CI had therefore never audited the largest
# runtime dependency while reporting a clean audit. That is the dependency-audit form
# of a silently shrinking denominator: coverage has to be explicit and fail closed.
#
# So the audit runs against a pinned list and then asserts nothing was skipped.
# Anything unauditable fails the gate instead of disappearing from the result. The
# population is the frozen lock rather than whatever happens to be installed: the
# "environment matches lock" check above is what makes this script audit the same
# dependency set locally and in CI.
#
# Local version identifiers are normalised for a closed set of packages, never in
# general. A local version is not decoration. foo==1.2.3+vendorpatch identifies a
# different artifact from upstream foo==1.2.3, so rewriting it would claim we audited
# something we did not. Only torch is listed, because only torch ships here as a +cpu
# wheel and its advisories are keyed to the base version. Any other package arriving
# with a local suffix stays unauditable and trips the coverage assertion, which is the
# intended outcome: a person decides whether base-version matching is valid for it.
NORMALISABLE_LOCAL_VERSION_PACKAGES="torch"
#
# Waived advisories. Each is bound to the assumptions it rests on, so it reopens by
# itself when those stop holding. A waiver that outlives its cause suppresses a future
# recurrence of the same identifier, which is why CVE-2026-3219 is not carried forward:
# pip is now 26.1.2 and that advisory no longer applies at all.
#
# PYSEC-2025-194 / CVE-2025-3000 / GHSA-rrmf-rvhw-rf47, torch, fixed in 2.13.0.
# Applicability established first: the GitHub-reviewed advisory records fixed 2.13.0
# with last known affected <= 2.12.1, and PyPI reports it for 2.12.1 unwithdrawn, so
# this version is affected. The PYSEC record's "last_affected: 2.6.0-NA" is malformed,
# which is why the sources appear to disagree. Waived on reachability: the flaw is
# memory corruption in torch.jit.script and the vector is local with privileges
# (CVSS AV:L/AC:L/PR:L). Supporting evidence beyond the tripwire below is that no
# TorchScript path exists in the inference service and checkpoints load through
# torch.load(..., weights_only=True).
WAIVED_TORCH_ADVISORY="PYSEC-2025-194"
WAIVED_TORCH_PACKAGE="torch"
WAIVED_TORCH_VERSION="2.12.1"
#
# PYSEC-2026-3447 / CVE-2026-59890, setuptools, fixed in 83.0.0. The flaw is in source
# distribution creation: MANIFEST.in exclusion rules can be bypassed through Unicode
# normalisation on macOS filesystems, so files meant to be excluded get packed into an
# sdist. The reachability condition is therefore whether sdists are built here, not
# whether anything imports setuptools; an earlier version of this waiver asserted the
# latter, which is close to unrelated to the advisory.
#
# The waiver rests primarily on an architectural fact rather than on a text search: no
# repository-controlled path publishes this project as a Python distribution, so nothing
# in its tracked build or deployment model performs sdist creation. That is a claim about
# automation, not about every possible human action. There is one workflow, ci.yml, with
# no release or publish pipeline; deployment is a container that copies backend/app and
# pyproject.toml rather than installing a built artifact; no MANIFEST.in exists; and no
# alternative build backend is configured. The checks below guard that state. They are
# tripwires, not proof that no sdist can ever be produced, since a person can always run
# a builder by hand and enumerating every spelling of every build command is not
# achievable.
#
# Remediation is blocked rather than merely inconvenient, and the evidence is recorded
# here because this claim is the reason a known advisory is carried at all. Constraining
# setuptools>=83.0.0 alongside the torch build this project requires yields:
#
#   Because torch==2.12.1+cpu depends on setuptools<82 and setuptools>=83.0.0,
#   we can conclude that torch==2.12.1+cpu cannot be used.
#
# So the +cpu build caps setuptools below 82, and the fix arrives with the torch 2.13
# upgrade, which also clears PYSEC-2025-194 outright. That cap is not visible in the
# installed darwin variant's metadata, where the requirement carries a different marker.
# Reading only that is how the constraint gets mistaken for folklore, which is the error
# made here before the resolver was asked directly.
WAIVED_SETUPTOOLS_ADVISORY="PYSEC-2026-3447"
WAIVED_SETUPTOOLS_PACKAGE="setuptools"
WAIVED_SETUPTOOLS_VERSION="81.0.0"

AUDIT_REQ="$(mktemp)"
AUDIT_JSON="$(mktemp)"
# One script-level cleanup. A RETURN trap would not be function-local: it fires again
# when later functions return, where its temp-file variable is out of scope.
trap 'rm -f "$AUDIT_REQ" "$AUDIT_JSON"' EXIT

# A waiver reasoned about one version says nothing about another, so any change to the
# waived package's version reopens the finding rather than inheriting the exception.
assert_waived_version() {
  local advisory="$1" package="$2" expected="$3" req="$4" actual
  actual="$(sed -nE "s/^${package}==([^ ]+)\$/\\1/p" "$req")"
  if [ "$actual" != "$expected" ]; then
    printf '%s is %s, but %s is waived only for %s. Re-establish applicability.\n' \
      "$package" "${actual:-absent}" "$advisory" "$expected"
    return 1
  fi
}

# A tripwire on tracked source, not a proof of unreachability. It establishes that no
# tracked Python names the API in any of its obvious import forms. It cannot see a
# dependency reaching TorchScript internally, so it supports the waiver rather than
# demonstrating it alone.
assert_torch_jit_unreferenced() {
  if git grep -qE 'torch\.jit|from torch import jit|from torch\.jit import' -- '*.py'; then
    printf 'TorchScript is now referenced in tracked source, so the reachability basis for %s no longer holds.\n' \
      "$WAIVED_TORCH_ADVISORY"
    return 1
  fi
}

# Regression tripwires for the packaging state described above, not proofs of
# impossibility. Either a MANIFEST.in appearing or tracked configuration gaining a build
# invocation means the packaging model changed and the waiver has to be re-argued.
assert_no_sdist_path() {
  if [ -e MANIFEST.in ]; then
    printf 'MANIFEST.in now exists, so the unexercised-sdist basis for %s no longer holds.\n' \
      "$WAIVED_SETUPTOOLS_ADVISORY"
    return 1
  fi
  # Matches build invocations rather than the word itself, and excludes this file, which
  # documents the rule and would otherwise trip it.
  if git grep -qE '(setup\.py[^;&|]*sdist|python[0-9.]* +-m +build|(uv|hatch|poetry|pdm|flit) +build)' \
    -- '*.yml' '*.yaml' '*.sh' '*.toml' 'Dockerfile*' ':!scripts/gate_python.sh'; then
    printf 'tracked configuration now builds a source distribution, so the basis for %s no longer holds.\n' \
      "$WAIVED_SETUPTOOLS_ADVISORY"
    return 1
  fi
}

audit_dependencies() {
  local req="$AUDIT_REQ" skipped normalise
  # Only the reviewed packages get their local version identifier removed.
  normalise="s/^(${NORMALISABLE_LOCAL_VERSION_PACKAGES})==([^+]+)\\+[A-Za-z0-9._]+\$/\\1==\\2/"

  # Drop the local project, which exists on no index and can never be audited.
  uv pip freeze | grep -vE '^(-e |harmonyrestorer([=<>@ ]|$))' | sed -E "$normalise" >"$req"

  assert_waived_version \
    "$WAIVED_TORCH_ADVISORY" "$WAIVED_TORCH_PACKAGE" "$WAIVED_TORCH_VERSION" "$req" || return 1
  assert_torch_jit_unreferenced || return 1
  assert_waived_version \
    "$WAIVED_SETUPTOOLS_ADVISORY" "$WAIVED_SETUPTOOLS_PACKAGE" "$WAIVED_SETUPTOOLS_VERSION" \
    "$req" || return 1
  assert_no_sdist_path || return 1

  "${UV_RUN[@]}" pip-audit --requirement "$req" --no-deps --format json \
    --ignore-vuln "$WAIVED_SETUPTOOLS_ADVISORY" \
    --ignore-vuln "$WAIVED_TORCH_ADVISORY" >"$AUDIT_JSON" || return 1

  # Fail closed on coverage: a package the service could not audit is not a pass.
  skipped="$(python3 -c "
import json, sys
report = json.load(open(sys.argv[1]))
for dep in report.get('dependencies', []):
    if dep.get('skip_reason'):
        print(f\"{dep.get('name')} {dep.get('version')}: {dep['skip_reason']}\")
" "$AUDIT_JSON")"
  if [ -n "$skipped" ]; then
    printf 'unaudited dependencies (coverage gap, not a pass):\n%s\n' "$skipped"
    return 1
  fi
  # Precise about what was established: torch was matched on its upstream base version
  # under the normalisation policy above, not on the exact local wheel identity.
  printf 'every pinned dependency had advisory coverage under the declared normalisation policy, no unwaived findings\n'
}

run "dependency audit (pip-audit, coverage asserted)" audit_dependencies

run "backend tests" \
  "${UV_RUN[@]}" pytest backend/tests/ -v --tb=short --cov=backend/app --cov-fail-under=80
run "dataset tests" "${UV_RUN[@]}" pytest dataset/tests/ -v --tb=short
run "benchmark tests" "${UV_RUN[@]}" pytest benchmark/tests/ -v --tb=short

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
