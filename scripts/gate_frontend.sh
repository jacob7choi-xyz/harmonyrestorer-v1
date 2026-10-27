#!/usr/bin/env bash
# The single definition of a green frontend gate.
#
# GitHub Actions invokes this script rather than restating the checks, so local runs and
# CI run the same checks; results can still differ with the Node version and the
# installed node_modules. The frontend checks were listed separately in CI and in the
# docs, and on 2026-10-25 the dependency audit, which only CI ran, failed a push whose
# local checks had passed. Add a check here and both consumers get it.
#
# Every check runs even after one fails, and the failures are listed at the end.
#
# Installing is not part of the gate: CI runs `npm ci` first, and locally the existing
# node_modules is used, so the gate never replaces it.
#
# Usage: scripts/gate_frontend.sh
# The Python gate is scripts/gate_python.sh.

set -euo pipefail

cd "$(dirname "$0")/../frontend"

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

run "lint (eslint)" npm run lint
run "types (tsc)" npm run type-check
run "dependency audit (npm)" npm audit --audit-level=high
run "build" npm run build
run "tests (vitest)" npm run test

printf '\n'
if [ ${#failed[@]} -gt 0 ]; then
  printf 'GATE FAILED (%d): %s\n' "${#failed[@]}" "${failed[*]}"
  exit 1
fi
printf 'GATE PASSED\n'
