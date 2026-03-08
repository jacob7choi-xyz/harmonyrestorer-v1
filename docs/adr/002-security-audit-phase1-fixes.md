# ADR-002: Security Audit Findings and Phase 1 Fix Plan

**Date:** 2026-03-07

**Status:** proposed

---

## Context

Two months after the enterprise refactor (ADR-001), a comprehensive codebase audit was conducted across all layers: backend core, services, routes, tests, frontend, Docker, nginx, CI/CD, and dependency management. The audit used 5 parallel review agents examining every file.

The codebase is functional and has a solid foundation, but the audit revealed **6 critical**, **9 high**, **10 medium**, and **7 low** severity issues across security, code quality, test coverage, and infrastructure.

### Critical Findings

1. **Race condition: download vs cleanup** — `cleanup_expired()` can delete a file between the `exists()` check and `FileResponse()` in the download endpoint, causing an unhandled 500 error.

2. **Race condition: process vs cleanup** — Cleanup can remove a job from `_jobs` while `process()` is mid-execution. The in-memory job object becomes orphaned — user can never retrieve status or download.

3. **No file content validation** — Upload route checks file extension only, not magic bytes. A PDF renamed to `.wav` passes validation and hits UVR, which could crash or produce undefined behavior.

4. **Rate limiter memory leak** — IP keys are never evicted from the `_requests` dict; only timestamps within each key are pruned. Under distributed traffic, dict grows unbounded until OOM.

5. **No security scanning in CI** — `bandit`, `pip-audit`, `detect-secrets` are installed as dev dependencies but never executed in the CI pipeline. No `npm audit` either.

6. **Python dependencies unpinned** — All use `>=` with no upper bound (e.g., `fastapi>=0.128.0`). No lockfile. Same `pip install` on different dates produces different versions.

### High Findings

7. CORS `allow_headers=["*"]` — overly permissive
8. Missing `response_model` on route endpoints — no OpenAPI schema, raw dicts returned
9. No `X-Forwarded-Proto` in nginx proxy config
10. No HSTS header in nginx
11. Frontend polling has no `AbortController` — memory leak on unmount
12. No client-side file size pre-validation
13. Max audio duration (10 min) documented but never enforced
14. Unused ML dependencies bloating install (`pytorch-lightning`, `omegaconf`, `matchering`, etc.)
15. No coverage threshold enforced in CI

### Test Suite Assessment

30 tests, 85% line coverage — but coverage is misleading:
- `DenoiserService` mocked entirely (74% uncovered)
- Zero concurrency/thread-safety tests
- Zero path traversal security tests
- Zero file I/O error tests (disk full, permissions)
- Tests create real files that are never cleaned up

## Decision

Fix the 6 critical and most impactful high-severity issues in a focused Phase 1 sprint. Group fixes by area to minimize churn:

### Fix 1: Race conditions in JobManager (`jobs.py`)
- Add a `downloading` state or reference count to prevent cleanup from deleting jobs/files that are actively being served
- Alternative: mark jobs as non-expirable while a download is in progress
- Ensure `process()` checks if job still exists after acquiring lock

### Fix 2: File content validation (`denoise.py`)
- Validate WAV magic bytes (`RIFF....WAVE`) at upload time
- For non-WAV formats (MP3, FLAC, OGG, M4A, AAC), validate their respective magic bytes
- Reject files that don't match their declared extension

### Fix 3: Rate limiter memory leak (`middleware.py`)
- After pruning timestamps, delete the IP key if the list is empty
- One-line fix: `if not self._requests[client_ip]: del self._requests[client_ip]`

### Fix 4: CORS headers (`main.py`)
- Replace `allow_headers=["*"]` with explicit list: `["Content-Type", "Accept"]`

### Fix 5: Nginx hardening (`nginx.conf`)
- Add `proxy_set_header X-Forwarded-Proto $scheme`
- Add `Strict-Transport-Security: max-age=31536000; includeSubDomains`

### Fix 6: CI security scanning (`ci.yml`)
- Add `bandit -r app/` to backend job
- Add `pip-audit` to backend job
- Add `npm audit --audit-level=high` to frontend job
- Add `pytest --cov=app --cov-fail-under=80`

## Consequences

### Positive
- Eliminates all known critical vulnerabilities
- Race conditions fixed — no more orphaned jobs or mid-download file deletions
- Malformed file uploads rejected before hitting UVR
- Rate limiter won't leak memory under sustained traffic
- CI catches security regressions automatically
- Coverage regression prevented by threshold enforcement

### Negative
- Magic byte validation adds complexity and must be maintained as new formats are supported
- `bandit` and `pip-audit` in CI will slow pipeline by ~15-30 seconds
- Stricter CORS headers could break custom clients that send non-standard headers (unlikely)
- Race condition fix adds lock complexity to JobManager

### Neutral
- Nginx changes only affect Docker/production deployments, not local dev
- Dependency pinning deferred to Phase 2 (requires choosing between pip-tools, Poetry, or uv)
- Frontend fixes (AbortController, file size validation) deferred to Phase 2

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| Fix everything at once (all 32 issues) | Too large; high risk of regression. Phased approach is safer |
| Use Redis for job storage to fix race conditions | Correct long-term, but overkill for MVP. Lock-based fix is simpler |
| Content validation via `python-magic` (libmagic) | Adds C library dependency; magic byte checks are simpler and sufficient |
| Move rate limiting to nginx entirely | Would fix memory leak, but loses per-endpoint granularity. Do both eventually |
| Switch to Poetry for dependency pinning | Good idea but separate concern; deferred to Phase 2 to keep this focused |
| Skip CI security scanning (rely on local checks) | Humans forget; CI doesn't. Non-negotiable for production readiness |
