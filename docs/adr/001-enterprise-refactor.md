# ADR-001: Enterprise Refactor — Monolith to Modular Architecture

**Date:** 2026-01-07 (approximate, reconstructed from git history)

**Status:** accepted

---

## Context

HarmonyRestorer started as a single-file FastAPI prototype (`main.py`) with all logic — routes, services, config, schemas — in one place. The frontend was a monolithic 600-line `App.tsx`. As the project matured past proof-of-concept, several problems emerged:

- **Maintainability:** Single-file backend made it hard to reason about responsibilities or test individual components.
- **Security:** No file size limits, no CORS restrictions, no rate limiting, no input validation. Stack traces leaked to users.
- **Testability:** No test suite existed. The monolith was untestable without major refactoring.
- **Deployability:** No containerization, no CI/CD pipeline.
- **Frontend:** 600-line component, 17 unused npm dependencies, no real API integration (UI was cosmetic).

## Decision

Execute an 8-phase enterprise refactor in a single session, committed as `6a7cc28` on the `develop` branch, then merged to `main`. Each phase built on the previous:

### Phase 1: Split main.py into modules
Decomposed into `config.py`, `schemas.py`, `exceptions.py`, `middleware.py`, `routes/health.py`, `routes/denoise.py`, `services/denoiser.py`, `services/jobs.py`. Clean dependency flow: Routes → Services → Config → Schemas.

### Phase 2: Security hardening
- 50 MB file size limit enforced at upload
- CORS restricted from `*` to `http://localhost:3000` (configurable via env)
- Rate limiter: 10 req/min per IP on upload endpoint
- Error sanitization: generic 500 messages, no stack traces to users
- UUID validation on all job ID parameters
- Filename sanitization: store with generated UUID, never trust client names

### Phase 3: Config management
`Settings` class with `@lru_cache` singleton. All magic values moved to env vars with sensible defaults. `.env.example` created.

### Phase 4: Dead code cleanup
574 lines deleted. OpGAN research code moved to `_archive/` directory (excluded from linting, testing, type checking).

### Phase 5: Job cleanup + health check
TTL-based background cleanup loop (default 1 hour). Health endpoint checks real disk space and directory existence.

### Phase 6: Test suite
22 tests covering API endpoints and unit logic. Fixtures for test client, mock denoiser, sample WAV bytes. All passing in 0.13s.

### Phase 7: Docker
Multi-stage Dockerfile for backend (non-root user, health check). Frontend Dockerfile with nginx (non-root, security headers, CSP). `docker-compose.yml` with resource limits, health checks, named volumes.

### Phase 8: CI/CD
GitHub Actions workflow with parallel backend + frontend jobs. Backend: black, isort, ruff, mypy, pytest. Frontend: eslint, tsc, vite build.

### Post-Phase Security Hardening (committed separately)
- UVR output path traversal validation (`is_relative_to()`)
- Thread safety via `threading.Lock` on all `JobManager` mutations
- Cleanup race fix: job deleted from dict before file deletion
- Docker resource limits (2 CPU/4G backend, 0.5 CPU/256M frontend)
- Nginx non-root user, security headers

### Quality Hardening (Tier 2, committed `db4e553`)
- `JobStatusEnum` (StrEnum) replaced raw status strings
- `Field(ge=-1, le=100)` on progress
- `MIN_DISK_BYTES` moved to config
- Additional tests: rate limiter 429, all 6 audio formats, stack trace suppression (30 tests total)

### Frontend Refactor (Tier 3, committed `b5d33eb`)
- Split `App.tsx` → 7 files: `types.ts`, `api/client.ts`, 5 components
- Wired real API calls (upload → poll status → download)
- `ErrorBoundary` added
- Removed 17 unused dependencies
- Added ESLint config, `.env.example`

## Consequences

### Positive
- **12 clean production files** with single responsibilities
- **30 passing tests** in 0.13s with 85% coverage
- **Fully containerized** with health checks and resource limits
- **CI/CD pipeline** catches regressions on every push
- **Security baseline** established (file limits, CORS, rate limiting, error sanitization)
- **Frontend functional** — actually talks to the backend now

### Negative
- **In-memory job storage** retained (MVP trade-off — production needs Redis/Postgres)
- **Single-worker concurrency** — no Celery/RQ, one job processes at a time
- **Test coverage misleading** — 85% line coverage but DenoiserService mocked entirely, zero concurrency tests
- **Rate limiter in-memory** — resets on restart, no distributed support

### Neutral
- OpGAN code archived but still in repo (may revisit for custom training)
- `_archive/` excluded from all tooling — invisible to linters, type checkers, tests

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| Incremental refactor over weeks | Too slow; monolith was blocking all progress |
| Microservices (separate denoiser service) | Overkill for MVP; adds deployment complexity |
| Django instead of FastAPI | FastAPI better suited for async file processing + simple API |
| Redis job queue from day one | Premature; in-memory dict sufficient for single-user MVP |
| Next.js frontend | Unnecessary SSR complexity for a single-page upload/download app |
