# Security Policy

## Supported Versions

Only the current `main` branch is actively maintained. No prior releases are supported.

| Version | Supported |
|---------|-----------|
| main    | Yes       |

## Reporting a Vulnerability

Please do not open a public GitHub issue for security vulnerabilities.

Report vulnerabilities via one of the following:

- **GitHub private advisory**: use the "Report a vulnerability" button on the [Security tab](https://github.com/jacob7choi-xyz/harmonyrestorer-v1/security/advisories/new)
- **Email**: jchoi26@colby.edu

Include a description of the issue, steps to reproduce, and potential impact. You can expect an acknowledgment within 72 hours and a resolution timeline within 14 days for confirmed issues.

## Scope

In scope:

- Backend API (file upload, job processing, download endpoints)
- File validation logic (magic byte checks, size and duration limits)
- Job management and file handling
- Dependency supply chain (pyproject.toml, uv.lock, package-lock.json)

Out of scope:

- The Vercel-hosted frontend (report to Vercel directly for platform issues)
- The GCP training infrastructure (decommissioned after model training)
- Denial-of-service via resource exhaustion beyond the implemented rate limits

## Supply Chain

### Python dependencies

Dependencies are managed with [uv](https://github.com/astral-sh/uv) and pinned via a committed `uv.lock`. CI runs `uv sync --frozen` to enforce the lockfile and prevent silent resolution drift.

**PyTorch index override**: `torch` and `torchaudio` are sourced exclusively from the official PyTorch CPU wheel index (`https://download.pytorch.org/whl/cpu`) via `[tool.uv.sources]` in `pyproject.toml`. This avoids pulling GPU wheels that are not needed in production and ensures the wheel source is explicit and auditable.

**Ignored CVEs**:

| CVE | Package | Reason |
|-----|---------|--------|
| CVE-2026-3219 | pip | Affects pip itself; no patched version available upstream. Monitored for fix. |
| CVE-2025-58438 | internetarchive 4.x | Fix requires v5 which has breaking API changes. `internetarchive` is used only in dataset acquisition scripts; training is complete and these scripts are not part of the production backend. |

Both ignores are documented in `.github/workflows/ci.yml` and reviewed on each dependency update cycle.

### JavaScript dependencies

Frontend dependencies are audited in CI via `npm audit --audit-level=high`. The `package-lock.json` is committed and updated as part of any dependency change.

## Dependency Update Policy

- Python: `uv lock --upgrade` run periodically; `uv.lock` committed after review
- JavaScript: `npm audit fix` run when vulnerabilities are reported; `package-lock.json` committed after review
- CVE ignores are re-evaluated on each update cycle and removed as soon as a patched version is available and tested
