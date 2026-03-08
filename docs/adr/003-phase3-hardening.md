# ADR-003: Phase 3 Hardening -- Duration, Accessibility, Tests, Logging

**Date:** 2026-03-07

**Status:** accepted

---

## Context

After Phase 1 (critical security fixes) and Phase 2 (quality hardening), the remaining audit findings fell into six categories:

1. **Unenforced rule**: Project docs specify a 10-minute max audio duration, but no code checked it -- backend or frontend.
2. **Accessibility gap**: The upload area used `<div onClick>` with no keyboard support, failing WCAG compliance.
3. **UI deception**: The SettingsPanel exposed 8 controls (noise reduction, reverb removal, voice isolation, etc.) that were never sent to the backend. UVR runs with zero configurable parameters.
4. **Test blind spots**: 30 tests covered happy paths but missed error handling in `process()`, download guard lifecycle, magic byte rejection, path traversal defense, and cleanup race conditions.
5. **Information exposure**: The `/health` endpoint was publicly accessible, exposing disk space and job counts.
6. **Log parsing**: Plain text logs are unparseable by production log aggregators (ELK, Datadog, etc.).

## Decision

### 1. Audio Duration Enforcement

**Backend** (`config.py`, `denoise.py`):
- Added `max_audio_duration_seconds = 600` to Settings (configurable via `MAX_AUDIO_DURATION_SECONDS` env var)
- After magic byte validation, uses `soundfile.info(BytesIO(content))` to read duration for WAV/FLAC/OGG
- Rejects with 400 if duration exceeds limit
- MP3/M4A/AAC skip the check (soundfile can't parse them without ffmpeg; backend rejects if UVR fails)

**Frontend** (`UploadArea.tsx`):
- Uses `new Audio()` + `loadedmetadata` event to check duration before upload
- Works for any format the browser can decode
- Falls through to backend validation if browser can't read metadata

### 2. Keyboard Accessibility

- Added `role="button"`, `tabIndex={0}`, `aria-label` to upload area div
- Added `onKeyDown` handler for Enter and Space keys
- Added `aria-label="Reset"` to the icon-only reset button

### 3. Settings Panel Removal

Removed entirely rather than wiring up, because:
- UVR's `audio-separator` exposes no configurable parameters (model choice is the only knob, set at startup)
- Adding fake backend handling would be worse than removing the facade
- Deleted `SettingsPanel.tsx`, `ProcessingSettings` interface
- Simplified `App.tsx` from 3-column grid to single-column layout (`max-w-2xl`)

### 4. Test Coverage Expansion (9 new tests, 39 total)

| Test | File | What it covers |
|------|------|----------------|
| `test_upload_rejects_mismatched_magic_bytes` | test_security.py | PDF content with .wav extension -> 400 |
| `test_upload_rejects_empty_file` | test_security.py | Zero-byte upload -> 400 |
| `test_path_traversal_filename_uses_uuid` | test_security.py | `../../etc/passwd.wav` filename neutralized by UUID |
| `test_process_sets_failed_on_denoiser_error` | test_jobs.py | Denoiser raises -> job status FAILED, progress -1 |
| `test_process_handles_vanished_job` | test_jobs.py | Job deleted before process() runs -> exits cleanly |
| `test_cleanup_skips_downloading_jobs` | test_jobs.py | Jobs in `_downloading` set are not cleaned up |
| `test_mark_downloading_returns_false_for_missing_job` | test_jobs.py | Returns False for nonexistent job |
| `test_unmark_downloading_is_idempotent` | test_jobs.py | No error if job was never marked |
| `test_download_after_job_cleanup` | test_denoise.py | Download returns 404 after cleanup removes job |

### 5. Nginx Health Check Restriction

Added IP allowlist to `/health` location block:
- `allow 127.0.0.1` (loopback)
- `allow 10.0.0.0/8` (Docker/private networks)
- `allow 172.16.0.0/12` (Docker bridge)
- `allow 192.168.0.0/16` (private LAN)
- `deny all` (everything else gets 403)

### 6. Structured Logging

- Added `JSONFormatter` class to `main.py` using stdlib `logging.Formatter`
- Controlled by `LOG_FORMAT` env var: `"json"` for production, `"text"` (default) for development
- JSON output: `{"timestamp": "ISO8601", "level": "INFO", "logger": "app.main", "message": "..."}`
- Exception info included in `"exception"` field when present
- No new dependencies

## Consequences

### Positive
- All documented security rules now have code enforcing them
- Upload area is keyboard-navigable (WCAG 2.1 compliant)
- UI no longer lies about capabilities
- Test suite covers error paths, security edge cases, and download guard lifecycle
- Health endpoint no longer leaks system info to the internet
- Production logs are machine-parseable

### Negative
- Duration check only works for WAV/FLAC/OGG on backend (MP3/M4A/AAC rely on frontend or UVR failure)
- Removing settings panel reduces perceived UI richness (but honesty > appearance)
- JSON logs are harder to read during local development (mitigated by defaulting to text)

### Neutral
- Test count grew from 30 to 39 (all passing in 0.20s)
- Frontend bundle size decreased slightly from SettingsPanel removal
- `soundfile` is a transitive dep of `audio-separator` -- no new install needed

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| Wire settings to backend as no-ops | Worse than removing: teaches users to expect behavior that doesn't exist |
| Use `librosa` for MP3 duration checking | Heavy dep (was just pruned in Phase 2); soundfile covers the main formats |
| Keep health endpoint public, strip sensitive fields | Still leaks job counts; easier to restrict access entirely |
| Use `python-json-logger` for structured logging | Unnecessary dependency for a 15-line formatter |
| Add `mutool` or `ffprobe` for universal duration checks | External binary dependency; not worth the complexity for MVP |
