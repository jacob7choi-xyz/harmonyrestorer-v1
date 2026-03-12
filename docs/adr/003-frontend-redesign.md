# ADR-003: Frontend Redesign — Spotify-esque Wizard with Animated Background

**Date:** 2026-03-12

**Status:** accepted

---

## Context

The frontend after ADR-001 was functional but visually flat: a plain dark background, basic file input, static progress bar. It lacked the polish expected of a production-quality audio tool. Specific problems:

- **No visual identity.** The UI was generic dark-mode with no distinctive character.
- **No audio preview.** Users couldn't listen to their file before processing.
- **No comparison flow.** After processing, users got a download link but no way to A/B compare original vs enhanced audio in-browser.
- **Accent color.** The initial Spotify green (#1DB954) risked brand confusion and was replaced with a distinctive soft blue (#5B8DEF).
- **No animation.** The background was a static solid color, giving the app a lifeless feel for an audio product.

## Decision

Redesign the frontend as a 3-step wizard (upload, processing, complete) with an animated canvas background, in-browser audio playback, and before/after comparison.

### Wizard Flow

1. **Upload step:** Drag-and-drop upload area with file validation (type, size, 10-minute duration check via Audio API). Inline audio preview with waveform scrub bar.
2. **Processing step:** Animated progress bar with status messages. Waveform visualization of the uploaded file.
3. **Complete step:** Side-by-side `ComparisonView` with synchronized original/enhanced `AudioPlayer` components. Download button and "New file" reset.

### Animated Background (TechnoBackground.tsx)

Full-viewport `<canvas>` element behind all content, rendered at 60fps via `requestAnimationFrame`:

- **Gravitational orb:** 3-layer radial gradient (gravity well, energy field, intense core) with slow 0.4Hz breathing animation (4% size variation). Conveys controlled power.
- **Radiating rings:** 4 concentric rings expanding outward from the orb center, fading as they grow. Speed scales with processing intensity.
- **Sparkle star field:** 150 4-pointed sparkles with independent twinkle phases and brightness. Drawn via `lineTo` path operations (not circles).
- **Ribbon system:** 6 layered silk ribbons with 3-pass rendering (wide/medium/narrow) for depth. Currently stashed but code retained for re-enabling.
- **Intensity states:** `idle` (gentle pulse), `processing` (amplified speed/size), `complete` (settled glow). All parameters lerp smoothly between states.
- **Performance:** Pre-allocated object pools eliminate GC pressure. No DOM manipulation in the animation loop. `ResizeObserver` handles viewport changes.

### Component Architecture

| Component | Responsibility |
|-----------|---------------|
| `App.tsx` | Wizard orchestrator, state machine, global drop zone, social footer |
| `TechnoBackground.tsx` | Full-screen animated canvas (orb, rings, sparkles) |
| `UploadArea.tsx` | Drag-and-drop with file validation and duration check |
| `AudioPlayer.tsx` | Play/pause button + waveform scrub bar (memoized) |
| `WaveformCanvas.tsx` | Canvas-rendered waveform with seek + resize (debounced) |
| `ComparisonView.tsx` | Before/after audio comparison layout |
| `ErrorBoundary.tsx` | Graceful error recovery with logging |

### Hooks

| Hook | Responsibility |
|------|---------------|
| `useAudioDecoder` | Web Audio API decode + `computePeaks` for waveform data |
| `useAudioPlayback` | Audio element lifecycle, RAF-driven playback state (30ms throttle) |

### API Client Hardening

- `pollUntilDone` now enforces a 5-minute timeout (`MAX_POLL_DURATION`)
- Unknown job statuses are rejected immediately (`KNOWN_STATUSES` validation)
- All fetch calls support `AbortController` cancellation

### Design Tokens

- Accent: `#5B8DEF` (soft blue), hover: `#7BA4F7`
- Background: `#121212`, cards: `#282828`, muted: `#727272`
- Semi-transparent UI with `backdrop-blur-md` over animated background
- Upload icon circle: `bg-white/5 border-white/10` (glass effect)

## Consequences

### Positive
- **Visual identity** distinct from generic dark-mode templates
- **Audio preview** before processing reduces wasted compute on wrong files
- **A/B comparison** lets users hear the difference immediately
- **Smooth animations** give the app a premium, audio-product feel
- **84 frontend tests** covering all components, hooks, and API client edge cases

### Negative
- **Canvas animation** consumes GPU cycles continuously (mitigated: `requestAnimationFrame` pauses in hidden tabs)
- **No SSR** — canvas animation is client-only, not relevant for this SPA but limits future SSR adoption
- **Ribbon code stashed** — dead code in the bundle until re-enabled or removed

### Neutral
- Accent color change from green to blue is purely aesthetic, no functional impact
- `AudioPlayer` memoization and `useAudioPlayback` throttling are micro-optimizations that primarily benefit low-end devices

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|--------------|
| CSS animations for background | Too limited for the organic, physics-inspired effects needed |
| Three.js / WebGL | Overkill for 2D effects; adds ~150KB dependency |
| Lottie animations | Pre-baked animations can't react to app state in real-time |
| Framer Motion for everything | Good for UI transitions but wrong tool for full-screen canvas animation |
| Keep Spotify green accent | Risk of brand confusion; blue is more neutral and distinctive |
