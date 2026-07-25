import { useState, useRef, useCallback, useEffect } from 'react';

/** Which recording the listener hears. Exactly one is ever audible. */
export type ComparisonSource = 'original' | 'restored';

/** Why a requested source change could not be committed. */
export type SwitchFailure = 'target-unavailable' | 'seek-failed' | 'play-failed';

export interface CrossfadeState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  originalReady: boolean;
  enhancedReady: boolean;
  /** Latest user selection. Survives pause; only a genuine failure rolls it back. */
  requestedSource: ComparisonSource;
  /** The audible source. Sole owner of playback-clock authority. */
  activeSource: ComparisonSource;
  switchError: SwitchFailure | null;
}

interface CrossfadeOptions {
  /** Loop both tracks continuously (used by the landing demo). */
  loop?: boolean;
  /** Which source is audible when a new pair loads. */
  initialSource?: ComparisonSource;
  /** Where the timeline boundary starts (1 = the whole track is restored). */
  initialDivider?: number;
  /** Handoff ramp length. Tests pass 0 to make transitions synchronous. */
  rampMs?: number;
}

/** Counters for choosing sync constants during development. Never reported anywhere. */
export interface SwitchStats {
  switches: number;
  switchesNeedingSeek: number;
}

interface UseCrossfadePlaybackReturn {
  state: CrossfadeState;
  originalRef: React.RefObject<HTMLAudioElement | null>;
  enhancedRef: React.RefObject<HTMLAudioElement | null>;
  play: () => void;
  pause: () => void;
  seek: (fraction: number) => void;
  setSource: (source: ComparisonSource) => void;
  /** Move the timeline boundary. Safe to call at pointer-move rate. */
  setDividerPosition: (position: number) => void;
  switchStats: React.RefObject<SwitchStats>;
}

/** Above this volume a source counts as audibly contributing. */
const AUDIBLE_FLOOR = 0.02;
/** Threshold in seconds. Only update state when currentTime drifts this far. */
const TIME_UPDATE_THRESHOLD = 0.03;

/* The three sync constants below are starting values to be tuned by listening
   and by the switchStats ratio, not derived from theory. Background sync is an
   optimization; the switch-time check is the correctness boundary. */

/** Inaudible track is snapped back once it drifts this far from the active clock. */
const BACKGROUND_SYNC_DRIFT = 0.05;
/** Target this close to the active clock at switch time needs no seek at all. */
const SWITCH_SYNC_TOLERANCE = 0.02;
/** Largest post-seek offset accepted when promoting a target to audible. */
const SWITCH_ACCEPT_TOLERANCE = 0.05;
/** A seek that has not landed within this long is treated as failed. */
const SEEK_TIMEOUT_MS = 400;
/** Within this distance of a track's end there is no meaningful playback left. */
const END_EPSILON = 0.05;
/** Fade length for the non-overlapping handoff, long enough to avoid a click. */
const DEFAULT_RAMP_MS = 12;

/* Dead zone around the divider, as a fraction of the strip. Playback crosses
   the boundary once and needs no hysteresis, but a drag held right on the
   playhead would otherwise chatter between sources on every pointer event. */
const DIVIDER_HYSTERESIS = 0.005;

const other = (source: ComparisonSource): ComparisonSource =>
  source === 'original' ? 'restored' : 'original';

/** Choose the audible source from where the playhead sits relative to the divider.
 *
 * The strip draws restored material left of the divider and original material
 * right of it, so the listener hears whichever region playback currently
 * occupies. The divider is a boundary in the timeline, not a blend amount.
 */
export function sourceForPlayhead(
  playhead: number,
  divider: number,
  current: ComparisonSource,
  band: number = DIVIDER_HYSTERESIS,
): ComparisonSource {
  if (playhead < divider - band) return 'restored';
  if (playhead > divider + band) return 'original';
  return current;
}

/** A play() interrupted by pause() or by the OS rejects with AbortError.
 *
 * That is cancellation, not evidence the target is broken. Our own pause
 * bumps the epoch and is caught by the epoch check, but interruptions we did
 * not initiate (incoming call, another app taking audio focus) arrive with no
 * epoch change and would otherwise be misread as a corrupt track.
 */
export function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

/** Whether a target still has playable material at the active clock position.
 *
 * The two recordings are not the same length: container padding makes a
 * compressed original typically run slightly longer than the restored PCM.
 */
export function isTargetPlayable(
  activeTime: number,
  targetDuration: number,
  endEpsilon: number = END_EPSILON,
): boolean {
  if (!Number.isFinite(targetDuration) || targetDuration <= 0) return true;
  return activeTime < targetDuration - endEpsilon;
}

export type PlaybackTransition = 'finish' | 'stop' | 'keep';

/** Decide the logical playback transition from the authoritative clock's state.
 *
 * DOM events and the RAF loop are only signals to call this; the event target
 * carries no authority. The active source is chosen before this decision,
 * never inside it. When the active element has both ended and paused (ended
 * elements are paused), finishing wins.
 */
export function resolvePlaybackTransition(
  activeEnded: boolean,
  activePaused: boolean,
  isPlaying: boolean,
): PlaybackTransition {
  if (!isPlaying) return 'keep';
  if (activeEnded) return 'finish';
  if (activePaused) return 'stop';
  return 'keep';
}

/**
 * Drive two audio elements as a binary A/B comparison. Exactly one source is
 * audible at any instant; the other keeps playing silently so switching does
 * not have to restart it.
 *
 * The state machine separates four distinct truths: what the user asked for
 * (requestedSource), what is actually audible and owns the clock
 * (activeSource), whether transport is running (isPlaying), and whether
 * in-flight asynchronous media work is still valid (an operation epoch).
 * Collapsing any of these into the others is what produced the dual-audible
 * comb filtering this hook replaces.
 */
export function useCrossfadePlayback(
  originalSrc: string | null,
  enhancedSrc: string | null,
  options: CrossfadeOptions = {},
): UseCrossfadePlaybackReturn {
  const {
    loop = false,
    initialSource = 'restored',
    initialDivider = 1,
    rampMs = DEFAULT_RAMP_MS,
  } = options;
  const originalRef = useRef<HTMLAudioElement | null>(null);
  const enhancedRef = useRef<HTMLAudioElement | null>(null);
  const rafRef = useRef<number>(0);
  const lastReportedTimeRef = useRef<number>(0);
  const isPlayingRef = useRef<boolean>(false);
  const requestedSourceRef = useRef<ComparisonSource>(initialSource);
  const activeSourceRef = useRef<ComparisonSource>(initialSource);
  /** Validity token for in-flight transport work. Never encodes user preference. */
  const epochRef = useRef<number>(0);
  /** Suspends background sync so it cannot race a foreground switch seek. */
  const switchInFlightRef = useRef<boolean>(false);
  /** Timeline boundary as a fraction of the track. */
  const dividerRef = useRef<number>(initialDivider);
  /** Latest known duration, readable from the tick without a stale closure. */
  const durationRef = useRef<number>(0);
  const switchStats = useRef<SwitchStats>({ switches: 0, switchesNeedingSeek: 0 });

  const [state, setState] = useState<CrossfadeState>({
    isPlaying: false,
    currentTime: 0,
    duration: 0,
    originalReady: false,
    enhancedReady: false,
    requestedSource: initialSource,
    activeSource: initialSource,
    switchError: null,
  });

  const elementFor = useCallback((source: ComparisonSource): HTMLAudioElement | null => {
    return source === 'original' ? originalRef.current : enhancedRef.current;
  }, []);

  /** Make exactly one source audible.
   *
   * Silences the outgoing element before raising the incoming one. The
   * reverse order leaves both above the audible floor for the width of two
   * assignments, which is brief but is exactly the state this hook exists to
   * make impossible.
   */
  const applyGains = useCallback(
    (audible: ComparisonSource): void => {
      const silent = elementFor(other(audible));
      const loud = elementFor(audible);
      if (silent) {
        silent.volume = 0;
        silent.muted = true;
      }
      if (loud) {
        loud.volume = 1;
        loud.muted = false;
      }
    },
    [elementFor],
  );

  /** Ramp one element's volume, aborting if the epoch moves on. */
  const rampVolume = useCallback(
    (el: HTMLAudioElement, to: number, epoch: number): Promise<void> => {
      if (rampMs <= 0) {
        el.volume = to;
        return Promise.resolve();
      }
      const from = el.volume;
      const start = performance.now();
      return new Promise<void>(resolve => {
        const step = (): void => {
          if (epoch !== epochRef.current) {
            resolve();
            return;
          }
          const progress = Math.min(1, (performance.now() - start) / rampMs);
          el.volume = from + (to - from) * progress;
          if (progress >= 1) {
            resolve();
            return;
          }
          requestAnimationFrame(step);
        };
        step();
      });
    },
    [rampMs],
  );

  /** Seek an element and wait for it to land, with a bounded timeout.
   *
   * Cleans up its listener and timer on every exit so an abandoned
   * transaction cannot leave a seeked handler behind.
   */
  const seekAndWait = useCallback(
    (el: HTMLAudioElement, time: number): Promise<boolean> => {
      return new Promise<boolean>(resolve => {
        let settled = false;
        const finish = (ok: boolean): void => {
          if (settled) return;
          settled = true;
          el.removeEventListener('seeked', onSeeked);
          clearTimeout(timer);
          resolve(ok);
        };
        const onSeeked = (): void => finish(true);
        const timer = setTimeout(() => finish(false), SEEK_TIMEOUT_MS);
        el.addEventListener('seeked', onSeeked);
        try {
          el.currentTime = time;
        } catch {
          finish(false);
        }
      });
    },
    [],
  );

  /** Roll intent back to reality after a genuine target failure. */
  const failSwitch = useCallback((epoch: number, reason: SwitchFailure): void => {
    if (epoch !== epochRef.current) return;
    requestedSourceRef.current = activeSourceRef.current;
    setState(prev => ({
      ...prev,
      requestedSource: activeSourceRef.current,
      switchError: reason,
    }));
  }, []);

  /** Drive reality toward intent.
   *
   * Every dangerous step happens while the current source is still audible,
   * so a failure at any point simply declines to commit. The handoff is
   * non-overlapping: the outgoing source reaches silence before the incoming
   * one becomes audible, which is what makes dual-audible states impossible
   * rather than merely unlikely.
   */
  const reconcile = useCallback(async (): Promise<void> => {
    const epoch = epochRef.current;
    const requested = requestedSourceRef.current;
    const active = activeSourceRef.current;
    if (requested === active) {
      applyGains(active);
      return;
    }

    const targetEl = elementFor(requested);
    const activeEl = elementFor(active);
    if (!targetEl || !activeEl) return;

    switchInFlightRef.current = true;
    try {
      const activeTime = activeEl.currentTime;
      if (!isTargetPlayable(activeTime, targetEl.duration)) {
        failSwitch(epoch, 'target-unavailable');
        return;
      }

      // Prepare the target while the listener still hears the current source
      if (Math.abs(targetEl.currentTime - activeTime) > SWITCH_SYNC_TOLERANCE) {
        switchStats.current.switchesNeedingSeek += 1;
        const landed = await seekAndWait(targetEl, activeTime);
        if (epoch !== epochRef.current) return;
        if (!landed) {
          failSwitch(epoch, 'seek-failed');
          return;
        }
        // The active clock advanced during the seek; re-measure against it
        if (Math.abs(targetEl.currentTime - activeEl.currentTime) > SWITCH_ACCEPT_TOLERANCE) {
          const retried = await seekAndWait(targetEl, activeEl.currentTime);
          if (epoch !== epochRef.current) return;
          if (!retried) {
            failSwitch(epoch, 'seek-failed');
            return;
          }
        }
      }

      if (isPlayingRef.current) {
        try {
          await targetEl.play();
        } catch (err) {
          if (epoch !== epochRef.current) return;
          if (isAbortError(err)) return;
          failSwitch(epoch, 'play-failed');
          return;
        }
        if (epoch !== epochRef.current) return;
      }

      // Non-overlapping handoff. Silent when paused, where a ramp buys nothing.
      const ramping = isPlayingRef.current;
      if (ramping) {
        await rampVolume(activeEl, 0, epoch);
        if (epoch !== epochRef.current) return;
      }
      activeEl.volume = 0;
      activeEl.muted = true;
      targetEl.volume = ramping ? 0 : 1;
      targetEl.muted = false;
      if (ramping) {
        await rampVolume(targetEl, 1, epoch);
        if (epoch !== epochRef.current) return;
      }

      activeSourceRef.current = requested;
      switchStats.current.switches += 1;
      setState(prev => ({ ...prev, activeSource: requested, switchError: null }));
    } finally {
      switchInFlightRef.current = false;
    }
  }, [applyGains, elementFor, failSwitch, rampVolume, seekAndWait]);

  const setSource = useCallback(
    (source: ComparisonSource): void => {
      if (source === requestedSourceRef.current) return;
      requestedSourceRef.current = source;
      epochRef.current += 1;
      switchInFlightRef.current = false;
      setState(prev => ({ ...prev, requestedSource: source, switchError: null }));
      void reconcile();
    },
    [reconcile],
  );

  /** Re-derive the audible source from the playhead's side of the divider.
   *
   * Called both when the divider moves and as playback advances, because
   * either one can carry the playhead across the boundary. Intent is read
   * from the ref, not React state: a drag fires far faster than state
   * settles, and a stale read would fight the hysteresis band.
   */
  const evaluateSource = useCallback((): void => {
    const activeEl = elementFor(activeSourceRef.current);
    const duration = durationRef.current;
    if (!activeEl || duration <= 0) return;
    const playhead = activeEl.currentTime / duration;
    setSource(sourceForPlayhead(playhead, dividerRef.current, requestedSourceRef.current));
  }, [elementFor, setSource]);

  const setDividerPosition = useCallback(
    (position: number): void => {
      dividerRef.current = Math.max(0, Math.min(1, position));
      evaluateSource();
    },
    [evaluateSource],
  );

  /** Recompute logical playback from the active element's actual state. */
  const reconcilePlayback = useCallback((): void => {
    const activeEl = elementFor(activeSourceRef.current);
    if (!activeEl) return;
    const transition = resolvePlaybackTransition(
      activeEl.ended,
      activeEl.paused,
      isPlayingRef.current,
    );
    if (transition === 'keep') return;
    isPlayingRef.current = false;
    originalRef.current?.pause();
    enhancedRef.current?.pause();
    cancelAnimationFrame(rafRef.current);
    if (transition === 'finish') {
      setState(prev => ({ ...prev, isPlaying: false, currentTime: prev.duration }));
    } else {
      setState(prev => ({ ...prev, isPlaying: false, currentTime: activeEl.currentTime }));
    }
  }, [elementFor]);

  useEffect(() => {
    const original = originalRef.current;
    const enhanced = enhancedRef.current;
    if (!original || !enhanced || !originalSrc || !enhancedSrc) return;

    epochRef.current += 1;
    isPlayingRef.current = false;
    requestedSourceRef.current = initialSource;
    activeSourceRef.current = initialSource;
    switchInFlightRef.current = false;
    dividerRef.current = initialDivider;
    durationRef.current = 0;
    setState(prev => ({
      ...prev,
      isPlaying: false,
      currentTime: 0,
      duration: 0,
      originalReady: false,
      enhancedReady: false,
      requestedSource: initialSource,
      activeSource: initialSource,
      switchError: null,
    }));
    original.src = originalSrc;
    enhanced.src = enhancedSrc;
    original.loop = loop;
    enhanced.loop = loop;
    lastReportedTimeRef.current = 0;
    applyGains(initialSource);

    /* Seeking is bounded by the shorter track so every position on the scrub
       bar exists in both recordings; container padding otherwise leaves a tail
       that only one of them can reach. */
    const onMeta = (): void => {
      const durations = [original.duration, enhanced.duration].filter(
        d => Number.isFinite(d) && d > 0,
      );
      const duration = durations.length === 2 ? Math.min(...durations) : (durations[0] ?? 0);
      durationRef.current = duration;
      setState(prev => ({
        ...prev,
        duration,
        originalReady: prev.originalReady || original.readyState > 0,
        enhancedReady: prev.enhancedReady || enhanced.readyState > 0,
      }));
    };
    const onOriginalError = (): void => {
      setState(prev => ({ ...prev, originalReady: false }));
    };
    const onEnhancedError = (): void => {
      setState(prev => ({ ...prev, enhancedReady: false }));
    };
    // ended and pause are signals to recompute logical state; the event target
    // carries no authority, so the silent track ending stops nothing
    const onPlaybackSignal = (): void => {
      reconcilePlayback();
    };

    original.addEventListener('loadedmetadata', onMeta);
    enhanced.addEventListener('loadedmetadata', onMeta);
    original.addEventListener('error', onOriginalError);
    enhanced.addEventListener('error', onEnhancedError);
    original.addEventListener('ended', onPlaybackSignal);
    enhanced.addEventListener('ended', onPlaybackSignal);
    original.addEventListener('pause', onPlaybackSignal);
    enhanced.addEventListener('pause', onPlaybackSignal);

    return () => {
      // Unmount and source replacement invalidate outstanding work and clean
      // up. They must not reconcile: that would write state after teardown.
      epochRef.current += 1;
      original.removeEventListener('loadedmetadata', onMeta);
      enhanced.removeEventListener('loadedmetadata', onMeta);
      original.removeEventListener('error', onOriginalError);
      enhanced.removeEventListener('error', onEnhancedError);
      original.removeEventListener('ended', onPlaybackSignal);
      enhanced.removeEventListener('ended', onPlaybackSignal);
      original.removeEventListener('pause', onPlaybackSignal);
      enhanced.removeEventListener('pause', onPlaybackSignal);
      isPlayingRef.current = false;
      switchInFlightRef.current = false;
      original.pause();
      enhanced.pause();
      cancelAnimationFrame(rafRef.current);
    };
  }, [
    originalSrc,
    enhancedSrc,
    loop,
    initialSource,
    initialDivider,
    applyGains,
    reconcilePlayback,
  ]);

  const startTicking = useCallback((): void => {
    cancelAnimationFrame(rafRef.current);
    const tick = (): void => {
      const activeEl = elementFor(activeSourceRef.current);
      const silentEl = elementFor(other(activeSourceRef.current));
      if (!activeEl || activeEl.paused || activeEl.ended) {
        reconcilePlayback();
        return;
      }
      if (Math.abs(activeEl.currentTime - lastReportedTimeRef.current) > TIME_UPDATE_THRESHOLD) {
        lastReportedTimeRef.current = activeEl.currentTime;
        setState(prev => ({ ...prev, currentTime: activeEl.currentTime }));
      }
      // Playback itself can carry the playhead across the boundary
      evaluateSource();
      /* Background sync only ever touches the silent track, and stands down
         while a switch owns that element's seek. */
      if (silentEl && !switchInFlightRef.current) {
        const drift = Math.abs(silentEl.currentTime - activeEl.currentTime);
        if (drift > BACKGROUND_SYNC_DRIFT) {
          silentEl.currentTime = activeEl.currentTime;
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [elementFor, evaluateSource, reconcilePlayback]);

  const play = useCallback((): void => {
    const epoch = ++epochRef.current;
    void (async () => {
      // Commit any change requested while paused first: with nothing audible
      // the handoff is silent, so the user hears the source they selected.
      if (requestedSourceRef.current !== activeSourceRef.current) {
        await reconcile();
        if (epoch !== epochRef.current) return;
      }
      const activeEl = elementFor(activeSourceRef.current);
      const silentEl = elementFor(other(activeSourceRef.current));
      if (!activeEl) return;
      cancelAnimationFrame(rafRef.current);
      applyGains(activeSourceRef.current);
      if (silentEl) silentEl.currentTime = activeEl.currentTime;

      const attempts = [activeEl.play()];
      if (silentEl) attempts.push(silentEl.play());
      const results = await Promise.allSettled(attempts);
      if (epoch !== epochRef.current) return;
      // Playback counts as started only if the audible element is playing
      if (results[0].status === 'fulfilled') {
        isPlayingRef.current = true;
        setState(prev => ({ ...prev, isPlaying: true }));
        startTicking();
      } else {
        isPlayingRef.current = false;
        setState(prev => ({ ...prev, isPlaying: false }));
      }
    })();
  }, [applyGains, elementFor, reconcile, startTicking]);

  const pause = useCallback((): void => {
    // Invalidates in-flight media work without touching the requested source:
    // pausing is not a change of mind. Settles; does not reconcile.
    epochRef.current += 1;
    isPlayingRef.current = false;
    switchInFlightRef.current = false;
    cancelAnimationFrame(rafRef.current);
    originalRef.current?.pause();
    enhancedRef.current?.pause();
    setState(prev => ({ ...prev, isPlaying: false }));
  }, []);

  const seek = useCallback(
    (fraction: number): void => {
      const activeEl = elementFor(activeSourceRef.current);
      const silentEl = elementFor(other(activeSourceRef.current));
      if (!activeEl || !activeEl.duration) return;
      // A scrub invalidates any switch prepared at the old position
      epochRef.current += 1;
      switchInFlightRef.current = false;
      const clamped = Math.max(0, Math.min(1, fraction));
      const t = clamped * activeEl.duration;
      activeEl.currentTime = t;
      if (silentEl) silentEl.currentTime = t;
      lastReportedTimeRef.current = t;
      setState(prev => ({ ...prev, currentTime: t }));
      if (requestedSourceRef.current !== activeSourceRef.current) {
        void reconcile();
      }
    },
    [elementFor, reconcile],
  );

  return {
    state,
    originalRef,
    enhancedRef,
    play,
    pause,
    seek,
    setSource,
    setDividerPosition,
    switchStats,
  };
}

export { AUDIBLE_FLOOR };
