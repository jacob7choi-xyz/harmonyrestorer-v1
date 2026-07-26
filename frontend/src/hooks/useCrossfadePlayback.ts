import { useState, useRef, useCallback, useEffect } from 'react';

/** Which recording the listener hears. Exactly one is ever audible. */
export type ComparisonSource = 'original' | 'restored';

/** Why a requested source change could not be committed. */
export type SwitchFailure = 'target-unavailable' | 'alignment-failed' | 'play-failed';

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
  alignmentFailures: number;
  /** Verified offset at the most recent commit, and the largest seen. */
  lastCommitOffset: number;
  maxCommitOffset: number;
}

interface UseCrossfadePlaybackReturn {
  state: CrossfadeState;
  originalRef: React.RefObject<HTMLAudioElement | null>;
  enhancedRef: React.RefObject<HTMLAudioElement | null>;
  play: () => void;
  pause: () => void;
  seek: (fraction: number) => void;
  setSource: (source: ComparisonSource) => void;
  /** Move the timeline boundary. Safe to call at pointer-move rate.
   *
   * pointerBand is the drag's stability margin as a fraction of the strip,
   * derived by the caller from a fixed pixel distance. */
  setDividerPosition: (position: number, pointerBand?: number) => void;
  switchStats: React.RefObject<SwitchStats>;
}

/** Above this volume a source counts as audibly contributing. */
const AUDIBLE_FLOOR = 0.02;
/** Threshold in seconds. Only update state when currentTime drifts this far. */
const TIME_UPDATE_THRESHOLD = 0.03;

/* Alignment is established once, at handoff. Nothing corrects the silent
   track between switches: see the animation loop for why continuous
   correction cannot converge. */

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

/* Dead zone applied when playback carries the playhead across the boundary,
   in seconds because that is the dimension of the problem it solves.

   Committing a source change swaps which decoder defines the playhead, and
   the incoming clock can sit behind the outgoing one. That offset is bounded
   by construction: a target further than SWITCH_SYNC_TOLERANCE is seeked and
   then accepted only within SWITCH_ACCEPT_TOLERANCE, so the offset at commit
   cannot exceed SWITCH_ACCEPT_TOLERANCE. A band wider than that bound keeps
   the newly committed source inside the dead zone on the next evaluation,
   which is what stops a crossing from bouncing straight back.

   Expressing this as a fraction of the track would make the lag grow with
   duration: 0.5% is 10 ms on a 2 second clip and 3 seconds on a 10 minute
   upload, so the playhead would visibly cross the divider while the audio
   waited. */
const BOUNDARY_BAND_SECONDS = 0.08;

const other = (source: ComparisonSource): ComparisonSource =>
  source === 'original' ? 'restored' : 'original';

/** Choose the audible source from where the playhead sits relative to the boundary.
 *
 * The strip draws restored material left of the divider and original material
 * right of it, so the listener hears whichever region playback currently
 * occupies. The divider is a boundary in the timeline, not a blend amount.
 *
 * Positions and band are seconds. Pointer stability is a separate concern
 * measured in pixels; see setDividerPosition.
 */
export function sourceForBoundary(
  playheadSeconds: number,
  boundarySeconds: number,
  current: ComparisonSource,
  bandSeconds: number = BOUNDARY_BAND_SECONDS,
): ComparisonSource {
  if (playheadSeconds < boundarySeconds - bandSeconds) return 'restored';
  if (playheadSeconds > boundarySeconds + bandSeconds) return 'original';
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
 * audible at any instant; the other keeps playing silently, which measured
 * better in Safari than pausing it and restarting at the handoff.
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
  /* Identifies the switch that owns background-sync suspension. An older
     overlapping transaction must not unsuspend sync while a newer one is
     still holding a seek open on the same element. */
  const switchOwnerRef = useRef<number | null>(null);
  /* Suppresses automatic re-demand of a source whose transition just failed.
     Geometry re-derives the same answer every frame, so without this a source
     that cannot align would be retried forever: each attempt is bounded but
     the sequence is not. The latch clears the moment geometry asks for the
     other side, so a fresh attempt requires a real change of intent rather
     than the passage of time. */
  const failedAutoSourceRef = useRef<ComparisonSource | null>(null);
  /** Timeline boundary as a fraction of the track. */
  const dividerRef = useRef<number>(initialDivider);
  /** Latest known duration, readable from the tick without a stale closure. */
  const durationRef = useRef<number>(0);
  /* Logical comparison position in seconds, and the single coordinate every
     consumer reads. Authority runs one way: explicit commands establish it,
     media elements reconcile toward it, and the active element advances it
     only while playback owns the position. A source commit never writes it,
     which is what stops changing sides from moving the playhead. */
  const transportRef = useRef<number>(0);
  /* Identifies the seek that currently owns transport, so the animation loop
     cannot overwrite a commanded position with a decoder that has not caught
     up. An epoch rather than a flag: overlapping seeks each need to know
     whether the authority is still theirs, and only the owner may release it.
     A boolean lets a stale operation's timeout free a newer one's claim. */
  const seekOwnerRef = useRef<number | null>(null);
  const switchStats = useRef<SwitchStats>({
    switches: 0,
    switchesNeedingSeek: 0,
    alignmentFailures: 0,
    lastCommitOffset: 0,
    maxCommitOffset: 0,
  });

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
    (el: HTMLAudioElement, time: number, tolerance: number): Promise<boolean> => {
      return new Promise<boolean>(resolve => {
        let settled = false;
        const finish = (ok: boolean): void => {
          if (settled) return;
          settled = true;
          el.removeEventListener('seeked', onSeeked);
          clearTimeout(timer);
          resolve(ok);
        };
        /* A seeked event is a wake-up, not a completion. Two overlapping
           seeks on one element produce events indistinguishable from each
           other, so the clock reaching the requested position is the only
           trustworthy completion condition. Expiry is judged the same way:
           a lost event on a decoder that actually arrived is a success. */
        const converged = (): boolean => Math.abs(el.currentTime - time) <= tolerance;
        const onSeeked = (): void => {
          // Another operation's event proves nothing about ours; keep waiting
          if (converged()) finish(true);
        };
        const timer = setTimeout(() => finish(converged()), SEEK_TIMEOUT_MS);
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

  /** Bring the target within tolerance of the authoritative position.
   *
   * The reference depends on transport state. Paused compares against the
   * frozen logical position, which does not move, so a single seek settles
   * it. Playing compares against the advancing active clock, so the landing
   * point is re-measured against live reality rather than against the value
   * we asked for: a seek can complete perfectly and still be stale.
   *
   * Success means verified alignment. A completed seek is not alignment, and
   * callers are given no way to confuse the two.
   */
  const alignTarget = useCallback(
    async (
      targetEl: HTMLAudioElement,
      referenceTime: () => number,
      epoch: number,
    ): Promise<'verified' | 'failed' | 'superseded'> => {
      if (Math.abs(targetEl.currentTime - referenceTime()) <= SWITCH_SYNC_TOLERANCE) {
        return 'verified';
      }
      switchStats.current.switchesNeedingSeek += 1;
      // One corrective seek plus one retry, then fail closed. Never a loop.
      for (let attempt = 0; attempt < 2; attempt += 1) {
        const landed = await seekAndWait(targetEl, referenceTime(), SWITCH_ACCEPT_TOLERANCE);
        if (epoch !== epochRef.current) return 'superseded';
        if (landed) return 'verified';
      }
      return 'failed';
    },
    [seekAndWait],
  );

  /** Roll intent back to reality after a genuine target failure. */
  const failSwitch = useCallback((epoch: number, reason: SwitchFailure): void => {
    if (epoch !== epochRef.current) return;
    failedAutoSourceRef.current = requestedSourceRef.current;
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

    switchOwnerRef.current = epoch;
    try {
      /* While playing the active decoder is the authority and it keeps
         moving; while paused the frozen transport position is, and it does
         not. Aligning against the wrong one is how a perfectly completed
         seek ends up stale. */
      const referenceTime = (): number =>
        isPlayingRef.current ? activeEl.currentTime : transportRef.current;

      if (!isTargetPlayable(referenceTime(), targetEl.duration)) {
        failSwitch(epoch, 'target-unavailable');
        return;
      }

      // Prepare the target while the listener still hears the current source
      const alignment = await alignTarget(targetEl, referenceTime, epoch);
      if (alignment === 'superseded') return;
      if (alignment === 'failed') {
        switchStats.current.alignmentFailures += 1;
        failSwitch(epoch, 'alignment-failed');
        return;
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

      /* The commit precondition is checked after the last await, immediately
         before authority transfers. Verifying before the ramps would only
         prove alignment held earlier: a precondition established before
         asynchronous work is not a precondition at the end of it. A failure
         here restores the outgoing source rather than leaving silence. */
      const commitOffset = Math.abs(targetEl.currentTime - referenceTime());
      if (commitOffset > SWITCH_ACCEPT_TOLERANCE) {
        switchStats.current.alignmentFailures += 1;
        if (ramping) activeEl.volume = 1;
        failSwitch(epoch, 'alignment-failed');
        return;
      }

      activeEl.volume = 0;
      activeEl.muted = true;
      targetEl.volume = ramping ? 0 : 1;
      targetEl.muted = false;
      activeSourceRef.current = requested;
      if (ramping) {
        await rampVolume(targetEl, 1, epoch);
      }
      switchStats.current.lastCommitOffset = commitOffset;
      switchStats.current.maxCommitOffset = Math.max(
        switchStats.current.maxCommitOffset,
        commitOffset,
      );
      switchStats.current.switches += 1;
      setState(prev => ({ ...prev, activeSource: requested, switchError: null }));
    } finally {
      if (switchOwnerRef.current === epoch) switchOwnerRef.current = null;
    }
  }, [alignTarget, applyGains, elementFor, failSwitch, rampVolume]);

  const setSource = useCallback(
    (source: ComparisonSource): void => {
      if (source === requestedSourceRef.current) return;
      requestedSourceRef.current = source;
      epochRef.current += 1;
      setState(prev => ({ ...prev, requestedSource: source, switchError: null }));
      void reconcile();
    },
    [reconcile],
  );

  /** Re-derive the audible source from the playhead's side of the boundary.
   *
   * Called both when the divider moves and as playback advances, because
   * either one can carry the playhead across the boundary. Intent is read
   * from the ref, not React state: a drag fires far faster than state
   * settles, and a stale read would decide against a position already
   * superseded. setSource returns early when the source is unchanged, so
   * this stays free to call on every animation frame.
   */
  const evaluateSource = useCallback(
    (bandSeconds: number): void => {
      const duration = durationRef.current;
      if (duration <= 0) return;
      const boundarySeconds = dividerRef.current * duration;
      const next = sourceForBoundary(
        transportRef.current,
        boundarySeconds,
        requestedSourceRef.current,
        bandSeconds,
      );
      if (next === failedAutoSourceRef.current) return;
      failedAutoSourceRef.current = null;
      setSource(next);
    },
    [setSource],
  );

  /** Move the timeline boundary.
   *
   * pointerBand is the drag's stability margin expressed as a fraction of the
   * strip, which callers derive from a fixed pixel distance. Pointer jitter is
   * spatial, so its tolerance belongs in pixels; it is converted to seconds
   * here only because the boundary comparison lives in seconds. It applies to
   * this drag alone and never to playback, where a pixel-derived margin would
   * reintroduce duration-scaled lag.
   */
  const setDividerPosition = useCallback(
    (position: number, pointerBand: number = 0): void => {
      dividerRef.current = Math.max(0, Math.min(1, position));
      evaluateSource(pointerBand * durationRef.current);
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
    switchOwnerRef.current = null;
    failedAutoSourceRef.current = null;
    dividerRef.current = initialDivider;
    durationRef.current = 0;
    // Replacing the pair is an explicit transport reset. Media load events
    // report facts and are never allowed to move the position themselves.
    transportRef.current = 0;
    seekOwnerRef.current = null;
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
      switchOwnerRef.current = null;
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
      if (!activeEl || activeEl.paused || activeEl.ended) {
        reconcilePlayback();
        return;
      }
      /* Playback advances transport only when no command owns the position,
         and never past the shared timeline: the comparison is defined on the
         intersection of the two recordings, so the longer one's tail is
         outside it even while that element keeps decoding. */
      if (seekOwnerRef.current === null) {
        transportRef.current = Math.min(activeEl.currentTime, durationRef.current);
      }
      if (Math.abs(transportRef.current - lastReportedTimeRef.current) > TIME_UPDATE_THRESHOLD) {
        lastReportedTimeRef.current = transportRef.current;
        setState(prev => ({ ...prev, currentTime: transportRef.current }));
      }
      // Playback itself can carry the playhead across the boundary
      evaluateSource(BOUNDARY_BAND_SECONDS);
      /* The silent track is deliberately left alone.

         Continuously nudging it toward the audible clock cannot converge
         where a seek costs more than the drift threshold it is correcting:
         the audible track advances by the seek's own latency, so the
         correction re-triggers the instant it lands. Measured in Safari at
         roughly four seeks per second on the inaudible element, 40 to 145 ms
         each, one of which blocked the animation loop for 494 ms and froze
         the playhead. Alignment is verified at handoff instead, which is the
         boundary that actually has to hold. */
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [elementFor, evaluateSource, reconcilePlayback]);

  const play = useCallback((): void => {
    const epoch = ++epochRef.current;
    /* Alignment is judged against a different reference once transport is
       running, so a demand that failed while paused deserves a fresh attempt
       rather than inheriting the latch. */
    failedAutoSourceRef.current = null;
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
      if (silentEl) silentEl.currentTime = transportRef.current;

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
    failedAutoSourceRef.current = null;
    cancelAnimationFrame(rafRef.current);
    originalRef.current?.pause();
    enhancedRef.current?.pause();
    setState(prev => ({ ...prev, isPlaying: false }));
  }, []);

  const seek = useCallback(
    (fraction: number): void => {
      const duration = durationRef.current;
      if (duration <= 0) return;
      // A scrub invalidates any switch prepared at the old position
      /* Invalidates the switch logically. Its physical seek may still be
         outstanding, and only its own finally may release the serialization
         flag: clearing it here would let background sync touch an element
         whose seeked event another operation is still waiting on. */
      const epoch = ++epochRef.current;
      failedAutoSourceRef.current = null;
      const clamped = Math.max(0, Math.min(1, fraction));
      /* The comparison timeline is the shared one, so the last reachable
         position is bounded by the shorter recording rather than by whichever
         element happens to be audible. */
      const t = Math.min(clamped * duration, Math.max(0, duration - END_EPSILON));

      // Intent lands immediately; the decoders catch up behind it
      transportRef.current = t;
      lastReportedTimeRef.current = t;
      setState(prev => ({ ...prev, currentTime: t }));

      seekOwnerRef.current = epoch;
      const activeEl = elementFor(activeSourceRef.current);
      const silentEl = elementFor(other(activeSourceRef.current));

      /* No separate release timer: seekAndWait is itself bounded, so this
         continuation always runs. A second timer would be another lifetime
         to own, and an older one firing could free a newer command's claim. */
      void (async () => {
        const converged = activeEl
          ? await seekAndWait(activeEl, t, SWITCH_ACCEPT_TOLERANCE)
          : true;
        // Release only what this command still owns
        if (seekOwnerRef.current === epoch) seekOwnerRef.current = null;
        // Supersession is checked before any mutation, not after
        if (epoch !== epochRef.current) return;
        if (!converged) {
          /* The decoder never arrived. Transport reflects where the media
             actually is rather than continuing to claim a position nothing
             reached, and the command ends here rather than silently. */
          if (activeEl) {
            transportRef.current = activeEl.currentTime;
            lastReportedTimeRef.current = activeEl.currentTime;
            setState(prev => ({ ...prev, currentTime: activeEl.currentTime }));
          }
          return;
        }
        if (silentEl) silentEl.currentTime = t;
        if (requestedSourceRef.current !== activeSourceRef.current) {
          void reconcile();
        }
      })();
    },
    [elementFor, reconcile, seekAndWait],
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

export { AUDIBLE_FLOOR, BOUNDARY_BAND_SECONDS, SWITCH_ACCEPT_TOLERANCE };
