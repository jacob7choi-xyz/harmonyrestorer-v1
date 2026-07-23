import { useState, useRef, useCallback, useEffect } from 'react';

export interface CrossfadeState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
  originalReady: boolean;
  enhancedReady: boolean;
}

interface CrossfadeOptions {
  /** Loop both tracks continuously (used by the landing demo). */
  loop?: boolean;
}

interface UseCrossfadePlaybackReturn {
  state: CrossfadeState;
  originalRef: React.RefObject<HTMLAudioElement | null>;
  enhancedRef: React.RefObject<HTMLAudioElement | null>;
  play: () => void;
  pause: () => void;
  seek: (fraction: number) => void;
  setMix: (mix: number) => void;
}

/** Max allowed clock divergence between the two elements before a hard resync. */
const DRIFT_THRESHOLD = 0.08;
/** Threshold in seconds. Only update state when currentTime drifts this far. */
const TIME_UPDATE_THRESHOLD = 0.03;
/** Below this volume the element is also muted (covers platforms that ignore volume). */
const MUTE_FLOOR = 0.02;
/** Below this volume a track counts as background and is safe to hard-resync. */
const BACKGROUND_VOLUME = 0.4;
/** Drift beyond this is corrected even on an audible track. */
const HARD_DRIFT_LIMIT = 0.5;


/** Decide whether the background track's clock may be snapped to the master.

    Seeking is frame-imprecise for compressed formats, so a snapped MP3 can
    land outside the drift threshold and re-trigger every frame; snapping an
    audible track therefore produces continuous stutter. Corrections are
    restricted to quiet tracks unless drift becomes severe. */
export function shouldResyncSlave(driftSeconds: number, slaveVolume: number): boolean {
  if (driftSeconds > HARD_DRIFT_LIMIT) return true;
  return driftSeconds > DRIFT_THRESHOLD && slaveVolume < BACKGROUND_VOLUME;
}


export type PlaybackTransition = 'finish' | 'stop' | 'keep';

/** Decide the logical playback transition from the authoritative clock's state.

    DOM events and the RAF loop are only signals to call this; the event
    target carries no authority. Master selection (by audibility) happens
    before this decision, never inside it. When the master has both ended
    and paused (ended elements are paused), finishing wins. */
export function resolvePlaybackTransition(
  masterEnded: boolean,
  masterPaused: boolean,
  isPlaying: boolean,
): PlaybackTransition {
  if (!isPlaying) return 'keep';
  if (masterEnded) return 'finish';
  if (masterPaused) return 'stop';
  return 'keep';
}

/**
 * Drive two audio elements in lockstep with a continuous crossfade between
 * them. mix = 0 plays only the original (noisy) track, mix = 1 plays only
 * the enhanced track, and values between blend the two with an equal-power
 * curve. Both elements always play together so the blend responds instantly.
 *
 * setMix writes element volumes directly without a React state update, so it
 * is safe to call at pointer-move frequency while dragging.
 */
export function useCrossfadePlayback(
  originalSrc: string | null,
  enhancedSrc: string | null,
  options: CrossfadeOptions = {},
): UseCrossfadePlaybackReturn {
  const { loop = false } = options;
  const originalRef = useRef<HTMLAudioElement | null>(null);
  const enhancedRef = useRef<HTMLAudioElement | null>(null);
  const rafRef = useRef<number>(0);
  const lastReportedTimeRef = useRef<number>(0);
  const mixRef = useRef<number>(0.5);
  // Logical playback intent, readable from event handlers without stale closures
  const isPlayingRef = useRef<boolean>(false);
  const [state, setState] = useState<CrossfadeState>({
    isPlaying: false,
    currentTime: 0,
    duration: 0,
    originalReady: false,
    enhancedReady: false,
  });

  const applyMix = useCallback((): void => {
    const original = originalRef.current;
    const enhanced = enhancedRef.current;
    // Equal-power curve keeps perceived loudness constant across the blend
    const originalVolume = Math.cos(mixRef.current * Math.PI * 0.5);
    const enhancedVolume = Math.sin(mixRef.current * Math.PI * 0.5);
    if (original) {
      original.volume = originalVolume;
      original.muted = originalVolume < MUTE_FLOOR;
    }
    if (enhanced) {
      enhanced.volume = enhancedVolume;
      enhanced.muted = enhancedVolume < MUTE_FLOOR;
    }
  }, []);

  const setMix = useCallback((mix: number): void => {
    mixRef.current = Math.max(0, Math.min(1, mix));
    applyMix();
  }, [applyMix]);

  /** The audible element is the clock master; the background one follows it.

      Seeking the track the listener can hear causes audible stutter, so the
      master flips with the mix and corrections land on the background track.
  */
  const getMaster = useCallback((): {
    master: HTMLAudioElement | null;
    slave: HTMLAudioElement | null;
  } => {
    const original = originalRef.current;
    const enhanced = enhancedRef.current;
    if (!original || !enhanced) {
      return { master: enhanced ?? original, slave: null };
    }
    return mixRef.current >= 0.5
      ? { master: enhanced, slave: original }
      : { master: original, slave: enhanced };
  }, []);

  /** Recompute logical playback from the authoritative clock's actual state.

      Idempotent: a self-induced pause has already cleared isPlayingRef, so
      the resulting element pause events resolve to keep.
  */
  const reconcilePlayback = useCallback((): void => {
    const { master } = getMaster();
    if (!master) return;
    const transition = resolvePlaybackTransition(
      master.ended,
      master.paused,
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
      setState(prev => ({ ...prev, isPlaying: false, currentTime: master.currentTime }));
    }
  }, [getMaster]);

  useEffect(() => {
    const original = originalRef.current;
    const enhanced = enhancedRef.current;
    if (!original || !enhanced || !originalSrc || !enhancedSrc) return;

    isPlayingRef.current = false;
    setState(prev => ({
      ...prev,
      isPlaying: false,
      currentTime: 0,
      duration: 0,
      originalReady: false,
      enhancedReady: false,
    }));
    original.src = originalSrc;
    enhanced.src = enhancedSrc;
    original.loop = loop;
    enhanced.loop = loop;
    lastReportedTimeRef.current = 0;
    applyMix();

    const onMeta = (): void => {
      setState(prev => ({
        ...prev,
        duration: Math.max(original.duration || 0, enhanced.duration || 0),
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
    // ended and pause events are signals to recompute logical state; the
    // event target itself carries no authority (looping elements never
    // fire ended, and a background-track pause must not stop playback)
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
      original.removeEventListener('loadedmetadata', onMeta);
      enhanced.removeEventListener('loadedmetadata', onMeta);
      original.removeEventListener('error', onOriginalError);
      enhanced.removeEventListener('error', onEnhancedError);
      original.removeEventListener('ended', onPlaybackSignal);
      enhanced.removeEventListener('ended', onPlaybackSignal);
      original.removeEventListener('pause', onPlaybackSignal);
      enhanced.removeEventListener('pause', onPlaybackSignal);
      isPlayingRef.current = false;
      original.pause();
      enhanced.pause();
      cancelAnimationFrame(rafRef.current);
    };
  }, [originalSrc, enhancedSrc, loop, applyMix, reconcilePlayback]);

  const startTicking = useCallback((): void => {
    cancelAnimationFrame(rafRef.current);
    const tick = (): void => {
      const { master, slave } = getMaster();
      // A paused or ended master (blend flipped onto a dead track, browser
      // interruption, natural end) must reconcile state, never exit silently
      if (!master || master.paused || master.ended) {
        reconcilePlayback();
        return;
      }
      if (Math.abs(master.currentTime - lastReportedTimeRef.current) > TIME_UPDATE_THRESHOLD) {
        lastReportedTimeRef.current = master.currentTime;
        setState(prev => ({ ...prev, currentTime: master.currentTime }));
      }
      // Bound clock drift, but never stutter the audible track
      if (slave) {
        const drift = Math.abs(slave.currentTime - master.currentTime);
        if (shouldResyncSlave(drift, slave.volume)) {
          slave.currentTime = master.currentTime;
        }
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [getMaster, reconcilePlayback]);

  const play = useCallback((): void => {
    const { master, slave } = getMaster();
    if (!master) return;
    cancelAnimationFrame(rafRef.current);
    if (slave) {
      slave.currentTime = master.currentTime;
    }
    const attempts = [master.play()];
    if (slave) attempts.push(slave.play());
    Promise.allSettled(attempts).then(results => {
      // Playback counts as started only if the audible element is playing
      if (results[0].status === 'fulfilled') {
        isPlayingRef.current = true;
        setState(prev => ({ ...prev, isPlaying: true }));
        startTicking();
      } else {
        isPlayingRef.current = false;
        setState(prev => ({ ...prev, isPlaying: false }));
      }
    });
  }, [getMaster, startTicking]);

  const pause = useCallback((): void => {
    // Clear intent before pausing the elements so their pause events
    // reconcile to keep instead of re-running the stop transition
    isPlayingRef.current = false;
    cancelAnimationFrame(rafRef.current);
    originalRef.current?.pause();
    enhancedRef.current?.pause();
    setState(prev => ({ ...prev, isPlaying: false }));
  }, []);

  const seek = useCallback((fraction: number): void => {
    const { master, slave } = getMaster();
    if (!master || !master.duration) return;
    const clamped = Math.max(0, Math.min(1, fraction));
    const t = clamped * master.duration;
    master.currentTime = t;
    if (slave) slave.currentTime = t;
    lastReportedTimeRef.current = t;
    setState(prev => ({ ...prev, currentTime: t }));
  }, [getMaster]);

  return { state, originalRef, enhancedRef, play, pause, seek, setMix };
}
