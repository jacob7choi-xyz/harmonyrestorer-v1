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

  /** The enhanced element is the clock master; the original follows it. */
  const getMaster = useCallback((): {
    master: HTMLAudioElement | null;
    slave: HTMLAudioElement | null;
  } => ({
    master: enhancedRef.current ?? originalRef.current,
    slave: enhancedRef.current ? originalRef.current : null,
  }), []);

  useEffect(() => {
    const original = originalRef.current;
    const enhanced = enhancedRef.current;
    if (!original || !enhanced || !originalSrc || !enhancedSrc) return;

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
    const onEnded = (e: Event): void => {
      // Looping elements never fire ended; only the master's end stops playback
      const { master } = getMaster();
      if (e.target !== master) return;
      original.pause();
      enhanced.pause();
      cancelAnimationFrame(rafRef.current);
      setState(prev => ({ ...prev, isPlaying: false, currentTime: prev.duration }));
    };

    original.addEventListener('loadedmetadata', onMeta);
    enhanced.addEventListener('loadedmetadata', onMeta);
    original.addEventListener('error', onOriginalError);
    enhanced.addEventListener('error', onEnhancedError);
    original.addEventListener('ended', onEnded);
    enhanced.addEventListener('ended', onEnded);

    return () => {
      original.removeEventListener('loadedmetadata', onMeta);
      enhanced.removeEventListener('loadedmetadata', onMeta);
      original.removeEventListener('error', onOriginalError);
      enhanced.removeEventListener('error', onEnhancedError);
      original.removeEventListener('ended', onEnded);
      enhanced.removeEventListener('ended', onEnded);
      original.pause();
      enhanced.pause();
      cancelAnimationFrame(rafRef.current);
    };
  }, [originalSrc, enhancedSrc, loop, applyMix, getMaster]);

  const startTicking = useCallback((): void => {
    cancelAnimationFrame(rafRef.current);
    const tick = (): void => {
      const { master, slave } = getMaster();
      if (master && !master.paused) {
        if (Math.abs(master.currentTime - lastReportedTimeRef.current) > TIME_UPDATE_THRESHOLD) {
          lastReportedTimeRef.current = master.currentTime;
          setState(prev => ({ ...prev, currentTime: master.currentTime }));
        }
        // Bound clock drift between the two elements
        if (slave && Math.abs(slave.currentTime - master.currentTime) > DRIFT_THRESHOLD) {
          slave.currentTime = master.currentTime;
        }
        rafRef.current = requestAnimationFrame(tick);
      }
    };
    rafRef.current = requestAnimationFrame(tick);
  }, [getMaster]);

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
      if (results[0].status === 'fulfilled') {
        setState(prev => ({ ...prev, isPlaying: true }));
        startTicking();
      } else {
        setState(prev => ({ ...prev, isPlaying: false }));
      }
    });
  }, [getMaster, startTicking]);

  const pause = useCallback((): void => {
    originalRef.current?.pause();
    enhancedRef.current?.pause();
    cancelAnimationFrame(rafRef.current);
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
