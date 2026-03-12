import { useState, useRef, useCallback, useEffect } from 'react';
import type { PlaybackState } from '../types';

interface UseAudioPlaybackReturn {
  state: PlaybackState;
  play: () => void;
  pause: () => void;
  seek: (fraction: number) => void;
  audioRef: React.RefObject<HTMLAudioElement | null>;
}

export function useAudioPlayback(src: string | null): UseAudioPlaybackReturn {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const rafRef = useRef<number>(0);
  const [state, setState] = useState<PlaybackState>({
    isPlaying: false,
    currentTime: 0,
    duration: 0,
  });

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !src) return;

    audio.src = src;

    const onMeta = (): void => {
      setState(prev => ({ ...prev, duration: audio.duration }));
    };
    const onEnded = (): void => {
      setState(prev => ({ ...prev, isPlaying: false, currentTime: audio.duration }));
      cancelAnimationFrame(rafRef.current);
    };

    audio.addEventListener('loadedmetadata', onMeta);
    audio.addEventListener('ended', onEnded);

    return () => {
      audio.removeEventListener('loadedmetadata', onMeta);
      audio.removeEventListener('ended', onEnded);
      audio.pause();
      cancelAnimationFrame(rafRef.current);
    };
  }, [src]);

  const startTicking = useCallback((): void => {
    cancelAnimationFrame(rafRef.current);
    const tick = (): void => {
      const audio = audioRef.current;
      if (audio && !audio.paused) {
        setState(prev => ({ ...prev, currentTime: audio.currentTime }));
        rafRef.current = requestAnimationFrame(tick);
      }
    };
    rafRef.current = requestAnimationFrame(tick);
  }, []);

  const play = useCallback((): void => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.play().catch(() => {
      setState(prev => ({ ...prev, isPlaying: false }));
      cancelAnimationFrame(rafRef.current);
    });
    setState(prev => ({ ...prev, isPlaying: true }));
    startTicking();
  }, [startTicking]);

  const pause = useCallback((): void => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.pause();
    cancelAnimationFrame(rafRef.current);
    setState(prev => ({ ...prev, isPlaying: false }));
  }, []);

  const seek = useCallback((fraction: number): void => {
    const audio = audioRef.current;
    if (!audio || !audio.duration) return;
    const clamped = Math.max(0, Math.min(1, fraction));
    audio.currentTime = clamped * audio.duration;
    setState(prev => ({ ...prev, currentTime: audio.currentTime }));
  }, []);

  return { state, play, pause, seek, audioRef };
}
