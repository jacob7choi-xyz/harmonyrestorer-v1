import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useAudioPlayback } from '../useAudioPlayback';

function createMockAudio(
  overrides: Partial<HTMLAudioElement> = {},
): HTMLAudioElement {
  return {
    play: vi.fn().mockResolvedValue(undefined),
    pause: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    currentTime: 0,
    duration: 0,
    paused: true,
    src: '',
    preload: '',
    ...overrides,
  } as unknown as HTMLAudioElement;
}

describe('useAudioPlayback', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('returns initial state with no src', () => {
    const { result } = renderHook(() => useAudioPlayback(null));
    expect(result.current.state).toEqual({
      isPlaying: false,
      currentTime: 0,
      duration: 0,
    });
  });

  it('exposes play, pause, seek functions', () => {
    const { result } = renderHook(() => useAudioPlayback(null));
    expect(typeof result.current.play).toBe('function');
    expect(typeof result.current.pause).toBe('function');
    expect(typeof result.current.seek).toBe('function');
  });

  it('provides an audio ref', () => {
    const { result } = renderHook(() => useAudioPlayback(null));
    expect(result.current.audioRef).toBeDefined();
  });

  describe('play()', () => {
    it('catches autoplay rejection and resets isPlaying to false', async () => {
      const mockAudio = createMockAudio({
        play: vi.fn().mockRejectedValue(new DOMException('NotAllowedError')),
      });

      const { result } = renderHook(() => useAudioPlayback(null));

      // Attach mock audio element to the ref
      act(() => {
        (
          result.current.audioRef as React.MutableRefObject<HTMLAudioElement | null>
        ).current = mockAudio;
      });

      await act(async () => {
        result.current.play();
        // Flush the rejected promise handler
        await vi.waitFor(() => {
          expect(result.current.state.isPlaying).toBe(false);
        });
      });

      expect(mockAudio.play).toHaveBeenCalledOnce();
      expect(result.current.state.isPlaying).toBe(false);
    });

    it('cancels previous animation frame via startTicking', () => {
      const cancelSpy = vi.spyOn(globalThis, 'cancelAnimationFrame');
      const mockAudio = createMockAudio();

      const { result } = renderHook(() => useAudioPlayback(null));

      act(() => {
        (
          result.current.audioRef as React.MutableRefObject<HTMLAudioElement | null>
        ).current = mockAudio;
      });

      act(() => {
        result.current.play();
      });

      // startTicking calls cancelAnimationFrame before scheduling a new one
      expect(cancelSpy).toHaveBeenCalled();
    });
  });

  describe('pause()', () => {
    it('cancels animation frame when pausing', () => {
      const cancelSpy = vi.spyOn(globalThis, 'cancelAnimationFrame');
      const mockAudio = createMockAudio();

      const { result } = renderHook(() => useAudioPlayback(null));

      act(() => {
        (
          result.current.audioRef as React.MutableRefObject<HTMLAudioElement | null>
        ).current = mockAudio;
      });

      act(() => {
        result.current.play();
      });

      cancelSpy.mockClear();

      act(() => {
        result.current.pause();
      });

      expect(cancelSpy).toHaveBeenCalled();
      expect(mockAudio.pause).toHaveBeenCalled();
      expect(result.current.state.isPlaying).toBe(false);
    });
  });

  describe('seek()', () => {
    it('clamps fraction below 0 to 0', () => {
      const mockAudio = createMockAudio({ duration: 100 });

      const { result } = renderHook(() => useAudioPlayback(null));

      act(() => {
        (
          result.current.audioRef as React.MutableRefObject<HTMLAudioElement | null>
        ).current = mockAudio;
      });

      act(() => {
        result.current.seek(-0.5);
      });

      expect(mockAudio.currentTime).toBe(0);
    });

    it('clamps fraction above 1 to 1', () => {
      const mockAudio = createMockAudio({ duration: 100 });

      const { result } = renderHook(() => useAudioPlayback(null));

      act(() => {
        (
          result.current.audioRef as React.MutableRefObject<HTMLAudioElement | null>
        ).current = mockAudio;
      });

      act(() => {
        result.current.seek(1.5);
      });

      // clamped to 1 * 100 = 100
      expect(mockAudio.currentTime).toBe(100);
    });

    it('sets currentTime correctly for valid fraction', () => {
      const mockAudio = createMockAudio({ duration: 200 });

      const { result } = renderHook(() => useAudioPlayback(null));

      act(() => {
        (
          result.current.audioRef as React.MutableRefObject<HTMLAudioElement | null>
        ).current = mockAudio;
      });

      act(() => {
        result.current.seek(0.25);
      });

      expect(mockAudio.currentTime).toBe(50);
    });

    it('does nothing when duration is 0', () => {
      const mockAudio = createMockAudio({ duration: 0 });

      const { result } = renderHook(() => useAudioPlayback(null));

      act(() => {
        (
          result.current.audioRef as React.MutableRefObject<HTMLAudioElement | null>
        ).current = mockAudio;
      });

      act(() => {
        result.current.seek(0.5);
      });

      // currentTime should remain at 0 since the guard returns early
      expect(mockAudio.currentTime).toBe(0);
    });

    it('does nothing when audioRef is null', () => {
      const { result } = renderHook(() => useAudioPlayback(null));

      // No mock audio attached -- audioRef.current is null
      act(() => {
        result.current.seek(0.5);
      });

      // State should remain unchanged
      expect(result.current.state.currentTime).toBe(0);
    });
  });
});
