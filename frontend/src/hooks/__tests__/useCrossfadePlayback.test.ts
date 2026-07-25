import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import {
  useCrossfadePlayback,
  resolvePlaybackTransition,
  isTargetPlayable,
  isAbortError,
  sourceForPlayhead,
  AUDIBLE_FLOOR,
  type ComparisonSource,
} from '../useCrossfadePlayback';

/** Effective audible gain, accounting for the mute flag. */
function gainOf(el: HTMLAudioElement): number {
  return el.muted ? 0 : el.volume;
}

interface FakeControls {
  setTime: (t: number) => void;
  setEnded: (ended: boolean) => void;
  /** Hold seeks open so a transaction can be interrupted mid-flight. */
  deferSeeks: (defer: boolean) => void;
  releaseSeek: () => void;
  playMock: ReturnType<typeof vi.fn>;
}

/**
 * Give a jsdom media element the behavior the hook depends on: a settable
 * clock that emits seeked, a duration, and a play() that can be made to fail.
 *
 * Every volume and muted write is checked against the peer element, so the
 * never-both-audible invariant is enforced continuously rather than only at
 * the points a test happens to assert.
 */
function stubMedia(
  el: HTMLAudioElement,
  duration: number,
  violations: string[],
  peer: () => HTMLAudioElement | null,
): FakeControls {
  let time = 0;
  let paused = true;
  let ended = false;
  let deferred = false;
  let pendingSeek: (() => void) | null = null;
  let volume = 1;
  let muted = false;

  const checkInvariant = (): void => {
    const otherEl = peer();
    if (!otherEl) return;
    if (gainOf(el) > AUDIBLE_FLOOR && gainOf(otherEl) > AUDIBLE_FLOOR) {
      violations.push(`both audible: ${gainOf(el)} and ${gainOf(otherEl)}`);
    }
  };

  Object.defineProperty(el, 'volume', {
    configurable: true,
    get: () => volume,
    set: (v: number) => {
      volume = v;
      checkInvariant();
    },
  });
  Object.defineProperty(el, 'muted', {
    configurable: true,
    get: () => muted,
    set: (v: boolean) => {
      muted = v;
      checkInvariant();
    },
  });
  Object.defineProperty(el, 'currentTime', {
    configurable: true,
    get: () => time,
    set: (v: number) => {
      const land = (): void => {
        time = v;
        el.dispatchEvent(new Event('seeked'));
      };
      if (deferred) {
        pendingSeek = land;
      } else {
        land();
      }
    },
  });
  Object.defineProperty(el, 'duration', { configurable: true, get: () => duration });
  Object.defineProperty(el, 'paused', { configurable: true, get: () => paused });
  Object.defineProperty(el, 'ended', { configurable: true, get: () => ended });

  const playMock = vi.fn(() => {
    paused = false;
    return Promise.resolve();
  });
  el.play = playMock as unknown as HTMLAudioElement['play'];
  el.pause = ((): void => {
    paused = true;
    el.dispatchEvent(new Event('pause'));
  }) as HTMLAudioElement['pause'];

  return {
    setTime: (t: number) => {
      time = t;
    },
    setEnded: (v: boolean) => {
      ended = v;
    },
    deferSeeks: (v: boolean) => {
      deferred = v;
    },
    releaseSeek: () => {
      const seek = pendingSeek;
      pendingSeek = null;
      seek?.();
    },
    playMock,
  };
}

interface Harness {
  result: { current: ReturnType<typeof useCrossfadePlayback> };
  original: HTMLAudioElement;
  enhanced: HTMLAudioElement;
  originalCtl: FakeControls;
  enhancedCtl: FakeControls;
  violations: string[];
  unmount: () => void;
}

function setup(
  initialSource: ComparisonSource = 'restored',
  initialDivider = 1,
): Harness {
  const violations: string[] = [];
  const original = document.createElement('audio');
  const enhanced = document.createElement('audio');
  // Container padding makes a compressed original run slightly longer
  const originalCtl = stubMedia(original, 182.12, violations, () => enhanced);
  const enhancedCtl = stubMedia(enhanced, 182.06, violations, () => original);

  const { result, unmount } = renderHook(() => {
    const playback = useCrossfadePlayback('orig.mp3', 'enh.wav', {
      initialSource,
      initialDivider,
      rampMs: 0,
    });
    playback.originalRef.current = original;
    playback.enhancedRef.current = enhanced;
    return playback;
  });

  return { result, original, enhanced, originalCtl, enhancedCtl, violations, unmount };
}

async function flush(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
  });
}

/** Publish durations so the divider can be interpreted as a timeline boundary. */
async function announceDuration(h: Harness): Promise<void> {
  await act(async () => {
    h.original.dispatchEvent(new Event('loadedmetadata'));
    h.enhanced.dispatchEvent(new Event('loadedmetadata'));
  });
}

async function startPlaying(h: Harness): Promise<void> {
  await act(async () => {
    h.result.current.play();
  });
  await waitFor(() => expect(h.result.current.state.isPlaying).toBe(true));
}

describe('pure decision helpers', () => {
  it('keeps playback when logically stopped regardless of element state', () => {
    expect(resolvePlaybackTransition(true, true, false)).toBe('keep');
  });

  it('finishes when the active element ended, since ended also implies paused', () => {
    expect(resolvePlaybackTransition(true, true, true)).toBe('finish');
  });

  it('stops on an unexpected pause of the active element', () => {
    expect(resolvePlaybackTransition(false, true, true)).toBe('stop');
  });

  it('keeps state when playback is healthy', () => {
    expect(resolvePlaybackTransition(false, false, true)).toBe('keep');
  });

  it('treats a target as unplayable within epsilon of its end', () => {
    expect(isTargetPlayable(182.09, 182.06)).toBe(false);
    expect(isTargetPlayable(100, 182.06)).toBe(true);
  });

  it('treats an unknown target duration as playable', () => {
    expect(isTargetPlayable(100, NaN)).toBe(true);
  });

  it('plays restored while the playhead sits left of the divider', () => {
    expect(sourceForPlayhead(0.1, 0.5, 'original')).toBe('restored');
    expect(sourceForPlayhead(0, 1, 'original')).toBe('restored');
  });

  it('plays original once the playhead is right of the divider', () => {
    expect(sourceForPlayhead(0.9, 0.5, 'restored')).toBe('original');
    expect(sourceForPlayhead(0.5, 0, 'restored')).toBe('original');
  });

  it('holds the current source inside the hysteresis band', () => {
    // A divider dragged onto the playhead would otherwise chatter
    expect(sourceForPlayhead(0.5, 0.5, 'original')).toBe('original');
    expect(sourceForPlayhead(0.5, 0.5, 'restored')).toBe('restored');
    expect(sourceForPlayhead(0.498, 0.5, 'original')).toBe('original');
  });

  it('switches when the divider crosses a stationary playhead', () => {
    // Dragging the boundary from right of the playhead to left of it
    expect(sourceForPlayhead(0.4, 0.6, 'restored')).toBe('restored');
    expect(sourceForPlayhead(0.4, 0.2, 'restored')).toBe('original');
  });

  it('classifies AbortError as cancellation rather than target failure', () => {
    expect(isAbortError(new DOMException('interrupted', 'AbortError'))).toBe(true);
    expect(isAbortError(new DOMException('decode', 'NotSupportedError'))).toBe(false);
    expect(isAbortError(new Error('boom'))).toBe(false);
  });
});

describe('useCrossfadePlayback state machine', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('lands on the restored source with the original fully silent', async () => {
    const h = setup();
    await flush();

    expect(h.result.current.state.requestedSource).toBe('restored');
    expect(h.result.current.state.activeSource).toBe('restored');
    expect(gainOf(h.enhanced)).toBeGreaterThan(AUDIBLE_FLOOR);
    expect(gainOf(h.original)).toBe(0);
    expect(h.violations).toEqual([]);
  });

  it('honors an explicit original-first initial source', async () => {
    const h = setup('original');
    await flush();

    expect(h.result.current.state.activeSource).toBe('original');
    expect(gainOf(h.original)).toBeGreaterThan(AUDIBLE_FLOOR);
    expect(gainOf(h.enhanced)).toBe(0);
    expect(h.violations).toEqual([]);
  });

  it('never leaves both sources audible across a switch', async () => {
    const h = setup();
    await flush();
    await startPlaying(h);

    await act(async () => {
      h.result.current.setSource('original');
    });
    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));

    expect(gainOf(h.original)).toBeGreaterThan(AUDIBLE_FLOOR);
    expect(gainOf(h.enhanced)).toBe(0);
    expect(h.violations).toEqual([]);
  });

  it('keeps the silent track playing so a switch does not have to restart it', async () => {
    const h = setup();
    await flush();
    await startPlaying(h);

    expect(h.originalCtl.playMock).toHaveBeenCalled();
    expect(h.enhancedCtl.playMock).toHaveBeenCalled();
  });

  it('lets the newest request win when intent reverses mid-switch', async () => {
    const h = setup();
    await flush();

    h.originalCtl.setTime(50);
    h.originalCtl.deferSeeks(true);
    await act(async () => {
      h.result.current.setSource('original');
    });
    // Reverse intent while the first transaction still waits on its seek
    await act(async () => {
      h.result.current.setSource('restored');
    });
    // The stale seek lands now and must not resurrect the abandoned switch
    await act(async () => {
      h.originalCtl.releaseSeek();
    });
    await flush();

    expect(h.result.current.state.requestedSource).toBe('restored');
    expect(h.result.current.state.activeSource).toBe('restored');
    expect(gainOf(h.enhanced)).toBeGreaterThan(AUDIBLE_FLOOR);
    expect(gainOf(h.original)).toBe(0);
    expect(h.violations).toEqual([]);
  });

  it('keeps the current source audible and rolls intent back when the target cannot play', async () => {
    const h = setup();
    await flush();
    await startPlaying(h);

    h.originalCtl.playMock.mockRejectedValueOnce(
      new DOMException('no decoder', 'NotSupportedError'),
    );
    await act(async () => {
      h.result.current.setSource('original');
    });
    await waitFor(() => expect(h.result.current.state.switchError).toBe('play-failed'));

    expect(h.result.current.state.activeSource).toBe('restored');
    expect(h.result.current.state.requestedSource).toBe('restored');
    expect(gainOf(h.enhanced)).toBeGreaterThan(AUDIBLE_FLOOR);
    expect(gainOf(h.original)).toBe(0);
    expect(h.violations).toEqual([]);
  });

  it('does not treat an interrupted play as a broken target', async () => {
    const h = setup();
    await flush();
    await startPlaying(h);

    h.originalCtl.playMock.mockRejectedValueOnce(
      new DOMException('interrupted by pause', 'AbortError'),
    );
    await act(async () => {
      h.result.current.setSource('original');
    });
    await flush();

    expect(h.result.current.state.switchError).toBeNull();
    expect(h.result.current.state.requestedSource).toBe('original');
    expect(h.result.current.state.activeSource).toBe('restored');
  });

  it('refuses to switch onto a target with no playable material left', async () => {
    const h = setup();
    await flush();
    await startPlaying(h);
    // Past the shorter restored track's end but inside the original's tail
    h.enhancedCtl.setTime(182.1);

    await act(async () => {
      h.result.current.setSource('original');
    });
    await waitFor(() => expect(h.result.current.state.switchError).toBe('target-unavailable'));

    expect(h.result.current.state.activeSource).toBe('restored');
    expect(h.result.current.state.requestedSource).toBe('restored');
  });

  it('preserves the requested source across pause and commits it on resume', async () => {
    const h = setup();
    await flush();
    await startPlaying(h);

    await act(async () => {
      h.result.current.pause();
    });
    await act(async () => {
      h.result.current.setSource('original');
    });

    // Pausing invalidates in-flight work but is not a change of mind
    expect(h.result.current.state.requestedSource).toBe('original');

    await startPlaying(h);
    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));
    expect(gainOf(h.original)).toBeGreaterThan(AUDIBLE_FLOOR);
    expect(gainOf(h.enhanced)).toBe(0);
    expect(h.violations).toEqual([]);
  });

  it('ignores an ended event from the silent track', async () => {
    const h = setup();
    await flush();
    await startPlaying(h);

    await act(async () => {
      h.originalCtl.setEnded(true);
      h.original.dispatchEvent(new Event('ended'));
    });

    expect(h.result.current.state.isPlaying).toBe(true);
  });

  it('finishes playback when the active track ends', async () => {
    const h = setup();
    await flush();
    await startPlaying(h);

    await act(async () => {
      h.enhancedCtl.setEnded(true);
      h.enhanced.dispatchEvent(new Event('ended'));
    });

    expect(h.result.current.state.isPlaying).toBe(false);
  });

  it('leaves both elements paused when a stale seek lands after unmount', async () => {
    const h = setup();
    await flush();
    await startPlaying(h);

    h.originalCtl.setTime(90);
    h.originalCtl.deferSeeks(true);
    await act(async () => {
      h.result.current.setSource('original');
    });

    h.unmount();
    await act(async () => {
      h.originalCtl.releaseSeek();
    });
    await flush();

    expect(h.original.paused).toBe(true);
    expect(h.enhanced.paused).toBe(true);
  });

  it('switches when the divider is dragged across a stationary playhead', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    // Halfway through the shorter track
    h.enhancedCtl.setTime(91);
    h.originalCtl.setTime(91);

    await act(async () => {
      h.result.current.setDividerPosition(0.2);
    });
    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));

    await act(async () => {
      h.result.current.setDividerPosition(0.8);
    });
    await waitFor(() => expect(h.result.current.state.activeSource).toBe('restored'));
    expect(h.violations).toEqual([]);
  });

  it('switches when playback carries the playhead past a stationary divider', async () => {
    const h = setup('restored', 0.5);
    await flush();
    await announceDuration(h);
    await startPlaying(h);

    // Playback has advanced beyond the boundary; the tick re-derives the source
    h.enhancedCtl.setTime(160);
    await act(async () => {
      h.result.current.setDividerPosition(0.5);
    });

    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));
    expect(h.violations).toEqual([]);
  });

  it('counts switches that needed a corrective seek', async () => {
    const h = setup();
    await flush();

    h.originalCtl.setTime(120);
    await act(async () => {
      h.result.current.setSource('original');
    });
    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));

    expect(h.result.current.switchStats.current.switches).toBe(1);
    expect(h.result.current.switchStats.current.switchesNeedingSeek).toBe(1);
  });

  it('does not seek a target that is already aligned', async () => {
    const h = setup();
    await flush();

    await act(async () => {
      h.result.current.setSource('original');
    });
    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));

    expect(h.result.current.switchStats.current.switchesNeedingSeek).toBe(0);
  });
});
