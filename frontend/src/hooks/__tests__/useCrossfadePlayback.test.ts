import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import {
  useCrossfadePlayback,
  resolvePlaybackTransition,
  isTargetPlayable,
  isAbortError,
  sourceForBoundary,
  AUDIBLE_FLOOR,
  BOUNDARY_BAND_SECONDS,
  SWITCH_ACCEPT_TOLERANCE,
  type ComparisonSource,
} from '../useCrossfadePlayback';

/** Effective audible gain, accounting for the mute flag. */
function gainOf(el: HTMLAudioElement): number {
  return el.muted ? 0 : el.volume;
}

interface FakeControls {
  setTime: (t: number) => void;
  /** Make seeks land this far from where they were asked to go. */
  setSeekOffset: (offset: number) => void;
  setEnded: (ended: boolean) => void;
  /** Hold seeks open so a transaction can be interrupted mid-flight. */
  deferSeeks: (defer: boolean) => void;
  releaseSeek: () => void;
  playMock: ReturnType<typeof vi.fn>;
  /** Outstanding seeked listeners, to prove waiters clean up after themselves. */
  seekedListeners: () => number;
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
  const pendingSeeks: Array<() => void> = [];
  let seekOffset = 0;
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
        time = v + seekOffset;
        el.dispatchEvent(new Event('seeked'));
      };
      if (deferred) {
        pendingSeeks.push(land);
      } else {
        land();
      }
    },
  });
  Object.defineProperty(el, 'duration', { configurable: true, get: () => duration });
  Object.defineProperty(el, 'paused', { configurable: true, get: () => paused });
  Object.defineProperty(el, 'ended', { configurable: true, get: () => ended });

  let seekedListeners = 0;
  const addListener = el.addEventListener.bind(el);
  const removeListener = el.removeEventListener.bind(el);
  el.addEventListener = ((type: string, fn: EventListener) => {
    if (type === 'seeked') seekedListeners += 1;
    addListener(type, fn);
  }) as HTMLAudioElement['addEventListener'];
  el.removeEventListener = ((type: string, fn: EventListener) => {
    if (type === 'seeked') seekedListeners -= 1;
    removeListener(type, fn);
  }) as HTMLAudioElement['removeEventListener'];

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
    setSeekOffset: (offset: number) => {
      seekOffset = offset;
    },
    setEnded: (v: boolean) => {
      ended = v;
    },
    deferSeeks: (v: boolean) => {
      deferred = v;
    },
    releaseSeek: () => {
      pendingSeeks.shift()?.();
    },
    playMock,
    seekedListeners: () => seekedListeners,
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
  opts: { loop?: boolean; durations?: [number, number] } = {},
): Harness {
  const { loop = false, durations = [182.12, 182.06] } = opts;
  const violations: string[] = [];
  const original = document.createElement('audio');
  const enhanced = document.createElement('audio');
  // Container padding makes a compressed original run slightly longer
  const originalCtl = stubMedia(original, durations[0], violations, () => enhanced);
  const enhancedCtl = stubMedia(enhanced, durations[1], violations, () => original);

  const { result, unmount } = renderHook(() => {
    const playback = useCrossfadePlayback('orig.mp3', 'enh.wav', {
      initialSource,
      initialDivider,
      loop,
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

  it('plays restored while the playhead sits before the boundary', () => {
    expect(sourceForBoundary(10, 91, 'original')).toBe('restored');
    expect(sourceForBoundary(0, 182, 'original')).toBe('restored');
  });

  it('plays original once the playhead is past the boundary', () => {
    expect(sourceForBoundary(160, 91, 'restored')).toBe('original');
    expect(sourceForBoundary(5, 0, 'restored')).toBe('original');
  });

  it('holds the current source inside the boundary band', () => {
    // A divider dragged onto the playhead would otherwise chatter
    expect(sourceForBoundary(91, 91, 'original')).toBe('original');
    expect(sourceForBoundary(91, 91, 'restored')).toBe('restored');
    expect(sourceForBoundary(90.95, 91, 'original')).toBe('original');
  });

  it('switches when the boundary crosses a stationary playhead', () => {
    // Dragging the boundary from after the playhead to before it
    expect(sourceForBoundary(72, 109, 'restored')).toBe('restored');
    expect(sourceForBoundary(72, 36, 'restored')).toBe('original');
  });

  it('leaves the boundary band wider than the largest accepted commit offset', () => {
    // Structural guard: if a handoff may commit further out than the boundary
    // band tolerates, a crossing can reverse itself purely from clock change.
    expect(SWITCH_ACCEPT_TOLERANCE).toBeLessThan(BOUNDARY_BAND_SECONDS);
  });

  it('holds just inside the band and switches just outside it, in both directions', () => {
    for (const boundary of [1, 91, 300]) {
      expect(sourceForBoundary(boundary + 0.07, boundary, 'restored')).toBe('restored');
      expect(sourceForBoundary(boundary + 0.09, boundary, 'restored')).toBe('original');
      expect(sourceForBoundary(boundary - 0.07, boundary, 'original')).toBe('original');
      expect(sourceForBoundary(boundary - 0.09, boundary, 'original')).toBe('restored');
    }
  });

  it('applies the same absolute band whatever the track length', () => {
    // The band is seconds, not a fraction of the track. Expressed as a
    // fraction it would be 10 ms on a short clip and seconds on a long one,
    // so a long track would keep playing the old source well past the divider.
    expect(sourceForBoundary(91.5, 91, 'restored')).toBe('original');
    expect(sourceForBoundary(1.5, 1, 'restored')).toBe('original');
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
    // Command the transport to the middle of the shared timeline
    await act(async () => {
      h.result.current.seek(0.5);
    });

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

  it('switches promptly on a long track instead of lagging by a fraction of it', async () => {
    const h = setup('restored', 0.5);
    await flush();
    await announceDuration(h);
    await startPlaying(h);

    // Boundary is 91.03s. Half a second past it is a sliver of a 182s strip,
    // and would have fallen inside a band expressed as a percentage.
    h.enhancedCtl.setTime(91.53);
    await act(async () => {
      h.result.current.setDividerPosition(0.5);
    });

    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));
    expect(h.violations).toEqual([]);
  });

  it('does not bounce back when handoff substitutes the clock across the boundary', async () => {
    const h = setup('restored', 0.5);
    await flush();
    await announceDuration(h);
    await startPlaying(h);

    h.enhancedCtl.setTime(91.2);
    await act(async () => {
      h.result.current.setDividerPosition(0.5);
    });
    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));

    // The incoming decoder trails the outgoing one by the largest offset the
    // handoff will accept, which lands it back on the restored side of the
    // boundary. Crossing must produce one transition, not an oscillation.
    h.originalCtl.setTime(91.0);
    for (let i = 0; i < 5; i++) {
      await act(async () => {
        h.result.current.setDividerPosition(0.5);
      });
    }

    expect(h.result.current.state.activeSource).toBe('original');
    expect(h.result.current.switchStats.current.switches).toBe(1);
    expect(h.violations).toEqual([]);
  });

  it('stays inert while the boundary decision is unchanged', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    await startPlaying(h);

    const playsBefore =
      h.originalCtl.playMock.mock.calls.length + h.enhancedCtl.playMock.mock.calls.length;
    const switchesBefore = h.result.current.switchStats.current.switches;
    const seeksBefore = h.result.current.switchStats.current.switchesNeedingSeek;

    // The evaluation runs on every animation frame; a stable decision must
    // not open a transaction, restart playback, or seek anything.
    for (let i = 0; i < 100; i++) {
      await act(async () => {
        h.result.current.setDividerPosition(1);
      });
    }

    expect(h.result.current.switchStats.current.switches).toBe(switchesBefore);
    expect(h.result.current.switchStats.current.switchesNeedingSeek).toBe(seeksBefore);
    expect(
      h.originalCtl.playMock.mock.calls.length + h.enhancedCtl.playMock.mock.calls.length,
    ).toBe(playsBefore);
    expect(h.result.current.state.activeSource).toBe('restored');
    expect(h.violations).toEqual([]);
  });

  it('does not reverse a paused switch when the new decoder reports a different time', async () => {
    // The exact incident: while paused there is no animation frame to fall
    // through to the temporal rule, so if the committed decoder's clock were
    // authoritative the next pointer event would flip the source straight back.
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    await act(async () => {
      h.result.current.seek(0.5);
    });
    const transportAfterSeek = h.result.current.state.currentTime;

    await act(async () => {
      h.result.current.setDividerPosition(0.2, 0.01);
    });
    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));

    // The newly audible decoder sits 40 ms behind, inside what the handoff
    // accepts but wider than the pointer tolerance
    h.originalCtl.setTime(transportAfterSeek - 0.04);
    for (let i = 0; i < 5; i++) {
      await act(async () => {
        h.result.current.setDividerPosition(0.2, 0.01);
      });
    }

    expect(h.result.current.state.activeSource).toBe('original');
    expect(h.result.current.state.currentTime).toBeCloseTo(transportAfterSeek, 5);
    expect(h.result.current.switchStats.current.switches).toBe(1);
    expect(h.violations).toEqual([]);
  });

  it('rejects a handoff whose seek lands but stays out of tolerance', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    // Paused, so the boundary rule and the request agree and no animation
    // frame re-requests the switch while alignment is being attempted
    await act(async () => {
      h.result.current.seek(0.5);
    });

    // Target is far enough away to require a corrective seek, and its seeks
    // complete but never land near enough. A completed seek is not alignment.
    h.originalCtl.setTime(10);
    h.originalCtl.setSeekOffset(5);
    await act(async () => {
      h.result.current.setDividerPosition(0, 0.01);
    });
    await waitFor(
      () => expect(h.result.current.state.switchError).toBe('alignment-failed'),
      { timeout: 3000 },
    );

    expect(h.result.current.state.activeSource).toBe('restored');
    expect(h.result.current.state.requestedSource).toBe('restored');
    expect(gainOf(h.enhanced)).toBeGreaterThan(AUDIBLE_FLOOR);
    expect(h.result.current.switchStats.current.alignmentFailures).toBe(1);
    expect(h.violations).toEqual([]);
  });

  it('records a verified offset within tolerance on every commit', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    await startPlaying(h);

    h.originalCtl.setTime(140);
    await act(async () => {
      h.result.current.setSource('original');
    });
    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));

    expect(h.result.current.switchStats.current.maxCommitOffset).toBeLessThanOrEqual(0.05);
  });

  it('keeps a commanded seek from being overwritten by a lagging decoder', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    await startPlaying(h);

    // The decoder has not caught up yet; the animation loop must not drag the
    // commanded position back to where the element still happens to be.
    h.enhancedCtl.deferSeeks(true);
    await act(async () => {
      h.result.current.seek(0.5);
    });
    const commanded = h.result.current.state.currentTime;
    await flush();
    await flush();

    expect(h.result.current.state.currentTime).toBeCloseTo(commanded, 5);
    expect(commanded).toBeGreaterThan(90);
  });

  it('keeps a seek inside the shared comparison timeline', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);

    await act(async () => {
      h.result.current.seek(1);
    });

    // The shorter recording bounds the timeline, not whichever is audible
    expect(h.result.current.state.currentTime).toBeLessThan(182.06);
    expect(h.result.current.state.currentTime).toBeGreaterThan(181.9);
  });

  it('does not let a superseded seek mutate the silent decoder', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);

    h.enhancedCtl.deferSeeks(true);
    await act(async () => {
      h.result.current.seek(0.2);
    });
    await act(async () => {
      h.result.current.seek(0.8);
    });
    const commanded = h.result.current.state.currentTime;

    // The first seek lands late. Its continuation is stale and must not write
    // its old target into either element on the way out.
    await act(async () => {
      h.enhancedCtl.releaseSeek();
    });
    await flush();

    expect(h.original.currentTime).not.toBeCloseTo(0.2 * 182.06, 2);
    expect(h.result.current.state.currentTime).toBeCloseTo(commanded, 5);
    expect(commanded).toBeGreaterThan(140);
  });

  it('does not let an older seek timeout release a newer seek', async () => {
    vi.useFakeTimers();
    try {
      const h = setup('restored', 1);
      await flush();
      await announceDuration(h);

      h.enhancedCtl.deferSeeks(true);
      await act(async () => {
        h.result.current.seek(0.2);
      });
      // Stagger the commands so only the older timeout comes due
      await act(async () => {
        vi.advanceTimersByTime(300);
      });
      await act(async () => {
        h.result.current.seek(0.8);
      });
      const commanded = h.result.current.state.currentTime;

      // The first seek's timeout fires while the second still owns transport.
      // A flag would have been freed here; an owner token is not.
      await act(async () => {
        vi.advanceTimersByTime(150);
      });

      expect(h.result.current.state.currentTime).toBeCloseTo(commanded, 5);
    } finally {
      vi.useRealTimers();
    }
  });

  it('attempts a failing boundary demand once rather than forever', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    await act(async () => {
      h.result.current.seek(0.5);
    });
    // Let the seek's continuation settle both clocks before displacing one,
    // otherwise it lands afterwards and removes the misalignment under test
    await flush();

    h.originalCtl.setTime(10);
    h.originalCtl.setSeekOffset(5);
    await act(async () => {
      h.result.current.setDividerPosition(0, 0.01);
    });
    await waitFor(
      () => expect(h.result.current.state.switchError).toBe('alignment-failed'),
      { timeout: 3000 },
    );

    // Geometry still demands the source that just failed. Each attempt is
    // bounded, but the sequence must be too.
    for (let i = 0; i < 20; i++) {
      await act(async () => {
        h.result.current.setDividerPosition(0, 0.01);
      });
    }
    await flush();

    expect(h.result.current.switchStats.current.alignmentFailures).toBe(1);
    expect(h.result.current.state.activeSource).toBe('restored');
  });

  it('retries only after the demand genuinely changes', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    await act(async () => {
      h.result.current.seek(0.5);
    });
    // Let the seek's continuation settle both clocks before displacing one,
    // otherwise it lands afterwards and removes the misalignment under test
    await flush();

    h.originalCtl.setTime(10);
    h.originalCtl.setSeekOffset(5);
    await act(async () => {
      h.result.current.setDividerPosition(0, 0.01);
    });
    await waitFor(
      () => expect(h.result.current.state.switchError).toBe('alignment-failed'),
      { timeout: 3000 },
    );

    // Geometry asks for the other side, then comes back: that is new intent
    await act(async () => {
      h.result.current.setDividerPosition(1, 0.01);
    });
    h.originalCtl.setSeekOffset(0);
    await act(async () => {
      h.result.current.setDividerPosition(0, 0.01);
    });

    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));
    expect(h.violations).toEqual([]);
  });

  it('services a newer intent after an older transaction is superseded', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    await act(async () => {
      h.result.current.seek(0.5);
    });

    // First transition blocks on a seek that has not landed
    h.originalCtl.deferSeeks(true);
    h.originalCtl.setTime(10);
    await act(async () => {
      h.result.current.setDividerPosition(0, 0.01);
    });
    expect(h.result.current.state.activeSource).toBe('restored');

    // A scrub supersedes it. Nothing else happens: no pointer move, no play,
    // no animation frame. The pending intent must still be serviced.
    h.originalCtl.deferSeeks(false);
    await act(async () => {
      h.result.current.seek(0.6);
    });

    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));
    expect(h.violations).toEqual([]);
  });

  it('removes its seek listener when a seek never converges', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    await act(async () => {
      h.result.current.seek(0.5);
    });
    await flush();

    /* Relative to a baseline, not zero: other subscribers may legitimately
       hold a seeked listener, and counting them would make this assert the
       observer rather than the waiter. */
    const baseline = h.originalCtl.seekedListeners();

    h.originalCtl.setTime(10);
    h.originalCtl.setSeekOffset(5);
    await act(async () => {
      h.result.current.setDividerPosition(0, 0.01);
    });
    await waitFor(
      () => expect(h.result.current.state.switchError).toBe('alignment-failed'),
      { timeout: 3000 },
    );

    // The listener deliberately survives unrelated events, so its removal on
    // the timeout path is what keeps waiters from accumulating.
    await flush();
    expect(h.originalCtl.seekedListeners()).toBe(baseline);
  });

  it('does not let a superseded transaction commit or roll back the winner', async () => {
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    await act(async () => {
      h.result.current.seek(0.5);
    });
    await flush();

    // First transaction blocks partway through aligning the target
    h.originalCtl.setTime(10);
    h.originalCtl.deferSeeks(true);
    await act(async () => {
      h.result.current.setDividerPosition(0, 0.01);
    });
    expect(h.result.current.state.activeSource).toBe('restored');

    // A scrub supersedes it and its own reconciliation completes the switch
    h.originalCtl.deferSeeks(false);
    await act(async () => {
      h.result.current.seek(0.6);
    });
    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));
    const switchesAfterWinner = h.result.current.switchStats.current.switches;

    // The abandoned transaction's seek finally lands. It must not commit
    // again, roll intent back, touch gains, or release the winner's claim.
    await act(async () => {
      h.originalCtl.releaseSeek();
    });
    await flush();

    expect(h.result.current.switchStats.current.switches).toBe(switchesAfterWinner);
    expect(h.result.current.state.activeSource).toBe('original');
    expect(h.result.current.state.requestedSource).toBe('original');
    expect(h.result.current.state.switchError).toBeNull();
    expect(gainOf(h.original)).toBeGreaterThan(AUDIBLE_FLOOR);
    expect(gainOf(h.enhanced)).toBe(0);
    expect(h.violations).toEqual([]);
  });

  it('converges on the last intent after the boundary is thrashed', async () => {
    /* The contract under pathological input is not that every intermediate
       crossing is honored audibly. It is that work stays bounded, nothing is
       left owned, and the final stable intent wins once input stops. */
    const h = setup('restored', 1);
    await flush();
    await announceDuration(h);
    await act(async () => {
      h.result.current.seek(0.5);
    });
    await flush();

    for (let i = 0; i < 20; i++) {
      await act(async () => {
        h.result.current.setDividerPosition(i % 2 === 0 ? 0 : 1, 0.01);
      });
    }
    // Settle on the original side
    await act(async () => {
      h.result.current.setDividerPosition(0, 0.01);
    });

    await waitFor(() => expect(h.result.current.state.activeSource).toBe('original'));
    expect(h.result.current.state.requestedSource).toBe('original');
    expect(h.result.current.state.switchError).toBeNull();
    // Bounded: at most one commit per demand, never a runaway sequence
    expect(h.result.current.switchStats.current.switches).toBeLessThanOrEqual(21);
    expect(gainOf(h.original)).toBeGreaterThan(AUDIBLE_FLOOR);
    expect(gainOf(h.enhanced)).toBe(0);
    expect(h.violations).toEqual([]);

    // Nothing keeps working after input stops
    const settled = h.result.current.switchStats.current.switches;
    await flush();
    await flush();
    expect(h.result.current.switchStats.current.switches).toBe(settled);
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
