import { describe, it, expect } from 'vitest';
import { resolvePlaybackTransition, shouldResyncSlave } from '../useCrossfadePlayback';

describe('shouldResyncSlave', () => {
  it('resyncs a quiet background track past the drift threshold', () => {
    expect(shouldResyncSlave(0.1, 0.0)).toBe(true);
    expect(shouldResyncSlave(0.1, 0.3)).toBe(true);
  });

  it('never stutters an audible track over small drift', () => {
    expect(shouldResyncSlave(0.1, 0.7)).toBe(false);
    expect(shouldResyncSlave(0.3, 1.0)).toBe(false);
  });

  it('leaves tracks alone below the drift threshold', () => {
    expect(shouldResyncSlave(0.05, 0.0)).toBe(false);
    expect(shouldResyncSlave(0.0, 1.0)).toBe(false);
  });

  it('corrects severe drift even on an audible track', () => {
    expect(shouldResyncSlave(0.6, 1.0)).toBe(true);
  });
});

describe('resolvePlaybackTransition', () => {
  it('finishes when the master ended during logical playback', () => {
    expect(resolvePlaybackTransition(true, true, true)).toBe('finish');
  });

  it('stops when the master is paused mid-playback, covering the dead-slave blend flip', () => {
    // The B2 regression: slave play() failed, user flips the blend so the
    // paused slave becomes master; the RAF reconciliation must stop cleanly
    expect(resolvePlaybackTransition(false, true, true)).toBe('stop');
  });

  it('keeps state when playback is healthy', () => {
    expect(resolvePlaybackTransition(false, false, true)).toBe('keep');
  });

  it('is idempotent for self-induced pauses', () => {
    // pause() clears logical intent before pausing the elements, so their
    // pause events arrive with isPlaying false and must not re-transition
    expect(resolvePlaybackTransition(false, true, false)).toBe('keep');
    expect(resolvePlaybackTransition(true, true, false)).toBe('keep');
  });

  it('prefers finishing over stopping when ended implies paused', () => {
    expect(resolvePlaybackTransition(true, true, true)).toBe('finish');
  });
});
