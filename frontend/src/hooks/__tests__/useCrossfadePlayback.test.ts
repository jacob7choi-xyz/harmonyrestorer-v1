import { describe, it, expect } from 'vitest';
import { shouldResyncSlave } from '../useCrossfadePlayback';

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
