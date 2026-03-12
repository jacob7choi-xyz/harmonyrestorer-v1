import { describe, it, expect } from 'vitest';
import { computePeaks } from '../useAudioDecoder';

describe('computePeaks', () => {
  it('returns correct number of peaks', () => {
    const data = new Float32Array(1000);
    const peaks = computePeaks(data, 10);
    expect(peaks.length).toBe(10);
  });

  it('extracts max absolute values per segment', () => {
    const data = new Float32Array(100);
    // Put a spike at position 5
    data[5] = 0.8;
    // Put a negative spike at position 55
    data[55] = -0.9;

    const peaks = computePeaks(data, 10);
    // First segment (0-9) should have 0.8
    expect(peaks[0]).toBeCloseTo(0.8);
    // Sixth segment (50-59) should have 0.9 (abs of -0.9)
    expect(peaks[5]).toBeCloseTo(0.9);
  });

  it('returns zeros for silent audio', () => {
    const data = new Float32Array(100);
    const peaks = computePeaks(data, 10);
    peaks.forEach(p => expect(p).toBe(0));
  });

  it('handles case where data is shorter than peak count', () => {
    const data = new Float32Array(5);
    data[0] = 0.5;
    const peaks = computePeaks(data, 10);
    // samplesPerPeak would be 0, so all peaks should be 0
    expect(peaks.length).toBe(10);
    peaks.forEach(p => expect(p).toBe(0));
  });
});
