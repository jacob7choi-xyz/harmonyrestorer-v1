import { useState, useEffect } from 'react';
import type { WaveformData } from '../types';

export const DEFAULT_PEAK_COUNT = 200;

function computePeaks(channelData: Float32Array, peakCount: number): Float32Array {
  const peaks = new Float32Array(peakCount);
  const samplesPerPeak = Math.floor(channelData.length / peakCount);
  if (samplesPerPeak === 0) return peaks;

  for (let i = 0; i < peakCount; i++) {
    let max = 0;
    const start = i * samplesPerPeak;
    const end = Math.min(start + samplesPerPeak, channelData.length);
    for (let j = start; j < end; j++) {
      const abs = Math.abs(channelData[j]);
      if (abs > max) max = abs;
    }
    peaks[i] = max;
  }
  return peaks;
}

/** Decode any audio Blob (or File) into waveform peaks and duration. */
export async function decodeBlobToWaveform(
  blob: Blob,
  peakCount: number = DEFAULT_PEAK_COUNT,
): Promise<WaveformData> {
  const arrayBuffer = await blob.arrayBuffer();
  const audioContext = new AudioContext();
  try {
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    const channelData = audioBuffer.getChannelData(0);
    return { peaks: computePeaks(channelData, peakCount), duration: audioBuffer.duration };
  } finally {
    await audioContext.close();
  }
}

export function useAudioDecoder(
  file: File | null,
  peakCount: number = DEFAULT_PEAK_COUNT,
): { waveform: WaveformData | null; error: string | null } {
  const [waveform, setWaveform] = useState<WaveformData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) return;

    let cancelled = false;

    async function decode(): Promise<void> {
      try {
        const result = await decodeBlobToWaveform(file!, peakCount);
        if (cancelled) return;
        setWaveform(result);
        setError(null);
      } catch {
        if (!cancelled) {
          setWaveform(null);
          setError('Waveform preview unavailable');
        }
      }
    }

    decode();
    return () => { cancelled = true; };
  }, [file, peakCount]);

  // When file is null, treat state as reset regardless of what's stored
  return { waveform: file ? waveform : null, error: file ? error : null };
}

// Export for testing
export { computePeaks };
