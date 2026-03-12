import { useState, useEffect } from 'react';
import type { WaveformData } from '../types';

const DEFAULT_PEAK_COUNT = 200;

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

export function useAudioDecoder(
  file: File | null,
  peakCount: number = DEFAULT_PEAK_COUNT,
): { waveform: WaveformData | null; error: string | null } {
  const [waveform, setWaveform] = useState<WaveformData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setWaveform(null);
      setError(null);
      return;
    }

    let cancelled = false;

    async function decode(): Promise<void> {
      try {
        const arrayBuffer = await file!.arrayBuffer();
        const audioContext = new AudioContext();
        try {
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
          if (cancelled) return;
          const channelData = audioBuffer.getChannelData(0);
          const peaks = computePeaks(channelData, peakCount);
          setWaveform({ peaks, duration: audioBuffer.duration });
          setError(null);
        } finally {
          await audioContext.close();
        }
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

  return { waveform, error };
}

// Export for testing
export { computePeaks };
