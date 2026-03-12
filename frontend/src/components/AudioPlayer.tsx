import { memo, useEffect } from 'react';
import { Play, Pause } from 'lucide-react';
import { useAudioPlayback } from '../hooks/useAudioPlayback';
import { WaveformCanvas } from './WaveformCanvas';

interface AudioPlayerProps {
  label: string;
  src: string | null;
  peaks: Float32Array | null;
  accentColor?: string;
  onPlay?: () => void;
  onPause?: () => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function AudioPlayerInner({
  label,
  src,
  peaks,
  accentColor = '#5B8DEF',
  onPlay,
  onPause,
}: AudioPlayerProps): React.JSX.Element {
  const { state, play, pause, seek, audioRef } = useAudioPlayback(src);

  const playhead = state.duration > 0 ? state.currentTime / state.duration : 0;

  const handlePlayPause = (): void => {
    if (state.isPlaying) {
      pause();
      onPause?.();
    } else {
      play();
      onPlay?.();
    }
  };

  // Notify parent when playback ends naturally
  useEffect(() => {
    if (!state.isPlaying && state.currentTime > 0 && state.currentTime >= state.duration) {
      onPause?.();
    }
  }, [state.isPlaying, state.currentTime, state.duration, onPause]);

  return (
    <div className="rounded-xl bg-[#282828] p-4">
      <div className="flex items-center justify-between mb-3">
        <span className="text-xs font-bold text-[#B3B3B3] uppercase tracking-widest">{label}</span>
        <span className="text-xs text-[#727272] tabular-nums">
          {formatTime(state.currentTime)} / {formatTime(state.duration)}
        </span>
      </div>

      {peaks ? (
        <WaveformCanvas
          peaks={peaks}
          playhead={playhead}
          onSeek={seek}
          accentColor={accentColor}
          baseColor="#404040"
          className="mb-3"
        />
      ) : (
        <div className="w-full h-16 bg-[#181818] rounded-lg flex items-center justify-center mb-3">
          <span className="text-sm text-[#727272]">No waveform available</span>
        </div>
      )}

      <button
        onClick={handlePlayPause}
        disabled={!src}
        className="w-10 h-10 rounded-full bg-[#5B8DEF] hover:bg-[#7BA4F7] hover:scale-105 disabled:bg-[#333333] disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center"
        aria-label={state.isPlaying ? 'Pause' : 'Play'}
      >
        {state.isPlaying ? (
          <Pause className="w-4 h-4 text-black" />
        ) : (
          <Play className="w-4 h-4 text-black ml-0.5" />
        )}
      </button>

      <audio ref={audioRef} preload="metadata" className="hidden" />
    </div>
  );
}

export const AudioPlayer = memo(AudioPlayerInner);
