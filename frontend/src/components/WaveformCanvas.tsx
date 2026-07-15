import { useRef, useEffect, useCallback } from 'react';

interface WaveformCanvasProps {
  peaks: Float32Array;
  playhead?: number;
  onSeek?: (fraction: number) => void;
  accentColor?: string;
  /** When set, the played region fills with a left-to-right gradient from accentColor to this color. */
  gradientEndColor?: string;
  baseColor?: string;
  className?: string;
}

const BAR_WIDTH = 2;
const BAR_GAP = 1;
const MIN_BAR_HEIGHT = 1;

function drawWaveform(
  canvas: HTMLCanvasElement,
  peaks: Float32Array,
  playhead: number,
  accentColor: string,
  baseColor: string,
  gradientEndColor?: string,
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;

  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const centerY = height / 2;
  const maxBarHeight = centerY - 2;
  const step = BAR_WIDTH + BAR_GAP;
  const barCount = Math.floor(width / step);
  const samplesPerBar = Math.max(1, Math.floor(peaks.length / barCount));
  const playheadX = playhead * width;

  let accentFill: string | CanvasGradient = accentColor;
  if (gradientEndColor) {
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, accentColor);
    gradient.addColorStop(1, gradientEndColor);
    accentFill = gradient;
  }

  for (let i = 0; i < barCount; i++) {
    const x = i * step;
    const sampleIndex = Math.min(Math.floor(i * peaks.length / barCount), peaks.length - 1);

    // Average nearby samples for smoother rendering
    let peak = 0;
    const start = sampleIndex;
    const end = Math.min(start + samplesPerBar, peaks.length);
    for (let j = start; j < end; j++) {
      if (peaks[j] > peak) peak = peaks[j];
    }

    const barHeight = Math.max(MIN_BAR_HEIGHT, peak * maxBarHeight);
    ctx.fillStyle = x < playheadX ? accentFill : baseColor;
    // Draw mirrored bar above and below center line
    ctx.fillRect(x, centerY - barHeight, BAR_WIDTH, barHeight);
    ctx.fillRect(x, centerY, BAR_WIDTH, barHeight);
  }
}

export function WaveformCanvas({
  peaks,
  playhead = 0,
  onSeek,
  accentColor = '#7c5fe8',
  gradientEndColor,
  baseColor = '#404040',
  className = '',
}: WaveformCanvasProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const resizeTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || peaks.length === 0) return;

    const draw = (): void => {
      drawWaveform(canvas, peaks, playhead, accentColor, baseColor, gradientEndColor);
    };
    draw();

    const observer = new ResizeObserver(() => {
      if (resizeTimerRef.current !== null) {
        clearTimeout(resizeTimerRef.current);
      }
      resizeTimerRef.current = setTimeout(draw, 100);
    });
    observer.observe(canvas);
    return () => {
      if (resizeTimerRef.current !== null) {
        clearTimeout(resizeTimerRef.current);
      }
      observer.disconnect();
    };
  }, [peaks, playhead, accentColor, baseColor, gradientEndColor]);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): void => {
      if (!onSeek) return;
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const fraction = (e.clientX - rect.left) / rect.width;
      onSeek(Math.max(0, Math.min(1, fraction)));
    },
    [onSeek],
  );

  return (
    <canvas
      ref={canvasRef}
      className={`w-full h-16 ${onSeek ? 'cursor-pointer' : ''} ${className}`}
      onClick={handleClick}
      role={onSeek ? 'slider' : 'img'}
      aria-label={onSeek ? 'Audio seek bar' : 'Audio waveform'}
      aria-valuemin={onSeek ? 0 : undefined}
      aria-valuemax={onSeek ? 100 : undefined}
      aria-valuenow={onSeek && playhead !== undefined ? Math.round(playhead * 100) : undefined}
      tabIndex={onSeek ? 0 : undefined}
      onKeyDown={onSeek ? (e: React.KeyboardEvent<HTMLCanvasElement>): void => {
        if (e.key === 'ArrowRight') onSeek(Math.min(1, (playhead ?? 0) + 0.05));
        if (e.key === 'ArrowLeft') onSeek(Math.max(0, (playhead ?? 0) - 0.05));
      } : undefined}
    />
  );
}
