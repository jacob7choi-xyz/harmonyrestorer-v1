import { useRef, useEffect, useCallback, useState } from 'react';
import { ChevronsLeftRight } from 'lucide-react';

export type TapeStripMode = 'demo' | 'compare' | 'processing' | 'file';

export interface TapeStripPalette {
  /** Restored-side bar color; a pair renders as a left-to-right gradient. */
  clean: string | readonly [string, string];
  noisy: string;
  divider: string;
  speckle: string;
}

interface TapeStripProps {
  /** Peaks of the noisy/original audio; drawn grainy right of the divider. */
  noisyPeaks: Float32Array | null;
  /** Peaks of the restored audio; drawn clean left of the divider. */
  cleanPeaks?: Float32Array | null;
  mode: TapeStripMode;
  /** Restoration progress 0..1; drives the divider in processing mode. */
  progress?: number;
  /** Playback position 0..1. */
  playhead?: number;
  onSeek?: (fraction: number) => void;
  /** Continuous mix updates while the divider moves (0 = noisy, 1 = restored). */
  onMixChange?: (mix: number) => void;
  palette?: TapeStripPalette;
  /** Draw ruler tick marks along the top edge. */
  ticks?: boolean;
  className?: string;
}

const BAR_WIDTH = 3;
const BAR_GAP = 2;
const MIN_BAR_HEIGHT = 2;
const SWEEP_SPEED = 0.5;
const SWEEP_RANGE = 0.35;
const JITTER_AMOUNT = 0.22;
const SPECKLE_COUNT = 32;
const TICK_MINOR_SPACING = 10;
const TICK_MAJOR_EVERY = 5;

const DEFAULT_PALETTE: TapeStripPalette = {
  clean: '#9d8bf0',
  noisy: 'rgba(232, 168, 32, 0.78)',
  divider: '#f0ede8',
  speckle: 'rgba(245, 212, 138, 0.35)',
};

function prefersReducedMotion(): boolean {
  return (
    typeof window.matchMedia === 'function' &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  );
}

function peakAt(peaks: Float32Array, fraction: number): number {
  if (peaks.length === 0) return 0;
  const index = Math.min(Math.floor(fraction * peaks.length), peaks.length - 1);
  return peaks[index];
}

function drawStrip(
  canvas: HTMLCanvasElement,
  noisyPeaks: Float32Array | null,
  cleanPeaks: Float32Array | null,
  divider: number,
  playhead: number,
  time: number,
  animate: boolean,
  palette: TapeStripPalette,
  ticks: boolean,
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  if (!noisyPeaks || noisyPeaks.length === 0) return;

  const centerY = height / 2;
  const maxBarHeight = centerY - 6;
  const step = BAR_WIDTH + BAR_GAP;
  const barCount = Math.floor(width / step);
  const dividerX = divider * width;
  const playheadX = playhead * width;

  let cleanFill: string | CanvasGradient;
  if (typeof palette.clean === 'string') {
    cleanFill = palette.clean;
  } else {
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, palette.clean[0]);
    gradient.addColorStop(1, palette.clean[1]);
    cleanFill = gradient;
  }

  if (ticks) {
    ctx.fillStyle = palette.divider;
    ctx.globalAlpha = 0.35;
    const tickCount = Math.floor(width / TICK_MINOR_SPACING);
    for (let t = 0; t <= tickCount; t++) {
      const tx = t * TICK_MINOR_SPACING;
      const tall = t % TICK_MAJOR_EVERY === 0;
      ctx.fillRect(tx, 0, 1, tall ? 8 : 4);
    }
    ctx.globalAlpha = 1;
  }

  for (let i = 0; i < barCount; i++) {
    const x = i * step;
    const fraction = i / barCount;
    const isClean = x < dividerX;
    // Restored side falls back to the noisy peaks when clean peaks are absent
    // (processing mode restores in place; the style change tells the story)
    const source = isClean && cleanPeaks && cleanPeaks.length > 0 ? cleanPeaks : noisyPeaks;
    let peak = peakAt(source, fraction);

    if (!isClean && animate) {
      // Grain: cheap deterministic per-bar wobble, no allocation per frame
      const wobble = Math.sin(i * 127.1 + time * 8) * 0.5 + 0.5;
      peak *= 1 + (wobble - 0.5) * JITTER_AMOUNT;
    }

    const barHeight = Math.max(MIN_BAR_HEIGHT, peak * maxBarHeight);
    ctx.fillStyle = isClean ? cleanFill : palette.noisy;
    ctx.globalAlpha = x < playheadX ? 1 : 0.62;
    ctx.fillRect(x, centerY - barHeight, BAR_WIDTH, barHeight);
    ctx.fillRect(x, centerY, BAR_WIDTH, barHeight);
  }
  ctx.globalAlpha = 1;

  // Static speckles on the noisy side
  if (animate) {
    ctx.fillStyle = palette.speckle;
    const noisyWidth = width - dividerX;
    if (noisyWidth > 8) {
      for (let k = 0; k < SPECKLE_COUNT; k++) {
        const seed = k * 73.7 + Math.floor(time * 12) * 37.3;
        const sx = dividerX + ((Math.sin(seed) * 0.5 + 0.5) * noisyWidth);
        const sy = (Math.sin(seed * 1.7) * 0.5 + 0.5) * height;
        ctx.fillRect(sx, sy, 1.5, 1.5);
      }
    }
  }

  // Divider line
  ctx.fillStyle = palette.divider;
  ctx.fillRect(dividerX - 1, 0, 2, height);

  // Traveling playhead: a bright line that moves with playback
  if (playhead > 0.001) {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.18)';
    ctx.fillRect(playheadX - 4, 0, 8, height);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
    ctx.fillRect(playheadX - 1, 0, 2, height);
  }
}

export function TapeStrip({
  noisyPeaks,
  cleanPeaks = null,
  mode,
  progress = 0,
  playhead = 0,
  onSeek,
  onMixChange,
  palette = DEFAULT_PALETTE,
  ticks = false,
  className = '',
}: TapeStripProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const thumbRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);
  const dividerRef = useRef<number>(0.5);
  const draggingRef = useRef<boolean>(false);
  const userInteractedRef = useRef<boolean>(false);
  const playheadRef = useRef<number>(playhead);
  const progressRef = useRef<number>(progress);
  // Settled divider value for aria; updated on drag end and keyboard moves
  const [dividerSettled, setDividerSettled] = useState(0.5);

  useEffect(() => {
    playheadRef.current = playhead;
    progressRef.current = progress;
  }, [playhead, progress]);

  const interactive = mode === 'demo' || mode === 'compare';

  const syncThumb = useCallback((value: number): void => {
    const thumb = thumbRef.current;
    if (thumb) {
      thumb.style.left = `${value * 100}%`;
    }
  }, []);

  const moveDivider = useCallback((value: number, settle: boolean): void => {
    const clamped = Math.max(0, Math.min(1, value));
    dividerRef.current = clamped;
    syncThumb(clamped);
    onMixChange?.(clamped);
    if (settle) setDividerSettled(clamped);
  }, [onMixChange, syncThumb]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const reduced = prefersReducedMotion();

    const frame = (now: number): void => {
      const time = now / 1000;

      if (mode === 'demo' && !userInteractedRef.current && !reduced) {
        // Idle sweep advertises the interaction and crossfades while playing
        const swept = 0.5 + Math.sin(time * SWEEP_SPEED) * SWEEP_RANGE;
        dividerRef.current = swept;
        syncThumb(swept);
        onMixChange?.(swept);
      }

      const divider = mode === 'processing' ? progressRef.current
        : mode === 'file' ? 0
        : dividerRef.current;

      drawStrip(
        canvas,
        noisyPeaks,
        cleanPeaks,
        divider,
        playheadRef.current,
        time,
        !reduced,
        palette,
        ticks,
      );

      if (!reduced) {
        rafRef.current = requestAnimationFrame(frame);
      }
    };

    rafRef.current = requestAnimationFrame(frame);

    const observer = new ResizeObserver(() => {
      if (reduced) {
        rafRef.current = requestAnimationFrame(frame);
      }
    });
    observer.observe(canvas);

    return () => {
      cancelAnimationFrame(rafRef.current);
      observer.disconnect();
    };
  }, [mode, noisyPeaks, cleanPeaks, onMixChange, syncThumb, palette, ticks]);

  // Redraw on playhead/progress changes under reduced motion (no RAF loop)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !prefersReducedMotion()) return;
    const divider = mode === 'processing' ? progress
      : mode === 'file' ? 0
      : dividerRef.current;
    drawStrip(canvas, noisyPeaks, cleanPeaks, divider, playhead, 0, false, palette, ticks);
  }, [mode, noisyPeaks, cleanPeaks, playhead, progress, palette, ticks]);

  const fractionFromEvent = useCallback((clientX: number): number => {
    const canvas = canvasRef.current;
    if (!canvas) return 0;
    const rect = canvas.getBoundingClientRect();
    return Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  }, []);

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>): void => {
    if (!onSeek || draggingRef.current) return;
    onSeek(fractionFromEvent(e.clientX));
  }, [onSeek, fractionFromEvent]);

  const handleThumbPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>): void => {
    if (!interactive) return;
    userInteractedRef.current = true;
    draggingRef.current = true;
    e.currentTarget.setPointerCapture(e.pointerId);
  }, [interactive]);

  const handleThumbPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>): void => {
    if (!draggingRef.current) return;
    moveDivider(fractionFromEvent(e.clientX), false);
  }, [moveDivider, fractionFromEvent]);

  const handleThumbPointerUp = useCallback((e: React.PointerEvent<HTMLDivElement>): void => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    e.currentTarget.releasePointerCapture(e.pointerId);
    moveDivider(fractionFromEvent(e.clientX), true);
  }, [moveDivider, fractionFromEvent]);

  const handleThumbKeyDown = useCallback((e: React.KeyboardEvent<HTMLDivElement>): void => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    userInteractedRef.current = true;
    const delta = e.key === 'ArrowRight' ? 0.05 : -0.05;
    moveDivider(dividerRef.current + delta, true);
  }, [moveDivider]);

  return (
    <div className={`relative h-40 select-none ${className}`}>
      <canvas
        ref={canvasRef}
        data-testid="tape-strip"
        className={`absolute inset-0 w-full h-full ${onSeek ? 'cursor-pointer' : ''}`}
        onClick={handleCanvasClick}
        role="img"
        aria-label="Audio waveform, split between restored and original"
      />
      {interactive && (
        <div
          ref={thumbRef}
          role="slider"
          tabIndex={0}
          aria-label="Restoration blend"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(dividerSettled * 100)}
          aria-valuetext={`${Math.round(dividerSettled * 100)}% restored`}
          onPointerDown={handleThumbPointerDown}
          onPointerMove={handleThumbPointerMove}
          onPointerUp={handleThumbPointerUp}
          onKeyDown={handleThumbKeyDown}
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-ink border border-glass-active shadow-lg cursor-ew-resize touch-none flex items-center justify-center"
          style={{ left: '50%' }}
        >
          <ChevronsLeftRight aria-hidden="true" className="w-3.5 h-3.5" style={{ color: 'var(--color-base)' }} />
        </div>
      )}
    </div>
  );
}
