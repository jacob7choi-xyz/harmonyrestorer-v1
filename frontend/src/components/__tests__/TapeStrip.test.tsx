import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { TapeStrip } from '../TapeStrip';

class MockResizeObserver {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}
globalThis.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;

const PEAKS = new Float32Array([0.5, 0.8, 0.3, 0.6]);

describe('TapeStrip', () => {
  beforeEach(() => {
    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(1);
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('renders the strip canvas with an accessible label', () => {
    render(<TapeStrip noisyPeaks={PEAKS} cleanPeaks={PEAKS} mode="demo" />);
    const canvas = screen.getByRole('img', {
      name: 'Audio waveform, split between restored and original',
    });
    expect(canvas.tagName).toBe('CANVAS');
  });

  it('renders the comparison slider in demo mode', () => {
    render(<TapeStrip noisyPeaks={PEAKS} cleanPeaks={PEAKS} mode="demo" />);
    const slider = screen.getByRole('slider', { name: 'Comparison divider' });
    expect(slider).toHaveAttribute('aria-valuemin', '0');
    expect(slider).toHaveAttribute('aria-valuemax', '100');
    expect(slider).toHaveAttribute('aria-valuenow', '50');
  });

  it('renders the comparison slider in compare mode', () => {
    render(<TapeStrip noisyPeaks={PEAKS} cleanPeaks={PEAKS} mode="compare" />);
    expect(screen.getByRole('slider', { name: 'Comparison divider' })).toBeInTheDocument();
  });

  it('does not render the slider in processing mode', () => {
    render(<TapeStrip noisyPeaks={PEAKS} mode="processing" progress={0.4} />);
    expect(screen.queryByRole('slider')).not.toBeInTheDocument();
  });

  it('does not render the slider in file mode', () => {
    render(<TapeStrip noisyPeaks={PEAKS} mode="file" />);
    expect(screen.queryByRole('slider')).not.toBeInTheDocument();
  });

  it('moves the divider with arrow keys and reports the position', () => {
    const onDividerChange = vi.fn();
    render(
      <TapeStrip noisyPeaks={PEAKS} cleanPeaks={PEAKS} mode="demo" onDividerChange={onDividerChange} />,
    );
    const slider = screen.getByRole('slider', { name: 'Comparison divider' });

    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(slider).toHaveAttribute('aria-valuenow', '55');
    expect(onDividerChange).toHaveBeenCalledWith(0.55, expect.any(Number));

    fireEvent.keyDown(slider, { key: 'ArrowLeft' });
    expect(slider).toHaveAttribute('aria-valuenow', '50');
  });

  it('derives the drag stability margin from a fixed pixel distance', () => {
    // Pointer jitter is spatial, so the margin must come from pixels and not
    // from a fraction of the track, which would scale with its duration.
    const onDividerChange = vi.fn();
    const rect = vi
      .spyOn(HTMLCanvasElement.prototype, 'getBoundingClientRect')
      .mockReturnValue({ width: 600, height: 96 } as DOMRect);

    render(
      <TapeStrip noisyPeaks={PEAKS} cleanPeaks={PEAKS} mode="demo" onDividerChange={onDividerChange} />,
    );
    fireEvent.keyDown(screen.getByRole('slider', { name: 'Comparison divider' }), {
      key: 'ArrowRight',
    });

    // 6 px of tolerance across a 600 px strip
    expect(onDividerChange).toHaveBeenCalledWith(0.55, 0.01);
    rect.mockRestore();
  });

  it('clamps the divider at the range edges', () => {
    render(<TapeStrip noisyPeaks={PEAKS} cleanPeaks={PEAKS} mode="demo" />);
    const slider = screen.getByRole('slider', { name: 'Comparison divider' });

    for (let i = 0; i < 15; i++) {
      fireEvent.keyDown(slider, { key: 'ArrowRight' });
    }
    expect(slider).toHaveAttribute('aria-valuenow', '100');
  });

  it('seeks on canvas click', () => {
    const onSeek = vi.fn();
    render(
      <TapeStrip noisyPeaks={PEAKS} cleanPeaks={PEAKS} mode="demo" onSeek={onSeek} />,
    );
    const canvas = screen.getByRole('img');
    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      left: 0, right: 400, top: 0, bottom: 160, width: 400, height: 160,
      x: 0, y: 0, toJSON: () => {},
    });

    fireEvent.click(canvas, { clientX: 100, clientY: 80 });
    expect(onSeek).toHaveBeenCalledWith(0.25);
  });

  it('has cursor-pointer on the canvas only when seekable', () => {
    const { rerender } = render(<TapeStrip noisyPeaks={PEAKS} mode="file" />);
    expect(screen.getByRole('img').className).not.toContain('cursor-pointer');

    rerender(<TapeStrip noisyPeaks={PEAKS} mode="file" onSeek={vi.fn()} />);
    expect(screen.getByRole('img').className).toContain('cursor-pointer');
  });

  it('renders without crashing when peaks are missing', () => {
    expect(() => render(<TapeStrip noisyPeaks={null} mode="demo" />)).not.toThrow();
  });

  it('accepts a gradient palette and ticks without crashing', () => {
    expect(() => render(
      <TapeStrip
        noisyPeaks={PEAKS}
        cleanPeaks={PEAKS}
        mode="demo"
        palette={{
          clean: ['#2b4bff', '#21c8ff'],
          noisy: 'rgba(14, 14, 14, 0.82)',
          divider: '#0e0e0e',
          speckle: 'rgba(14, 14, 14, 0.28)',
        }}
        ticks
      />,
    )).not.toThrow();
  });
});
