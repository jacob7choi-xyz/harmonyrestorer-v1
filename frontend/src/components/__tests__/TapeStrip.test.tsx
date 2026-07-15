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

  it('renders the blend slider in demo mode', () => {
    render(<TapeStrip noisyPeaks={PEAKS} cleanPeaks={PEAKS} mode="demo" />);
    const slider = screen.getByRole('slider', { name: 'Restoration blend' });
    expect(slider).toHaveAttribute('aria-valuemin', '0');
    expect(slider).toHaveAttribute('aria-valuemax', '100');
    expect(slider).toHaveAttribute('aria-valuenow', '50');
  });

  it('renders the blend slider in compare mode', () => {
    render(<TapeStrip noisyPeaks={PEAKS} cleanPeaks={PEAKS} mode="compare" />);
    expect(screen.getByRole('slider', { name: 'Restoration blend' })).toBeInTheDocument();
  });

  it('does not render the slider in processing mode', () => {
    render(<TapeStrip noisyPeaks={PEAKS} mode="processing" progress={0.4} />);
    expect(screen.queryByRole('slider')).not.toBeInTheDocument();
  });

  it('does not render the slider in file mode', () => {
    render(<TapeStrip noisyPeaks={PEAKS} mode="file" />);
    expect(screen.queryByRole('slider')).not.toBeInTheDocument();
  });

  it('moves the blend with arrow keys and reports the mix', () => {
    const onMixChange = vi.fn();
    render(
      <TapeStrip noisyPeaks={PEAKS} cleanPeaks={PEAKS} mode="demo" onMixChange={onMixChange} />,
    );
    const slider = screen.getByRole('slider', { name: 'Restoration blend' });

    fireEvent.keyDown(slider, { key: 'ArrowRight' });
    expect(slider).toHaveAttribute('aria-valuenow', '55');
    expect(onMixChange).toHaveBeenCalledWith(0.55);

    fireEvent.keyDown(slider, { key: 'ArrowLeft' });
    expect(slider).toHaveAttribute('aria-valuenow', '50');
  });

  it('clamps the blend at the range edges', () => {
    render(<TapeStrip noisyPeaks={PEAKS} cleanPeaks={PEAKS} mode="demo" />);
    const slider = screen.getByRole('slider', { name: 'Restoration blend' });

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
});
