import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { WaveformCanvas } from '../WaveformCanvas';

// jsdom doesn't have ResizeObserver
class MockResizeObserver {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}
globalThis.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;

// jsdom doesn't support canvas getContext, so mock it
const mockFillRect = vi.fn();
const mockClearRect = vi.fn();
const mockScale = vi.fn();

HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
  fillRect: mockFillRect,
  clearRect: mockClearRect,
  scale: mockScale,
  fillStyle: '',
}) as unknown as typeof HTMLCanvasElement.prototype.getContext;

describe('WaveformCanvas', () => {
  beforeEach(() => {
    mockFillRect.mockClear();
    mockClearRect.mockClear();
    mockScale.mockClear();
  });

  it('renders a canvas element', () => {
    const peaks = new Float32Array([0.5, 0.8, 0.3]);
    render(<WaveformCanvas peaks={peaks} />);
    expect(screen.getByRole('img', { name: 'Audio waveform' })).toBeInTheDocument();
  });

  it('has cursor-pointer class when onSeek is provided', () => {
    const peaks = new Float32Array([0.5]);
    render(<WaveformCanvas peaks={peaks} onSeek={vi.fn()} />);
    const canvas = screen.getByRole('img');
    expect(canvas.className).toContain('cursor-pointer');
  });

  it('does not have cursor-pointer without onSeek', () => {
    const peaks = new Float32Array([0.5]);
    render(<WaveformCanvas peaks={peaks} />);
    const canvas = screen.getByRole('img');
    expect(canvas.className).not.toContain('cursor-pointer');
  });

  it('calls onSeek with fraction on click', () => {
    const onSeek = vi.fn();
    const peaks = new Float32Array([0.5, 0.8]);
    render(<WaveformCanvas peaks={peaks} onSeek={onSeek} />);
    const canvas = screen.getByRole('img');

    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      left: 0, right: 200, top: 0, bottom: 64, width: 200, height: 64,
      x: 0, y: 0, toJSON: () => {},
    });

    fireEvent.click(canvas, { clientX: 100, clientY: 32 });
    expect(onSeek).toHaveBeenCalledWith(0.5);
  });

  it('clamps seek fraction to 0-1 range', () => {
    const onSeek = vi.fn();
    const peaks = new Float32Array([0.5]);
    render(<WaveformCanvas peaks={peaks} onSeek={onSeek} />);
    const canvas = screen.getByRole('img');

    vi.spyOn(canvas, 'getBoundingClientRect').mockReturnValue({
      left: 100, right: 300, top: 0, bottom: 64, width: 200, height: 64,
      x: 100, y: 0, toJSON: () => {},
    });

    // Click before canvas start
    fireEvent.click(canvas, { clientX: 50, clientY: 32 });
    expect(onSeek).toHaveBeenCalledWith(0);
  });

  it('does not draw when peaks array is empty', () => {
    const peaks = new Float32Array(0);
    render(<WaveformCanvas peaks={peaks} />);
    expect(screen.getByRole('img', { name: 'Audio waveform' })).toBeInTheDocument();
    expect(mockFillRect).not.toHaveBeenCalled();
    expect(mockClearRect).not.toHaveBeenCalled();
  });

  it('applies custom className', () => {
    const peaks = new Float32Array([0.5]);
    render(<WaveformCanvas peaks={peaks} className="my-custom" />);
    const canvas = screen.getByRole('img');
    expect(canvas.className).toContain('my-custom');
  });
});
