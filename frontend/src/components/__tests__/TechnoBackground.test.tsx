import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import { TechnoBackground } from '../TechnoBackground';

class MockResizeObserver {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}
globalThis.ResizeObserver = MockResizeObserver as unknown as typeof ResizeObserver;

const mockFillRect = vi.fn();
const mockClearRect = vi.fn();
const mockScale = vi.fn();
const mockCreateLinearGradient = vi.fn().mockReturnValue({
  addColorStop: vi.fn(),
});

HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue({
  fillRect: mockFillRect,
  clearRect: mockClearRect,
  scale: mockScale,
  createLinearGradient: mockCreateLinearGradient,
  fillStyle: '',
  globalAlpha: 1,
}) as unknown as typeof HTMLCanvasElement.prototype.getContext;

describe('TechnoBackground', () => {
  beforeEach(() => {
    mockFillRect.mockClear();
    mockClearRect.mockClear();
    mockScale.mockClear();
    vi.spyOn(window, 'requestAnimationFrame').mockReturnValue(1);
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {});
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('renders a canvas element', () => {
    const { getByTestId } = render(<TechnoBackground intensity="idle" />);
    expect(getByTestId('techno-background')).toBeInTheDocument();
    expect(getByTestId('techno-background').tagName).toBe('CANVAS');
  });

  it('accepts idle intensity without crashing', () => {
    expect(() => render(<TechnoBackground intensity="idle" />)).not.toThrow();
  });

  it('accepts processing intensity without crashing', () => {
    expect(() => render(<TechnoBackground intensity="processing" />)).not.toThrow();
  });

  it('accepts complete intensity without crashing', () => {
    expect(() => render(<TechnoBackground intensity="complete" />)).not.toThrow();
  });

  it('has fixed positioning', () => {
    const { getByTestId } = render(<TechnoBackground intensity="idle" />);
    const canvas = getByTestId('techno-background');
    expect(canvas.className).toContain('fixed');
    expect(canvas.className).toContain('inset-0');
  });

  it('cancels animation frame on unmount', () => {
    const cancelSpy = vi.spyOn(window, 'cancelAnimationFrame');
    const { unmount } = render(<TechnoBackground intensity="idle" />);
    unmount();
    expect(cancelSpy).toHaveBeenCalled();
  });

  it('is hidden from accessibility tree', () => {
    const { getByTestId } = render(<TechnoBackground intensity="idle" />);
    expect(getByTestId('techno-background')).toHaveAttribute('aria-hidden', 'true');
  });

  it('handles getContext returning null', () => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = vi.fn().mockReturnValue(null) as unknown as typeof HTMLCanvasElement.prototype.getContext;
    expect(() => render(<TechnoBackground intensity="idle" />)).not.toThrow();
    HTMLCanvasElement.prototype.getContext = original;
  });
});
