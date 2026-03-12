import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { AudioPlayer } from '../AudioPlayer';

// Mock the hooks and child components to isolate AudioPlayer logic
vi.mock('../../hooks/useAudioPlayback', () => ({
  useAudioPlayback: () => ({
    state: { isPlaying: false, currentTime: 0, duration: 120 },
    play: vi.fn(),
    pause: vi.fn(),
    seek: vi.fn(),
    audioRef: { current: null },
  }),
}));

vi.mock('../WaveformCanvas', () => ({
  WaveformCanvas: (props: { peaks: Float32Array }) => (
    <div data-testid="waveform-canvas" data-peaks-length={props.peaks.length} />
  ),
}));

describe('AudioPlayer', () => {
  it('renders label text', () => {
    render(<AudioPlayer label="Original" src={null} peaks={null} />);
    expect(screen.getByText('Original')).toBeInTheDocument();
  });

  it('shows time display', () => {
    render(<AudioPlayer label="Test" src="blob:test" peaks={null} />);
    expect(screen.getByText('0:00 / 2:00')).toBeInTheDocument();
  });

  it('renders play button when not playing', () => {
    render(<AudioPlayer label="Test" src="blob:test" peaks={null} />);
    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
  });

  it('disables play button when no src', () => {
    render(<AudioPlayer label="Test" src={null} peaks={null} />);
    expect(screen.getByRole('button', { name: 'Play' })).toBeDisabled();
  });

  it('renders waveform canvas when peaks provided', () => {
    const peaks = new Float32Array([0.5, 0.8]);
    render(<AudioPlayer label="Test" src="blob:test" peaks={peaks} />);
    expect(screen.getByTestId('waveform-canvas')).toBeInTheDocument();
  });

  it('shows fallback when no peaks', () => {
    render(<AudioPlayer label="Test" src="blob:test" peaks={null} />);
    expect(screen.getByText('No waveform available')).toBeInTheDocument();
  });

  it('renders hidden audio element', () => {
    const { container } = render(<AudioPlayer label="Test" src="blob:test" peaks={null} />);
    const audio = container.querySelector('audio');
    expect(audio).toBeInTheDocument();
    expect(audio?.className).toContain('hidden');
  });
});
