import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ComparisonView } from '../ComparisonView';

// Mock AudioPlayer to isolate ComparisonView
vi.mock('../AudioPlayer', () => ({
  AudioPlayer: ({ label, src }: { label: string; src: string | null }) => (
    <div data-testid={`player-${label.toLowerCase()}`} data-src={src}>
      {label}
    </div>
  ),
}));

describe('ComparisonView', () => {
  it('renders both Original and Enhanced labels', () => {
    render(
      <ComparisonView
        originalSrc={null}
        enhancedSrc={null}
        originalPeaks={null}
        enhancedPeaks={null}
      />,
    );
    expect(screen.getByText('Original')).toBeInTheDocument();
    expect(screen.getByText('Enhanced')).toBeInTheDocument();
  });

  it('passes correct src to each player', () => {
    render(
      <ComparisonView
        originalSrc="blob:original"
        enhancedSrc="blob:enhanced"
        originalPeaks={null}
        enhancedPeaks={null}
      />,
    );
    expect(screen.getByTestId('player-original')).toHaveAttribute('data-src', 'blob:original');
    expect(screen.getByTestId('player-enhanced')).toHaveAttribute('data-src', 'blob:enhanced');
  });

  it('renders two player containers', () => {
    render(
      <ComparisonView
        originalSrc={null}
        enhancedSrc={null}
        originalPeaks={null}
        enhancedPeaks={null}
      />,
    );
    expect(screen.getByTestId('player-original')).toBeInTheDocument();
    expect(screen.getByTestId('player-enhanced')).toBeInTheDocument();
  });
});
