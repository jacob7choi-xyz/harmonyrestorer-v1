import { AudioPlayer } from './AudioPlayer';

interface ComparisonViewProps {
  originalSrc: string | null;
  enhancedSrc: string | null;
  originalPeaks: Float32Array | null;
  enhancedPeaks: Float32Array | null;
}

export function ComparisonView({
  originalSrc,
  enhancedSrc,
  originalPeaks,
  enhancedPeaks,
}: ComparisonViewProps): React.JSX.Element {
  return (
    <div className="space-y-4">
      <AudioPlayer
        label="Original"
        src={originalSrc}
        peaks={originalPeaks}
        accentColor="#B3B3B3"
      />
      <AudioPlayer
        label="Enhanced"
        src={enhancedSrc}
        peaks={enhancedPeaks}
        accentColor="#5B8DEF"
      />
    </div>
  );
}
