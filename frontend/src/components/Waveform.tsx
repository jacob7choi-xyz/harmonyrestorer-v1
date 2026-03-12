import { useState, useEffect, useMemo } from 'react';

const IDLE_HEIGHTS: number[] = Array(32).fill(0.1);

export function Waveform({ isActive }: { isActive: boolean }) {
  const [animatedHeights, setAnimatedHeights] = useState<number[]>(IDLE_HEIGHTS);

  useEffect(() => {
    if (!isActive) {
      return;
    }

    const interval = setInterval(() => {
      setAnimatedHeights(prev => prev.map(() => Math.random() * 0.9 + 0.1));
    }, 100);

    return () => {
      clearInterval(interval);
      setAnimatedHeights(IDLE_HEIGHTS);
    };
  }, [isActive]);

  const heights = useMemo(
    () => (isActive ? animatedHeights : IDLE_HEIGHTS),
    [isActive, animatedHeights],
  );

  return (
    <div className="flex items-end justify-center space-x-1 h-16 px-4">
      {heights.map((height, i) => (
        <div
          key={i}
          className="w-1.5 bg-gradient-to-t from-blue-500 via-blue-400 to-blue-300 rounded-full transition-all duration-200 ease-out"
          style={{ height: `${height * 100}%`, minHeight: '3px' }}
        />
      ))}
    </div>
  );
}
