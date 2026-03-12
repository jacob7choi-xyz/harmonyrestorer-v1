import { useRef, useEffect } from 'react';

type Intensity = 'idle' | 'processing' | 'complete';

interface TechnoBackgroundProps {
  intensity: Intensity;
}

interface Bar {
  x: number;
  baseHeight: number;
  phase: number;
  speed: number;
  hueOffset: number;
}

interface IntensityConfig {
  amplitude: number;
  speedMultiplier: number;
  chromaSpeed: number;
}

const INTENSITY_CONFIG: Record<Intensity, IntensityConfig> = {
  idle: { amplitude: 0.3, speedMultiplier: 1, chromaSpeed: 0.15 },
  processing: { amplitude: 0.8, speedMultiplier: 2, chromaSpeed: 0.3 },
  complete: { amplitude: 0.4, speedMultiplier: 1.2, chromaSpeed: 0.15 },
};

const BAR_WIDTH = 5;
const BAR_GAP = 3;
const LERP_SPEED = 0.03;

function lerp(current: number, target: number, t: number): number {
  return current + (target - current) * t;
}

function generateBars(width: number, height: number): Bar[] {
  const totalBarWidth = BAR_WIDTH + BAR_GAP;
  const count = Math.max(1, Math.floor(width / totalBarWidth));
  const bars: Bar[] = [];

  for (let i = 0; i < count; i++) {
    bars.push({
      x: i * totalBarWidth + BAR_GAP,
      baseHeight: (0.2 + Math.random() * 0.4) * height,
      phase: Math.random() * Math.PI * 2,
      speed: 0.3 + Math.random() * 1.2,
      hueOffset: (Math.random() - 0.5) * 30,
    });
  }

  return bars;
}

export function TechnoBackground({ intensity }: TechnoBackgroundProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const barsRef = useRef<Bar[]>([]);
  const currentAmplitude = useRef(INTENSITY_CONFIG.idle.amplitude);
  const currentSpeedMult = useRef(INTENSITY_CONFIG.idle.speedMultiplier);
  const currentChromaSpeed = useRef(INTENSITY_CONFIG.idle.chromaSpeed);
  const baseHue = useRef(120);
  const intensityRef = useRef<Intensity>(intensity);

  useEffect(() => {
    intensityRef.current = intensity;
  }, [intensity]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    function resize(): void {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas!.getBoundingClientRect();
      canvas!.width = rect.width * dpr;
      canvas!.height = rect.height * dpr;
      ctx!.scale(dpr, dpr);
      barsRef.current = generateBars(rect.width, rect.height);
    }

    resize();

    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    function animate(now: number): void {
      const rect = canvas!.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;

      const target = INTENSITY_CONFIG[intensityRef.current];
      currentAmplitude.current = lerp(currentAmplitude.current, target.amplitude, LERP_SPEED);
      currentSpeedMult.current = lerp(currentSpeedMult.current, target.speedMultiplier, LERP_SPEED);
      currentChromaSpeed.current = lerp(currentChromaSpeed.current, target.chromaSpeed, LERP_SPEED);

      baseHue.current = (baseHue.current + currentChromaSpeed.current) % 360;

      ctx!.clearRect(0, 0, w, h);

      const time = now / 1000;
      const bars = barsRef.current;
      const amp = currentAmplitude.current;
      const speedMult = currentSpeedMult.current;
      const hue = baseHue.current;

      for (let i = 0; i < bars.length; i++) {
        const bar = bars[i];
        const oscillation = Math.sin(time * bar.speed * speedMult + bar.phase);
        const barHeight = bar.baseHeight * (1 + oscillation * amp);
        const barHue = (hue + bar.hueOffset + 360) % 360;
        const y = h - barHeight;

        // Glow layer
        ctx!.globalAlpha = 0.15;
        ctx!.fillStyle = `hsl(${barHue}, 80%, 55%)`;
        ctx!.fillRect(bar.x - 2, y - 2, BAR_WIDTH + 4, barHeight + 4);

        // Solid bar with vertical gradient
        ctx!.globalAlpha = 1;
        const gradient = ctx!.createLinearGradient(0, y, 0, h);
        gradient.addColorStop(0, `hsla(${barHue}, 85%, 60%, 0.9)`);
        gradient.addColorStop(0.6, `hsla(${barHue}, 80%, 45%, 0.5)`);
        gradient.addColorStop(1, `hsla(${barHue}, 75%, 30%, 0.1)`);
        ctx!.fillStyle = gradient;
        ctx!.fillRect(bar.x, y, BAR_WIDTH, barHeight);
      }

      // Reset alpha
      ctx!.globalAlpha = 1;

      rafRef.current = requestAnimationFrame(animate);
    }

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafRef.current);
      observer.disconnect();
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      data-testid="techno-background"
      className="fixed inset-0 w-full h-full"
      style={{ zIndex: 0 }}
      aria-hidden="true"
    />
  );
}
