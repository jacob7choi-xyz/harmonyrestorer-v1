import { useRef, useEffect } from 'react';

type Intensity = 'idle' | 'processing' | 'complete';

interface TechnoBackgroundProps {
  intensity: Intensity;
}

interface Star {
  x: number;
  y: number;
  size: number;
  phase: number;
  speed: number;
  brightness: number;
}

interface Ribbon {
  angleOffset: number;
  side: -1 | 1;
  speedMult: number;
  widthMult: number;
  lengthMult: number;
  phaseOffset: number;
  hue: number;
}

interface IntensityConfig {
  ribbonAmplitude: number;
  ribbonSpeed: number;
  ringPulse: number;
  glowSize: number;
  starBrightness: number;
}

const CONFIG: Record<Intensity, IntensityConfig> = {
  idle: {
    ribbonAmplitude: 0.8,
    ribbonSpeed: 1.0,
    ringPulse: 0.5,
    glowSize: 1.0,
    starBrightness: 0.7,
  },
  processing: {
    ribbonAmplitude: 1.2,
    ribbonSpeed: 2.5,
    ringPulse: 1.0,
    glowSize: 1.5,
    starBrightness: 1.0,
  },
  complete: {
    ribbonAmplitude: 0.9,
    ribbonSpeed: 1.2,
    ringPulse: 0.6,
    glowSize: 1.1,
    starBrightness: 0.8,
  },
};

const TAU = Math.PI * 2;
const LERP_SPEED = 0.025;
const STAR_COUNT = 150;
const RING_COUNT = 4;
const RIBBON_POINTS = 60;

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function generateStars(w: number, h: number): Star[] {
  const stars: Star[] = [];
  for (let i = 0; i < STAR_COUNT; i++) {
    stars.push({
      x: Math.random() * w,
      y: Math.random() * h,
      size: 0.8 + Math.random() * 2,
      phase: Math.random() * TAU,
      speed: 0.5 + Math.random() * 2,
      brightness: 0.5 + Math.random() * 0.5,
    });
  }
  return stars;
}

function generateRibbons(): Ribbon[] {
  return [
    { angleOffset: -0.15, side: -1, speedMult: 1.0, widthMult: 0.6, lengthMult: 1.0, phaseOffset: 0, hue: 200 },
    { angleOffset: 0.1, side: -1, speedMult: 0.8, widthMult: 0.5, lengthMult: 0.95, phaseOffset: 1.2, hue: 270 },
    { angleOffset: -0.05, side: -1, speedMult: 1.2, widthMult: 0.4, lengthMult: 0.85, phaseOffset: 2.5, hue: 240 },
    { angleOffset: -0.15, side: 1, speedMult: 1.0, widthMult: 0.6, lengthMult: 1.0, phaseOffset: 0.3, hue: 200 },
    { angleOffset: 0.1, side: 1, speedMult: 0.8, widthMult: 0.5, lengthMult: 0.95, phaseOffset: 1.5, hue: 270 },
    { angleOffset: -0.05, side: 1, speedMult: 1.2, widthMult: 0.4, lengthMult: 0.85, phaseOffset: 2.8, hue: 240 },
  ];
}

interface Point {
  x: number;
  y: number;
}

interface RibbonPointPair {
  upper: Point[];
  lower: Point[];
}

/** Pre-allocate a reusable pair of point arrays for ribbon computation. */
function createRibbonPointPair(): RibbonPointPair {
  const count = RIBBON_POINTS + 1;
  const upper: Point[] = new Array<Point>(count);
  const lower: Point[] = new Array<Point>(count);
  for (let i = 0; i < count; i++) {
    upper[i] = { x: 0, y: 0 };
    lower[i] = { x: 0, y: 0 };
  }
  return { upper, lower };
}

function computeRibbonPoints(
  cx: number,
  cy: number,
  ribbon: Ribbon,
  time: number,
  amplitude: number,
  speed: number,
  maxLen: number,
  widthScale: number,
  out: RibbonPointPair,
): void {
  const points = RIBBON_POINTS;
  const length = maxLen * ribbon.lengthMult;
  const baseWidth = maxLen * 0.18 * ribbon.widthMult * widthScale;
  const dir = ribbon.side;
  const t = time * ribbon.speedMult * speed;
  const { upper, lower } = out;

  for (let i = 0; i <= points; i++) {
    const frac = i / points;
    const dist = frac * length;

    const wave1 = Math.sin(t * 1.0 + frac * 4.5 + ribbon.phaseOffset) * amplitude;
    const wave2 = Math.sin(t * 0.6 - frac * 3 + ribbon.phaseOffset * 1.5) * amplitude * 0.5;
    const wave3 = Math.sin(t * 1.8 + frac * 8 + ribbon.phaseOffset * 0.7) * amplitude * 0.2;
    const displacement = (wave1 + wave2 + wave3) * maxLen * 0.2;

    // Wider envelope -- peaks earlier, fades more gradually
    const env = Math.pow(Math.sin(frac * Math.PI), 0.7) * (1 - frac * 0.2);
    const halfWidth = baseWidth * env * 0.5;

    // More dramatic curve path
    const baseAngle = ribbon.angleOffset + frac * 0.35 * dir;
    const px = cx + dir * dist * Math.cos(baseAngle);
    const py = cy - dist * Math.sin(Math.abs(baseAngle)) * 0.5 + displacement;

    const perpX = -Math.sin(baseAngle) * halfWidth;
    const perpY = Math.cos(baseAngle) * halfWidth;

    upper[i].x = px + perpX;
    upper[i].y = py + perpY;
    lower[i].x = px - perpX;
    lower[i].y = py - perpY;
  }
}

function smoothPath(ctx: CanvasRenderingContext2D, pts: Point[]): void {
  if (pts.length < 2) return;
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length - 1; i++) {
    const cpx = (pts[i].x + pts[i + 1].x) / 2;
    const cpy = (pts[i].y + pts[i + 1].y) / 2;
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, cpx, cpy);
  }
  const last = pts[pts.length - 1];
  ctx.lineTo(last.x, last.y);
}

// Pre-allocated point pair pools: 3 passes per drawRibbon call
const PASS_COUNT = 3;
const PASS_CONFIGS: ReadonlyArray<{ widthScale: number; alpha: number }> = [
  { widthScale: 1.2, alpha: 0.25 },
  { widthScale: 0.7, alpha: 0.6 },
  { widthScale: 0.35, alpha: 0.85 },
];
const ribbonPointPool: RibbonPointPair[] = Array.from(
  { length: PASS_COUNT },
  () => createRibbonPointPair(),
);

function drawRibbon(
  ctx: CanvasRenderingContext2D,
  cx: number,
  cy: number,
  ribbon: Ribbon,
  time: number,
  amplitude: number,
  speed: number,
  maxLen: number,
): void {
  const h = ribbon.hue;
  const dir = ribbon.side;
  const length = maxLen * ribbon.lengthMult;
  const endX = cx + dir * length * 0.8;

  // Draw 3 layered passes at different widths for rich opaque silk effect
  for (let p = 0; p < PASS_COUNT; p++) {
    const pass = PASS_CONFIGS[p];
    const out = ribbonPointPool[p];
    computeRibbonPoints(cx, cy, ribbon, time, amplitude, speed, maxLen, pass.widthScale, out);

    const gradient = ctx.createLinearGradient(cx, cy, endX, cy);
    gradient.addColorStop(0, `hsla(${h}, 80%, 55%, 0.0)`);
    gradient.addColorStop(0.04, `hsla(${h}, 90%, 60%, ${pass.alpha * 0.9})`);
    gradient.addColorStop(0.15, `hsla(${h + 15}, 95%, 65%, ${pass.alpha})`);
    gradient.addColorStop(0.4, `hsla(${h + 35}, 85%, 60%, ${pass.alpha * 0.85})`);
    gradient.addColorStop(0.7, `hsla(${h + 25}, 75%, 55%, ${pass.alpha * 0.5})`);
    gradient.addColorStop(1, `hsla(${h + 20}, 60%, 50%, 0.0)`);

    ctx.beginPath();
    smoothPath(ctx, out.upper);
    // Traverse lower edge in reverse -- avoids allocating a reversed copy
    const lower = out.lower;
    for (let i = lower.length - 1; i >= 0; i--) {
      ctx.lineTo(lower[i].x, lower[i].y);
    }
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
  }
}

export function TechnoBackground({ intensity }: TechnoBackgroundProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const starsRef = useRef<Star[]>([]);
  const ribbonsRef = useRef<Ribbon[]>(generateRibbons());
  const intensityRef = useRef<Intensity>(intensity);

  const curRibbonAmp = useRef(CONFIG.idle.ribbonAmplitude);
  const curRibbonSpd = useRef(CONFIG.idle.ribbonSpeed);
  const curRingPulse = useRef(CONFIG.idle.ringPulse);
  const curGlowSize = useRef(CONFIG.idle.glowSize);
  const curStarBright = useRef(CONFIG.idle.starBrightness);

  useEffect(() => {
    intensityRef.current = intensity;
  }, [intensity]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const glowCanvas = document.createElement('canvas');
    const glowCtx = glowCanvas.getContext('2d');
    if (!glowCtx) return;

    function resize(): void {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas!.getBoundingClientRect();
      const w = rect.width * dpr;
      const h = rect.height * dpr;
      canvas!.width = w;
      canvas!.height = h;
      glowCanvas.width = w;
      glowCanvas.height = h;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      glowCtx!.setTransform(dpr, 0, 0, dpr, 0, 0);
      starsRef.current = generateStars(rect.width, rect.height);
    }

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    function animate(now: number): void {
      const rect = canvas!.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const cx = w * 0.5;
      const cy = h * 0.48;
      const time = now / 1000;

      // Lerp all parameters
      const target = CONFIG[intensityRef.current];
      curRibbonAmp.current = lerp(curRibbonAmp.current, target.ribbonAmplitude, LERP_SPEED);
      curRibbonSpd.current = lerp(curRibbonSpd.current, target.ribbonSpeed, LERP_SPEED);
      curRingPulse.current = lerp(curRingPulse.current, target.ringPulse, LERP_SPEED);
      curGlowSize.current = lerp(curGlowSize.current, target.glowSize, LERP_SPEED);
      curStarBright.current = lerp(curStarBright.current, target.starBrightness, LERP_SPEED);

      const rAmp = curRibbonAmp.current;
      const rSpd = curRibbonSpd.current;
      const ringPulse = curRingPulse.current;
      const glowSz = curGlowSize.current;
      const starBr = curStarBright.current;

      const maxLen = Math.max(w, h) * 0.85;
      const ribbons = ribbonsRef.current;

      // -- Glow canvas: ribbons for bloom --
      glowCtx!.clearRect(0, 0, w, h);
      for (let i = 0; i < ribbons.length; i++) {
        drawRibbon(glowCtx!, cx, cy, ribbons[i], time, rAmp, rSpd, maxLen);
      }

      // -- Main canvas --
      ctx!.clearRect(0, 0, w, h);

      // Star field
      const stars = starsRef.current;
      for (let i = 0; i < stars.length; i++) {
        const s = stars[i];
        const twinkle = (Math.sin(time * s.speed + s.phase) + 1) * 0.5;
        ctx!.globalAlpha = s.brightness * twinkle * starBr;
        ctx!.fillStyle = '#c8d0ff';
        ctx!.beginPath();
        ctx!.arc(s.x, s.y, s.size, 0, TAU);
        ctx!.fill();
      }
      ctx!.globalAlpha = 1;

      // Concentric rings
      for (let i = 0; i < RING_COUNT; i++) {
        const baseRadius = (i + 1) * maxLen * 0.28;
        const pulse = Math.sin(time * 0.8 + i * 1.5) * ringPulse;
        const radius = baseRadius * (1 + pulse * 0.08);
        const alpha = 0.12 + pulse * 0.08;
        ctx!.globalAlpha = alpha;
        ctx!.strokeStyle = 'hsla(260, 70%, 60%, 1)';
        ctx!.lineWidth = 2;
        ctx!.beginPath();
        ctx!.arc(cx, cy, radius, 0, TAU);
        ctx!.stroke();
      }
      ctx!.globalAlpha = 1;

      // Bloom layer 1: tight glow
      ctx!.filter = 'blur(18px)';
      ctx!.globalAlpha = 1.0;
      ctx!.drawImage(glowCanvas, 0, 0, w, h, 0, 0, w, h);

      // Bloom layer 2: medium spread
      ctx!.filter = 'blur(40px)';
      ctx!.globalAlpha = 0.7;
      ctx!.drawImage(glowCanvas, 0, 0, w, h, 0, 0, w, h);

      // Bloom layer 3: wide ambient wash
      ctx!.filter = 'blur(80px)';
      ctx!.globalAlpha = 0.4;
      ctx!.drawImage(glowCanvas, 0, 0, w, h, 0, 0, w, h);

      // Reset filter and alpha after bloom passes
      ctx!.filter = 'none';
      ctx!.globalAlpha = 1;

      // Sharp ribbons on top
      for (let i = 0; i < ribbons.length; i++) {
        drawRibbon(ctx!, cx, cy, ribbons[i], time, rAmp, rSpd, maxLen);
      }

      // Gravitational orb -- slow, deep, controlled breath
      const breathe = 1 + Math.sin(time * 0.4) * 0.04;
      const orbR = maxLen * 0.2 * glowSz * breathe;

      // Layer 1: massive gravity well -- you feel it before you see it
      const wellR = orbR * 6;
      const wellGrad = ctx!.createRadialGradient(cx, cy, 0, cx, cy, wellR);
      wellGrad.addColorStop(0, 'hsla(220, 80%, 60%, 0.25)');
      wellGrad.addColorStop(0.15, 'hsla(240, 70%, 50%, 0.15)');
      wellGrad.addColorStop(0.4, 'hsla(260, 60%, 40%, 0.06)');
      wellGrad.addColorStop(1, 'hsla(280, 50%, 30%, 0)');
      ctx!.fillStyle = wellGrad;
      ctx!.fillRect(cx - wellR, cy - wellR, wellR * 2, wellR * 2);

      // Layer 2: dense energy field
      const midR = orbR * 3;
      const midGrad = ctx!.createRadialGradient(cx, cy, 0, cx, cy, midR);
      midGrad.addColorStop(0, 'hsla(210, 90%, 75%, 0.45)');
      midGrad.addColorStop(0.25, 'hsla(230, 85%, 60%, 0.25)');
      midGrad.addColorStop(0.6, 'hsla(260, 75%, 50%, 0.06)');
      midGrad.addColorStop(1, 'hsla(280, 60%, 40%, 0)');
      ctx!.fillStyle = midGrad;
      ctx!.fillRect(cx - midR, cy - midR, midR * 2, midR * 2);

      // Layer 3: intense core
      const coreGrad = ctx!.createRadialGradient(cx, cy, 0, cx, cy, orbR);
      coreGrad.addColorStop(0, 'hsla(200, 100%, 95%, 1.0)');
      coreGrad.addColorStop(0.08, 'hsla(210, 95%, 88%, 0.9)');
      coreGrad.addColorStop(0.25, 'hsla(225, 90%, 72%, 0.5)');
      coreGrad.addColorStop(0.5, 'hsla(250, 85%, 55%, 0.15)');
      coreGrad.addColorStop(1, 'hsla(270, 70%, 45%, 0)');
      ctx!.fillStyle = coreGrad;
      ctx!.fillRect(cx - orbR, cy - orbR, orbR * 2, orbR * 2);

      // Subtle vignette -- just darken the very edges
      const vig = ctx!.createRadialGradient(cx, cy, w * 0.3, cx, h * 0.5, w * 0.95);
      vig.addColorStop(0, 'rgba(0, 0, 0, 0)');
      vig.addColorStop(0.7, 'rgba(0, 0, 0, 0.1)');
      vig.addColorStop(1, 'rgba(0, 0, 0, 0.45)');
      ctx!.fillStyle = vig;
      ctx!.fillRect(0, 0, w, h);

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
