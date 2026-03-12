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

interface Bar {
  x: number;
  baseHeight: number;
  waveOffset: number;
  phase: number;
  speed: number;
}

interface IntensityConfig {
  ribbonAmplitude: number;
  ribbonSpeed: number;
  ringPulse: number;
  glowSize: number;
  starBrightness: number;
  barAmplitude: number;
  barSpeed: number;
}

const CONFIG: Record<Intensity, IntensityConfig> = {
  idle: {
    ribbonAmplitude: 0.6,
    ribbonSpeed: 0.8,
    ringPulse: 0.3,
    glowSize: 0.8,
    starBrightness: 0.5,
    barAmplitude: 0.3,
    barSpeed: 0.8,
  },
  processing: {
    ribbonAmplitude: 1.0,
    ribbonSpeed: 2.0,
    ringPulse: 0.8,
    glowSize: 1.2,
    starBrightness: 0.9,
    barAmplitude: 0.9,
    barSpeed: 2.2,
  },
  complete: {
    ribbonAmplitude: 0.7,
    ribbonSpeed: 1.0,
    ringPulse: 0.4,
    glowSize: 0.9,
    starBrightness: 0.6,
    barAmplitude: 0.4,
    barSpeed: 1.0,
  },
};

const LERP_SPEED = 0.025;
const STAR_COUNT = 150;
const RING_COUNT = 4;
const RIBBON_POINTS = 60;
const BAR_WIDTH = 14;
const BAR_GAP = 8;

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function generateStars(w: number, h: number): Star[] {
  const stars: Star[] = [];
  for (let i = 0; i < STAR_COUNT; i++) {
    stars.push({
      x: Math.random() * w,
      y: Math.random() * h,
      size: 0.5 + Math.random() * 1.5,
      phase: Math.random() * Math.PI * 2,
      speed: 0.5 + Math.random() * 2,
      brightness: 0.3 + Math.random() * 0.7,
    });
  }
  return stars;
}

function generateBars(w: number, h: number): Bar[] {
  const totalW = BAR_WIDTH + BAR_GAP;
  const count = Math.max(1, Math.floor(w / totalW));
  const bars: Bar[] = [];
  for (let i = 0; i < count; i++) {
    const norm = i / count;
    bars.push({
      x: i * totalW + BAR_GAP * 0.5,
      baseHeight: (0.08 + Math.random() * 0.22) * h,
      waveOffset: norm * Math.PI * 4,
      phase: Math.random() * Math.PI * 2,
      speed: 0.4 + Math.random() * 0.5,
    });
  }
  return bars;
}

function generateRibbons(): Ribbon[] {
  return [
    { angleOffset: -0.15, side: -1, speedMult: 1.0, widthMult: 1.0, lengthMult: 1.0, phaseOffset: 0, hue: 200 },
    { angleOffset: 0.1, side: -1, speedMult: 0.8, widthMult: 0.7, lengthMult: 0.85, phaseOffset: 1.2, hue: 270 },
    { angleOffset: -0.05, side: -1, speedMult: 1.2, widthMult: 0.5, lengthMult: 0.7, phaseOffset: 2.5, hue: 240 },
    { angleOffset: -0.15, side: 1, speedMult: 1.0, widthMult: 1.0, lengthMult: 1.0, phaseOffset: 0.3, hue: 200 },
    { angleOffset: 0.1, side: 1, speedMult: 0.8, widthMult: 0.7, lengthMult: 0.85, phaseOffset: 1.5, hue: 270 },
    { angleOffset: -0.05, side: 1, speedMult: 1.2, widthMult: 0.5, lengthMult: 0.7, phaseOffset: 2.8, hue: 240 },
  ];
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
): { upper: { x: number; y: number }[]; lower: { x: number; y: number }[] } {
  const points = RIBBON_POINTS;
  const length = maxLen * ribbon.lengthMult;
  const baseWidth = maxLen * 0.3 * ribbon.widthMult * widthScale;
  const dir = ribbon.side;
  const t = time * ribbon.speedMult * speed;

  const upper: { x: number; y: number }[] = [];
  const lower: { x: number; y: number }[] = [];

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

    upper.push({ x: px + perpX, y: py + perpY });
    lower.push({ x: px - perpX, y: py - perpY });
  }

  return { upper, lower };
}

function smoothPath(ctx: CanvasRenderingContext2D, pts: { x: number; y: number }[]): void {
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

  // Draw 3 layered passes at different widths for translucent silk effect
  const passes = [
    { widthScale: 1.6, alpha: 0.12 },
    { widthScale: 1.0, alpha: 0.3 },
    { widthScale: 0.5, alpha: 0.5 },
  ];

  for (const pass of passes) {
    const { upper, lower } = computeRibbonPoints(
      cx, cy, ribbon, time, amplitude, speed, maxLen, pass.widthScale,
    );

    const gradient = ctx.createLinearGradient(cx, cy, endX, cy);
    gradient.addColorStop(0, `hsla(${h}, 80%, 65%, 0.0)`);
    gradient.addColorStop(0.06, `hsla(${h}, 85%, 65%, ${pass.alpha * 0.8})`);
    gradient.addColorStop(0.2, `hsla(${h + 15}, 90%, 72%, ${pass.alpha})`);
    gradient.addColorStop(0.45, `hsla(${h + 35}, 80%, 78%, ${pass.alpha * 0.7})`);
    gradient.addColorStop(0.75, `hsla(${h + 25}, 70%, 82%, ${pass.alpha * 0.3})`);
    gradient.addColorStop(1, `hsla(${h + 20}, 60%, 80%, 0.0)`);

    ctx.beginPath();
    smoothPath(ctx, upper);
    // Reverse lower edge
    const revLower = [...lower].reverse();
    for (const pt of revLower) {
      ctx.lineTo(pt.x, pt.y);
    }
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();
  }
}

function drawBars(
  ctx: CanvasRenderingContext2D,
  bars: Bar[],
  w: number,
  h: number,
  time: number,
  amp: number,
  spd: number,
): void {
  const baseline = h;

  for (let i = 0; i < bars.length; i++) {
    const bar = bars[i];
    const sweep1 = Math.sin(time * 1.2 * spd + bar.waveOffset);
    const sweep2 = Math.sin(time * 0.7 * spd - bar.waveOffset * 1.5) * 0.5;
    const jitter = Math.sin(time * bar.speed * spd * 0.5 + bar.phase) * 0.15;
    const oscillation = (sweep1 + sweep2 + jitter) / 1.65;
    const barHeight = Math.max(3, bar.baseHeight * (1 + oscillation * amp));
    const y = baseline - barHeight;

    // Distance from center for hue shift
    const centerDist = Math.abs((bar.x + BAR_WIDTH / 2) / w - 0.5) * 2;
    const barHue = 220 + centerDist * 60;

    // Body gradient
    const grad = ctx.createLinearGradient(0, y, 0, baseline);
    grad.addColorStop(0, `hsla(${barHue}, 80%, 70%, 0.8)`);
    grad.addColorStop(0.15, `hsla(${barHue}, 85%, 55%, 0.7)`);
    grad.addColorStop(0.5, `hsla(${barHue}, 88%, 40%, 0.4)`);
    grad.addColorStop(1, `hsla(${barHue}, 90%, 25%, 0.05)`);

    ctx.globalAlpha = 1;
    ctx.fillStyle = grad;

    // Rounded top
    const r = Math.min(BAR_WIDTH / 2, barHeight / 2);
    if (barHeight > r * 2) {
      ctx.beginPath();
      ctx.moveTo(bar.x, baseline);
      ctx.lineTo(bar.x, y + r);
      ctx.quadraticCurveTo(bar.x, y, bar.x + r, y);
      ctx.lineTo(bar.x + BAR_WIDTH - r, y);
      ctx.quadraticCurveTo(bar.x + BAR_WIDTH, y, bar.x + BAR_WIDTH, y + r);
      ctx.lineTo(bar.x + BAR_WIDTH, baseline);
      ctx.closePath();
      ctx.fill();
    } else {
      ctx.fillRect(bar.x, y, BAR_WIDTH, barHeight);
    }

    // Subtle inner glow
    const coreW = BAR_WIDTH * 0.35;
    const coreX = bar.x + (BAR_WIDTH - coreW) / 2;
    const coreGrad = ctx.createLinearGradient(0, y, 0, y + barHeight * 0.5);
    coreGrad.addColorStop(0, `hsla(${barHue}, 40%, 95%, 0.4)`);
    coreGrad.addColorStop(1, `hsla(${barHue}, 60%, 70%, 0)`);
    ctx.fillStyle = coreGrad;
    ctx.fillRect(coreX, y + r, coreW, barHeight * 0.5);
  }

  ctx.globalAlpha = 1;
}

export function TechnoBackground({ intensity }: TechnoBackgroundProps): React.JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const starsRef = useRef<Star[]>([]);
  const barsRef = useRef<Bar[]>([]);
  const ribbonsRef = useRef<Ribbon[]>(generateRibbons());
  const intensityRef = useRef<Intensity>(intensity);

  const curRibbonAmp = useRef(CONFIG.idle.ribbonAmplitude);
  const curRibbonSpd = useRef(CONFIG.idle.ribbonSpeed);
  const curRingPulse = useRef(CONFIG.idle.ringPulse);
  const curGlowSize = useRef(CONFIG.idle.glowSize);
  const curStarBright = useRef(CONFIG.idle.starBrightness);
  const curBarAmp = useRef(CONFIG.idle.barAmplitude);
  const curBarSpd = useRef(CONFIG.idle.barSpeed);

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
      barsRef.current = generateBars(rect.width, rect.height);
    }

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    function animate(now: number): void {
      const rect = canvas!.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const cx = w * 0.5;
      const cy = h * 0.62;
      const time = now / 1000;

      // Lerp all parameters
      const target = CONFIG[intensityRef.current];
      curRibbonAmp.current = lerp(curRibbonAmp.current, target.ribbonAmplitude, LERP_SPEED);
      curRibbonSpd.current = lerp(curRibbonSpd.current, target.ribbonSpeed, LERP_SPEED);
      curRingPulse.current = lerp(curRingPulse.current, target.ringPulse, LERP_SPEED);
      curGlowSize.current = lerp(curGlowSize.current, target.glowSize, LERP_SPEED);
      curStarBright.current = lerp(curStarBright.current, target.starBrightness, LERP_SPEED);
      curBarAmp.current = lerp(curBarAmp.current, target.barAmplitude, LERP_SPEED);
      curBarSpd.current = lerp(curBarSpd.current, target.barSpeed, LERP_SPEED);

      const rAmp = curRibbonAmp.current;
      const rSpd = curRibbonSpd.current;
      const ringPulse = curRingPulse.current;
      const glowSz = curGlowSize.current;
      const starBr = curStarBright.current;
      const bAmp = curBarAmp.current;
      const bSpd = curBarSpd.current;

      const maxLen = Math.max(w, h) * 0.55;
      const ribbons = ribbonsRef.current;

      // -- Glow canvas: ribbons + bars for bloom --
      glowCtx!.clearRect(0, 0, w, h);
      for (let i = 0; i < ribbons.length; i++) {
        drawRibbon(glowCtx!, cx, cy, ribbons[i], time, rAmp, rSpd, maxLen);
      }
      drawBars(glowCtx!, barsRef.current, w, h, time, bAmp, bSpd);

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
        ctx!.arc(s.x, s.y, s.size, 0, Math.PI * 2);
        ctx!.fill();
      }
      ctx!.globalAlpha = 1;

      // Concentric rings
      for (let i = 0; i < RING_COUNT; i++) {
        const baseRadius = (i + 1) * maxLen * 0.28;
        const pulse = Math.sin(time * 0.8 + i * 1.5) * ringPulse;
        const radius = baseRadius * (1 + pulse * 0.08);
        const alpha = 0.06 + pulse * 0.04;
        ctx!.globalAlpha = alpha;
        ctx!.strokeStyle = 'hsla(260, 60%, 50%, 1)';
        ctx!.lineWidth = 1.5;
        ctx!.beginPath();
        ctx!.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx!.stroke();
      }
      ctx!.globalAlpha = 1;

      // Bloom layer 1: tight
      ctx!.save();
      ctx!.filter = 'blur(14px)';
      ctx!.globalAlpha = 0.75;
      ctx!.drawImage(glowCanvas, 0, 0, w, h, 0, 0, w, h);
      ctx!.restore();

      // Bloom layer 2: wide ambient
      ctx!.save();
      ctx!.filter = 'blur(45px)';
      ctx!.globalAlpha = 0.3;
      ctx!.drawImage(glowCanvas, 0, 0, w, h, 0, 0, w, h);
      ctx!.restore();

      // Sharp ribbons on top
      for (let i = 0; i < ribbons.length; i++) {
        drawRibbon(ctx!, cx, cy, ribbons[i], time, rAmp, rSpd, maxLen);
      }

      // Central glow orb
      const orbR = maxLen * 0.12 * glowSz;
      const orbGrad = ctx!.createRadialGradient(cx, cy, 0, cx, cy, orbR);
      orbGrad.addColorStop(0, 'hsla(210, 80%, 85%, 0.7)');
      orbGrad.addColorStop(0.3, 'hsla(230, 90%, 65%, 0.3)');
      orbGrad.addColorStop(0.7, 'hsla(260, 80%, 50%, 0.08)');
      orbGrad.addColorStop(1, 'hsla(280, 70%, 40%, 0)');
      ctx!.fillStyle = orbGrad;
      ctx!.fillRect(cx - orbR, cy - orbR, orbR * 2, orbR * 2);

      // Wide ambient orb
      const ambR = orbR * 3;
      const ambGrad = ctx!.createRadialGradient(cx, cy, 0, cx, cy, ambR);
      ambGrad.addColorStop(0, 'hsla(220, 70%, 70%, 0.15)');
      ambGrad.addColorStop(0.4, 'hsla(260, 60%, 50%, 0.06)');
      ambGrad.addColorStop(1, 'hsla(280, 50%, 30%, 0)');
      ctx!.fillStyle = ambGrad;
      ctx!.fillRect(cx - ambR, cy - ambR, ambR * 2, ambR * 2);

      // Sharp bars on top of bloom
      drawBars(ctx!, barsRef.current, w, h, time, bAmp, bSpd);

      // Vignette
      const vig = ctx!.createRadialGradient(cx, cy, w * 0.15, cx, h * 0.5, w * 0.85);
      vig.addColorStop(0, 'rgba(0, 0, 0, 0)');
      vig.addColorStop(0.5, 'rgba(0, 0, 0, 0.2)');
      vig.addColorStop(1, 'rgba(0, 0, 0, 0.7)');
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
