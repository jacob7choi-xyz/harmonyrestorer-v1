import { useState, useCallback, useRef, useEffect } from 'react';
import { Download, Play, Pause, RotateCcw, Music, Github, Linkedin, Instagram, Youtube } from 'lucide-react';
import { uploadAudio, pollUntilDone, getDownloadUrl } from './api/client';
import { UploadArea } from './components/UploadArea';
import { TapeStrip, type TapeStripPalette } from './components/TapeStrip';
import { Analytics } from '@vercel/analytics/react';
import { useAudioDecoder, decodeBlobToWaveform } from './hooks/useAudioDecoder';
import { useCrossfadePlayback } from './hooks/useCrossfadePlayback';
import type { ProcessingStatus, WizardStep } from './types';

const INITIAL_STATUS: ProcessingStatus = {
  status: 'idle',
  progress: 0,
  message: 'Ready to enhance your audio',
};

const SAMPLE_NOISY_URL = '/sample-noisy.wav';
const SAMPLE_RESTORED_URL = '/sample-restored.wav';

const FILM_GRAIN = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='120' height='120' filter='url(%23n)' opacity='0.6'/%3E%3C/svg%3E")`;

/* Warm sepia for the original audio; iridescent violet-to-cyan for the restored side. */
const AURORA_PALETTE: TapeStripPalette = {
  clean: ['#7c5cff', '#4fd1ff'],
  noisy: 'rgba(232, 168, 32, 0.78)',
  divider: '#f0ede8',
  speckle: 'rgba(245, 212, 138, 0.35)',
};

const CTA_GRADIENT =
  'bg-gradient-to-r from-[#7c5cff] to-[#4fd1ff] text-white shadow-[0_0_30px_rgba(124,92,255,0.3)]';

interface DemoAssets {
  noisyUrl: string;
  cleanUrl: string;
  noisyPeaks: Float32Array;
  cleanPeaks: Float32Array;
}

function deriveStep(status: ProcessingStatus['status']): WizardStep {
  switch (status) {
    case 'uploading':
    case 'queued':
    case 'processing':
      return 'processing';
    case 'completed':
      return 'complete';
    default:
      return 'upload';
  }
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

interface StripStageProps {
  children: React.ReactNode;
}

/** Full-bleed stage for the tape strip with the amber/violet horizon glow. */
function StripStage({ children }: StripStageProps): React.JSX.Element {
  return (
    <div className="relative left-1/2 w-screen -translate-x-1/2">
      <div aria-hidden="true" className="pointer-events-none absolute inset-0">
        <div className="absolute left-[12%] top-1/2 h-56 w-[45%] -translate-y-1/2 rounded-full bg-violet-glow blur-3xl" />
        <div className="absolute right-[12%] top-1/2 h-56 w-[45%] -translate-y-1/2 rounded-full bg-amber-glow blur-3xl" />
      </div>
      {children}
    </div>
  );
}

interface StripLabelsProps {
  time?: string;
}

/** Side labels tying the strip's two halves to the color narrative. */
function StripLabels({ time }: StripLabelsProps): React.JSX.Element {
  return (
    <div className="mt-3 flex items-center justify-between text-[0.65rem] font-semibold uppercase tracking-[0.25em]">
      <span className="text-violet-soft">Restored</span>
      {time && <span className="font-mono normal-case tracking-normal text-ink-muted">{time}</span>}
      <span className="text-amber-soft">Original</span>
    </div>
  );
}

interface PlayButtonProps {
  isPlaying: boolean;
  onClick: () => void;
  disabled?: boolean;
}

function PlayButton({ isPlaying, onClick, disabled = false }: PlayButtonProps): React.JSX.Element {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={isPlaying ? 'Pause' : 'Play'}
      className="flex h-12 w-12 items-center justify-center rounded-full bg-ink text-[#0d0d0f] transition-all hover:scale-105 disabled:cursor-not-allowed disabled:opacity-40"
    >
      {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="ml-0.5 h-5 w-5" />}
    </button>
  );
}

export default function HarmonyRestorer(): React.JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<ProcessingStatus>(INITIAL_STATUS);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSampleLoading, setIsSampleLoading] = useState(false);
  const [demo, setDemo] = useState<DemoAssets | null>(null);
  const [originalBlobUrl, setOriginalBlobUrl] = useState<string | null>(null);
  const [enhancedBlobUrl, setEnhancedBlobUrl] = useState<string | null>(null);
  const [enhancedPeaks, setEnhancedPeaks] = useState<Float32Array | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const { waveform } = useAudioDecoder(file);
  const step = deriveStep(status.status);

  const demoPlayback = useCrossfadePlayback(
    demo?.noisyUrl ?? null,
    demo?.cleanUrl ?? null,
    { loop: true },
  );
  const resultPlayback = useCrossfadePlayback(originalBlobUrl, enhancedBlobUrl);
  const { originalRef: demoOriginalRef, enhancedRef: demoEnhancedRef } = demoPlayback;
  const { originalRef: resultOriginalRef, enhancedRef: resultEnhancedRef } = resultPlayback;

  // Load the landing demo pair; the page degrades gracefully without it
  useEffect(() => {
    let cancelled = false;
    const urls: string[] = [];

    async function load(): Promise<void> {
      try {
        const [noisyRes, cleanRes] = await Promise.all([
          fetch(SAMPLE_NOISY_URL),
          fetch(SAMPLE_RESTORED_URL),
        ]);
        if (!noisyRes.ok || !cleanRes.ok) return;
        const [noisyBlob, cleanBlob] = await Promise.all([noisyRes.blob(), cleanRes.blob()]);
        const [noisyWf, cleanWf] = await Promise.all([
          decodeBlobToWaveform(noisyBlob),
          decodeBlobToWaveform(cleanBlob),
        ]);
        if (cancelled) return;
        const noisyUrl = URL.createObjectURL(noisyBlob);
        const cleanUrl = URL.createObjectURL(cleanBlob);
        urls.push(noisyUrl, cleanUrl);
        setDemo({ noisyUrl, cleanUrl, noisyPeaks: noisyWf.peaks, cleanPeaks: cleanWf.peaks });
      } catch {
        // Demo strip simply stays hidden if assets fail to load or decode
      }
    }

    load();
    return () => {
      cancelled = true;
      urls.forEach(u => URL.revokeObjectURL(u));
    };
  }, []);

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (originalBlobUrl) URL.revokeObjectURL(originalBlobUrl);
      if (enhancedBlobUrl) URL.revokeObjectURL(enhancedBlobUrl);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const resetState = useCallback((): void => {
    abortRef.current?.abort();
    if (originalBlobUrl) URL.revokeObjectURL(originalBlobUrl);
    if (enhancedBlobUrl) URL.revokeObjectURL(enhancedBlobUrl);
    setFile(null);
    setStatus(INITIAL_STATUS);
    setIsProcessing(false);
    setOriginalBlobUrl(null);
    setEnhancedBlobUrl(null);
    setEnhancedPeaks(null);
  }, [originalBlobUrl, enhancedBlobUrl]);

  const processAudio = useCallback(async (): Promise<void> => {
    if (!file) return;

    setIsProcessing(true);
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setStatus({ status: 'uploading', progress: 0, message: 'Uploading audio file...' });

    try {
      const { job_id } = await uploadAudio(file, controller.signal);
      setStatus({ status: 'queued', progress: 10, message: 'Queued for processing...', jobId: job_id });

      const result = await pollUntilDone(job_id, (update) => {
        setStatus({
          status: update.status,
          progress: update.progress,
          message: update.message,
          jobId: update.job_id,
          downloadUrl: update.download_url ?? undefined,
          processingTime: update.processing_time ?? undefined,
        });
      }, controller.signal);

      const downloadUrl = getDownloadUrl(result.job_id);

      // Create blob URL from original file for in-browser playback
      const origUrl = URL.createObjectURL(file);
      setOriginalBlobUrl(origUrl);

      try {
        const res = await fetch(downloadUrl, { signal: controller.signal });
        if (!res.ok) throw new Error(`Download failed: ${res.status}`);
        const blob = await res.blob();
        const enhUrl = URL.createObjectURL(blob);
        setEnhancedBlobUrl(enhUrl);

        try {
          const enhancedWaveform = await decodeBlobToWaveform(blob);
          setEnhancedPeaks(enhancedWaveform.peaks);
        } catch (err) {
          console.warn('Failed to decode enhanced audio waveform:', err);
        }
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') throw err;
        console.warn('Failed to fetch enhanced audio for playback:', err);
      }

      setStatus({
        status: 'completed',
        progress: 100,
        message: 'Audio enhancement complete',
        jobId: result.job_id,
        downloadUrl,
        processingTime: result.processing_time ?? undefined,
      });
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      setStatus({
        status: 'failed',
        progress: 0,
        message: err instanceof Error ? err.message : 'Processing failed',
      });
    } finally {
      setIsProcessing(false);
    }
  }, [file]);

  const handleFileSelect = useCallback((selectedFile: File): void => {
    if (isProcessing) {
      abortRef.current?.abort();
      setIsProcessing(false);
    }
    if (originalBlobUrl) URL.revokeObjectURL(originalBlobUrl);
    if (enhancedBlobUrl) URL.revokeObjectURL(enhancedBlobUrl);
    setFile(selectedFile);
    setStatus(INITIAL_STATUS);
    setOriginalBlobUrl(null);
    setEnhancedBlobUrl(null);
    setEnhancedPeaks(null);
  }, [isProcessing, originalBlobUrl, enhancedBlobUrl]);

  const handleTrySample = useCallback(async (): Promise<void> => {
    setIsSampleLoading(true);
    try {
      const res = await fetch(SAMPLE_NOISY_URL);
      if (!res.ok) throw new Error(`Sample fetch failed: ${res.status}`);
      const blob = await res.blob();
      handleFileSelect(new File([blob], 'sample-noisy.wav', { type: 'audio/wav' }));
    } catch {
      setStatus({
        status: 'failed',
        progress: 0,
        message: 'Could not load the sample. Try uploading a file instead.',
      });
    } finally {
      setIsSampleLoading(false);
    }
  }, [handleFileSelect]);

  // Global drop zone
  const handleGlobalDrop = useCallback((e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      const dropped = files[0];
      const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024;
      if (dropped.type.startsWith('audio/')) {
        if (dropped.size > MAX_FILE_SIZE_BYTES) return;
        handleFileSelect(dropped);
      }
    }
  }, [handleFileSelect]);

  const handleGlobalDragOver = useCallback((e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
  }, []);

  const handleDemoPlayPause = useCallback((): void => {
    if (demoPlayback.state.isPlaying) {
      demoPlayback.pause();
    } else {
      demoPlayback.play();
    }
  }, [demoPlayback]);

  const handleResultPlayPause = useCallback((): void => {
    if (resultPlayback.state.isPlaying) {
      resultPlayback.pause();
    } else {
      resultPlayback.play();
    }
  }, [resultPlayback]);

  const demoPlayhead = demoPlayback.state.duration > 0
    ? demoPlayback.state.currentTime / demoPlayback.state.duration
    : 0;
  const resultPlayhead = resultPlayback.state.duration > 0
    ? resultPlayback.state.currentTime / resultPlayback.state.duration
    : 0;

  return (
    <div
      className="min-h-screen overflow-x-clip bg-base"
      onDrop={handleGlobalDrop}
      onDragOver={handleGlobalDragOver}
    >
      <div aria-hidden="true" className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <div
          className="absolute left-1/2 top-[36%] h-[70vh] w-[110vw] rounded-full blur-[120px] animate-aurora-a"
          style={{
            transform: 'translate(-50%, -50%)',
            background: 'radial-gradient(ellipse at center, rgba(124, 92, 255, 0.3), transparent 60%)',
          }}
        />
        <div
          className="absolute left-[72%] top-[58%] h-[50vh] w-[65vw] rounded-full blur-[110px] animate-aurora-b"
          style={{
            transform: 'translate(-50%, -50%)',
            background: 'radial-gradient(ellipse at center, rgba(79, 209, 255, 0.22), transparent 60%)',
          }}
        />
        <div
          className="absolute left-[24%] top-[62%] h-[42vh] w-[52vw] rounded-full blur-[110px] animate-aurora-c"
          style={{
            transform: 'translate(-50%, -50%)',
            background: 'radial-gradient(ellipse at center, rgba(232, 168, 32, 0.14), transparent 60%)',
          }}
        />
      </div>

      <div
        aria-hidden="true"
        className="pointer-events-none fixed inset-0 z-20 opacity-[0.05]"
        style={{ backgroundImage: FILM_GRAIN }}
      />

      <div className="relative z-10 mx-auto max-w-3xl px-6 py-14">
        <p className="mb-16 text-center text-xs font-semibold uppercase tracking-[0.35em] text-ink-muted">
          HarmonyRestorer
        </p>

        {/* Step 1: Upload */}
        {step === 'upload' && (
          <div className="animate-fade-in">
            <header className="mb-12 text-center">
              <h1 className="font-display mb-5 text-5xl leading-[1.02] text-ink sm:text-7xl">
                Hear it the way it was <span className="italic text-amber-soft">recorded</span>
              </h1>
              <p className="mx-auto max-w-md text-lg text-ink-secondary">
                AI-powered audio restoration.
              </p>
            </header>

            {!file && demo && (
              <section aria-label="Live restoration demo" className="mb-14">
                <StripStage>
                  <TapeStrip
                    noisyPeaks={demo.noisyPeaks}
                    cleanPeaks={demo.cleanPeaks}
                    mode="demo"
                    playhead={demoPlayhead}
                    onSeek={demoPlayback.seek}
                    onMixChange={demoPlayback.setMix}
                    palette={AURORA_PALETTE}
                  />
                </StripStage>
                <StripLabels
                  time={`${formatTime(demoPlayback.state.currentTime)} / ${formatTime(demoPlayback.state.duration)}`}
                />
                <div className="mt-6 flex flex-col items-center gap-4">
                  <PlayButton
                    isPlaying={demoPlayback.state.isPlaying}
                    onClick={handleDemoPlayPause}
                  />
                  <p className="max-w-sm text-center text-xs leading-relaxed text-ink-muted">
                    Bach, degraded to tape-era noise and restored live by the model.
                    Press play, then drag the line to hear the difference.
                  </p>
                </div>
              </section>
            )}

            {file && waveform && (
              <section aria-label="Your recording" className="mb-10">
                <StripStage>
                  <TapeStrip noisyPeaks={waveform.peaks} mode="file" palette={AURORA_PALETTE} />
                </StripStage>
              </section>
            )}

            <div className="space-y-6">
              <UploadArea
                onFileSelect={handleFileSelect}
                isProcessing={isProcessing}
                currentFile={file}
              />

              {!file && (
                <button
                  onClick={handleTrySample}
                  disabled={isSampleLoading || isProcessing}
                  className="mx-auto flex items-center gap-2 text-sm text-ink-secondary transition-colors hover:text-amber-soft disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Music className="h-4 w-4" />
                  <span>{isSampleLoading ? 'Loading sample...' : 'No file handy? Try a sample'}</span>
                </button>
              )}

              {status.status === 'failed' && (
                <div className="rounded-md border border-destructive/20 bg-destructive-bg p-4 text-center">
                  <p className="text-sm text-destructive">{status.message}</p>
                </div>
              )}

              {file && (
                <button
                  onClick={processAudio}
                  disabled={isProcessing}
                  className={`mx-auto flex items-center justify-center gap-2 rounded-full px-10 py-3.5 font-bold transition-all hover:scale-[1.03] hover:brightness-110 disabled:cursor-not-allowed disabled:bg-none disabled:bg-white/5 disabled:text-ink-muted disabled:shadow-none disabled:hover:scale-100 ${CTA_GRADIENT}`}
                >
                  <span>Enhance</span>
                </button>
              )}
            </div>
          </div>
        )}

        {/* Step 2: Processing */}
        {step === 'processing' && (
          <div className="animate-fade-in">
            <header className="mb-12 text-center">
              <h1 className="font-display text-4xl text-ink sm:text-5xl">
                Restoring your recording
              </h1>
            </header>

            {waveform && (
              <StripStage>
                <TapeStrip
                  noisyPeaks={waveform.peaks}
                  mode="processing"
                  progress={status.progress / 100}
                  palette={AURORA_PALETTE}
                />
              </StripStage>
            )}

            <div className="mt-8 text-center">
              <p className="text-sm font-medium text-ink">{status.message}</p>
              <p className="mt-2 font-mono text-xs text-ink-muted">
                {Math.round(status.progress)}%
              </p>
            </div>
          </div>
        )}

        {/* Step 3: Complete */}
        {step === 'complete' && (
          <div className="animate-fade-in">
            <header className="mb-12 text-center">
              <h1 className="font-display mb-3 text-5xl text-ink sm:text-6xl">
                <span className="italic bg-gradient-to-r from-[#a78bff] to-[#4fd1ff] bg-clip-text text-transparent">
                  Restored
                </span>
              </h1>
              {status.processingTime != null && (
                <p className="text-sm text-ink-secondary">
                  Enhanced in {status.processingTime.toFixed(1)}s
                </p>
              )}
            </header>

            <StripStage>
              <TapeStrip
                noisyPeaks={waveform?.peaks ?? null}
                cleanPeaks={enhancedPeaks}
                mode="compare"
                playhead={resultPlayhead}
                onSeek={resultPlayback.seek}
                onMixChange={resultPlayback.setMix}
                palette={AURORA_PALETTE}
              />
            </StripStage>
            <StripLabels
              time={`${formatTime(resultPlayback.state.currentTime)} / ${formatTime(resultPlayback.state.duration)}`}
            />

            <div className="mt-6 flex flex-col items-center gap-4">
              <PlayButton
                isPlaying={resultPlayback.state.isPlaying}
                onClick={handleResultPlayPause}
                disabled={!originalBlobUrl && !enhancedBlobUrl}
              />
              <p className="max-w-sm text-center text-xs text-ink-muted">
                Drag the line while it plays to compare your original with the restoration.
              </p>
            </div>

            <div className="mt-10 flex justify-center gap-3">
              {status.downloadUrl && (
                <a
                  href={status.downloadUrl}
                  download
                  className={`flex items-center justify-center gap-2 rounded-full px-8 py-3.5 font-bold transition-all hover:scale-[1.02] hover:brightness-110 ${CTA_GRADIENT}`}
                >
                  <Download className="h-4 w-4" />
                  <span>Download</span>
                </a>
              )}

              <button
                onClick={resetState}
                className="flex items-center justify-center gap-2 rounded-full border border-glass bg-white/5 px-6 py-3.5 font-medium text-ink transition-colors hover:bg-white/10"
              >
                <RotateCcw className="h-4 w-4" />
                <span>New file</span>
              </button>
            </div>
          </div>
        )}

        <footer className="mt-20 border-t border-glass pt-8 text-center">
          <p className="mb-5 text-sm font-medium text-ink-secondary">Powered by OpGAN AI denoising</p>

          <div className="mb-5 flex items-center justify-center gap-8">
            <a href="https://github.com/jacob7choi-xyz" target="_blank" rel="noopener noreferrer" aria-label="GitHub" className="text-ink-muted transition-colors hover:text-amber-soft">
              <Github className="h-5 w-5" />
            </a>
            <a href="https://www.linkedin.com/in/jacobjchoi/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" className="text-ink-muted transition-colors hover:text-amber-soft">
              <Linkedin className="h-5 w-5" />
            </a>
            <a href="https://x.com/jacob7choii" target="_blank" rel="noopener noreferrer" aria-label="X" className="text-ink-muted transition-colors hover:text-amber-soft">
              <svg viewBox="0 0 24 24" className="h-5 w-5" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" /></svg>
            </a>
            <a href="https://www.instagram.com/jacob7choi/" target="_blank" rel="noopener noreferrer" aria-label="Instagram" className="text-ink-muted transition-colors hover:text-amber-soft">
              <Instagram className="h-5 w-5" />
            </a>
            <a href="https://youtube.com/@jacob7choi?si=QUGG9m33dLDOHoxM" target="_blank" rel="noopener noreferrer" aria-label="YouTube" className="text-ink-muted transition-colors hover:text-amber-soft">
              <Youtube className="h-5 w-5" />
            </a>
          </div>

          <p className="text-xs text-ink-muted">&copy; {new Date().getFullYear()} <a href="https://jacobjchoi.xyz/" target="_blank" rel="noopener noreferrer" className="underline transition-colors hover:text-ink">Jacob J. Choi</a></p>
        </footer>
      </div>

      <audio ref={demoOriginalRef} preload="auto" className="hidden" aria-hidden="true" />
      <audio ref={demoEnhancedRef} preload="auto" className="hidden" aria-hidden="true" />
      <audio ref={resultOriginalRef} preload="auto" className="hidden" aria-hidden="true" />
      <audio ref={resultEnhancedRef} preload="auto" className="hidden" aria-hidden="true" />
      <Analytics />
    </div>
  );
}
