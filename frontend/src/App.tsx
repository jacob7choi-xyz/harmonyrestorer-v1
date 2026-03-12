import { useState, useCallback, useRef, useEffect } from 'react';
import { Download, Loader2, RotateCcw, Github, Linkedin, Instagram, Youtube } from 'lucide-react';
import { uploadAudio, pollUntilDone, getDownloadUrl } from './api/client';
import { UploadArea } from './components/UploadArea';
import { AudioPlayer } from './components/AudioPlayer';
import { WaveformCanvas } from './components/WaveformCanvas';
import { ComparisonView } from './components/ComparisonView';
import { TechnoBackground } from './components/TechnoBackground';
import { useAudioDecoder, computePeaks } from './hooks/useAudioDecoder';
import type { ProcessingStatus, WizardStep } from './types';

const INITIAL_STATUS: ProcessingStatus = {
  status: 'idle',
  progress: 0,
  message: 'Ready to enhance your audio',
};

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


export default function HarmonyRestorer(): React.JSX.Element {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<ProcessingStatus>(INITIAL_STATUS);
  const [isProcessing, setIsProcessing] = useState(false);
  const [originalBlobUrl, setOriginalBlobUrl] = useState<string | null>(null);
  const [enhancedBlobUrl, setEnhancedBlobUrl] = useState<string | null>(null);
  const [previewBlobUrl, setPreviewBlobUrl] = useState<string | null>(null);
  const [enhancedPeaks, setEnhancedPeaks] = useState<Float32Array | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const { waveform } = useAudioDecoder(file);
  const step = deriveStep(status.status);

  // Cleanup blob URLs on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (previewBlobUrl) URL.revokeObjectURL(previewBlobUrl);
      if (originalBlobUrl) URL.revokeObjectURL(originalBlobUrl);
      if (enhancedBlobUrl) URL.revokeObjectURL(enhancedBlobUrl);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const resetState = useCallback((): void => {
    abortRef.current?.abort();
    if (previewBlobUrl) URL.revokeObjectURL(previewBlobUrl);
    if (originalBlobUrl) URL.revokeObjectURL(originalBlobUrl);
    if (enhancedBlobUrl) URL.revokeObjectURL(enhancedBlobUrl);
    setFile(null);
    setStatus(INITIAL_STATUS);
    setIsProcessing(false);
    setPreviewBlobUrl(null);
    setOriginalBlobUrl(null);
    setEnhancedBlobUrl(null);
    setEnhancedPeaks(null);
  }, [previewBlobUrl, originalBlobUrl, enhancedBlobUrl]);

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
        const blob = await res.blob();
        const enhUrl = URL.createObjectURL(blob);
        setEnhancedBlobUrl(enhUrl);

        // Decode enhanced audio for waveform visualization.
        // A new AudioContext is created and closed per upload rather than using a
        // singleton. This is acceptable because uploads are infrequent and it avoids
        // managing AudioContext lifetime/suspension across the component tree.
        try {
          const arrayBuffer = await blob.arrayBuffer();
          const audioCtx = new AudioContext();
          try {
            const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
            const channelData = audioBuffer.getChannelData(0);
            setEnhancedPeaks(computePeaks(channelData, 200));
          } finally {
            await audioCtx.close();
          }
        } catch (err) {
          console.warn('Failed to decode enhanced audio waveform:', err);
        }
      } catch (err) {
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
    if (previewBlobUrl) URL.revokeObjectURL(previewBlobUrl);
    if (originalBlobUrl) URL.revokeObjectURL(originalBlobUrl);
    if (enhancedBlobUrl) URL.revokeObjectURL(enhancedBlobUrl);
    setFile(selectedFile);
    setPreviewBlobUrl(URL.createObjectURL(selectedFile));
    setStatus(INITIAL_STATUS);
    setOriginalBlobUrl(null);
    setEnhancedBlobUrl(null);
    setEnhancedPeaks(null);
  }, [isProcessing, previewBlobUrl, originalBlobUrl, enhancedBlobUrl]);

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

  const bgIntensity = step === 'processing' ? 'processing'
    : step === 'complete' ? 'complete'
    : 'idle';

  return (
    <div
      className="min-h-screen bg-[#121212]"
      onDrop={handleGlobalDrop}
      onDragOver={handleGlobalDragOver}
    >
      <TechnoBackground intensity={bgIntensity} />
      <div className="relative z-10 mx-auto max-w-2xl px-6 py-16">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-3xl font-bold text-white tracking-tight mb-2">
            HarmonyRestorer
          </h1>
          <p className="text-[#B3B3B3]">
            AI-powered audio restoration. Upload, enhance, compare.
          </p>
        </header>

        {/* Step 1: Upload */}
        {step === 'upload' && (
          <div className="space-y-6 animate-fade-in">
            <UploadArea
              onFileSelect={handleFileSelect}
              isProcessing={isProcessing}
              currentFile={file}
            />

            {waveform && (
              <AudioPlayer
                label="Preview"
                src={previewBlobUrl}
                peaks={waveform.peaks}
                accentColor="#B3B3B3"
              />
            )}

            {status.status === 'failed' && (
              <div className="bg-[#E34040]/10 rounded-lg p-4">
                <p className="text-sm text-[#E34040]">{status.message}</p>
              </div>
            )}

            <button
              onClick={processAudio}
              disabled={!file || isProcessing}
              className="w-full flex items-center justify-center gap-2 bg-[#5B8DEF] hover:bg-[#7BA4F7] hover:scale-[1.02] disabled:bg-[#333333]/50 disabled:backdrop-blur-md disabled:text-[#727272] text-black font-bold py-3.5 px-6 rounded-full transition-all disabled:cursor-not-allowed disabled:hover:scale-100"
            >
              <span>Enhance</span>
            </button>
          </div>
        )}

        {/* Step 2: Processing */}
        {step === 'processing' && (
          <div className="space-y-6 animate-fade-in">
            {waveform && (
              <div className="bg-[#282828] rounded-xl p-4">
                <WaveformCanvas peaks={waveform.peaks} accentColor="#5B8DEF" baseColor="#404040" />
              </div>
            )}

            <div className="bg-[#282828] rounded-xl p-6">
              <div className="flex items-center gap-3 mb-4">
                <Loader2 className="w-5 h-5 text-[#5B8DEF] animate-spin" />
                <span className="text-sm font-medium text-white">{status.message}</span>
              </div>

              <div className="w-full bg-[#404040] rounded-full h-1 overflow-hidden">
                <div
                  className="h-full bg-[#5B8DEF] rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${status.progress}%` }}
                />
              </div>

              <p className="text-xs text-[#727272] mt-2 text-right">
                {Math.round(status.progress)}%
              </p>
            </div>
          </div>
        )}

        {/* Step 3: Complete */}
        {step === 'complete' && (
          <div className="space-y-6 animate-fade-in">
            {status.processingTime != null && (
              <p className="text-sm text-[#B3B3B3] text-center">
                Enhanced in {status.processingTime.toFixed(1)}s
              </p>
            )}

            <ComparisonView
              originalSrc={originalBlobUrl}
              enhancedSrc={enhancedBlobUrl}
              originalPeaks={waveform?.peaks ?? null}
              enhancedPeaks={enhancedPeaks}
            />

            <div className="flex gap-3">
              {status.downloadUrl && (
                <a
                  href={status.downloadUrl}
                  download
                  className="flex-1 flex items-center justify-center gap-2 bg-[#5B8DEF] hover:bg-[#7BA4F7] hover:scale-[1.02] text-black font-bold py-3.5 px-6 rounded-full transition-all"
                >
                  <Download className="w-4 h-4" />
                  <span>Download</span>
                </a>
              )}

              <button
                onClick={resetState}
                className="flex items-center justify-center gap-2 bg-[#282828] hover:bg-[#333333] text-white font-medium py-3.5 px-5 rounded-full transition-colors"
              >
                <RotateCcw className="w-4 h-4" />
                <span>New file</span>
              </button>
            </div>
          </div>
        )}

        <footer className="text-center mt-16 pt-8 border-t border-[#282828]">
          <p className="text-[#B3B3B3] text-sm font-medium mb-6">Powered by UVR AI denoising</p>

          <div className="flex items-center justify-center gap-12 mb-6">
            <a href="https://github.com/jacob7choi-xyz" target="_blank" rel="noopener noreferrer" aria-label="GitHub" className="text-[#727272] hover:text-white transition-colors">
              <Github className="w-5 h-5" />
            </a>
            <a href="https://www.linkedin.com/in/jacobjchoi/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn" className="text-[#727272] hover:text-white transition-colors">
              <Linkedin className="w-5 h-5" />
            </a>
            <a href="https://x.com/jacob7choii" target="_blank" rel="noopener noreferrer" aria-label="X" className="text-[#727272] hover:text-white transition-colors">
              <svg viewBox="0 0 24 24" className="w-5 h-5" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" /></svg>
            </a>
            <a href="https://www.instagram.com/jacob7choi/" target="_blank" rel="noopener noreferrer" aria-label="Instagram" className="text-[#727272] hover:text-white transition-colors">
              <Instagram className="w-5 h-5" />
            </a>
            <a href="https://youtube.com/@jacob7choi?si=QUGG9m33dLDOHoxM" target="_blank" rel="noopener noreferrer" aria-label="YouTube" className="text-[#727272] hover:text-white transition-colors">
              <Youtube className="w-5 h-5" />
            </a>
          </div>

          <p className="text-[#727272] text-xs">&copy; 2026 <a href="https://jacobjchoi.xyz/" target="_blank" rel="noopener noreferrer" className="underline hover:text-white transition-colors">Jacob J. Choi</a></p>
        </footer>
      </div>
    </div>
  );
}
