import { useState, useCallback, useRef, useEffect } from 'react';
import { Download, Loader2, RotateCcw, Brain, Crown, Sparkles, BarChart3 } from 'lucide-react';
import { uploadAudio, pollUntilDone, getDownloadUrl } from './api/client';
import { UploadArea } from './components/UploadArea';
import { ProgressCard } from './components/ProgressCard';
import { SettingsPanel } from './components/SettingsPanel';
import { Waveform } from './components/Waveform';
import type { ProcessingSettings, ProcessingStatus } from './types';

const INITIAL_STATUS: ProcessingStatus = {
  status: 'idle',
  progress: 0,
  message: 'Ready to enhance your audio',
};

const INITIAL_SETTINGS: ProcessingSettings = {
  noise_reduction: 'medium',
  enhance_speech: true,
  remove_reverb: false,
  isolate_voice: false,
  boost_clarity: true,
  output_format: 'wav',
  quality: 'high',
  preserve_dynamics: true,
};

export default function HarmonyRestorer() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<ProcessingStatus>(INITIAL_STATUS);
  const [settings, setSettings] = useState<ProcessingSettings>(INITIAL_SETTINGS);
  const [isProcessing, setIsProcessing] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // Cancel in-flight requests on unmount
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
    };
  }, []);

  const processAudio = useCallback(async () => {
    if (!file) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsProcessing(true);
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

      setStatus({
        status: 'completed',
        progress: 100,
        message: 'Audio enhancement complete',
        jobId: result.job_id,
        downloadUrl: getDownloadUrl(result.job_id),
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

  const handleFileSelect = (selectedFile: File) => {
    setFile(selectedFile);
    setStatus(INITIAL_STATUS);
  };

  const handleReset = () => {
    setFile(null);
    setStatus(INITIAL_STATUS);
    setIsProcessing(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 relative overflow-hidden">
      {/* Background blurs */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl" />
        <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
      </div>

      <div className="relative z-10 container mx-auto px-6 py-8 max-w-6xl">
        {/* Header */}
        <header className="text-center mb-12">
          <div className="inline-flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
              <Crown className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-white to-white/80 bg-clip-text text-transparent">
              HarmonyRestorer
            </h1>
            <Sparkles className="w-5 h-5 text-blue-400 ml-2" />
          </div>
          <p className="text-xl text-white/70 max-w-2xl mx-auto font-medium leading-relaxed">
            Professional audio restoration powered by advanced AI.
            Achieve studio-quality results with one tap.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main content */}
          <div className="lg:col-span-2 space-y-6">
            <UploadArea onFileSelect={handleFileSelect} isProcessing={isProcessing} currentFile={file} />

            {file && (
              <div className="bg-white/5 backdrop-blur-xl rounded-3xl p-6 border border-white/20 shadow-lg">
                <h3 className="text-lg font-semibold text-white/90 mb-4 flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-blue-400" />
                  <span>Waveform</span>
                </h3>
                <Waveform isActive={isProcessing} />
              </div>
            )}

            <ProgressCard status={status} />

            {/* Controls */}
            <div className="flex space-x-4">
              <button
                onClick={processAudio}
                disabled={!file || isProcessing}
                className="flex-1 bg-blue-500 hover:bg-blue-600 disabled:bg-white/10 disabled:text-white/30 text-white font-semibold py-4 px-6 rounded-2xl transition-all duration-300 flex items-center justify-center space-x-2 disabled:cursor-not-allowed shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40"
              >
                {isProcessing ? <Loader2 className="w-5 h-5 animate-spin" /> : <Brain className="w-5 h-5" />}
                <span>{isProcessing ? 'Processing...' : 'Enhance Audio'}</span>
              </button>

              {status.downloadUrl && (
                <a
                  href={status.downloadUrl}
                  download
                  className="bg-green-500 hover:bg-green-600 text-white font-semibold py-4 px-6 rounded-2xl transition-all duration-300 flex items-center space-x-2 shadow-lg shadow-green-500/30"
                >
                  <Download className="w-5 h-5" />
                  <span>Download</span>
                </a>
              )}

              <button
                onClick={handleReset}
                className="bg-white/10 hover:bg-white/20 text-white/70 hover:text-white font-semibold py-4 px-4 rounded-2xl transition-all duration-300 border border-white/20"
              >
                <RotateCcw className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Settings sidebar */}
          <div>
            <SettingsPanel settings={settings} onSettingsChange={setSettings} disabled={isProcessing} />
          </div>
        </div>

        <footer className="text-center mt-16 pt-8 border-t border-white/10">
          <p className="text-white/50 text-sm">Powered by UVR AI denoising</p>
        </footer>
      </div>
    </div>
  );
}
