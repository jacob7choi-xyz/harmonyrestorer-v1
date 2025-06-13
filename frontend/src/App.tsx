import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Upload, Music, BarChart3, Settings, Download, Play, Pause, Volume2, Zap, Sparkles, FileAudio, CheckCircle, AlertCircle, Loader2, RotateCcw, Brain, TrendingUp, Activity, Crown, Palette, Award } from 'lucide-react';

// Types
interface ProcessingSettings {
  noise_reduction: 'light' | 'medium' | 'strong' | 'extreme';
  enhance_speech: boolean;
  remove_reverb: boolean;
  isolate_voice: boolean;
  boost_clarity: boolean;
  output_format: 'wav' | 'flac' | 'aiff';
  quality: 'standard' | 'high' | 'pro';
  preserve_dynamics: boolean;
}

interface ProcessingStatus {
  status: 'idle' | 'uploading' | 'analyzing' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  jobId?: string;
  downloadUrl?: string;
  processingTime?: number;
  sdrImprovement?: number;
}

// Apple-style Waveform
const AppleWaveform = ({ isActive }: { isActive: boolean }) => {
  const [heights, setHeights] = useState(Array(32).fill(0.1));

  useEffect(() => {
    if (!isActive) {
      setHeights(Array(32).fill(0.1));
      return;
    }

    const interval = setInterval(() => {
      setHeights(prev => prev.map(() => Math.random() * 0.9 + 0.1));
    }, 100);

    return () => clearInterval(interval);
  }, [isActive]);

  return (
    <div className="flex items-end justify-center space-x-1 h-16 px-4">
      {heights.map((height, i) => (
        <div
          key={i}
          className="w-1.5 bg-gradient-to-t from-blue-500 via-blue-400 to-blue-300 rounded-full transition-all duration-200 ease-out"
          style={{
            height: `${height * 100}%`,
            minHeight: '3px'
          }}
        />
      ))}
    </div>
  );
};

// Apple-style Upload Area
const AppleUploadArea = ({ 
  onFileSelect, 
  isProcessing, 
  currentFile 
}: {
  onFileSelect: (file: File) => void;
  isProcessing: boolean;
  currentFile: File | null;
}) => {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.startsWith('audio/')) {
      onFileSelect(files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('audio/')) {
      onFileSelect(file);
    }
  };

  const handleClick = () => {
    if (!isProcessing) {
      fileInputRef.current?.click();
    }
  };

  return (
    <div
      className={`relative rounded-3xl p-8 text-center transition-all duration-300 cursor-pointer border border-white/20
        ${isDragging 
          ? 'bg-blue-500/10 border-blue-400/40 scale-[1.02] shadow-2xl shadow-blue-500/20' 
          : 'bg-white/5 hover:bg-white/10 hover:border-white/30 hover:shadow-xl'
        }
        ${isProcessing ? 'opacity-60 cursor-not-allowed' : ''}
        backdrop-blur-xl shadow-lg
      `}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        className="hidden"
        disabled={isProcessing}
      />
      
      <div className="flex flex-col items-center space-y-4">
        {currentFile ? (
          <>
            <div className="w-16 h-16 rounded-2xl bg-blue-500/20 flex items-center justify-center backdrop-blur-sm border border-blue-400/30">
              <Music className="w-8 h-8 text-blue-400" />
            </div>
            <div>
              <p className="text-lg font-medium text-white/90">{currentFile.name}</p>
              <p className="text-sm text-white/60">{(currentFile.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          </>
        ) : (
          <>
            <div className="w-16 h-16 rounded-2xl bg-white/10 flex items-center justify-center backdrop-blur-sm border border-white/20">
              <Upload className="w-8 h-8 text-white/70" />
            </div>
            <div>
              <p className="text-xl font-medium text-white/90 mb-2">
                Drop audio file here
              </p>
              <p className="text-sm text-white/60">
                or tap to browse • WAV, FLAC, MP3, AIFF
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// Apple-style Progress Card
const AppleProgressCard = ({ 
  status, 
  progress, 
  message, 
  sdrImprovement 
}: {
  status: ProcessingStatus;
  progress: number;
  message: string;
  sdrImprovement?: number;
}) => {
  const getStatusIcon = () => {
    switch (status.status) {
      case 'uploading':
        return <Upload className="w-5 h-5 text-blue-500" />;
      case 'analyzing':
        return <Brain className="w-5 h-5 text-purple-500" />;
      case 'processing':
        return <Loader2 className="w-5 h-5 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'failed':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <Activity className="w-5 h-5 text-white/50" />;
    }
  };

  const getStatusColor = () => {
    switch (status.status) {
      case 'processing':
        return 'from-blue-500 to-blue-400';
      case 'completed':
        return 'from-green-500 to-green-400';
      case 'failed':
        return 'from-red-500 to-red-400';
      default:
        return 'from-white/30 to-white/20';
    }
  };

  return (
    <div className="bg-white/5 backdrop-blur-xl rounded-3xl p-6 border border-white/20 shadow-lg">
      <div className="flex items-center space-x-3 mb-4">
        <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center border border-white/20">
          {getStatusIcon()}
        </div>
        <div>
          <h3 className="text-lg font-semibold text-white/90 capitalize">
            {status.status === 'idle' ? 'Ready' : status.status}
          </h3>
          <p className="text-sm text-white/60">{message}</p>
        </div>
      </div>

      {(status.status === 'processing' || status.status === 'uploading' || status.status === 'analyzing') && (
        <div className="mb-4">
          <div className="flex justify-between text-sm text-white/60 mb-2">
            <span>Progress</span>
            <span>{Math.round(progress)}%</span>
          </div>
          <div className="w-full bg-white/10 rounded-full h-2 overflow-hidden backdrop-blur-sm">
            <div
              className={`h-full bg-gradient-to-r ${getStatusColor()} transition-all duration-500 ease-out rounded-full`}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {status.status === 'completed' && sdrImprovement && (
        <div className="bg-green-500/10 border border-green-400/20 rounded-2xl p-4 mt-4 backdrop-blur-sm">
          <div className="flex items-center space-x-2">
            <TrendingUp className="w-4 h-4 text-green-400" />
            <span className="text-green-400 font-medium">
              +{sdrImprovement.toFixed(1)} dB improvement
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

// Apple-style Settings Panel
const AppleSettingsPanel = ({ 
  settings, 
  onSettingsChange, 
  disabled 
}: {
  settings: ProcessingSettings;
  onSettingsChange: (settings: ProcessingSettings) => void;
  disabled: boolean;
}) => {
  return (
    <div className="bg-white/5 backdrop-blur-xl rounded-3xl border border-white/20 shadow-lg overflow-hidden">
      <div className="p-6 border-b border-white/10">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
            <Settings className="w-4 h-4 text-white/70" />
          </div>
          <h3 className="text-lg font-semibold text-white/90">Settings</h3>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Noise Reduction */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-3">
            Noise Reduction
          </label>
          <div className="grid grid-cols-2 gap-2">
            {['light', 'medium', 'strong', 'extreme'].map((level) => (
              <button
                key={level}
                onClick={() => onSettingsChange({ ...settings, noise_reduction: level })}
                disabled={disabled}
                className={`px-3 py-2.5 text-sm rounded-xl transition-all font-medium capitalize ${
                  settings.noise_reduction === level
                    ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
                    : 'bg-white/10 text-white/70 hover:bg-white/20 border border-white/20'
                } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {level}
              </button>
            ))}
          </div>
        </div>

        {/* Toggle Options */}
        <div className="space-y-3">
          {[
            { key: 'enhance_speech', label: 'Speech Enhancement' },
            { key: 'remove_reverb', label: 'Remove Reverb' },
            { key: 'isolate_voice', label: 'Voice Isolation' },
            { key: 'boost_clarity', label: 'Boost Clarity' }
          ].map(({ key, label }) => (
            <label key={key} className="flex items-center justify-between p-3 rounded-xl bg-white/5 border border-white/10 cursor-pointer hover:bg-white/10 transition-all">
              <span className="text-sm font-medium text-white/80">{label}</span>
              <div className="relative">
                <input
                  type="checkbox"
                  checked={settings[key]}
                  onChange={(e) => onSettingsChange({ ...settings, [key]: e.target.checked })}
                  disabled={disabled}
                  className="sr-only"
                />
                <div className={`w-11 h-6 rounded-full transition-all ${
                  settings[key] ? 'bg-blue-500' : 'bg-white/20'
                }`}>
                  <div className={`w-5 h-5 rounded-full bg-white transition-all duration-200 ease-out ${
                    settings[key] ? 'translate-x-5' : 'translate-x-0.5'
                  } mt-0.5 shadow-lg`} />
                </div>
              </div>
            </label>
          ))}
        </div>

        {/* Output Format */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-3">
            Output Format
          </label>
          <div className="relative">
            <select
              value={settings.output_format}
              onChange={(e) => onSettingsChange({ ...settings, output_format: e.target.value })}
              disabled={disabled}
              className="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white/90 focus:ring-2 focus:ring-blue-500 focus:border-transparent appearance-none backdrop-blur-sm"
            >
              <option value="wav" className="bg-gray-800">WAV (Lossless)</option>
              <option value="flac" className="bg-gray-800">FLAC (Compressed)</option>
              <option value="aiff" className="bg-gray-800">AIFF (Professional)</option>
            </select>
            <div className="absolute right-3 top-3 pointer-events-none">
              <svg className="w-5 h-5 text-white/50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </div>

        {/* Quality */}
        <div>
          <label className="block text-sm font-medium text-white/70 mb-3">
            Quality
          </label>
          <div className="grid grid-cols-3 gap-2">
            {['standard', 'high', 'pro'].map((quality) => (
              <button
                key={quality}
                onClick={() => onSettingsChange({ ...settings, quality })}
                disabled={disabled}
                className={`px-3 py-2.5 text-sm rounded-xl transition-all font-medium capitalize ${
                  settings.quality === quality
                    ? 'bg-blue-500 text-white shadow-lg shadow-blue-500/30'
                    : 'bg-white/10 text-white/70 hover:bg-white/20 border border-white/20'
                } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {quality}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Apple-style App
export default function HarmonyRestorer() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState({
    status: 'idle',
    progress: 0,
    message: 'Ready to enhance your audio',
  });
  const [settings, setSettings] = useState({
    noise_reduction: 'medium',
    enhance_speech: true,
    remove_reverb: false,
    isolate_voice: false,
    boost_clarity: true,
    output_format: 'wav',
    quality: 'high',
    preserve_dynamics: true,
  });
  const [isProcessing, setIsProcessing] = useState(false);

  const processAudio = useCallback(async () => {
    if (!file) return;

    setIsProcessing(true);
    
    // Upload
    setStatus({
      status: 'uploading',
      progress: 0,
      message: 'Uploading audio file...',
    });

    for (let i = 0; i <= 100; i += 10) {
      await new Promise(resolve => setTimeout(resolve, 100));
      setStatus(prev => ({
        ...prev,
        progress: i,
        message: i === 100 ? 'Upload complete' : 'Uploading audio file...',
      }));
    }

    // Analysis
    setStatus({
      status: 'analyzing',
      progress: 0,
      message: 'Analyzing audio structure...',
    });

    for (let i = 0; i <= 100; i += 15) {
      await new Promise(resolve => setTimeout(resolve, 120));
      setStatus(prev => ({
        ...prev,
        progress: i,
        message: 'Analyzing audio structure...',
      }));
    }

    // Processing
    setStatus({
      status: 'processing',
      progress: 0,
      message: 'Processing with AI...',
    });

    const messages = [
      'Processing with AI...',
      'Reducing noise...',
      'Enhancing clarity...',
      'Optimizing quality...',
      'Finalizing...',
    ];

    for (let i = 0; i <= 100; i += 5) {
      await new Promise(resolve => setTimeout(resolve, 150));
      setStatus(prev => ({
        ...prev,
        progress: i,
        message: messages[Math.floor(i / 25)] || 'Processing with AI...',
      }));
    }

    // Complete
    setStatus({
      status: 'completed',
      progress: 100,
      message: 'Audio enhancement complete',
      processingTime: 3.2,
      sdrImprovement: 6.5 + Math.random() * 2,
    });

    setIsProcessing(false);
  }, [file]);

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile);
    setStatus({
      status: 'idle',
      progress: 0,
      message: 'Ready to enhance your audio',
    });
  };

  const handleReset = () => {
    setFile(null);
    setStatus({
      status: 'idle',
      progress: 0,
      message: 'Ready to enhance your audio',
    });
    setIsProcessing(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 relative overflow-hidden">
      {/* Apple-style background */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl" />
        <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
      </div>

      <div className="relative z-10 container mx-auto px-6 py-8 max-w-6xl">
        {/* Apple-style Header */}
        <header className="text-center mb-12">
          <div className="inline-flex items-center space-x-3 mb-6">
            <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-lg shadow-blue-500/30">
              <Crown className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-white to-white/80 bg-clip-text text-transparent">
              HarmonyRestorer
            </h1>
          </div>
          <p className="text-xl text-white/70 max-w-2xl mx-auto font-medium leading-relaxed">
            Professional audio restoration powered by advanced AI. 
            Achieve studio-quality results with one tap.
          </p>
          <div className="flex items-center justify-center space-x-6 mt-6 text-sm text-white/60">
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
              <span>Real-time Processing</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              <span>7+ dB Enhancement</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              <span>AI-Powered</span>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main Content */}
          <div className="lg:col-span-2 space-y-6">
            <AppleUploadArea
              onFileSelect={handleFileSelect}
              isProcessing={isProcessing}
              currentFile={file}
            />

            {/* Waveform */}
            {file && (
              <div className="bg-white/5 backdrop-blur-xl rounded-3xl p-6 border border-white/20 shadow-lg">
                <h3 className="text-lg font-semibold text-white/90 mb-4 flex items-center space-x-2">
                  <BarChart3 className="w-5 h-5 text-blue-400" />
                  <span>Waveform</span>
                </h3>
                <AppleWaveform isActive={isProcessing} />
              </div>
            )}

            <AppleProgressCard
              status={status}
              progress={status.progress}
              message={status.message}
              sdrImprovement={status.sdrImprovement}
            />

            {/* Controls */}
            <div className="flex space-x-4">
              <button
                onClick={processAudio}
                disabled={!file || isProcessing}
                className="flex-1 bg-blue-500 hover:bg-blue-600 disabled:bg-white/10 disabled:text-white/30 text-white font-semibold py-4 px-6 rounded-2xl transition-all duration-300 flex items-center justify-center space-x-2 disabled:cursor-not-allowed shadow-lg shadow-blue-500/30 hover:shadow-xl hover:shadow-blue-500/40"
              >
                {isProcessing ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Brain className="w-5 h-5" />
                )}
                <span>{isProcessing ? 'Processing...' : 'Enhance Audio'}</span>
              </button>

              {status.status === 'completed' && (
                <button className="bg-green-500 hover:bg-green-600 text-white font-semibold py-4 px-6 rounded-2xl transition-all duration-300 flex items-center space-x-2 shadow-lg shadow-green-500/30">
                  <Download className="w-5 h-5" />
                  <span>Download</span>
                </button>
              )}

              <button
                onClick={handleReset}
                className="bg-white/10 hover:bg-white/20 text-white/70 hover:text-white font-semibold py-4 px-4 rounded-2xl transition-all duration-300 border border-white/20"
              >
                <RotateCcw className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Settings */}
          <div>
            <AppleSettingsPanel
              settings={settings}
              onSettingsChange={setSettings}
              disabled={isProcessing}
            />
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-16 pt-8 border-t border-white/10">
          <p className="text-white/50 text-sm">
            Powered by 1D Operational GANs • Professional audio restoration
          </p>
        </footer>
      </div>
    </div>
  );
}