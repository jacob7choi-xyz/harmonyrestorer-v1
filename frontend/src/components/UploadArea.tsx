import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileAudio } from 'lucide-react';

const MAX_FILE_SIZE_MB = 50;
const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024;
const MAX_DURATION_MINUTES = 10;
const MAX_DURATION_SECONDS = MAX_DURATION_MINUTES * 60;

interface UploadAreaProps {
  onFileSelect: (file: File) => void;
  isProcessing: boolean;
  currentFile: File | null;
}

export function UploadArea({ onFileSelect, isProcessing, currentFile }: UploadAreaProps): React.JSX.Element {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const durationCheckCleanupRef = useRef<(() => void) | null>(null);

  // Clean up any pending duration check on unmount
  useEffect(() => {
    return () => {
      durationCheckCleanupRef.current?.();
    };
  }, []);

  const checkDuration = (file: File): Promise<boolean> => {
    // Clean up any previous pending check
    durationCheckCleanupRef.current?.();

    return new Promise((resolve) => {
      const audio = new Audio();
      const url = URL.createObjectURL(file);
      audio.src = url;
      let settled = false;

      const cleanup = (): void => {
        if (durationCheckCleanupRef.current === onUnmount) {
          durationCheckCleanupRef.current = null;
        }
        audio.pause();
        audio.removeAttribute('src');
        URL.revokeObjectURL(url);
        clearTimeout(timeoutId);
      };

      const onUnmount = (): void => {
        if (!settled) {
          settled = true;
          cleanup();
          resolve(false);
        }
      };

      durationCheckCleanupRef.current = onUnmount;

      // Timeout after 5s if audio metadata never loads
      const timeoutId = setTimeout(() => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(true); // Let backend validate
      }, 5000);

      audio.addEventListener('loadedmetadata', () => {
        if (settled) return;
        settled = true;
        cleanup();
        if (audio.duration > MAX_DURATION_SECONDS) {
          setError(
            `Audio too long (${Math.ceil(audio.duration / 60)} min). Max ${MAX_DURATION_MINUTES} minutes.`
          );
          resolve(false);
        } else {
          resolve(true);
        }
      });

      audio.addEventListener('error', () => {
        if (settled) return;
        settled = true;
        cleanup();
        // Can't read duration -- let the backend validate
        resolve(true);
      });
    });
  };

  const validateAndSelect = async (file: File): Promise<void> => {
    setError(null);
    if (!file.type.startsWith('audio/')) {
      setError('Please select an audio file (WAV, MP3, FLAC, OGG, M4A, AAC)');
      return;
    }
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setError(`File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_FILE_SIZE_MB} MB.`);
      return;
    }
    const durationOk = await checkDuration(file);
    if (!durationOk) return;
    onFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    // Only clear drag state when leaving the component itself, not nested children
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      validateAndSelect(files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      validateAndSelect(file);
    }
  };

  const handleClick = () => {
    if (!isProcessing) {
      fileInputRef.current?.click();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={currentFile ? `Selected: ${currentFile.name}. Press to change file.` : 'Upload audio file'}
      className={`rounded-xl p-8 text-center transition-all cursor-pointer border
        ${isDragging
          ? 'bg-[#5B8DEF]/10 border-[#5B8DEF]/50 scale-[1.02] backdrop-blur-md'
          : 'bg-[#282828]/50 backdrop-blur-md border-white/5 hover:bg-[#333333]/50 hover:border-[#5B8DEF]/30'}
        ${isProcessing ? 'opacity-50 cursor-not-allowed' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={handleClick}
      onKeyDown={handleKeyDown}
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
            <div className="w-16 h-16 rounded-full bg-[#5B8DEF]/20 flex items-center justify-center">
              <FileAudio className="w-8 h-8 text-[#5B8DEF]" />
            </div>
            <div>
              <p className="text-lg font-medium text-white">{currentFile.name}</p>
              <p className="text-sm text-[#B3B3B3]">{(currentFile.size / 1024 / 1024).toFixed(2)} MB</p>
            </div>
          </>
        ) : (
          <>
            <div className="w-16 h-16 rounded-full bg-white/5 border border-white/10 flex items-center justify-center">
              <Upload className="w-8 h-8 text-[#B3B3B3]" />
            </div>
            <div>
              <p className="text-xl font-medium text-white mb-2">Drop audio file here</p>
              <p className="text-sm text-[#B3B3B3]">or tap to browse -- WAV, FLAC, MP3, OGG, M4A, AAC</p>
            </div>
          </>
        )}

        {error && (
          <p className="text-sm text-[#E34040] mt-2">{error}</p>
        )}
      </div>
    </div>
  );
}
