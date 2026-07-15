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
        // Can't read duration; let the backend validate
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

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    // Only clear drag state when leaving the component itself, not nested children
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>): void => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      validateAndSelect(files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>): void => {
    const file = e.target.files?.[0];
    if (file) {
      validateAndSelect(file);
    }
  };

  const handleClick = (): void => {
    if (!isProcessing) {
      fileInputRef.current?.click();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>): void => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      handleClick();
    }
  };

  return (
    <div className="text-center">
      <div
        role="button"
        tabIndex={0}
        aria-label={currentFile ? `Selected: ${currentFile.name}. Press to change file.` : 'Upload audio file'}
        aria-disabled={isProcessing}
        className={`inline-flex items-center gap-3 rounded-full px-8 py-4 transition-all cursor-pointer border font-bold
          ${isDragging
            ? 'bg-amber-glow border-amber/60 scale-105 text-amber-soft'
            : currentFile
              ? 'bg-white/5 border-glass text-ink hover:border-amber/40'
              : 'bg-amber border-transparent text-on-amber hover:bg-amber-deep hover:scale-[1.03] animate-pulse-glow'}
          ${isProcessing ? 'opacity-50 cursor-not-allowed animate-none' : ''}`}
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

        {currentFile ? (
          <>
            <FileAudio className="w-5 h-5 text-amber shrink-0" />
            <span className="font-medium">{currentFile.name}</span>
            <span className="text-sm font-normal text-ink-secondary">
              {(currentFile.size / 1024 / 1024).toFixed(2)} MB
            </span>
          </>
        ) : (
          <>
            <Upload className="w-5 h-5 shrink-0" />
            <span>Restore your audio</span>
          </>
        )}
      </div>

      <p className="mt-4 text-xs text-ink-muted tracking-wide">
        {currentFile
          ? 'Tap the pill to choose a different file'
          : 'or drop a file anywhere. WAV, FLAC, MP3, OGG, M4A, AAC, up to 50 MB'}
      </p>

      {error && (
        <p className="text-sm text-destructive mt-3">{error}</p>
      )}
    </div>
  );
}
