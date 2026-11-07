import React, { useState, useRef, useEffect } from 'react';
import { Upload, FileAudio } from 'lucide-react';
import { MAX_UPLOAD_MB } from '../config';
import { validateAudioFile } from '../fileValidation';

interface UploadAreaProps {
  onFileSelect: (file: File) => void;
  isProcessing: boolean;
  currentFile: File | null;
}

export function UploadArea({ onFileSelect, isProcessing, currentFile }: UploadAreaProps): React.JSX.Element {
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // The check in flight; a newer selection or unmounting aborts it
  const validationRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      validationRef.current?.abort();
    };
  }, []);

  const validateAndSelect = async (file: File): Promise<void> => {
    validationRef.current?.abort();
    const controller = new AbortController();
    validationRef.current = controller;
    setError(null);
    const message = await validateAudioFile(file, controller.signal);
    if (controller.signal.aborted) return;
    if (message) {
      setError(message);
      return;
    }
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
    // The page also handles drops; without this its handler would act on the file
    // too, including one this check refuses
    e.stopPropagation();
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
            ? 'bg-violet-glow border-violet/60 scale-105 text-violet-soft'
            : currentFile
              ? 'bg-white/5 border-glass text-ink hover:border-violet/40'
              : 'bg-gradient-to-r from-[#7c5cff] to-[#4fd1ff] border-transparent text-white hover:scale-[1.03] hover:brightness-110 animate-pulse-glow'}
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
          : `or drop a file anywhere. WAV, FLAC, MP3, OGG, M4A, AAC, up to ${MAX_UPLOAD_MB} MB. Restored audio is rendered at 16 kHz mono.`}
      </p>

      {error && (
        <p className="text-sm text-destructive mt-3">{error}</p>
      )}
    </div>
  );
}
