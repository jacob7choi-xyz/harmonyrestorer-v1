import React, { useState, useRef } from 'react';
import { Upload, FileAudio } from 'lucide-react';

interface UploadAreaProps {
  onFileSelect: (file: File) => void;
  isProcessing: boolean;
  currentFile: File | null;
}

export function UploadArea({ onFileSelect, isProcessing, currentFile }: UploadAreaProps) {
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
          : 'bg-white/5 hover:bg-white/10 hover:border-white/30 hover:shadow-xl'}
        ${isProcessing ? 'opacity-60 cursor-not-allowed' : ''}
        backdrop-blur-xl shadow-lg`}
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
              <FileAudio className="w-8 h-8 text-blue-400" />
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
              <p className="text-xl font-medium text-white/90 mb-2">Drop audio file here</p>
              <p className="text-sm text-white/60">or tap to browse — WAV, FLAC, MP3, OGG, M4A, AAC</p>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
