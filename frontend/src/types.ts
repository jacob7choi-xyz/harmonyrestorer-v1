export interface ProcessingStatus {
  status: 'idle' | 'uploading' | 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  jobId?: string;
  downloadUrl?: string;
  processingTime?: number;
}

export type WizardStep = 'upload' | 'processing' | 'complete';

export interface WaveformData {
  peaks: Float32Array;
  duration: number;
}

export interface PlaybackState {
  isPlaying: boolean;
  currentTime: number;
  duration: number;
}
