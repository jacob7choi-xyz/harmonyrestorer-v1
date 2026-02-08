export interface ProcessingSettings {
  noise_reduction: 'light' | 'medium' | 'strong' | 'extreme';
  enhance_speech: boolean;
  remove_reverb: boolean;
  isolate_voice: boolean;
  boost_clarity: boolean;
  output_format: 'wav' | 'flac' | 'aiff';
  quality: 'standard' | 'high' | 'pro';
  preserve_dynamics: boolean;
}

export interface ProcessingStatus {
  status: 'idle' | 'uploading' | 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  jobId?: string;
  downloadUrl?: string;
  processingTime?: number;
}
