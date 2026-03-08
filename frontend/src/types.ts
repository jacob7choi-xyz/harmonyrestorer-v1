export interface ProcessingStatus {
  status: 'idle' | 'uploading' | 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  jobId?: string;
  downloadUrl?: string;
  processingTime?: number;
}
