const API_BASE = '/api/v1';

interface UploadResponse {
  job_id: string;
  status: string;
  message: string;
}

interface JobStatusResponse {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number;
  message: string;
  completed_at: string | null;
  download_url: string | null;
  processing_time: number | null;
}

export async function uploadAudio(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/denoise`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail ?? `Upload failed (${res.status})`);
  }

  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/status/${jobId}`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Status check failed' }));
    throw new Error(err.detail ?? `Status check failed (${res.status})`);
  }

  return res.json();
}

export function getDownloadUrl(jobId: string): string {
  return `${API_BASE}/download/${jobId}`;
}

export async function pollUntilDone(
  jobId: string,
  onUpdate: (status: JobStatusResponse) => void,
  intervalMs = 1500,
): Promise<JobStatusResponse> {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const status = await getJobStatus(jobId);
        onUpdate(status);

        if (status.status === 'completed') {
          resolve(status);
        } else if (status.status === 'failed') {
          reject(new Error(status.message || 'Processing failed'));
        } else {
          setTimeout(poll, intervalMs);
        }
      } catch (err) {
        reject(err);
      }
    };

    poll();
  });
}
