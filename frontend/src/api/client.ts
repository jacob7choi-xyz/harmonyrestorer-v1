const API_BASE = import.meta.env.VITE_API_URL ?? '/api/v1';

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

export async function uploadAudio(
  file: File,
  signal?: AbortSignal,
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/denoise`, {
    method: 'POST',
    body: formData,
    signal,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail ?? `Upload failed (${res.status})`);
  }

  return res.json();
}

export async function getJobStatus(
  jobId: string,
  signal?: AbortSignal,
): Promise<JobStatusResponse> {
  const res = await fetch(`${API_BASE}/status/${jobId}`, { signal });

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
  signal?: AbortSignal,
  intervalMs = 1500,
): Promise<JobStatusResponse> {
  return new Promise((resolve, reject) => {
    let timeoutId: ReturnType<typeof setTimeout>;

    const cleanup = (): void => {
      clearTimeout(timeoutId);
      if (signal) signal.removeEventListener('abort', onAbort);
    };

    const onAbort = (): void => {
      cleanup();
      reject(new DOMException('Polling aborted', 'AbortError'));
    };

    if (signal?.aborted) {
      reject(new DOMException('Polling aborted', 'AbortError'));
      return;
    }

    signal?.addEventListener('abort', onAbort);

    const poll = async (): Promise<void> => {
      if (signal?.aborted) return;

      try {
        const status = await getJobStatus(jobId, signal);
        onUpdate(status);

        if (status.status === 'completed') {
          cleanup();
          resolve(status);
        } else if (status.status === 'failed') {
          cleanup();
          reject(new Error(status.message || 'Processing failed'));
        } else {
          timeoutId = setTimeout(poll, intervalMs);
        }
      } catch (err) {
        cleanup();
        if (err instanceof DOMException && err.name === 'AbortError') return;
        reject(err);
      }
    };

    poll();
  });
}
