const API_BASE = import.meta.env.VITE_API_URL ?? '/api/v1';

/** A non-ok HTTP response, with its status so callers can tell a 429 from a failure. */
export class HttpError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
  }
}

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
    throw new HttpError(err.detail ?? `Status check failed (${res.status})`, res.status);
  }

  return res.json();
}

export function getDownloadUrl(jobId: string, format: string = 'wav'): string {
  return `${API_BASE}/download/${jobId}?format=${format}`;
}

const MAX_POLL_DURATION = 300_000;

const HTTP_TOO_MANY_REQUESTS = 429;

// A throttled status check waits this long at most before the next try.
const MAX_THROTTLED_DELAY_MS = 10_000;

const KNOWN_STATUSES = new Set(['queued', 'processing', 'completed', 'failed']);

export async function pollUntilDone(
  jobId: string,
  onUpdate: (status: JobStatusResponse) => void,
  signal?: AbortSignal,
  intervalMs = 1500,
): Promise<JobStatusResponse> {
  return new Promise((resolve, reject) => {
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    const startTime = Date.now();
    let throttledDelayMs = intervalMs;

    const cleanup = (): void => {
      if (timeoutId !== undefined) clearTimeout(timeoutId);
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

      if (Date.now() - startTime > MAX_POLL_DURATION) {
        cleanup();
        reject(new Error('Processing timed out'));
        return;
      }

      try {
        const status = await getJobStatus(jobId, signal);
        throttledDelayMs = intervalMs;
        try {
          onUpdate(status);
        } catch (callbackErr) {
          cleanup();
          reject(callbackErr);
          return;
        }

        if (status.status === 'completed') {
          cleanup();
          resolve(status);
        } else if (status.status === 'failed') {
          cleanup();
          reject(new Error(status.message || 'Processing failed'));
        } else if (!KNOWN_STATUSES.has(status.status)) {
          cleanup();
          reject(new Error(`Unexpected job status: ${status.status}`));
        } else {
          timeoutId = setTimeout(poll, intervalMs);
        }
      } catch (err) {
        // A backend with one execution environment (the Lambda deployment in ADR 009)
        // answers 429 while another request holds it, so a throttled check says
        // nothing about this job. Back off and ask again.
        if (err instanceof HttpError && err.status === HTTP_TOO_MANY_REQUESTS) {
          timeoutId = setTimeout(poll, throttledDelayMs);
          throttledDelayMs = Math.min(throttledDelayMs * 2, MAX_THROTTLED_DELAY_MS);
          return;
        }
        cleanup();
        if (err instanceof DOMException && err.name === 'AbortError') {
          reject(err);
          return;
        }
        reject(err);
      }
    };

    poll();
  });
}
