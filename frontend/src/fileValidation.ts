import { MAX_UPLOAD_BYTES, MAX_UPLOAD_MB } from './config';

const MAX_DURATION_MINUTES = 10;
const MAX_DURATION_SECONDS = MAX_DURATION_MINUTES * 60;
// How long to wait for the browser to read the duration before leaving it to the server
const METADATA_TIMEOUT_MS = 5000;

/**
 * Reads the file's duration in seconds through an Audio element. Resolves null when
 * the browser cannot read it, when it takes longer than the timeout, or on abort.
 */
function readDurationSeconds(file: File, signal?: AbortSignal): Promise<number | null> {
  return new Promise((resolve) => {
    const audio = new Audio();
    const url = URL.createObjectURL(file);
    let settled = false;

    const finish = (seconds: number | null): void => {
      if (settled) return;
      settled = true;
      clearTimeout(timeoutId);
      signal?.removeEventListener('abort', onAbort);
      audio.pause();
      audio.removeAttribute('src');
      URL.revokeObjectURL(url);
      resolve(seconds);
    };
    const onAbort = (): void => finish(null);
    const timeoutId = setTimeout(() => finish(null), METADATA_TIMEOUT_MS);

    audio.addEventListener('loadedmetadata', () => finish(audio.duration));
    audio.addEventListener('error', () => finish(null));
    signal?.addEventListener('abort', onAbort);
    if (signal?.aborted) {
      finish(null);
      return;
    }
    audio.src = url;
  });
}

/**
 * Checks a file before it is selected for upload: its type, its size, and, when the
 * browser can read it, its duration. The server makes its own checks of the size, the
 * content and the decoded duration (backend/app/routes/denoise.py,
 * backend/app/services/audio_decode.py).
 *
 * @returns A message for the user, or null when the file may be selected. Callers
 *   that pass a signal check it after awaiting, since an aborted check resolves null.
 */
export async function validateAudioFile(file: File, signal?: AbortSignal): Promise<string | null> {
  if (!file.type.startsWith('audio/')) {
    return 'Please select an audio file (WAV, MP3, FLAC, OGG, M4A, AAC)';
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return `File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Max ${MAX_UPLOAD_MB} MB.`;
  }
  const seconds = await readDurationSeconds(file, signal);
  if (seconds !== null && seconds > MAX_DURATION_SECONDS) {
    return `Audio too long (${Math.ceil(seconds / 60)} min). Max ${MAX_DURATION_MINUTES} minutes.`;
  }
  return null;
}
