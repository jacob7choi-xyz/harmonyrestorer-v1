import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import App from '../App';

// Mock API client
vi.mock('../api/client', () => ({
  uploadAudio: vi.fn(),
  pollUntilDone: vi.fn(),
  getDownloadUrl: vi.fn(
    (id: string, format = 'wav') => `/api/v1/download/${id}?format=${format}`,
  ),
}));

// Mock useAudioDecoder (AudioContext not available in jsdom); returns peaks
// when a file is selected so the preview strip sections render
vi.mock('../hooks/useAudioDecoder', () => ({
  useAudioDecoder: (file: File | null) => ({
    waveform: file ? { peaks: new Float32Array([0.5, 0.8]), duration: 2 } : null,
    error: null,
  }),
  decodeBlobToWaveform: vi.fn().mockResolvedValue({ peaks: new Float32Array(0), duration: 0 }),
}));

// Mock TapeStrip to avoid canvas/RAF complexity
vi.mock('../components/TapeStrip', () => ({
  TapeStrip: () => <div data-testid="tape-strip">TapeStrip</div>,
}));

// Mock fetch for blob download in the processing flow
const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

// Mock URL.createObjectURL / revokeObjectURL (jsdom stubs); unique URLs so
// revocation can be asserted per created URL
let blobUrlCounter = 0;
globalThis.URL.createObjectURL = vi.fn(() => `blob:mock-${++blobUrlCounter}`);
globalThis.URL.revokeObjectURL = vi.fn();

import { uploadAudio, pollUntilDone } from '../api/client';

const mockUpload = vi.mocked(uploadAudio);
const mockPoll = vi.mocked(pollUntilDone);

function createAudioFile(name = 'song.wav'): File {
  const file = new File(['audio-data'], name, { type: 'audio/wav' });
  Object.defineProperty(file, 'size', { value: 1024 });
  return file;
}

/**
 * Select a file through the hidden input, advancing fake timers to clear
 * the 5s duration check in UploadArea. Caller must have fake timers active.
 */
async function selectFileWithFakeTimers(name = 'song.wav'): Promise<void> {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement;
  const file = createAudioFile(name);

  await act(async () => {
    fireEvent.change(input, { target: { files: [file] } });
    await vi.advanceTimersByTimeAsync(5100);
  });
}

/**
 * Select a file and then switch back to real timers so that async
 * processing mocks resolve normally. Returns after the file is selected.
 */
async function selectFileAndRestoreTimers(name = 'song.wav'): Promise<void> {
  await selectFileWithFakeTimers(name);
  vi.useRealTimers();
}

describe('App (HarmonyRestorer)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      blob: () => Promise.resolve(new Blob(['audio'])),
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // Static render tests (no fake timers needed)

  it('renders the wordmark, headline, and subtitle', () => {
    render(<App />);
    expect(screen.getByText('HarmonyRestorer')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /hear it the way it was/i })).toBeInTheDocument();
    expect(screen.getByText(/ai-powered audio restoration/i)).toBeInTheDocument();
  });

  it('renders the upload pill on initial load', () => {
    render(<App />);
    expect(screen.getByText(/restore your audio/i)).toBeInTheDocument();
  });

  it('does not show the Enhance button before a file is selected', () => {
    render(<App />);
    expect(screen.queryByRole('button', { name: /enhance/i })).not.toBeInTheDocument();
  });

  it('renders the footer', () => {
    render(<App />);
    expect(screen.getByText(/powered by opgan ai denoising/i)).toBeInTheDocument();
  });

  // Sample loading tests (handleFileSelect is called directly; no duration check)

  it('loads the bundled sample when "Try a sample" is clicked', async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      blob: () => Promise.resolve(new Blob(['audio'], { type: 'audio/wav' })),
    });
    render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /try a sample/i }));
    });

    await waitFor(() => {
      expect(screen.getByText('sample-noisy.wav')).toBeInTheDocument();
    });
    expect(screen.getByRole('button', { name: /enhance/i })).toBeEnabled();
  });

  it('shows an error and selects no file when the sample fails to load', async () => {
    mockFetch.mockRejectedValue(new Error('network down'));
    render(<App />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /try a sample/i }));
    });

    await waitFor(() => {
      expect(screen.getByText(/could not load the sample/i)).toBeInTheDocument();
    });
    expect(screen.queryByText('sample-noisy.wav')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /enhance/i })).not.toBeInTheDocument();
  });

  // File selection tests (need fake timers for duration check)

  it('enables the Enhance button after selecting a file', async () => {
    vi.useFakeTimers();
    render(<App />);
    await selectFileWithFakeTimers();

    const button = screen.getByRole('button', { name: /enhance/i });
    expect(button).toBeEnabled();
  });

  it('shows file name in upload area after selecting a file', async () => {
    vi.useFakeTimers();
    render(<App />);
    await selectFileWithFakeTimers('my_track.wav');

    expect(screen.getByText('my_track.wav')).toBeInTheDocument();
  });

  it('revokes every tracked blob URL on unmount', async () => {
    vi.useFakeTimers();
    const { unmount } = render(<App />);
    await selectFileWithFakeTimers();
    vi.useRealTimers();

    const created = (
      globalThis.URL.createObjectURL as ReturnType<typeof vi.fn>
    ).mock.results.map(r => r.value as string);
    expect(created.length).toBeGreaterThan(0);

    unmount();

    const revoked = (
      globalThis.URL.revokeObjectURL as ReturnType<typeof vi.fn>
    ).mock.calls.map(c => c[0] as string);
    for (const url of created) {
      expect(revoked).toContain(url);
    }
  });

  it('offers original playback after selecting a file', async () => {
    vi.useFakeTimers();
    render(<App />);
    await selectFileWithFakeTimers();

    expect(screen.getByRole('button', { name: 'Play' })).toBeInTheDocument();
    expect(screen.getByTestId('tape-strip')).toBeInTheDocument();
  });

  // Processing flow tests (fake timers for file select, real timers for async flow)

  it('runs the full processing flow: upload, poll, complete', async () => {
    mockUpload.mockResolvedValue({
      job_id: 'test-123',
      status: 'queued',
      message: 'Queued',
    });
    mockPoll.mockImplementation(async (jobId, onUpdate) => {
      const result = {
        job_id: jobId,
        status: 'completed' as const,
        progress: 100,
        message: 'Done',
        completed_at: null,
        download_url: `/api/v1/download/${jobId}`,
        processing_time: 2.5,
      };
      onUpdate(result);
      return result;
    });

    vi.useFakeTimers();
    render(<App />);
    await selectFileAndRestoreTimers();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
    });

    await waitFor(() => {
      expect(screen.getByText(/download/i)).toBeInTheDocument();
    });

    expect(mockUpload).toHaveBeenCalledTimes(1);
    expect(mockPoll).toHaveBeenCalledWith('test-123', expect.any(Function), expect.any(Object));

    expect(screen.getByText(/enhanced in 2\.5s/i)).toBeInTheDocument();
    const downloadLink = screen.getByRole('link', { name: /download/i });
    expect(downloadLink).toHaveAttribute('href', '/api/v1/download/test-123?format=wav');
  });

  function mockCompletedJob(jobId: string): void {
    mockUpload.mockResolvedValue({ job_id: jobId, status: 'queued', message: 'Queued' });
    mockPoll.mockImplementation(async (id, onUpdate) => {
      const result = {
        job_id: id,
        status: 'completed' as const,
        progress: 100,
        message: 'Done',
        completed_at: null,
        download_url: `/api/v1/download/${id}`,
        processing_time: 1.0,
      };
      onUpdate(result);
      return result;
    });
  }

  it('enables comparison playback when the restored audio loads', async () => {
    mockCompletedJob('ok-1');
    vi.useFakeTimers();
    render(<App />);
    await selectFileAndRestoreTimers();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
    });
    await waitFor(() => {
      expect(screen.getByRole('link', { name: /download/i })).toBeInTheDocument();
    });

    expect(screen.getByRole('button', { name: 'Play' })).toBeEnabled();
    expect(screen.queryByText(/couldn't be loaded for playback/i)).not.toBeInTheDocument();
  });

  it('says so and disables playback when the restored audio cannot be fetched', async () => {
    mockCompletedJob('busy-1');
    mockFetch.mockResolvedValue({
      ok: false,
      status: 429,
      blob: () => Promise.resolve(new Blob([])),
    });
    vi.useFakeTimers();
    render(<App />);
    await selectFileAndRestoreTimers();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
    });
    await waitFor(() => {
      expect(screen.getByRole('link', { name: /download/i })).toBeInTheDocument();
    });

    expect(screen.getByRole('status')).toHaveTextContent(/couldn't be loaded for playback/i);
    expect(screen.getByRole('button', { name: 'Play' })).toBeDisabled();
    expect(screen.getByRole('link', { name: /download/i })).toHaveAttribute(
      'href',
      '/api/v1/download/busy-1?format=wav',
    );
  });

  it('switches the download link format from the picker', async () => {
    mockUpload.mockResolvedValue({
      job_id: 'fmt-321',
      status: 'queued',
      message: 'Queued',
    });
    mockPoll.mockImplementation(async (jobId, onUpdate) => {
      const result = {
        job_id: jobId,
        status: 'completed' as const,
        progress: 100,
        message: 'Done',
        completed_at: null,
        download_url: `/api/v1/download/${jobId}`,
        processing_time: 1.0,
      };
      onUpdate(result);
      return result;
    });

    vi.useFakeTimers();
    render(<App />);
    await selectFileAndRestoreTimers();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
    });
    await waitFor(() => {
      expect(screen.getByRole('link', { name: /download/i })).toBeInTheDocument();
    });

    // Default follows the uploaded .wav; picking MP3 rewrites the link
    fireEvent.click(screen.getByRole('button', { name: /^mp3$/i }));
    expect(screen.getByRole('link', { name: /download/i })).toHaveAttribute(
      'href',
      '/api/v1/download/fmt-321?format=mp3',
    );
  });

  it('shows error message when upload fails', async () => {
    mockUpload.mockRejectedValue(new Error('Server unavailable'));

    vi.useFakeTimers();
    render(<App />);
    await selectFileAndRestoreTimers();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
    });

    await waitFor(() => {
      expect(screen.getByText('Server unavailable')).toBeInTheDocument();
    });

    // Failed status maps to 'upload' step; file remains selected so user can retry
    expect(screen.getByRole('button', { name: /enhance/i })).toBeEnabled();
  });

  it('shows error message when polling reports failure', async () => {
    mockUpload.mockResolvedValue({
      job_id: 'fail-456',
      status: 'queued',
      message: 'Queued',
    });
    mockPoll.mockRejectedValue(new Error('Denoising engine crashed'));

    vi.useFakeTimers();
    render(<App />);
    await selectFileAndRestoreTimers();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
    });

    await waitFor(() => {
      expect(screen.getByText('Denoising engine crashed')).toBeInTheDocument();
    });
  });

  it('resets to upload step when "New file" is clicked after completion', async () => {
    mockUpload.mockResolvedValue({
      job_id: 'reset-789',
      status: 'queued',
      message: 'Queued',
    });
    mockPoll.mockImplementation(async (jobId, onUpdate) => {
      const result = {
        job_id: jobId,
        status: 'completed' as const,
        progress: 100,
        message: 'Done',
        completed_at: null,
        download_url: `/api/v1/download/${jobId}`,
        processing_time: 1.0,
      };
      onUpdate(result);
      return result;
    });

    vi.useFakeTimers();
    render(<App />);
    await selectFileAndRestoreTimers();

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
    });

    await waitFor(() => {
      expect(screen.getByText(/new file/i)).toBeInTheDocument();
    });

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /new file/i }));
    });

    expect(screen.getByText(/restore your audio/i)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /enhance/i })).not.toBeInTheDocument();
  });

  // Drag and drop. A drop on any element bubbles to the page's drop handler.

  describe('dropped files', () => {
    afterEach(() => {
      vi.unstubAllGlobals();
    });

    /** An Audio stand-in whose metadata reports the given duration. */
    function stubAudioDuration(seconds: number): void {
      class FakeAudio {
        duration = seconds;
        private listeners = new Map<string, () => void>();
        addEventListener(type: string, listener: () => void): void {
          this.listeners.set(type, listener);
        }
        set src(_url: string) {
          setTimeout(() => this.listeners.get('loadedmetadata')?.(), 0);
        }
        pause(): void {}
        removeAttribute(): void {}
      }
      vi.stubGlobal('Audio', FakeAudio);
    }

    async function dropOn(target: Element, file: File): Promise<void> {
      await act(async () => {
        fireEvent.drop(target, { dataTransfer: { files: [file] } });
        await vi.advanceTimersByTimeAsync(5100);
      });
    }

    it('keeps the running job when a file is dropped while it processes', async () => {
      let pollSignal: AbortSignal | undefined;
      mockUpload.mockResolvedValue({ job_id: 'busy-1', status: 'queued', message: 'Queued' });
      mockPoll.mockImplementation((_id, _onUpdate, signal) => {
        pollSignal = signal;
        return new Promise(() => {});
      });

      vi.useFakeTimers();
      render(<App />);
      await selectFileWithFakeTimers();
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(screen.getByRole('heading', { name: /restoring your recording/i })).toBeInTheDocument();

      await dropOn(screen.getByRole('heading', { name: /restoring your recording/i }), createAudioFile('other.wav'));

      expect(pollSignal?.aborted).toBe(false);
      expect(screen.getByRole('heading', { name: /restoring your recording/i })).toBeInTheDocument();
    });

    it('says why when a dropped file is not audio, and selects nothing', async () => {
      vi.useFakeTimers();
      render(<App />);

      await dropOn(screen.getByRole('heading', { name: /hear it the way it was/i }),
        new File(['hello'], 'readme.txt', { type: 'text/plain' }));

      expect(screen.getByRole('alert')).toHaveTextContent(/please select an audio file/i);
      expect(screen.queryByRole('button', { name: /enhance/i })).not.toBeInTheDocument();
    });

    it('says why when a dropped file is too long, and selects nothing', async () => {
      stubAudioDuration(700);
      vi.useFakeTimers();
      render(<App />);

      await dropOn(screen.getByRole('heading', { name: /hear it the way it was/i }), createAudioFile('long.wav'));

      expect(screen.getByRole('alert')).toHaveTextContent(/audio too long/i);
      expect(screen.queryByRole('button', { name: /enhance/i })).not.toBeInTheDocument();
    });

    it('selects a valid file dropped anywhere and clears an earlier message', async () => {
      vi.useFakeTimers();
      render(<App />);
      const heading = screen.getByRole('heading', { name: /hear it the way it was/i });
      await dropOn(heading, new File(['hello'], 'readme.txt', { type: 'text/plain' }));
      expect(screen.getByRole('alert')).toBeInTheDocument();

      await dropOn(heading, createAudioFile('dropped.wav'));

      expect(screen.getByText('dropped.wav')).toBeInTheDocument();
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });

    it('drops a pending check of a dropped file when an upload starts', async () => {
      let pollSignal: AbortSignal | undefined;
      mockUpload.mockResolvedValue({ job_id: 'busy-2', status: 'queued', message: 'Queued' });
      mockPoll.mockImplementation((_id, _onUpdate, signal) => {
        pollSignal = signal;
        return new Promise(() => {});
      });

      vi.useFakeTimers();
      render(<App />);
      await selectFileWithFakeTimers();
      // jsdom never loads the metadata, so this check waits out its 5 s timeout
      await act(async () => {
        fireEvent.drop(screen.getByRole('heading', { name: /hear it the way it was/i }), {
          dataTransfer: { files: [createAudioFile('late.wav')] },
        });
      });
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
        await vi.advanceTimersByTimeAsync(5100);
      });

      expect(pollSignal?.aborted).toBe(false);
      expect(screen.getByRole('heading', { name: /restoring your recording/i })).toBeInTheDocument();
    });

    it('keeps the running job when a sample fetch started earlier finishes', async () => {
      mockUpload.mockResolvedValue({ job_id: 'busy-3', status: 'queued', message: 'Queued' });
      mockPoll.mockImplementation(() => new Promise(() => {}));

      vi.useFakeTimers();
      render(<App />);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      let pollSignal: AbortSignal | undefined;
      mockPoll.mockImplementation((_id, _onUpdate, signal) => {
        pollSignal = signal;
        return new Promise(() => {});
      });
      // Set after the page's own fetches on load, so only the sample click gets it
      let resolveSample: (response: unknown) => void = () => {};
      mockFetch.mockImplementationOnce(() => new Promise((resolve) => { resolveSample = resolve; }));
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /try a sample/i }));
      });
      await dropOn(screen.getByRole('heading', { name: /hear it the way it was/i }), createAudioFile('mine.wav'));
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(screen.getByRole('heading', { name: /restoring your recording/i })).toBeInTheDocument();

      await act(async () => {
        resolveSample({ ok: true, blob: () => Promise.resolve(new Blob(['audio'], { type: 'audio/wav' })) });
        await vi.advanceTimersByTimeAsync(0);
      });

      expect(pollSignal?.aborted).toBe(false);
      expect(screen.getByRole('heading', { name: /restoring your recording/i })).toBeInTheDocument();
    });

    it('selects a dropped file after a job has failed', async () => {
      mockUpload.mockRejectedValue(new Error('Server unavailable'));
      vi.useFakeTimers();
      render(<App />);
      await selectFileWithFakeTimers();
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(screen.getByText('Server unavailable')).toBeInTheDocument();

      await dropOn(screen.getByRole('heading', { name: /hear it the way it was/i }), createAudioFile('retry.wav'));

      expect(screen.getByText('retry.wav')).toBeInTheDocument();
    });

    it('clears a refused-drop message when "New file" is clicked', async () => {
      mockCompletedJob('done-1');
      vi.useFakeTimers();
      render(<App />);
      await selectFileWithFakeTimers();
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /enhance/i }));
        await vi.advanceTimersByTimeAsync(0);
      });
      await dropOn(screen.getByRole('button', { name: /new file/i }),
        new File(['hello'], 'readme.txt', { type: 'text/plain' }));
      expect(screen.getByRole('alert')).toBeInTheDocument();

      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /new file/i }));
      });

      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
    });

    it('does not select a too-long file dropped on the upload pill', async () => {
      stubAudioDuration(700);
      vi.useFakeTimers();
      render(<App />);

      await dropOn(screen.getByRole('button', { name: /upload audio file/i }), createAudioFile('long.wav'));

      expect(screen.getByText(/audio too long/i)).toBeInTheDocument();
      expect(screen.queryByRole('alert')).not.toBeInTheDocument();
      expect(screen.queryByRole('button', { name: /enhance/i })).not.toBeInTheDocument();
    });
  });
});
