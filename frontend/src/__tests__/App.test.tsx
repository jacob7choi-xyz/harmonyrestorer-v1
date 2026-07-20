import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import App from '../App';

// Mock API client
vi.mock('../api/client', () => ({
  uploadAudio: vi.fn(),
  pollUntilDone: vi.fn(),
  getDownloadUrl: vi.fn((id: string) => `/api/v1/download/${id}`),
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

// Mock URL.createObjectURL / revokeObjectURL (jsdom stubs)
globalThis.URL.createObjectURL = vi.fn(() => 'blob:mock-url');
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
    expect(downloadLink).toHaveAttribute('href', '/api/v1/download/test-123');
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
});
