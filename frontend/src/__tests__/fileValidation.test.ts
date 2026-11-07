import { describe, it, expect, vi, afterEach } from 'vitest';
import { validateAudioFile } from '../fileValidation';
import { MAX_UPLOAD_BYTES } from '../config';

/** An Audio stand-in that fires one event when its source is set. */
function stubAudio(event: 'loadedmetadata' | 'error' | null, duration = 0): void {
  class FakeAudio {
    duration = duration;
    private listeners = new Map<string, () => void>();
    addEventListener(type: string, listener: () => void): void {
      this.listeners.set(type, listener);
    }
    set src(_url: string) {
      if (event) setTimeout(() => this.listeners.get(event)?.(), 0);
    }
    pause(): void {}
    removeAttribute(): void {}
  }
  vi.stubGlobal('Audio', FakeAudio);
}

function audioFile(size = 1024): File {
  const file = new File(['audio'], 'song.wav', { type: 'audio/wav' });
  Object.defineProperty(file, 'size', { value: size });
  return file;
}

describe('validateAudioFile', () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it('refuses a file whose type is not audio', async () => {
    const result = await validateAudioFile(new File(['x'], 'a.txt', { type: 'text/plain' }));
    expect(result).toMatch(/please select an audio file/i);
  });

  it('refuses a file over the upload limit', async () => {
    const result = await validateAudioFile(audioFile(MAX_UPLOAD_BYTES + 1));
    expect(result).toMatch(/file too large/i);
  });

  it('refuses audio longer than 10 minutes', async () => {
    stubAudio('loadedmetadata', 601);
    expect(await validateAudioFile(audioFile())).toBe('Audio too long (11 min). Max 10 minutes.');
  });

  it('accepts audio of exactly 10 minutes', async () => {
    stubAudio('loadedmetadata', 600);
    expect(await validateAudioFile(audioFile())).toBeNull();
  });

  it('leaves a file the browser cannot read to the server', async () => {
    stubAudio('error');
    expect(await validateAudioFile(audioFile())).toBeNull();
  });

  it('leaves a file whose metadata never loads to the server after 5 s', async () => {
    vi.useFakeTimers();
    stubAudio(null);
    const pending = validateAudioFile(audioFile());
    await vi.advanceTimersByTimeAsync(5000);
    expect(await pending).toBeNull();
  });

  it('settles at once and releases the object URL when aborted', async () => {
    stubAudio(null);
    const revoke = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => {});
    const controller = new AbortController();
    const pending = validateAudioFile(audioFile(), controller.signal);
    controller.abort();
    expect(await pending).toBeNull();
    expect(revoke).toHaveBeenCalledTimes(1);
  });
});
