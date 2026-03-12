import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { uploadAudio, getJobStatus, getDownloadUrl, pollUntilDone } from '../client'

vi.stubEnv('VITE_API_URL', '/api/v1')

function mockFetchResponse(data: unknown, ok = true, status = 200): void {
  global.fetch = vi.fn().mockResolvedValueOnce({
    ok,
    status,
    json: () => Promise.resolve(data),
  })
}

describe('uploadAudio', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('sends file as FormData and returns response', async () => {
    const mockResponse = { job_id: 'abc-123', status: 'queued', message: 'Uploaded' }
    mockFetchResponse(mockResponse)

    const file = new File(['audio data'], 'test.wav', { type: 'audio/wav' })
    const result = await uploadAudio(file)

    expect(result).toEqual(mockResponse)
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/denoise'),
      expect.objectContaining({ method: 'POST' }),
    )
  })

  it('throws on non-ok response', async () => {
    mockFetchResponse({ detail: 'File too large' }, false, 413)

    const file = new File(['data'], 'test.wav', { type: 'audio/wav' })
    await expect(uploadAudio(file)).rejects.toThrow('File too large')
  })

  it('throws generic message when error response has no detail', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () => Promise.reject(new Error('parse error')),
    })

    const file = new File(['data'], 'test.wav', { type: 'audio/wav' })
    await expect(uploadAudio(file)).rejects.toThrow('Upload failed')
  })

  it('falls back to status code when detail field is missing', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 422,
      json: () => Promise.resolve({}),
    })

    const file = new File(['data'], 'test.wav', { type: 'audio/wav' })
    await expect(uploadAudio(file)).rejects.toThrow('Upload failed (422)')
  })

  it('passes AbortSignal to fetch', async () => {
    const mockResponse = { job_id: 'abc', status: 'queued', message: 'ok' }
    mockFetchResponse(mockResponse)

    const controller = new AbortController()
    const file = new File(['data'], 'test.wav', { type: 'audio/wav' })
    await uploadAudio(file, controller.signal)

    expect(global.fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ signal: controller.signal }),
    )
  })
})

describe('getJobStatus', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('fetches status for given job ID', async () => {
    const mockStatus = {
      job_id: 'abc-123',
      status: 'processing',
      progress: 50,
      message: 'Processing...',
      completed_at: null,
      download_url: null,
      processing_time: null,
    }
    mockFetchResponse(mockStatus)

    const result = await getJobStatus('abc-123')
    expect(result).toEqual(mockStatus)
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/status/abc-123'),
      expect.any(Object),
    )
  })

  it('throws on non-ok response', async () => {
    mockFetchResponse({ detail: 'Not found' }, false, 404)
    await expect(getJobStatus('bad-id')).rejects.toThrow('Not found')
  })

  it('falls back to status code when detail field is missing', async () => {
    global.fetch = vi.fn().mockResolvedValueOnce({
      ok: false,
      status: 502,
      json: () => Promise.resolve({}),
    })
    await expect(getJobStatus('abc')).rejects.toThrow('Status check failed (502)')
  })

  it('passes AbortSignal to fetch', async () => {
    const mockStatus = {
      job_id: 'abc',
      status: 'processing',
      progress: 0,
      message: 'Working',
      completed_at: null,
      download_url: null,
      processing_time: null,
    }
    mockFetchResponse(mockStatus)

    const controller = new AbortController()
    await getJobStatus('abc', controller.signal)

    expect(global.fetch).toHaveBeenCalledWith(
      expect.any(String),
      expect.objectContaining({ signal: controller.signal }),
    )
  })
})

describe('getDownloadUrl', () => {
  it('returns correct download URL', () => {
    const url = getDownloadUrl('abc-123')
    expect(url).toContain('/download/abc-123')
  })
})

describe('pollUntilDone', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('resolves when status is completed', async () => {
    const completedStatus = {
      job_id: 'abc',
      status: 'completed' as const,
      progress: 100,
      message: 'Done',
      completed_at: '2026-01-01',
      download_url: '/download/abc',
      processing_time: 5.0,
    }
    mockFetchResponse(completedStatus)

    const onUpdate = vi.fn()
    const promise = pollUntilDone('abc', onUpdate)

    await vi.advanceTimersByTimeAsync(0)
    const result = await promise

    expect(result).toEqual(completedStatus)
    expect(onUpdate).toHaveBeenCalledWith(completedStatus)
  })

  it('rejects when status is failed', async () => {
    const failedStatus = {
      job_id: 'abc',
      status: 'failed' as const,
      progress: 0,
      message: 'Processing error',
      completed_at: null,
      download_url: null,
      processing_time: null,
    }
    mockFetchResponse(failedStatus)

    const onUpdate = vi.fn()
    const promise = pollUntilDone('abc', onUpdate)

    // Attach the catch handler before advancing timers to avoid unhandled rejection
    const rejection = promise.catch((e: Error) => e)

    await vi.advanceTimersByTimeAsync(0)
    const error = await rejection
    expect(error).toBeInstanceOf(Error)
    expect(error.message).toBe('Processing error')
  })

  it('polls multiple times before completing', async () => {
    const processingStatus = {
      job_id: 'abc',
      status: 'processing' as const,
      progress: 50,
      message: 'Working...',
      completed_at: null,
      download_url: null,
      processing_time: null,
    }
    const completedStatus = {
      job_id: 'abc',
      status: 'completed' as const,
      progress: 100,
      message: 'Done',
      completed_at: '2026-01-01',
      download_url: '/download/abc',
      processing_time: 5.0,
    }

    // First call returns processing, second returns completed
    global.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: () => Promise.resolve(processingStatus) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: () => Promise.resolve(completedStatus) })

    const onUpdate = vi.fn()
    const promise = pollUntilDone('abc', onUpdate)

    // First poll: processing
    await vi.advanceTimersByTimeAsync(0)
    expect(onUpdate).toHaveBeenCalledWith(processingStatus)

    // Second poll after interval
    await vi.advanceTimersByTimeAsync(1500)
    const result = await promise

    expect(result).toEqual(completedStatus)
    expect(onUpdate).toHaveBeenCalledTimes(2)
    expect(global.fetch).toHaveBeenCalledTimes(2)
  })

  it('rejects on network error during polling', async () => {
    global.fetch = vi.fn().mockRejectedValueOnce(new TypeError('Failed to fetch'))

    const onUpdate = vi.fn()
    const promise = pollUntilDone('abc', onUpdate)
    const rejection = promise.catch((e: Error) => e)

    await vi.advanceTimersByTimeAsync(0)
    const error = await rejection
    expect(error).toBeInstanceOf(TypeError)
    expect(error.message).toBe('Failed to fetch')
  })

  it('rejects when aborted mid-poll', async () => {
    const controller = new AbortController()
    const abortError = new DOMException('The operation was aborted', 'AbortError')
    global.fetch = vi.fn().mockRejectedValueOnce(abortError)

    const onUpdate = vi.fn()
    const promise = pollUntilDone('abc', onUpdate, controller.signal)
    const rejection = promise.catch((e: unknown) => e)

    await vi.advanceTimersByTimeAsync(0)
    const error = await rejection
    expect(error).toBeInstanceOf(DOMException)
    expect((error as DOMException).name).toBe('AbortError')
  })

  it('rejects when aborted between polls', async () => {
    const processingStatus = {
      job_id: 'abc',
      status: 'processing' as const,
      progress: 50,
      message: 'Working...',
      completed_at: null,
      download_url: null,
      processing_time: null,
    }

    // First poll succeeds with processing, second never happens (aborted)
    global.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: () => Promise.resolve(processingStatus) })

    const controller = new AbortController()
    const onUpdate = vi.fn()
    const promise = pollUntilDone('abc', onUpdate, controller.signal)
    const rejection = promise.catch((e: unknown) => e)

    // First poll completes: status is processing, setTimeout scheduled
    await vi.advanceTimersByTimeAsync(0)
    expect(onUpdate).toHaveBeenCalledWith(processingStatus)

    // Abort before the next poll fires
    controller.abort()

    await vi.advanceTimersByTimeAsync(1500)
    const error = await rejection
    expect(error).toBeInstanceOf(DOMException)
    expect((error as DOMException).name).toBe('AbortError')
  })

  it('rejects when onUpdate callback throws on completed', async () => {
    const completedStatus = {
      job_id: 'abc',
      status: 'completed' as const,
      progress: 100,
      message: 'Done',
      completed_at: '2026-01-01',
      download_url: '/download/abc',
      processing_time: 5.0,
    }
    mockFetchResponse(completedStatus)

    const onUpdate = vi.fn().mockImplementation(() => {
      throw new Error('render crashed')
    })
    const promise = pollUntilDone('abc', onUpdate)
    const rejection = promise.catch((e: Error) => e)

    await vi.advanceTimersByTimeAsync(0)
    const error = await rejection
    expect(error).toBeInstanceOf(Error)
    expect(error.message).toBe('render crashed')
  })

  it('rejects when onUpdate throws mid-sequence and stops polling', async () => {
    const processingStatus = {
      job_id: 'abc',
      status: 'processing' as const,
      progress: 50,
      message: 'Working...',
      completed_at: null,
      download_url: null,
      processing_time: null,
    }

    global.fetch = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: () => Promise.resolve(processingStatus) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: () => Promise.resolve(processingStatus) })

    const onUpdate = vi.fn().mockImplementation(() => {
      throw new Error('state update failed')
    })
    const promise = pollUntilDone('abc', onUpdate)
    const rejection = promise.catch((e: Error) => e)

    await vi.advanceTimersByTimeAsync(0)
    const error = await rejection
    expect(error.message).toBe('state update failed')
    // Should not have scheduled a second poll
    expect(global.fetch).toHaveBeenCalledTimes(1)
  })

  it('rejects immediately if signal already aborted', async () => {
    const controller = new AbortController()
    controller.abort()

    const onUpdate = vi.fn()
    await expect(
      pollUntilDone('abc', onUpdate, controller.signal),
    ).rejects.toThrow('aborted')
  })
})
