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

  it('rejects immediately if signal already aborted', async () => {
    const controller = new AbortController()
    controller.abort()

    const onUpdate = vi.fn()
    await expect(
      pollUntilDone('abc', onUpdate, controller.signal),
    ).rejects.toThrow('aborted')
  })
})
