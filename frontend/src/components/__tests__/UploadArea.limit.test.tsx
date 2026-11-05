import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { UploadArea } from '../UploadArea'

vi.mock('../../config', () => ({
  MAX_UPLOAD_MB: 4,
  MAX_UPLOAD_BYTES: 4 * 1024 * 1024,
}))

function selectFileOfSize(bytes: number): void {
  const input = document.querySelector('input[type="file"]') as HTMLInputElement
  const file = new File(['x'], 'track.wav', { type: 'audio/wav' })
  Object.defineProperty(file, 'size', { value: bytes })
  fireEvent.change(input, { target: { files: [file] } })
}

describe('UploadArea with a configured upload limit', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('shows the configured limit in the prompt', () => {
    render(<UploadArea onFileSelect={vi.fn()} isProcessing={false} currentFile={null} />)
    expect(screen.getByText(/up to 4 MB\./)).toBeInTheDocument()
  })

  it('rejects a file one byte over the limit and names the limit', async () => {
    const onFileSelect = vi.fn()
    render(<UploadArea onFileSelect={onFileSelect} isProcessing={false} currentFile={null} />)

    selectFileOfSize(4 * 1024 * 1024 + 1)

    await vi.waitFor(() => {
      expect(screen.getByText(/Max 4 MB\./)).toBeInTheDocument()
    })
    expect(onFileSelect).not.toHaveBeenCalled()
  })

  it('accepts a file exactly at the limit', async () => {
    vi.useFakeTimers()
    const onFileSelect = vi.fn()
    render(<UploadArea onFileSelect={onFileSelect} isProcessing={false} currentFile={null} />)

    selectFileOfSize(4 * 1024 * 1024)
    // jsdom never fires loadedmetadata, so the 5 s duration-check timeout lets the file through.
    await vi.advanceTimersByTimeAsync(5100)

    expect(onFileSelect).toHaveBeenCalledTimes(1)
    expect(screen.queryByText(/file too large/i)).not.toBeInTheDocument()
  })
})
