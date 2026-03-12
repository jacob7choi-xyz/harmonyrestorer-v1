import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { UploadArea } from '../UploadArea'

describe('UploadArea', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders upload prompt when no file selected', () => {
    render(<UploadArea onFileSelect={vi.fn()} isProcessing={false} currentFile={null} />)
    expect(screen.getByText(/drop audio file here/i)).toBeInTheDocument()
  })

  it('shows file name when file is selected', () => {
    const file = new File(['data'], 'my_song.wav', { type: 'audio/wav' })
    Object.defineProperty(file, 'size', { value: 1024 * 1024 })

    render(<UploadArea onFileSelect={vi.fn()} isProcessing={false} currentFile={file} />)
    expect(screen.getByText('my_song.wav')).toBeInTheDocument()
  })

  it('shows file size when file is selected', () => {
    const file = new File(['data'], 'track.wav', { type: 'audio/wav' })
    Object.defineProperty(file, 'size', { value: 2 * 1024 * 1024 })

    render(<UploadArea onFileSelect={vi.fn()} isProcessing={false} currentFile={file} />)
    expect(screen.getByText('2.00 MB')).toBeInTheDocument()
  })

  it('shows error for non-audio file', async () => {
    const onFileSelect = vi.fn()
    render(<UploadArea onFileSelect={onFileSelect} isProcessing={false} currentFile={null} />)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const textFile = new File(['hello'], 'readme.txt', { type: 'text/plain' })

    fireEvent.change(input, { target: { files: [textFile] } })

    await vi.waitFor(() => {
      expect(screen.getByText(/please select an audio file/i)).toBeInTheDocument()
    })
    expect(onFileSelect).not.toHaveBeenCalled()
  })

  it('shows error for oversized file', async () => {
    const onFileSelect = vi.fn()
    render(<UploadArea onFileSelect={onFileSelect} isProcessing={false} currentFile={null} />)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const bigFile = new File(['x'], 'big.wav', { type: 'audio/wav' })
    Object.defineProperty(bigFile, 'size', { value: 51 * 1024 * 1024 })

    fireEvent.change(input, { target: { files: [bigFile] } })

    await vi.waitFor(() => {
      expect(screen.getByText(/file too large/i)).toBeInTheDocument()
    })
    expect(onFileSelect).not.toHaveBeenCalled()
  })

  it('disables input when processing', () => {
    render(<UploadArea onFileSelect={vi.fn()} isProcessing={true} currentFile={null} />)
    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    expect(input.disabled).toBe(true)
  })

  it('has correct aria-label when no file selected', () => {
    render(<UploadArea onFileSelect={vi.fn()} isProcessing={false} currentFile={null} />)
    const area = screen.getByRole('button')
    expect(area).toHaveAttribute('aria-label', 'Upload audio file')
  })

  it('has correct aria-label when file is selected', () => {
    const file = new File(['data'], 'song.wav', { type: 'audio/wav' })
    render(<UploadArea onFileSelect={vi.fn()} isProcessing={false} currentFile={file} />)
    const area = screen.getByRole('button')
    expect(area).toHaveAttribute('aria-label', 'Selected: song.wav. Press to change file.')
  })

  it('calls onFileSelect for valid audio file', async () => {
    vi.useFakeTimers()
    const onFileSelect = vi.fn()

    render(<UploadArea onFileSelect={onFileSelect} isProcessing={false} currentFile={null} />)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    const audioFile = new File(['audio'], 'song.wav', { type: 'audio/wav' })
    Object.defineProperty(audioFile, 'size', { value: 1024 })
    fireEvent.change(input, { target: { files: [audioFile] } })

    // Duration check creates an Audio element. In jsdom, loadedmetadata won't fire,
    // so the 5s timeout resolves true, allowing onFileSelect to be called.
    await vi.advanceTimersByTimeAsync(5100)

    expect(onFileSelect).toHaveBeenCalledWith(audioFile)
  })
})
