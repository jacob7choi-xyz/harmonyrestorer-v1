import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { UploadArea } from '../UploadArea'

describe('UploadArea', () => {
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

  it('accepts click to open file dialog', () => {
    render(<UploadArea onFileSelect={vi.fn()} isProcessing={false} currentFile={null} />)
    const area = screen.getByRole('button')
    expect(area).toBeInTheDocument()
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
})
