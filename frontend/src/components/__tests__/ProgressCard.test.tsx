import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import { ProgressCard } from '../ProgressCard'
import type { ProcessingStatus } from '../../types'

describe('ProgressCard', () => {
  it('shows Ready for idle status', () => {
    const status: ProcessingStatus = { status: 'idle', progress: 0, message: 'Ready to enhance' }
    render(<ProgressCard status={status} />)
    expect(screen.getByText('Ready')).toBeInTheDocument()
    expect(screen.getByText('Ready to enhance')).toBeInTheDocument()
  })

  it('shows progress bar during processing', () => {
    const status: ProcessingStatus = { status: 'processing', progress: 50, message: 'Working...' }
    render(<ProgressCard status={status} />)
    expect(screen.getByText('50%')).toBeInTheDocument()
    expect(screen.getByText('Progress')).toBeInTheDocument()
  })

  it('shows progress bar during uploading', () => {
    const status: ProcessingStatus = { status: 'uploading', progress: 30, message: 'Uploading...' }
    render(<ProgressCard status={status} />)
    expect(screen.getByText('30%')).toBeInTheDocument()
  })

  it('hides progress bar for completed status', () => {
    const status: ProcessingStatus = { status: 'completed', progress: 100, message: 'Done' }
    render(<ProgressCard status={status} />)
    expect(screen.queryByText('Progress')).not.toBeInTheDocument()
  })

  it('shows processing time when completed', () => {
    const status: ProcessingStatus = {
      status: 'completed',
      progress: 100,
      message: 'Done',
      processingTime: 3.5,
    }
    render(<ProgressCard status={status} />)
    expect(screen.getByText(/Processed in 3.5s/i)).toBeInTheDocument()
  })

  it('shows failed status', () => {
    const status: ProcessingStatus = { status: 'failed', progress: 0, message: 'Something broke' }
    render(<ProgressCard status={status} />)
    expect(screen.getByText('failed')).toBeInTheDocument()
    expect(screen.getByText('Something broke')).toBeInTheDocument()
  })
})
