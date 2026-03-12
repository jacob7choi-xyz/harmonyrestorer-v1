import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { ReactNode } from 'react'
import { ErrorBoundary } from '../ErrorBoundary'

let shouldThrow = true

function ConditionalThrower(): ReactNode {
  if (shouldThrow) {
    throw new Error('Test explosion')
  }
  return <div>Recovered content</div>
}

function AlwaysThrower(): ReactNode {
  throw new Error('Test explosion')
}

describe('ErrorBoundary', () => {
  const originalError = console.error
  beforeEach(() => {
    console.error = vi.fn()
    shouldThrow = true
  })
  afterEach(() => {
    console.error = originalError
  })

  it('renders children when no error', () => {
    render(
      <ErrorBoundary>
        <div>Child content</div>
      </ErrorBoundary>,
    )
    expect(screen.getByText('Child content')).toBeInTheDocument()
  })

  it('renders error UI when child throws', () => {
    render(
      <ErrorBoundary>
        <AlwaysThrower />
      </ErrorBoundary>,
    )
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText('Test explosion')).toBeInTheDocument()
  })

  it('shows Try Again button in error state', () => {
    render(
      <ErrorBoundary>
        <AlwaysThrower />
      </ErrorBoundary>,
    )
    expect(screen.getByText('Try Again')).toBeInTheDocument()
  })

  it('recovers when child stops throwing after Try Again', async () => {
    const user = userEvent.setup()

    render(
      <ErrorBoundary>
        <ConditionalThrower />
      </ErrorBoundary>,
    )

    // Initially throws, so error UI is shown
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()

    // Stop throwing before clicking Try Again
    shouldThrow = false
    await user.click(screen.getByText('Try Again'))

    // Now the child renders successfully
    expect(screen.getByText('Recovered content')).toBeInTheDocument()
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument()
  })
})
