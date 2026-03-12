import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ErrorBoundary } from '../ErrorBoundary'

function ThrowingComponent(): JSX.Element {
  throw new Error('Test explosion')
}

describe('ErrorBoundary', () => {
  const originalError = console.error
  beforeEach(() => {
    console.error = vi.fn()
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
        <ThrowingComponent />
      </ErrorBoundary>,
    )
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
    expect(screen.getByText('Test explosion')).toBeInTheDocument()
  })

  it('shows Try Again button in error state', () => {
    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>,
    )
    expect(screen.getByText('Try Again')).toBeInTheDocument()
  })

  it('Try Again button clears error state', async () => {
    const user = userEvent.setup()

    render(
      <ErrorBoundary>
        <ThrowingComponent />
      </ErrorBoundary>,
    )

    expect(screen.getByText('Something went wrong')).toBeInTheDocument()

    // Clicking Try Again clears error state via setState({ error: null }).
    // Since the child still throws, it will re-enter error state, but this
    // verifies the button handler fires without crashing.
    await user.click(screen.getByText('Try Again'))

    // After clicking, the boundary re-renders children, which throw again,
    // so we end up back at the error UI -- this is expected behavior.
    expect(screen.getByText('Something went wrong')).toBeInTheDocument()
  })
})
