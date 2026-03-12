import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, act } from '@testing-library/react'
import { Waveform } from '../Waveform'

describe('Waveform', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('renders 32 bars', () => {
    const { container } = render(<Waveform isActive={false} />)
    const bars = container.querySelectorAll('[class*="bg-gradient"]')
    expect(bars.length).toBe(32)
  })

  it('bars have minimal height when inactive', () => {
    const { container } = render(<Waveform isActive={false} />)
    const bars = container.querySelectorAll('[class*="bg-gradient"]')
    bars.forEach((bar) => {
      expect((bar as HTMLElement).style.height).toBe('10%')
    })
  })

  it('updates heights after interval tick when active', () => {
    const { container } = render(<Waveform isActive={true} />)
    const bars = container.querySelectorAll('[class*="bg-gradient"]')

    // All start at 10%
    const initialHeights = Array.from(bars).map((b) => (b as HTMLElement).style.height)
    expect(initialHeights.every((h) => h === '10%')).toBe(true)

    // Advance past one interval tick (100ms)
    act(() => {
      vi.advanceTimersByTime(100)
    })

    // At least some bars should have changed from 10%
    const updatedHeights = Array.from(bars).map((b) => (b as HTMLElement).style.height)
    const changed = updatedHeights.filter((h) => h !== '10%')
    expect(changed.length).toBeGreaterThan(0)
  })

  it('resets heights when switching from active to inactive', () => {
    const { container, rerender } = render(<Waveform isActive={true} />)

    // Let animation run
    act(() => {
      vi.advanceTimersByTime(100)
    })

    // Switch to inactive
    rerender(<Waveform isActive={false} />)

    const bars = container.querySelectorAll('[class*="bg-gradient"]')
    bars.forEach((bar) => {
      expect((bar as HTMLElement).style.height).toBe('10%')
    })
  })

  it('stops updating heights after deactivation', () => {
    const { container, rerender } = render(<Waveform isActive={true} />)

    // Let animation run
    act(() => {
      vi.advanceTimersByTime(100)
    })

    // Deactivate
    rerender(<Waveform isActive={false} />)

    // Capture heights after reset
    const bars = container.querySelectorAll('[class*="bg-gradient"]')
    const heightsAfterReset = Array.from(bars).map((b) => (b as HTMLElement).style.height)

    // Advance more time -- heights should stay at 10%
    act(() => {
      vi.advanceTimersByTime(500)
    })

    const heightsLater = Array.from(bars).map((b) => (b as HTMLElement).style.height)
    expect(heightsLater).toEqual(heightsAfterReset)
  })

  it('clears interval on unmount', () => {
    const clearIntervalSpy = vi.spyOn(global, 'clearInterval')
    const { unmount } = render(<Waveform isActive={true} />)
    unmount()
    expect(clearIntervalSpy).toHaveBeenCalled()
    clearIntervalSpy.mockRestore()
  })
})
