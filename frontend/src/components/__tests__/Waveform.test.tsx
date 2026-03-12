import { describe, it, expect, vi } from 'vitest'
import { render } from '@testing-library/react'
import { Waveform } from '../Waveform'

describe('Waveform', () => {
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

  it('starts animation interval when active', () => {
    const setIntervalSpy = vi.spyOn(global, 'setInterval')
    render(<Waveform isActive={true} />)
    expect(setIntervalSpy).toHaveBeenCalledWith(expect.any(Function), 100)
    setIntervalSpy.mockRestore()
  })

  it('clears interval on unmount', () => {
    const clearIntervalSpy = vi.spyOn(global, 'clearInterval')
    const { unmount } = render(<Waveform isActive={true} />)
    unmount()
    expect(clearIntervalSpy).toHaveBeenCalled()
    clearIntervalSpy.mockRestore()
  })
})
