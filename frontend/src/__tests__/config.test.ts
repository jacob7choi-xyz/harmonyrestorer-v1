import { describe, it, expect } from 'vitest'
import { parseMaxUploadMb, MAX_UPLOAD_MB, MAX_UPLOAD_BYTES } from '../config'

describe('parseMaxUploadMb', () => {
  it('defaults to 50 when the setting is unset', () => {
    expect(parseMaxUploadMb(undefined)).toBe(50)
  })

  it('reads a positive whole number of MB', () => {
    expect(parseMaxUploadMb('4')).toBe(4)
  })

  it.each([
    ['empty', ''],
    ['not a number', 'abc'],
    ['zero', '0'],
    ['negative', '-4'],
    ['fractional', '4.5'],
    ['exponent form', '4e1'],
    ['surrounding whitespace', ' 4 '],
    ['too large to count in bytes exactly', '9999999999999'],
  ])('falls back to 50 for an invalid value (%s)', (_label, raw) => {
    expect(parseMaxUploadMb(raw)).toBe(50)
  })
})

describe('upload limit exports', () => {
  it('counts a MB as 1024 * 1024 bytes', () => {
    expect(MAX_UPLOAD_BYTES).toBe(MAX_UPLOAD_MB * 1024 * 1024)
  })
})
