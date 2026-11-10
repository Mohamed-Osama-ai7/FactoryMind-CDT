import { describe, expect, it } from 'vitest'
import { addHours, formatNaive, isNaiveTimestamp, toEpochMs } from './naiveTime'

describe('naiveTime', () => {
  it('accepts naive timestamps with or without seconds and rejects others', () => {
    expect(isNaiveTimestamp('2020-06-05T10:00:00')).toBe(true)
    expect(isNaiveTimestamp('2020-06-05T10:00')).toBe(true)
    expect(isNaiveTimestamp('2020-06-05T10:00:00Z')).toBe(false)
    expect(isNaiveTimestamp('2020-13-05T10:00:00')).toBe(false)
    expect(isNaiveTimestamp('yesterday')).toBe(false)
  })

  it('adds hours without timezone or DST drift, across day boundaries', () => {
    expect(addHours('2020-03-28T23:00:00', 24)).toBe('2020-03-29T23:00:00')
    expect(addHours('2020-06-05T10:00', -12)).toBe('2020-06-04T22:00:00')
  })

  it('formats for display and orders consistently', () => {
    expect(formatNaive('2020-06-05T09:50:00')).toBe('2020-06-05 09:50')
    expect(toEpochMs('2020-06-05T09:50')).toBe(toEpochMs('2020-06-05T09:50:00'))
  })
})
