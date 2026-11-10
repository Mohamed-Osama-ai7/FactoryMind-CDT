/**
 * Helpers for the dataset's naive timestamps ("YYYY-MM-DDTHH:mm:ss", local train time with no
 * timezone). Arithmetic is done in UTC so the browser's timezone and DST never shift values.
 */

const NAIVE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?$/

export function isNaiveTimestamp(value: string): boolean {
  return NAIVE.test(value) && !Number.isNaN(toEpochMs(value))
}

/** Milliseconds treating the naive timestamp as if it were UTC (stable, timezone-free). */
export function toEpochMs(value: string): number {
  const withSeconds = value.length === 16 ? `${value}:00` : value
  return Date.parse(`${withSeconds}Z`)
}

export function fromEpochMs(ms: number): string {
  return new Date(ms).toISOString().slice(0, 19)
}

export function addHours(value: string, hours: number): string {
  return fromEpochMs(toEpochMs(value) + hours * 3_600_000)
}

/** "2020-06-05T09:50:00" -> "2020-06-05 09:50" */
export function formatNaive(value: string): string {
  return value.replace('T', ' ').slice(0, 16)
}
