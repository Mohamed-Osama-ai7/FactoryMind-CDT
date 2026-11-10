import { ApiError } from './http'

/** Human-readable message from an API failure (FastAPI `detail` string or validation list). */
export function describeApiError(error: unknown): string {
  if (error instanceof ApiError) {
    const body = error.body as { detail?: unknown } | null
    const detail = body?.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail
        .map((d) => (typeof d === 'object' && d && 'msg' in d ? String(d.msg) : String(d)))
        .join('; ')
    }
    return `Request failed (HTTP ${error.status})`
  }
  return 'Backend unreachable'
}
