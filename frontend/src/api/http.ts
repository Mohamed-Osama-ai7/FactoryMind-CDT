/** Error raised for any non-2xx API response, so TanStack Query treats it as a failure. */
export class ApiError extends Error {
  readonly status: number
  readonly body: unknown

  constructor(status: number, body: unknown) {
    super(`API request failed with status ${status}`)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

const NO_CONTENT = new Set([204, 205, 304])

function parseBody(text: string): unknown {
  if (text === '') return null
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

/**
 * Fetch wrapper used by the orval-generated client (see orval.config.ts).
 * Returns `{ data, status, headers }` and throws ApiError on non-2xx responses.
 */
export async function apiFetch<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options)
  const body = NO_CONTENT.has(response.status) ? null : parseBody(await response.text())
  if (!response.ok) {
    throw new ApiError(response.status, body)
  }
  return { data: body, status: response.status, headers: response.headers } as T
}
