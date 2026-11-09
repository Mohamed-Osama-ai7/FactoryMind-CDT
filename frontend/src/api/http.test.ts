import { describe, expect, it, vi } from 'vitest'
import { ApiError, apiFetch } from './http'

describe('apiFetch', () => {
  it('returns data, status and headers for a 2xx JSON response', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"a":1}', { status: 200 })))
    const result = await apiFetch<{ data: { a: number }; status: number }>('/x')
    expect(result.status).toBe(200)
    expect(result.data).toEqual({ a: 1 })
  })

  it('returns null data for 204 No Content', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(null, { status: 204 })))
    const result = await apiFetch<{ data: unknown }>('/x')
    expect(result.data).toBeNull()
  })

  it('throws ApiError carrying status and body on non-2xx', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(new Response('{"detail":"nope"}', { status: 422 })),
    )
    const error = await apiFetch('/x').catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect(error).toMatchObject({ status: 422, body: { detail: 'nope' } })
  })
})
