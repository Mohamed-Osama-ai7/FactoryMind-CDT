import { describe, expect, it, vi } from 'vitest'
import { setTokenProvider, setUnauthorizedHandler } from '../auth/token'
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

  describe('authentication', () => {
    it('attaches the bearer token to protected calls', async () => {
      const f = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
      vi.stubGlobal('fetch', f)
      await apiFetch('/api/v1/me')
      expect(new Headers(f.mock.calls[0]?.[1]?.headers).get('Authorization')).toBe(
        'Bearer test-token',
      )
    })

    it('does not touch the network without a token and signals unauthorized', async () => {
      const f = vi.fn()
      const unauthorized = vi.fn()
      vi.stubGlobal('fetch', f)
      setTokenProvider(() => null)
      setUnauthorizedHandler(unauthorized)
      await expect(apiFetch('/api/v1/incidents')).rejects.toMatchObject({ status: 401 })
      expect(f).not.toHaveBeenCalled()
      expect(unauthorized).toHaveBeenCalledOnce()
    })

    it('keeps health public: no token required and none sent', async () => {
      const f = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }))
      vi.stubGlobal('fetch', f)
      setTokenProvider(() => null)
      await apiFetch('/api/v1/health')
      expect(new Headers(f.mock.calls[0]?.[1]?.headers).has('Authorization')).toBe(false)
    })

    it('notifies on a 401 from the server', async () => {
      const unauthorized = vi.fn()
      setUnauthorizedHandler(unauthorized)
      vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 401 })))
      await expect(apiFetch('/api/v1/me')).rejects.toBeInstanceOf(ApiError)
      expect(unauthorized).toHaveBeenCalledOnce()
    })
  })
})
