import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { SystemStatus } from './SystemStatus'

function renderWithClient() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={client}>
      <SystemStatus />
    </QueryClientProvider>,
  )
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('SystemStatus', () => {
  it('shows backend version and read-only operating mode when healthy', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      jsonResponse({
        status: 'ok',
        version: '0.1.0',
        environment: 'test',
        operating_mode: 'advisory-read-only',
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    renderWithClient()

    expect(await screen.findByText('Backend online')).toBeInTheDocument()
    expect(screen.getByText('0.1.0')).toBeInTheDocument()
    expect(screen.getByText('advisory-read-only')).toBeInTheDocument()
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/v1/health',
      expect.objectContaining({ method: 'GET' }),
    )
  })

  it('reports the backend as unreachable on an HTTP error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(jsonResponse({ detail: 'boom' }, 503)))

    renderWithClient()

    expect(await screen.findByText('Backend unreachable')).toBeInTheDocument()
  })
})
