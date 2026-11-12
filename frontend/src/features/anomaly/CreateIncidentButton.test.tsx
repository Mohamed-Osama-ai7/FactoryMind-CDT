import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { CreateIncidentButton } from './CreateIncidentButton'

function stubApi(assetIds: string[], permissions = ['anomaly:read', 'incident:create']) {
  const fetchMock = vi.fn().mockImplementation((url: string) => {
    const isMe = String(url).startsWith('/api/v1/me')
    const body = isMe
      ? {
          actor_id: 'eve',
          tenant_id: 'demo',
          roles: ['engineer'],
          permissions,
          asset_ids: assetIds,
        }
      : { id: 'a'.repeat(32), status: 'open' }
    return Promise.resolve(
      new Response(JSON.stringify(body), {
        status: isMe ? 200 : 201,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

function renderButton() {
  render(
    <QueryClientProvider
      client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}
    >
      <CreateIncidentButton
        rangeStart="2020-06-05T00:00:00"
        rangeEnd="2020-06-05T06:00:00"
        episodeStart="2020-06-05T09:50:00"
      />
    </QueryClientProvider>,
  )
}

describe('CreateIncidentButton', () => {
  it('always posts the replayed telemetry asset, not the first granted asset', async () => {
    const fetchMock = stubApi(['ai4i-milling-bench', 'metropt3-apu-1'])
    renderButton()
    await userEvent.click(await screen.findByRole('button', { name: /Create incident/i }))
    const post = fetchMock.mock.calls.find(([, init]) => init?.method === 'POST')
    expect(JSON.parse(String(post?.[1]?.body)).asset_id).toBe('metropt3-apu-1')
  })

  it('is hidden when the replayed asset is not granted', async () => {
    const fetchMock = stubApi(['ai4i-milling-bench'])
    renderButton()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    expect(screen.queryByRole('button', { name: /Create incident/i })).not.toBeInTheDocument()
  })

  it('is hidden when incident:create permission is absent even if metropt3-apu-1 is granted', async () => {
    const fetchMock = stubApi(['metropt3-apu-1'], ['anomaly:read', 'incident:read'])
    renderButton()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    expect(screen.queryByRole('button', { name: /Create incident/i })).not.toBeInTheDocument()
  })
})
