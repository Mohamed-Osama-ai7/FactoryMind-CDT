import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { AuditViewer } from './AuditViewer'

describe('AuditViewer', () => {
  it('shows permission notice when principal lacks audit:read', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            actor_id: 'eng-1',
            tenant_id: 'acme',
            roles: ['engineer'],
            permissions: ['incident:read'],
            asset_ids: ['metropt3-apu-1'],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <AuditViewer />
      </QueryClientProvider>,
    )
    expect(await screen.findByText(/do not hold the/)).toBeInTheDocument()
  })

  it('lists audit events and verifies SHA-256 chain integrity', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        if (String(url).startsWith('/api/v1/me')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                actor_id: 'aud-1',
                tenant_id: 'acme',
                roles: ['auditor'],
                permissions: ['incident:read', 'audit:read'],
                asset_ids: ['metropt3-apu-1'],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        if (String(url).startsWith('/api/v1/audit/verify')) {
          return Promise.resolve(
            new Response(JSON.stringify({ valid: true, events: 2, first_invalid_seq: null }), {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            }),
          )
        }
        return Promise.resolve(
          new Response(
            JSON.stringify({
              items: [
                {
                  seq: 2,
                  ts: '2026-11-11T15:00:00',
                  actor_id: 'eng-1',
                  roles: 'engineer',
                  action: 'outcome.create',
                  object_type: 'incident',
                  object_id: 'a'.repeat(32),
                  outcome: 'allowed',
                  request_id: 'req-2',
                  details: { outcome: 'confirmed_leak' },
                  hash: 'abcdef0123456789abcdef',
                },
              ],
              limit: 25,
              offset: 0,
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <AuditViewer />
      </QueryClientProvider>,
    )

    expect(await screen.findByText('eng-1')).toBeInTheDocument()
    expect(screen.getByRole('cell', { name: 'outcome.create' })).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Verify SHA-256 hash chain' }))
    expect(
      await screen.findByText('Hash chain intact across 2 audit event(s).'),
    ).toBeInTheDocument()
  })

  it('supports action filtering without action=null, pagination, broken chain alert, and me error', async () => {
    const requestedUrls: string[] = []
    const makeRow = (seq: number, action: string) => ({
      seq,
      ts: '2026-11-11T15:00:00',
      actor_id: `eng-${seq}`,
      roles: 'engineer',
      action,
      object_type: 'incident',
      object_id: 'a'.repeat(32),
      outcome: 'allowed',
      request_id: `req-${seq}`,
      details: {},
      hash: 'abcdef0123456789abcdef',
    })

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        const urlStr = String(url)
        requestedUrls.push(urlStr)
        if (urlStr.startsWith('/api/v1/me')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                actor_id: 'aud-1',
                tenant_id: 'acme',
                roles: ['auditor'],
                permissions: ['incident:read', 'audit:read'],
                asset_ids: ['metropt3-apu-1'],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        if (urlStr.startsWith('/api/v1/audit/verify')) {
          return Promise.resolve(
            new Response(JSON.stringify({ valid: false, events: 30, first_invalid_seq: 17 }), {
              status: 200,
              headers: { 'Content-Type': 'application/json' },
            }),
          )
        }
        const parsed = new URL(urlStr, 'http://localhost')
        const offset = Number(parsed.searchParams.get('offset') ?? '0')
        const action = parsed.searchParams.get('action') ?? 'incident.create'
        const items =
          offset === 0
            ? Array.from({ length: 25 }, (_, idx) => makeRow(idx + 1, action))
            : [makeRow(26, action)]
        return Promise.resolve(
          new Response(JSON.stringify({ items, limit: 25, offset }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          }),
        )
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <AuditViewer />
      </QueryClientProvider>,
    )

    expect(await screen.findByText('eng-1')).toBeInTheDocument()
    const initialEventsUrl = requestedUrls.find((u) => u.startsWith('/api/v1/audit/events'))
    expect(initialEventsUrl).toBeTruthy()
    expect(initialEventsUrl).not.toContain('action=null')

    // Click Next page when 25 rows returned
    await userEvent.click(screen.getByRole('button', { name: 'Next' }))
    expect(await screen.findByText('eng-26')).toBeInTheDocument()
    expect(screen.getByText('Offset: 25')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Next' })).toBeDisabled()

    // Click action filter resets offset to 0 and omits null when cleared
    await userEvent.click(screen.getByRole('button', { name: 'outcome.create' }))
    expect(await screen.findByText('eng-1')).toBeInTheDocument()
    expect(screen.getByText('Offset: 0')).toBeInTheDocument()
    expect(
      requestedUrls.some((u) => u.includes('/api/v1/audit/events?action=outcome.create')),
    ).toBe(true)

    // Verify broken hash-chain warning
    await userEvent.click(screen.getByRole('button', { name: 'Verify SHA-256 hash chain' }))
    expect(
      await screen.findByText('Hash chain break detected at seq 17 (30 total).'),
    ).toBeInTheDocument()
  })

  it('shows an error alert instead of permission denied if /me fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'Auth unavailable' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    )
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <AuditViewer />
      </QueryClientProvider>,
    )
    expect(await screen.findByRole('alert')).toHaveTextContent(
      /Could not verify audit permissions: Auth unavailable/,
    )
  })
})
