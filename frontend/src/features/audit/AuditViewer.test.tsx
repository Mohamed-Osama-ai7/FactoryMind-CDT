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
})
