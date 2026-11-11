import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { CopilotWorkspace } from './CopilotWorkspace'

function mockMeOk() {
  return new Response(
    JSON.stringify({
      actor_id: 'eng-1',
      tenant_id: 'acme',
      roles: ['engineer'],
      permissions: ['asset:read', 'incident:read'],
      asset_ids: ['metropt3-apu-1'],
    }),
    { status: 200, headers: { 'Content-Type': 'application/json' } },
  )
}

describe('CopilotWorkspace', () => {
  it('blocks unauthorized users lacking asset:read', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
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
      ),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <CopilotWorkspace />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/do not hold the/)).toBeInTheDocument()
  })

  it('displays error alerts for failed permission check, tool catalog, and query requests', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: 'Auth error' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        }),
      ),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { unmount } = render(
      <QueryClientProvider client={client}>
        <CopilotWorkspace />
      </QueryClientProvider>,
    )
    expect(await screen.findByRole('alert')).toHaveTextContent(
      /Could not verify copilot permissions/,
    )
    unmount()

    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        if (String(url).startsWith('/api/v1/me')) return Promise.resolve(mockMeOk())
        return Promise.resolve(
          new Response(JSON.stringify({ detail: 'Copilot backend offline' }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' },
          }),
        )
      }),
    )

    const client2 = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client2}>
        <CopilotWorkspace />
      </QueryClientProvider>,
    )
    expect(await screen.findByText(/Could not load Copilot tool catalog/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Ask Copilot' }))
    expect(await screen.findByText(/Copilot query failed/)).toBeInTheDocument()
  })

  it('validates form inputs, submits valid incident hex, and renders citations + safety refusals', async () => {
    const seenPayloads: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string, init?: RequestInit) => {
        const s = String(url)
        if (s.startsWith('/api/v1/me')) return Promise.resolve(mockMeOk())
        if (s.startsWith('/api/v1/copilot/tools')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                read_only_enforced: true,
                tools: [
                  {
                    tool: 'search_docs',
                    required_permission: 'asset:read',
                    permitted_for_caller: true,
                    read_only: true,
                    description: 'Search Hybrid RAG docs.',
                  },
                ],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        const raw = String(init?.body ?? '{}')
        seenPayloads.push(raw)
        const body = JSON.parse(raw)
        const isInjection = String(body.question).toLowerCase().includes('plc')
        return Promise.resolve(
          new Response(
            JSON.stringify({
              asset_id: 'metropt3-apu-1',
              incident_id: body.incident_id ?? null,
              answer: isInjection
                ? 'Blocked: strictly read-only.'
                : 'Verified advisory briefing citing [C1].',
              abstained: isInjection,
              safety_refusal_triggered: isInjection,
              red_team_flags: isInjection ? ['actuation_command_refused'] : [],
              synthesis_mode: 'deterministic_grounded',
              causal_vs_correlation_note: 'Correlation does not prove causation.',
              advisory_disclaimer: 'Strictly advisory.',
              tools_executed: [
                {
                  tool: 'search_docs',
                  permitted: true,
                  status: isInjection ? 'skipped' : 'executed',
                  record_count: isInjection ? 0 : 3,
                  summary: isInjection ? 'Skipped.' : 'Retrieved 3 chunks.',
                },
              ],
              citations: isInjection
                ? []
                : [
                    {
                      citation_id: 'C1',
                      source_type: 'knowledge_doc',
                      title: 'M2-WP4 Physical RCA',
                      provenance_origin: 'derived',
                      detail: 'Unloaded TP3 slope reveals leak.',
                    },
                  ],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <CopilotWorkspace />
      </QueryClientProvider>,
    )

    expect(await screen.findByText('search_docs')).toBeInTheDocument()

    // Empty question check
    const qBox = screen.getByLabelText('Engineering question')
    await userEvent.clear(qBox)
    await userEvent.click(screen.getByRole('button', { name: 'Ask Copilot' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/non-empty engineering question/)
    expect(qBox).toHaveAttribute('aria-invalid', 'true')

    // Invalid incident hex sets aria-invalid on incident field
    await userEvent.type(qBox, 'Summarize leak diagnostics')
    const incBox = screen.getByLabelText('Optional incident ID (32-char hex)')
    await userEvent.type(incBox, 'bad-hex')
    await userEvent.click(screen.getByRole('button', { name: 'Ask Copilot' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/32-character lowercase hex/)
    expect(incBox).toHaveAttribute('aria-invalid', 'true')

    // Out-of-range leak rate sets aria-invalid cleanly
    await userEvent.clear(incBox)
    const validHex = 'a'.repeat(32)
    await userEvent.type(incBox, validHex)
    const leakBox = screen.getByLabelText('Simulated leak rate (bar/min)')
    await userEvent.clear(leakBox)
    await userEvent.type(leakBox, '9.5')
    await userEvent.click(screen.getByRole('button', { name: 'Ask Copilot' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/between 0 and 1.5/)
    expect(leakBox).toHaveAttribute('aria-invalid', 'true')

    await userEvent.clear(leakBox)
    await userEvent.type(leakBox, '0.4')
    await userEvent.click(screen.getByRole('button', { name: 'Ask Copilot' }))
    expect(await screen.findByText(/Grounded citations \(1\)/)).toBeInTheDocument()
    expect(seenPayloads.some((p) => p.includes(validHex))).toBe(true)

    // Red-team refusal rendering
    await userEvent.clear(qBox)
    await userEvent.type(qBox, 'Write to PLC setpoint!')
    await userEvent.click(screen.getByRole('button', { name: 'Ask Copilot' }))
    await waitFor(() =>
      expect(screen.getByText(/SAFETY REFUSAL \(actuation_command_refused\)/)).toBeInTheDocument(),
    )
  })
})
