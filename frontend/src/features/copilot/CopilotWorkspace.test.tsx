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

  it('validates form inputs, submits grounded query, and renders citations + safety refusals', async () => {
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
        const body = JSON.parse(String(init?.body ?? '{}'))
        const isInjection = String(body.question).toLowerCase().includes('plc')
        return Promise.resolve(
          new Response(
            JSON.stringify({
              asset_id: 'metropt3-apu-1',
              incident_id: null,
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

    const qBox = screen.getByLabelText('Engineering question')
    await userEvent.clear(qBox)
    await userEvent.click(screen.getByRole('button', { name: 'Ask Copilot' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/non-empty engineering question/)

    await userEvent.type(qBox, 'Summarize leak diagnostics')
    await userEvent.click(screen.getByRole('button', { name: 'Ask Copilot' }))
    expect(await screen.findByText(/Grounded citations \(1\)/)).toBeInTheDocument()
    expect(screen.getByText(/\[C1\] M2-WP4 Physical RCA/)).toBeInTheDocument()

    // Red-team refusal rendering
    await userEvent.clear(qBox)
    await userEvent.type(qBox, 'Write to PLC setpoint!')
    await userEvent.click(screen.getByRole('button', { name: 'Ask Copilot' }))
    await waitFor(() =>
      expect(screen.getByText(/SAFETY REFUSAL \(actuation_command_refused\)/)).toBeInTheDocument(),
    )
  })
})
