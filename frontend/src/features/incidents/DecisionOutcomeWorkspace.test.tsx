import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import type { IncidentDetail } from '../../api/generated/model'
import { DecisionPanel } from './DecisionPanel'
import { OutcomeVerificationPanel } from './OutcomeVerificationPanel'

const sampleIncident: IncidentDetail = {
  id: 'a'.repeat(32),
  tenant_id: 'acme',
  asset_id: 'metropt3-apu-1',
  status: 'open',
  version: 1,
  detector: 'duty_rule',
  model_version: 'v1',
  episode_start: '2020-05-29T23:30:00',
  episode_end: '2020-05-30T06:00:00',
  raised_at: '2020-05-30T00:00:00',
  peak_score: 0.9,
  created_by: 'eng-1',
  created_at: '2020-05-30T06:10:00',
  evidence: [],
  decisions: [],
  observations: [],
  outcomes: [],
}

describe('DecisionPanel and OutcomeVerificationPanel', () => {
  it('enforces separation of duties notice when creator matches current approver', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            actor_id: 'eng-1',
            tenant_id: 'acme',
            roles: ['engineer', 'approver'],
            permissions: ['incident:read', 'decision:create', 'outcome:create'],
            asset_ids: ['metropt3-apu-1'],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    )
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <DecisionPanel incident={sampleIncident} />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/Separation of duties active/)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: 'Submit review decision' })).toBeNull()
  })

  it('allows separate approver decision submit and outcome verification logging', async () => {
    const fetchMock = vi.fn().mockImplementation((url: string, init?: RequestInit) => {
      if (String(url).startsWith('/api/v1/me')) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              actor_id: 'app-2',
              tenant_id: 'acme',
              roles: ['approver'],
              permissions: ['incident:read', 'decision:create', 'outcome:create'],
              asset_ids: ['metropt3-apu-1'],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }
      if (String(url).endsWith('/decisions') && init?.method === 'POST') {
        return Promise.resolve(
          new Response(JSON.stringify({ ...sampleIncident, status: 'confirmed', version: 2 }), {
            status: 201,
            headers: { 'Content-Type': 'application/json' },
          }),
        )
      }
      return Promise.resolve(
        new Response(
          JSON.stringify({
            id: 'out-1',
            incident_id: sampleIncident.id,
            asset_id: 'metropt3-apu-1',
            actor_id: 'app-2',
            outcome: 'confirmed_leak',
            maintenance_action: 'Tightened purge hose',
            verified_at: '2020-05-30T06:30:00',
            notes: 'Soap bubble confirmed.',
            created_at: '2020-05-30T07:00:00',
            origin: 'human_observation',
          }),
          { status: 201, headers: { 'Content-Type': 'application/json' } },
        ),
      )
    })
    vi.stubGlobal('fetch', fetchMock)

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <DecisionPanel incident={sampleIncident} />
        <OutcomeVerificationPanel incident={sampleIncident} />
      </QueryClientProvider>,
    )

    const rationale = await screen.findByLabelText(/Engineering rationale/)
    await userEvent.type(rationale, 'Corroborated air leak duty rise.')
    await userEvent.click(screen.getByRole('button', { name: 'Submit review decision' }))
    expect(await screen.findByText(/Decision recorded and audited/)).toBeInTheDocument()

    await userEvent.type(
      screen.getByLabelText(/Physical check \/ maintenance summary/),
      'Tightened purge hose',
    )
    await userEvent.type(screen.getByLabelText(/Verification notes/), 'Soap bubble confirmed.')
    await userEvent.click(screen.getByRole('button', { name: 'Record outcome' }))
    expect(await screen.findByText(/Outcome recorded and audited/)).toBeInTheDocument()
  })
})
