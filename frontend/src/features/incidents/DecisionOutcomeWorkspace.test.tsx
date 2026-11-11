import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import {
  getGetIncidentQueryKey,
  getListIncidentsQueryKey,
} from '../../api/generated/incidents/incidents'
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

  it('supports defer workflow, invalidates incident on 409 Conflict, and renders history cleanly', async () => {
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
        const payload = JSON.parse(String(init.body)) as { expected_version: number }
        if (payload.expected_version === 2) {
          return Promise.resolve(
            new Response(JSON.stringify({ detail: 'Incident changed or already decided.' }), {
              status: 409,
              headers: { 'Content-Type': 'application/json' },
            }),
          )
        }
        return Promise.resolve(
          new Response(JSON.stringify({ ...sampleIncident, status: 'open', version: 2 }), {
            status: 201,
            headers: { 'Content-Type': 'application/json' },
          }),
        )
      }
      return Promise.resolve(new Response('{}', { status: 404 }))
    })
    vi.stubGlobal('fetch', fetchMock)

    const populated: IncidentDetail = {
      ...sampleIncident,
      version: 2,
      decisions: [
        {
          id: 'dec-1',
          decision: 'defer',
          rationale: 'Waiting for maintenance crew report.',
          actor_id: 'app-1',
          incident_version: 1,
          created_at: '2020-05-30T06:20:00',
        },
      ],
      outcomes: [
        {
          id: 'out-1',
          incident_id: sampleIncident.id,
          asset_id: 'metropt3-apu-1',
          actor_id: 'eng-2',
          outcome: 'sensor_fault_confirmed',
          maintenance_action: 'Inspected TP3 pressure transducer harness',
          verified_at: null,
          notes: 'Intermittent loose pin caused zero drop.',
          created_at: '2020-05-30T08:00:00',
          origin: 'human_observation',
        },
      ],
    }

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const invalidateSpy = vi.spyOn(client, 'invalidateQueries')

    render(
      <QueryClientProvider client={client}>
        <DecisionPanel incident={populated} />
        <OutcomeVerificationPanel incident={populated} />
      </QueryClientProvider>,
    )

    expect(screen.getByText(/Waiting for maintenance crew report\./)).toBeInTheDocument()
    expect(
      screen.getByText(/Confirmed sensor \/ instrumentation fault · eng-2/),
    ).toBeInTheDocument()
    expect(screen.getByText(/recorded UTC/)).toBeInTheDocument()

    // Whitespace-only rationale shows validation error instead of silently ignoring
    const rationale = await screen.findByLabelText(/Engineering rationale/)
    await userEvent.click(screen.getByLabelText(/Defer pending check/))
    await userEvent.type(rationale, '   ')
    await userEvent.click(screen.getByRole('button', { name: 'Submit review decision' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/non-empty engineering rationale/)

    // 409 Conflict refreshes both incident detail and list queries
    await userEvent.clear(rationale)
    await userEvent.type(rationale, 'Re-deferring with stale v2 token.')
    await userEvent.click(screen.getByRole('button', { name: 'Submit review decision' }))

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /Incident changed or already decided/,
    )
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: getGetIncidentQueryKey(populated.id),
    })
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: getListIncidentsQueryKey(),
    })
  })
})
