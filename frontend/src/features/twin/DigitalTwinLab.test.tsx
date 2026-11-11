import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { DigitalTwinLab } from './DigitalTwinLab'

function mockMeOk() {
  return new Response(
    JSON.stringify({
      actor_id: 'eng-1',
      tenant_id: 'acme',
      roles: ['engineer'],
      permissions: ['asset:read'],
      asset_ids: ['metropt3-apu-1'],
    }),
    { status: 200, headers: { 'Content-Type': 'application/json' } },
  )
}

describe('DigitalTwinLab', () => {
  it('blocks callers lacking asset:read', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            actor_id: 'aud-1',
            tenant_id: 'acme',
            roles: ['auditor'],
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
        <DigitalTwinLab />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/do not hold the/)).toBeInTheDocument()
  })

  it('displays errors cleanly when twin model or simulation fails', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        if (String(url).startsWith('/api/v1/me')) return Promise.resolve(mockMeOk())
        return Promise.resolve(
          new Response(JSON.stringify({ detail: 'Twin offline' }), {
            status: 500,
            headers: { 'Content-Type': 'application/json' },
          }),
        )
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <DigitalTwinLab />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/Simulation failed/)).toBeInTheDocument()
    expect(screen.getByText(/Could not load digital twin specification/)).toBeInTheDocument()
  })

  it('validates numeric bounds cleanly and simulates non-step-aligned continuous scenarios', async () => {
    const seenBodies: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string, init?: RequestInit) => {
        const s = String(url)
        if (s.startsWith('/api/v1/me')) return Promise.resolve(mockMeOk())
        if (s.startsWith('/api/v1/twin/model')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                asset_id: 'metropt3-apu-1',
                model_version: 'apu-mass-balance-v1',
                provenance_origin: 'advisory_physics_twin',
                governing_equations: ['dP_TP3/dt = k_charge * u - k_drain - k_leak'],
                parameters: [
                  {
                    name: 'k_charge_bar_per_min',
                    value: 0.948,
                    unit: 'bar/min',
                    identifiable: true,
                    fit_source: 'live_pre_eval_fit',
                    uncertainty_note: 'Steady slope.',
                  },
                ],
                unidentifiable_parameters: [
                  {
                    name: 'reservoir_volume_m3',
                    reason: 'Reservoir vessel volume not metered.',
                    assumption_or_proxy: 'Normalized pressure rates in bar/min.',
                  },
                ],
                validation: {
                  split_policy: 'Fit strictly on <2020-04-01; validate >=2020-04-01.',
                  train_cal_samples: 165547,
                  heldout_normal_segments: 168,
                  mae_10min_bar: 0.2526,
                  mae_60min_bar: 0.5218,
                  rmse_60min_bar: 0.5853,
                  constraint_violations: 0,
                  step_halving_max_diff_bar: 0.395,
                  leak_vs_normal_residual_ratio: 3.78,
                  provenance_origin: 'empirical_validation',
                },
                advisory_note: 'Advisory surrogate only.',
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        seenBodies.push(String(init?.body ?? ''))
        return Promise.resolve(
          new Response(
            JSON.stringify({
              asset_id: 'metropt3-apu-1',
              scenario: 'maintenance_repair',
              provenance_origin: 'simulated_projection',
              params_used: [],
              unidentifiable_parameters: [],
              trajectory: [
                {
                  minute: 0,
                  tp3_bar: 9.1,
                  reservoirs_bar: 9.1,
                  oil_temp_c: 62.0,
                  motor_loaded: false,
                  rolling_duty: 0.0,
                  lps_active: false,
                  regime: 'unloaded',
                  provenance_origin: 'simulated_projection',
                },
                {
                  minute: 45,
                  tp3_bar: 9.6,
                  reservoirs_bar: 9.6,
                  oil_temp_c: 64.2,
                  motor_loaded: false,
                  rolling_duty: 0.12,
                  lps_active: false,
                  regime: 'unloaded',
                  provenance_origin: 'simulated_projection',
                },
              ],
              summary: {
                scenario: 'maintenance_repair',
                mean_load_duty: 0.32,
                peak_load_duty: 0.78,
                min_tp3_bar: 7.4,
                max_tp3_bar: 10.1,
                lps_excursion_minutes: 0.0,
                time_above_duty_threshold_minutes: 12.0,
                final_oil_temp_c: 64.2,
                constraint_violations: 0,
                advisory_interpretation: 'Repair restores pressure stability.',
              },
              advisory_note: 'Counterfactual advisory rollout.',
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <DigitalTwinLab />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/simulated_projection/)).toBeInTheDocument()
    expect(await screen.findByText(/reservoir_volume_m3/)).toBeInTheDocument()
    expect(screen.getByText(/k_charge_bar_per_min: 0.948 bar\/min/)).toBeInTheDocument()
    expect(screen.getByText(/3\.78x/)).toBeInTheDocument()

    // Empty numeric input shows inline alert instead of crashing or snapping to 0
    const horizonInput = screen.getByLabelText('Horizon (minutes, 10–240)')
    await userEvent.clear(horizonInput)
    expect(horizonInput).toHaveValue(null)
    await userEvent.click(screen.getByRole('button', { name: 'Run scenario simulation' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/Horizon \(minutes\) is required/)

    // Non-step-aligned continuous float inputs (horizon=15, leak=0.33, repair>=horizon check)
    await userEvent.type(horizonInput, '15')
    await userEvent.selectOptions(screen.getByLabelText('Scenario'), 'maintenance_repair')
    await userEvent.click(screen.getByRole('button', { name: 'Run scenario simulation' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/strictly less than horizon/)

    const repairInput = screen.getByLabelText('Repair intervention minute')
    await userEvent.clear(repairInput)
    await userEvent.type(repairInput, '10')
    const leakInput = screen.getByLabelText('Leak loss rate (bar/min)')
    await userEvent.clear(leakInput)
    await userEvent.type(leakInput, '0.33')
    await userEvent.click(screen.getByRole('button', { name: 'Run scenario simulation' }))

    await waitFor(() =>
      expect(
        seenBodies.some(
          (b) =>
            b.includes('"scenario":"maintenance_repair"') &&
            b.includes('"horizon_minutes":15') &&
            b.includes('"leak_rate_bar_per_min":0.33'),
        ),
      ).toBe(true),
    )
  })
})
