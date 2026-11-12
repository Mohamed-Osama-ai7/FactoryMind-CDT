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

  it('surfaces explicit HTTP 503 unavailability when calibration telemetry is missing', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        if (String(url).startsWith('/api/v1/me')) return Promise.resolve(mockMeOk())
        return Promise.resolve(
          new Response(
            JSON.stringify({
              detail:
                'Digital twin calibration telemetry not found. Cannot initialize calibrated model without pre-evaluation telemetry.',
            }),
            {
              status: 503,
              headers: { 'Content-Type': 'application/json' },
            },
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

    const alerts = await screen.findAllByText(/calibration telemetry not found/i)
    expect(alerts).toHaveLength(2)
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
                model_version: 'apu-mass-balance-v2',
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
                  mae_10min_bar: 0.3325,
                  mae_60min_bar: 0.503,
                  rmse_60min_bar: 0.56,
                  closed_loop_mae_10min_bar: 0.328,
                  closed_loop_mae_60min_bar: 0.6008,
                  closed_loop_rmse_60min_bar: 0.68,
                  max_supported_closed_loop_horizon_minutes: 30,
                  max_supported_teacher_forced_horizon_minutes: 60,
                  max_supported_duty_horizon_minutes: 30,
                  unsupported_horizons: ['60m (closed_loop)'],
                  horizon_evaluations: [
                    {
                      horizon_minutes: 10,
                      segment_count: 227,
                      teacher_forced_tp3_mae_bar: 0.3325,
                      teacher_forced_tp3_median_mae_bar: 0.2345,
                      teacher_forced_tp3_rmse_bar: 0.39,
                      closed_loop_tp3_mae_bar: 0.328,
                      closed_loop_tp3_median_mae_bar: 0.2597,
                      closed_loop_tp3_rmse_bar: 0.39,
                      persistence_tp3_mae_bar: 1.5354,
                      persistence_tp3_median_mae_bar: 0.8427,
                      train_mean_tp3_mae_bar: 1.0014,
                      train_mean_tp3_median_mae_bar: 0.6671,
                      teacher_forced_skill_vs_persistence: 0.7834,
                      teacher_forced_skill_vs_train_mean: 0.6679,
                      closed_loop_skill_vs_persistence: 0.7864,
                      closed_loop_skill_vs_train_mean: 0.6725,
                      teacher_forced_supported: true,
                      closed_loop_supported: true,
                      duty_cycle_mae: 0.0599,
                      duty_cycle_median_mae: 0.0333,
                      persistence_duty_mae: 0.317,
                      train_mean_duty_mae: 0.2482,
                      train_mean_duty_median_mae: 0.1185,
                      duty_skill_vs_train_mean: 0.7587,
                      duty_cycle_supported: true,
                      observed_cycles_per_segment: 0.3,
                      simulated_cycles_per_segment: 0.19,
                      cycle_count_mae: 0.14,
                      closed_loop_oil_mae_c: 2.72,
                      persistence_oil_mae_c: 7.74,
                      train_mean_oil_mae_c: 7.12,
                      support_note: 'Closed-loop TP3 beats both baselines.',
                    },
                    {
                      horizon_minutes: 60,
                      segment_count: 168,
                      teacher_forced_tp3_mae_bar: 0.503,
                      teacher_forced_tp3_median_mae_bar: 0.4242,
                      teacher_forced_tp3_rmse_bar: 0.56,
                      closed_loop_tp3_mae_bar: 0.6008,
                      closed_loop_tp3_median_mae_bar: 0.6266,
                      closed_loop_tp3_rmse_bar: 0.68,
                      persistence_tp3_mae_bar: 1.8765,
                      persistence_tp3_median_mae_bar: 0.8236,
                      train_mean_tp3_mae_bar: 0.5821,
                      train_mean_tp3_median_mae_bar: 0.5146,
                      teacher_forced_skill_vs_persistence: 0.7319,
                      teacher_forced_skill_vs_train_mean: 0.1359,
                      closed_loop_skill_vs_persistence: 0.6798,
                      closed_loop_skill_vs_train_mean: -0.0321,
                      teacher_forced_supported: true,
                      closed_loop_supported: false,
                      duty_cycle_mae: 0.0496,
                      duty_cycle_median_mae: 0.0333,
                      persistence_duty_mae: 0.4732,
                      train_mean_duty_mae: 0.0568,
                      train_mean_duty_median_mae: 0.0319,
                      duty_skill_vs_train_mean: 0.1268,
                      duty_cycle_supported: false,
                      observed_cycles_per_segment: 2.95,
                      simulated_cycles_per_segment: 2.02,
                      cycle_count_mae: 1.19,
                      closed_loop_oil_mae_c: 4.75,
                      persistence_oil_mae_c: 9.89,
                      train_mean_oil_mae_c: 5.9,
                      support_note: 'UNSUPPORTED closed-loop TP3 forecast.',
                    },
                  ],
                  sensitivity_checks: [
                    {
                      perturbation: 'k_charge -15%',
                      evaluation_scope: 'pre_evaluation_only',
                      segment_count: 77,
                      baseline_tp3_mae_10min_bar: 0.3463,
                      perturbed_tp3_mae_10min_bar: 0.3443,
                      delta_tp3_mae_bar: -0.002,
                      baseline_duty_mae_10min: 0.06,
                      perturbed_duty_mae_10min: 0.0782,
                      delta_duty_mae: 0.0182,
                    },
                  ],
                  calibration_notes: ['Fitted pre-eval.'],
                  constraint_violations: 0,
                  step_halving_max_diff_bar: 0.395,
                  leak_vs_normal_residual_ratio: 2.914,
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
              horizon_supported: false,
              horizon_support_status: 'unsupported_closed_loop_horizon',
              horizon_warning: 'Exceeds validated closed-loop horizon of 30 min.',
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
    expect(await screen.findByText(/Horizon validation limit/)).toBeInTheDocument()
    expect(await screen.findByText(/reservoir_volume_m3/)).toBeInTheDocument()
    expect(screen.getByText(/k_charge_bar_per_min: 0.948 bar\/min/)).toBeInTheDocument()
    expect(screen.getByText(/Unsupported prediction horizons/)).toBeInTheDocument()
    expect(screen.getByText(/k_charge -15%/)).toBeInTheDocument()
    expect(screen.getByText(/2\.91x/)).toBeInTheDocument()
    expect(screen.getByText(/168×60m seg.*All-Horizon Clamps/i)).toBeInTheDocument()

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
