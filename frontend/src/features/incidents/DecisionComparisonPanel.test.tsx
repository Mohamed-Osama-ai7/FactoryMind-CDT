import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { DecisionComparisonPanel } from './DecisionComparisonPanel'

describe('DecisionComparisonPanel', () => {
  it('displays stratified impact tiers, Pareto badges, and updates economic assumptions', async () => {
    const seenUrls: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        const s = String(url)
        seenUrls.push(s)
        return Promise.resolve(
          new Response(
            JSON.stringify({
              asset_id: 'metropt3-apu-1',
              incident_id: 'a'.repeat(32),
              provenance_origin: 'advisory_decision_comparison',
              recommended_option: 'repair_now',
              recommendation_note: 'Advisory ranking favors repair_now.',
              uncertainty_warnings: ['Power kW is an engineering estimate.'],
              impact_quantities: [
                {
                  key: 'measured_load_duty',
                  label: 'Measured Loaded Duty Fraction',
                  value: 0.85,
                  unit: 'fraction',
                  tier: 'measured',
                  basis_note: 'Observed sensor fraction.',
                },
                {
                  key: 'estimated_power_kw',
                  label: 'Estimated Mean APU Power Draw',
                  value: 4.1,
                  unit: 'kW',
                  tier: 'engineering_estimate',
                  basis_note: 'Rated kW proxy.',
                },
                {
                  key: 'assumed_unplanned_rate',
                  label: 'Configured Unplanned Disruption Rate',
                  value: 1800,
                  unit: 'EUR/hr',
                  tier: 'configurable_economic_assumption',
                  basis_note: 'Assumed rate.',
                },
                {
                  key: 'projected_monitor_lps_min',
                  label: 'Simulated Monitor LPS Excursion',
                  value: 45,
                  unit: 'min',
                  tier: 'simulated_projection',
                  basis_note: 'Twin rollout.',
                },
              ],
              options: [
                {
                  option: 'repair_now',
                  title: 'Immediate Depot Hold & Repair',
                  feasible: true,
                  constraint_violations: [],
                  pareto_dominated: false,
                  projected_energy_kwh: 18.4,
                  projected_excess_energy_kwh: 1.2,
                  projected_lps_risk_minutes: 0,
                  projected_unplanned_disruption_hours: 0.25,
                  expected_total_cost_eur: 928.31,
                  bounded_utility_score: 0.62,
                  rationale: 'Intervenes immediately.',
                },
                {
                  option: 'monitor',
                  title: 'Continue Service',
                  feasible: false,
                  constraint_violations: ['LPS excursion projected (45.0 min).'],
                  pareto_dominated: true,
                  projected_energy_kwh: 34.2,
                  projected_excess_energy_kwh: 17.0,
                  projected_lps_risk_minutes: 45,
                  projected_unplanned_disruption_hours: 0.9,
                  expected_total_cost_eur: 1746.16,
                  bounded_utility_score: 0.01,
                  rationale: 'Unplanned risk.',
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
        <DecisionComparisonPanel incidentId={'a'.repeat(32)} dominantHypothesis="air_leakage" />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/Advisory choice: repair_now/)).toBeInTheDocument()
    expect(screen.getByText('measured')).toBeInTheDocument()
    expect(screen.getByText('Observed sensor fraction.')).toBeInTheDocument()
    expect(screen.getByText('Pareto optimal')).toBeInTheDocument()
    expect(screen.getByText('Dominated')).toBeInTheDocument()
    expect(screen.getByText('(Recommended)')).toBeInTheDocument()
    expect(seenUrls.some((u) => u.includes('dominant_hypothesis=air_leakage'))).toBe(true)

    // Out-of-range and empty validation cleared smoothly on edit
    const tariffInput = screen.getByLabelText('EUR/kWh')
    await userEvent.clear(tariffInput)
    await userEvent.click(screen.getByRole('button', { name: 'Update comparison' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/Electricity EUR\/kWh is required/)

    await userEvent.type(tariffInput, '9.9')
    expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Update comparison' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/between 0.01 and 2/)

    await userEvent.clear(tariffInput)
    await userEvent.type(tariffInput, '0.42')
    await userEvent.click(screen.getByRole('button', { name: 'Update comparison' }))
    await waitFor(() =>
      expect(seenUrls.some((u) => u.includes('electricity_eur_per_kwh=0.42'))).toBe(true),
    )
  })

  it('defers query while rcaPending and displays API errors cleanly', async () => {
    const fetchSpy = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: 'Comparison service unavailable' }), {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    vi.stubGlobal('fetch', fetchSpy)

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const { rerender } = render(
      <QueryClientProvider client={client}>
        <DecisionComparisonPanel incidentId={'b'.repeat(32)} rcaPending={true} />
      </QueryClientProvider>,
    )

    expect(screen.getByRole('status')).toHaveTextContent(/Computing bounded maintenance/)
    expect(fetchSpy).not.toHaveBeenCalled()

    rerender(
      <QueryClientProvider client={client}>
        <DecisionComparisonPanel
          incidentId={'b'.repeat(32)}
          rcaPending={false}
          rcaUnavailable={true}
        />
      </QueryClientProvider>,
    )

    expect(await screen.findByRole('alert')).toHaveTextContent(
      /Could not evaluate maintenance options/,
    )
    expect(screen.getByText(/RCA hypothesis context unavailable/)).toBeInTheDocument()
  })
})
