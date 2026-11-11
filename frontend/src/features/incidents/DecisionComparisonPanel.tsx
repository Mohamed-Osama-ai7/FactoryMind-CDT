import { type FormEvent, useId, useState } from 'react'
import { describeApiError } from '../../api/errors'
import type {
  GetIncidentDecisionComparisonDominantHypothesis,
  GetIncidentDecisionComparisonParams,
} from '../../api/generated/model'
import { useGetIncidentDecisionComparison } from '../../api/generated/optimization/optimization'

export interface DecisionComparisonPanelProps {
  incidentId: string
  dominantHypothesis?: GetIncidentDecisionComparisonDominantHypothesis
}

const TIER_BADGE = {
  measured: 'border-ok/60 bg-ok/10 text-ok',
  engineering_estimate: 'border-accent/60 bg-accent/15 text-ink',
  configurable_economic_assumption: 'border-warn/60 bg-warn/10 text-warn',
  simulated_projection: 'border-critical/60 bg-critical/10 text-critical',
} as const

interface AssumptionForm {
  electricityEurPerKwh: string
  plannedMaintenanceEur: string
  disruptionEurPerHr: string
  leakRateBarPerMin: string
  horizonHours: string
}

function parseFiniteBounded(
  raw: string,
  label: string,
  lo: number,
  hi: number,
): { value: number; error: null } | { value: null; error: string } {
  const trimmed = raw.trim()
  if (!trimmed) {
    return { value: null, error: `${label} is required.` }
  }
  const n = Number(trimmed)
  if (!Number.isFinite(n) || n < lo || n > hi) {
    return { value: null, error: `${label} must be between ${lo} and ${hi}.` }
  }
  return { value: n, error: null }
}

/** Stratified impact estimation & constrained repair_now vs schedule vs monitor comparison. */
export function DecisionComparisonPanel({
  incidentId,
  dominantHypothesis,
}: DecisionComparisonPanelProps) {
  const idPrefix = useId()
  const [form, setForm] = useState<AssumptionForm>({
    electricityEurPerKwh: '0.18',
    plannedMaintenanceEur: '650',
    disruptionEurPerHr: '1800',
    leakRateBarPerMin: '0.35',
    horizonHours: '8',
  })
  const [applied, setApplied] = useState<GetIncidentDecisionComparisonParams>({
    electricity_eur_per_kwh: 0.18,
    planned_depot_maintenance_eur: 650,
    unplanned_service_disruption_eur_per_hr: 1800,
    leak_rate_bar_per_min: 0.35,
    planning_horizon_hours: 8,
  })
  const [validationError, setValidationError] = useState<string | null>(null)

  const query = useGetIncidentDecisionComparison(incidentId, {
    ...applied,
    ...(dominantHypothesis ? { dominant_hypothesis: dominantHypothesis } : {}),
  })

  const onApply = (e: FormEvent) => {
    e.preventDefault()
    const tariff = parseFiniteBounded(form.electricityEurPerKwh, 'Electricity EUR/kWh', 0.01, 2.0)
    if (tariff.value === null) {
      setValidationError(tariff.error)
      return
    }
    const depot = parseFiniteBounded(
      form.plannedMaintenanceEur,
      'Planned maintenance EUR',
      50,
      50000,
    )
    if (depot.value === null) {
      setValidationError(depot.error)
      return
    }
    const disruption = parseFiniteBounded(
      form.disruptionEurPerHr,
      'Unplanned disruption EUR/hr',
      100,
      100000,
    )
    if (disruption.value === null) {
      setValidationError(disruption.error)
      return
    }
    const leak = parseFiniteBounded(form.leakRateBarPerMin, 'Leak rate bar/min', 0, 1.5)
    if (leak.value === null) {
      setValidationError(leak.error)
      return
    }
    const horizon = parseFiniteBounded(form.horizonHours, 'Horizon hours', 1, 72)
    if (horizon.value === null) {
      setValidationError(horizon.error)
      return
    }

    setValidationError(null)
    setApplied({
      electricity_eur_per_kwh: tariff.value,
      planned_depot_maintenance_eur: depot.value,
      unplanned_service_disruption_eur_per_hr: disruption.value,
      leak_rate_bar_per_min: leak.value,
      planning_horizon_hours: horizon.value,
    })
  }

  const data = query.data?.status === 200 ? query.data.data : undefined
  const errorAlertId = `${idPrefix}-validation`

  return (
    <section
      aria-labelledby={`${idPrefix}-title`}
      className="flex flex-col gap-3 rounded-lg border border-border bg-panel p-4 text-sm"
    >
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <h2 id={`${idPrefix}-title`} className="font-medium">
            Impact estimation &amp; maintenance option comparison
          </h2>
          <p className="text-xs text-muted">
            Clean separation of measured telemetry, engineering kW proxy, configurable EUR
            assumptions, and twin simulations. Advisory comparison only.
          </p>
        </div>
        {data && (
          <span className="rounded border border-accent/60 bg-accent/15 px-2 py-0.5 font-mono text-xs text-ink">
            Advisory choice: {data.recommended_option}
          </span>
        )}
      </div>

      <form
        noValidate
        onSubmit={onApply}
        aria-describedby={validationError ? errorAlertId : undefined}
        className="grid grid-cols-2 gap-2 rounded border border-border p-3 text-xs sm:grid-cols-3 lg:grid-cols-6"
      >
        <label htmlFor={`${idPrefix}-tariff`} className="flex flex-col gap-1">
          <span className="text-muted">EUR/kWh</span>
          <input
            id={`${idPrefix}-tariff`}
            type="number"
            step="any"
            value={form.electricityEurPerKwh}
            onChange={(e) => setForm((p) => ({ ...p, electricityEurPerKwh: e.target.value }))}
            className="rounded border border-border bg-surface px-2 py-1 text-ink"
          />
        </label>
        <label htmlFor={`${idPrefix}-depot`} className="flex flex-col gap-1">
          <span className="text-muted">Planned EUR</span>
          <input
            id={`${idPrefix}-depot`}
            type="number"
            step="any"
            value={form.plannedMaintenanceEur}
            onChange={(e) => setForm((p) => ({ ...p, plannedMaintenanceEur: e.target.value }))}
            className="rounded border border-border bg-surface px-2 py-1 text-ink"
          />
        </label>
        <label htmlFor={`${idPrefix}-disrupt`} className="flex flex-col gap-1">
          <span className="text-muted">Disruption EUR/h</span>
          <input
            id={`${idPrefix}-disrupt`}
            type="number"
            step="any"
            value={form.disruptionEurPerHr}
            onChange={(e) => setForm((p) => ({ ...p, disruptionEurPerHr: e.target.value }))}
            className="rounded border border-border bg-surface px-2 py-1 text-ink"
          />
        </label>
        <label htmlFor={`${idPrefix}-leak`} className="flex flex-col gap-1">
          <span className="text-muted">Leak bar/min</span>
          <input
            id={`${idPrefix}-leak`}
            type="number"
            step="any"
            value={form.leakRateBarPerMin}
            onChange={(e) => setForm((p) => ({ ...p, leakRateBarPerMin: e.target.value }))}
            className="rounded border border-border bg-surface px-2 py-1 text-ink"
          />
        </label>
        <label htmlFor={`${idPrefix}-horizon`} className="flex flex-col gap-1">
          <span className="text-muted">Horizon (hr)</span>
          <input
            id={`${idPrefix}-horizon`}
            type="number"
            step="any"
            value={form.horizonHours}
            onChange={(e) => setForm((p) => ({ ...p, horizonHours: e.target.value }))}
            className="rounded border border-border bg-surface px-2 py-1 text-ink"
          />
        </label>
        <div className="flex items-end">
          <button
            type="submit"
            disabled={query.isFetching}
            className="w-full rounded bg-accent px-2.5 py-1 font-medium text-white disabled:opacity-50"
          >
            {query.isFetching ? 'Recalculating…' : 'Update comparison'}
          </button>
        </div>
        {validationError && (
          <p id={errorAlertId} role="alert" className="col-span-full text-critical">
            {validationError}
          </p>
        )}
      </form>

      {query.isPending && <p role="status">Computing bounded maintenance trade-offs…</p>}
      {query.isError && (
        <p role="alert" className="rounded border border-critical p-3 text-xs text-critical">
          Could not evaluate maintenance options: {describeApiError(query.error)}
        </p>
      )}

      {data && (
        <div className="flex flex-col gap-3 text-xs">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-5">
            {data.impact_quantities.map((q) => (
              <div key={q.key} className="rounded border border-border p-2">
                <span
                  className={`inline-block rounded border px-1.5 py-0.5 font-mono text-[10px] ${TIER_BADGE[q.tier]}`}
                >
                  {q.tier}
                </span>
                <p className="mt-1 font-mono text-sm font-medium text-ink">
                  {q.value} {q.unit}
                </p>
                <p className="text-muted">{q.label}</p>
              </div>
            ))}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <caption className="mb-1 text-left text-muted">{data.recommendation_note}</caption>
              <thead>
                <tr className="text-left text-muted">
                  {[
                    'Option',
                    'Feasible',
                    'Energy (kWh)',
                    'LPS risk (min)',
                    'Disruption (h)',
                    'Expected cost (EUR)',
                    'Utility',
                    'Constraints / rationale',
                  ].map((h) => (
                    <th key={h} scope="col" className="pr-3 font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.options.map((opt) => (
                  <tr key={opt.option} className="border-t border-border align-top">
                    <th
                      scope="row"
                      className="py-1.5 pr-3 text-left font-mono font-medium text-ink"
                    >
                      {opt.option}
                      {opt.option === data.recommended_option ? ' *' : ''}
                    </th>
                    <td className="py-1.5 pr-3 font-mono">{opt.feasible ? 'YES' : 'BLOCKED'}</td>
                    <td className="py-1.5 pr-3 font-mono">{opt.projected_energy_kwh.toFixed(1)}</td>
                    <td className="py-1.5 pr-3 font-mono">
                      {opt.projected_lps_risk_minutes.toFixed(1)}
                    </td>
                    <td className="py-1.5 pr-3 font-mono">
                      {opt.projected_unplanned_disruption_hours.toFixed(2)}
                    </td>
                    <td className="py-1.5 pr-3 font-mono">
                      {opt.expected_total_cost_eur.toFixed(2)}
                    </td>
                    <td className="py-1.5 pr-3 font-mono">
                      {opt.bounded_utility_score.toFixed(3)}
                    </td>
                    <td className="py-1.5 text-muted">
                      {opt.constraint_violations.length > 0
                        ? opt.constraint_violations.join(' ')
                        : opt.rationale}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {data.uncertainty_warnings.length > 0 && (
            <ul className="space-y-1 text-muted">
              {data.uncertainty_warnings.map((w) => (
                <li key={w}>• {w}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  )
}
