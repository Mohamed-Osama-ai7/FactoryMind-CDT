import { type FormEvent, useEffect, useId, useRef, useState } from 'react'
import { describeApiError } from '../../api/errors'
import {
  useGetTwinModel,
  useSimulateTwinScenario,
} from '../../api/generated/digital-twin/digital-twin'
import { useMe } from '../../api/generated/incidents/incidents'
import type { TwinSimulationRequestScenario } from '../../api/generated/model'

type TwinScenarioKind = TwinSimulationRequestScenario

const SCENARIO_LABELS: Record<TwinScenarioKind, string> = {
  baseline: 'Baseline nominal operation',
  leak: 'Injected pneumatic air leak',
  continued_operation: 'Continued operation (unrepaired progressive leak)',
  maintenance_repair: 'Counterfactual maintenance repair intervention',
}

interface TwinFormState {
  horizonMinutes: string
  leakRate: string
  demandFactor: string
  repairAtMinute: string
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
    return { value: null, error: `${label} must be a finite number between ${lo} and ${hi}.` }
  }
  return { value: n, error: null }
}

/** Physics-Informed Digital Twin calibration, held-out validation & counterfactual simulator. */
export function DigitalTwinLab() {
  const idPrefix = useId()
  const me = useMe()
  const canRead = me.data?.data.permissions.includes('asset:read') ?? false

  const [scenario, setScenario] = useState<TwinScenarioKind>('leak')
  const [form, setForm] = useState<TwinFormState>({
    horizonMinutes: '60',
    leakRate: '0.45',
    demandFactor: '1.0',
    repairAtMinute: '30',
  })
  const [validationError, setValidationError] = useState<string | null>(null)
  const didAutoRunRef = useRef(false)

  const modelQuery = useGetTwinModel(
    { asset_id: 'metropt3-apu-1' },
    { query: { enabled: canRead } },
  )
  const simMutation = useSimulateTwinScenario()

  // Trigger an initial default 'leak' rollout once authorized (StrictMode-safe via ref guard)
  useEffect(() => {
    if (canRead && !didAutoRunRef.current) {
      didAutoRunRef.current = true
      simMutation.mutate({
        data: {
          asset_id: 'metropt3-apu-1',
          scenario: 'leak',
          horizon_minutes: 60,
          leak_rate_bar_per_min: 0.45,
          demand_factor: 1.0,
        },
      })
    }
  }, [canRead, simMutation.mutate])

  if (me.isPending) return <p role="status">Checking digital twin permissions…</p>
  if (me.isError) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Physics-Informed Digital Twin &amp; Counterfactual Simulator
        </h1>
        <p role="alert" className="mt-2 rounded border border-critical p-3 text-sm text-critical">
          Could not verify digital twin permissions: {describeApiError(me.error)}
        </p>
      </section>
    )
  }
  if (!canRead) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Physics-Informed Digital Twin &amp; Counterfactual Simulator
        </h1>
        <p
          role="status"
          className="mt-2 rounded border border-border bg-panel p-4 text-sm text-muted"
        >
          You do not hold the <span className="font-mono">asset:read</span> permission required to
          inspect calibrated digital twin models or run advisory simulations.
        </p>
      </section>
    )
  }

  const modelData = modelQuery.data?.status === 200 ? modelQuery.data.data : undefined
  const simData = simMutation.data?.status === 200 ? simMutation.data.data : undefined

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    const horizon = parseFiniteBounded(form.horizonMinutes, 'Horizon (minutes)', 10, 240)
    if (horizon.value === null) {
      setValidationError(horizon.error)
      return
    }
    const leak = parseFiniteBounded(form.leakRate, 'Leak loss rate (bar/min)', 0, 1.5)
    if (leak.value === null) {
      setValidationError(leak.error)
      return
    }
    const demand = parseFiniteBounded(form.demandFactor, 'Downstream demand multiplier', 0.5, 3.0)
    if (demand.value === null) {
      setValidationError(demand.error)
      return
    }

    let repairMinute: number | undefined
    if (scenario === 'maintenance_repair') {
      const repair = parseFiniteBounded(form.repairAtMinute, 'Repair intervention minute', 1, 239)
      if (repair.value === null) {
        setValidationError(repair.error)
        return
      }
      if (repair.value >= horizon.value) {
        setValidationError(
          `Repair intervention minute (${repair.value}) must be strictly less than horizon (${horizon.value} min).`,
        )
        return
      }
      repairMinute = repair.value
    }

    setValidationError(null)
    simMutation.mutate({
      data: {
        asset_id: 'metropt3-apu-1',
        scenario,
        horizon_minutes: horizon.value,
        leak_rate_bar_per_min: leak.value,
        demand_factor: demand.value,
        ...(repairMinute !== undefined ? { repair_at_minute: repairMinute } : {}),
      },
    })
  }

  const validationAlertId = `${idPrefix}-form-error`

  return (
    <section aria-labelledby={`${idPrefix}-title`} className="flex max-w-5xl flex-col gap-6">
      <header>
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Physics-Informed Digital Twin &amp; Counterfactual Simulator
        </h1>
        <p className="mt-1 text-sm text-muted">
          Normalized APU compressor mass-balance and governor dynamics fitted strictly on
          pre-evaluation telemetry. All rollouts carry explicit simulated-projection provenance and
          never actuate physical equipment.
        </p>
      </header>

      <section aria-labelledby={`${idPrefix}-sim`} className="flex flex-col gap-3">
        <h2 id={`${idPrefix}-sim`} className="text-lg font-medium">
          Controlled scenario simulator
        </h2>
        <form
          noValidate
          onSubmit={onSubmit}
          aria-describedby={validationError ? validationAlertId : undefined}
          className="grid grid-cols-1 gap-3 rounded-lg border border-border bg-panel p-4 text-xs sm:grid-cols-2 lg:grid-cols-3"
        >
          <label htmlFor={`${idPrefix}-scenario`} className="flex flex-col gap-1">
            <span className="text-muted">Scenario</span>
            <select
              id={`${idPrefix}-scenario`}
              value={scenario}
              onChange={(e) => {
                setScenario(e.target.value as TwinScenarioKind)
                if (validationError) setValidationError(null)
              }}
              className="rounded border border-border bg-surface px-2 py-1.5 text-ink"
            >
              {(Object.keys(SCENARIO_LABELS) as TwinScenarioKind[]).map((key) => (
                <option key={key} value={key}>
                  {SCENARIO_LABELS[key]}
                </option>
              ))}
            </select>
          </label>
          <label htmlFor={`${idPrefix}-horizon`} className="flex flex-col gap-1">
            <span className="text-muted">Horizon (minutes, 10–240)</span>
            <input
              id={`${idPrefix}-horizon`}
              type="number"
              min={10}
              max={240}
              step="any"
              value={form.horizonMinutes}
              onChange={(e) => {
                setForm((prev) => ({ ...prev, horizonMinutes: e.target.value }))
                if (validationError) setValidationError(null)
              }}
              className="rounded border border-border bg-surface px-2 py-1.5 text-ink"
            />
          </label>
          <label htmlFor={`${idPrefix}-leak`} className="flex flex-col gap-1">
            <span className="text-muted">Leak loss rate (bar/min)</span>
            <input
              id={`${idPrefix}-leak`}
              type="number"
              min={0}
              max={1.5}
              step="any"
              value={form.leakRate}
              onChange={(e) => {
                setForm((prev) => ({ ...prev, leakRate: e.target.value }))
                if (validationError) setValidationError(null)
              }}
              className="rounded border border-border bg-surface px-2 py-1.5 text-ink"
            />
          </label>
          <label htmlFor={`${idPrefix}-demand`} className="flex flex-col gap-1">
            <span className="text-muted">Downstream demand multiplier</span>
            <input
              id={`${idPrefix}-demand`}
              type="number"
              min={0.5}
              max={3.0}
              step="any"
              value={form.demandFactor}
              onChange={(e) => {
                setForm((prev) => ({ ...prev, demandFactor: e.target.value }))
                if (validationError) setValidationError(null)
              }}
              className="rounded border border-border bg-surface px-2 py-1.5 text-ink"
            />
          </label>
          <label htmlFor={`${idPrefix}-repair`} className="flex flex-col gap-1">
            <span className="text-muted">Repair intervention minute</span>
            <input
              id={`${idPrefix}-repair`}
              type="number"
              min={1}
              max={239}
              step="any"
              value={form.repairAtMinute}
              disabled={scenario !== 'maintenance_repair'}
              onChange={(e) => {
                setForm((prev) => ({ ...prev, repairAtMinute: e.target.value }))
                if (validationError) setValidationError(null)
              }}
              className="rounded border border-border bg-surface px-2 py-1.5 text-ink disabled:opacity-50"
            />
          </label>
          <div className="flex items-end">
            <button
              type="submit"
              disabled={simMutation.isPending}
              className="w-full rounded bg-accent px-3 py-1.5 font-medium text-white disabled:opacity-50"
            >
              {simMutation.isPending ? 'Simulating…' : 'Run scenario simulation'}
            </button>
          </div>
          {validationError && (
            <p id={validationAlertId} role="alert" className="col-span-full text-critical">
              {validationError}
            </p>
          )}
        </form>

        {simMutation.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-xs text-critical">
            Simulation failed: {describeApiError(simMutation.error)}
          </p>
        )}

        {simData && (
          <div className="flex flex-col gap-3 rounded-lg border border-border bg-panel p-4 text-xs">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="rounded border border-warn/60 bg-warn/10 px-2 py-0.5 font-mono text-warn">
                Provenance: {simData.provenance_origin} (Counterfactual Advisory)
              </span>
              <span className="text-muted">{simData.summary.advisory_interpretation}</span>
            </div>

            <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <div className="rounded border border-border p-2">
                <dt className="text-muted">Mean / Peak Duty</dt>
                <dd className="font-mono text-sm text-ink">
                  {simData.summary.mean_load_duty.toFixed(2)} /{' '}
                  {simData.summary.peak_load_duty.toFixed(2)}
                </dd>
              </div>
              <div className="rounded border border-border p-2">
                <dt className="text-muted">TP3 Min / Max (bar)</dt>
                <dd className="font-mono text-sm text-ink">
                  {simData.summary.min_tp3_bar.toFixed(2)} /{' '}
                  {simData.summary.max_tp3_bar.toFixed(2)}
                </dd>
              </div>
              <div className="rounded border border-border p-2">
                <dt className="text-muted">LPS (&lt;7 bar) / Duty Breach</dt>
                <dd className="font-mono text-sm text-ink">
                  {simData.summary.lps_excursion_minutes.toFixed(1)}m /{' '}
                  {simData.summary.time_above_duty_threshold_minutes.toFixed(1)}m
                </dd>
              </div>
              <div className="rounded border border-border p-2">
                <dt className="text-muted">Final Oil / Violations</dt>
                <dd className="font-mono text-sm text-ink">
                  {simData.summary.final_oil_temp_c.toFixed(1)} °C /{' '}
                  {simData.summary.constraint_violations}
                </dd>
              </div>
            </dl>

            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <caption className="mb-1 text-left text-muted">
                  Sampled trajectory points ({simData.trajectory.length} points, advisory
                  counterfactual)
                </caption>
                <thead>
                  <tr className="text-left text-muted">
                    {[
                      'Minute',
                      'TP3 (bar)',
                      'Reservoirs (bar)',
                      'Rolling duty',
                      'Regime',
                      'Oil (°C)',
                      'LPS active',
                    ].map((h) => (
                      <th key={h} scope="col" className="pr-3 font-normal">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {simData.trajectory
                    .filter((_, idx) => idx % 5 === 0 || idx === simData.trajectory.length - 1)
                    .map((pt) => (
                      <tr key={pt.minute} className="border-t border-border">
                        <th scope="row" className="py-1 pr-3 text-left font-mono font-normal">
                          {pt.minute.toFixed(0)}
                        </th>
                        <td className="py-1 pr-3 font-mono">{pt.tp3_bar.toFixed(2)}</td>
                        <td className="py-1 pr-3 font-mono">{pt.reservoirs_bar.toFixed(2)}</td>
                        <td className="py-1 pr-3 font-mono">{pt.rolling_duty.toFixed(2)}</td>
                        <td className="py-1 pr-3 font-mono">{pt.regime}</td>
                        <td className="py-1 pr-3 font-mono">{pt.oil_temp_c.toFixed(1)}</td>
                        <td className="py-1 font-mono">{pt.lps_active ? 'YES' : 'no'}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      <section aria-labelledby={`${idPrefix}-cal`} className="flex flex-col gap-3">
        <h2 id={`${idPrefix}-cal`} className="text-lg font-medium">
          Held-out validation &amp; unidentifiable bounds
        </h2>
        {modelQuery.isPending && (
          <p role="status">Loading calibrated digital twin specification…</p>
        )}
        {modelQuery.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-xs text-critical">
            Could not load digital twin specification: {describeApiError(modelQuery.error)}
          </p>
        )}
        {modelData && (
          <div className="flex flex-col gap-3 rounded-lg border border-border bg-panel p-4 text-xs">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-muted">
                Model: <span className="font-mono text-ink">{modelData.model_version}</span> ·
                Provenance:{' '}
                <span className="font-mono text-ink">{modelData.provenance_origin}</span> · Split:{' '}
                {modelData.validation.split_policy}
              </p>
              <span className="text-muted">{modelData.advisory_note}</span>
            </div>

            <dl className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <div className="rounded border border-border p-2">
                <dt className="text-muted">Held-out Segments</dt>
                <dd className="font-mono text-sm text-ink">
                  {modelData.validation.heldout_normal_segments} (1h rollouts)
                </dd>
              </div>
              <div className="rounded border border-border p-2">
                <dt className="text-muted">10-min / 60-min MAE</dt>
                <dd className="font-mono text-sm text-ink">
                  {modelData.validation.mae_10min_bar.toFixed(4)} /{' '}
                  {modelData.validation.mae_60min_bar.toFixed(4)} bar
                </dd>
              </div>
              <div className="rounded border border-border p-2">
                <dt className="text-muted">60-min RMSE / Step Diff</dt>
                <dd className="font-mono text-sm text-ink">
                  {modelData.validation.rmse_60min_bar.toFixed(4)} /{' '}
                  {modelData.validation.step_halving_max_diff_bar.toFixed(3)} bar
                </dd>
              </div>
              <div className="rounded border border-border p-2">
                <dt className="text-muted">Leak Residual Magnification</dt>
                <dd className="font-mono text-sm text-ink">
                  {modelData.validation.leak_vs_normal_residual_ratio.toFixed(2)}x (
                  {modelData.validation.constraint_violations} viol)
                </dd>
              </div>
            </dl>

            {modelData.governing_equations.length > 0 && (
              <div>
                <h3 className="mb-1.5 font-medium text-muted">Governing equations</h3>
                <ul className="flex flex-col gap-1 font-mono text-[11px] text-ink">
                  {modelData.governing_equations.map((eq) => (
                    <li key={eq} className="rounded border border-border px-2 py-1">
                      {eq}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {modelData.parameters.length > 0 && (
              <div>
                <h3 className="mb-1.5 font-medium text-muted">
                  Calibrated physical parameters ({modelData.parameters.length})
                </h3>
                <div className="grid grid-cols-1 gap-1.5 sm:grid-cols-2">
                  {modelData.parameters.map((param) => (
                    <div key={param.name} className="rounded border border-border p-2">
                      <span className="font-mono font-medium text-ink">
                        {param.name}: {param.value} {param.unit}
                      </span>
                      <p className="mt-0.5 text-muted">{param.uncertainty_note}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div>
              <h3 className="mb-1.5 font-medium text-muted">
                Explicitly unidentifiable physical parameters
              </h3>
              <ul className="flex flex-col gap-1.5">
                {modelData.unidentifiable_parameters.map((u) => (
                  <li key={u.name} className="rounded border border-border p-2">
                    <span className="font-mono font-medium text-ink">{u.name}</span> —{' '}
                    <span className="text-muted">{u.reason}</span>{' '}
                    <span className="text-ink">Proxy: {u.assumption_or_proxy}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </section>
    </section>
  )
}
