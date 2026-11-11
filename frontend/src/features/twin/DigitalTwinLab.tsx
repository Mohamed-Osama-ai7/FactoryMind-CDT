import { type FormEvent, useEffect, useId, useState } from 'react'
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

/** Physics-Informed Digital Twin calibration, held-out validation & counterfactual simulator. */
export function DigitalTwinLab() {
  const idPrefix = useId()
  const me = useMe()
  const canRead = me.data?.data.permissions.includes('asset:read') ?? false

  const [scenario, setScenario] = useState<TwinScenarioKind>('leak')
  const [horizonMinutes, setHorizonMinutes] = useState<number>(60)
  const [leakRate, setLeakRate] = useState<number>(0.45)
  const [demandFactor, setDemandFactor] = useState<number>(1.0)
  const [repairAtMinute, setRepairAtMinute] = useState<number>(30)

  const modelQuery = useGetTwinModel(
    { asset_id: 'metropt3-apu-1' },
    { query: { enabled: canRead } },
  )
  const simMutation = useSimulateTwinScenario()

  const runSimulation = (targetScenario: TwinScenarioKind) => {
    simMutation.mutate({
      data: {
        asset_id: 'metropt3-apu-1',
        scenario: targetScenario,
        horizon_minutes: horizonMinutes,
        leak_rate_bar_per_min: leakRate,
        demand_factor: demandFactor,
        repair_at_minute: repairAtMinute,
      },
    })
  }

  // Trigger an initial baseline/leak comparison simulation cleanly once authorized
  useEffect(() => {
    if (canRead && !simMutation.data && !simMutation.isPending && !simMutation.isError) {
      simMutation.mutate({
        data: {
          asset_id: 'metropt3-apu-1',
          scenario: 'leak',
          horizon_minutes: 60,
          leak_rate_bar_per_min: 0.45,
          demand_factor: 1.0,
          repair_at_minute: 30,
        },
      })
    }
  }, [canRead, simMutation.data, simMutation.isPending, simMutation.isError, simMutation.mutate])

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
    runSimulation(scenario)
  }

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
          onSubmit={onSubmit}
          className="grid grid-cols-1 gap-3 rounded-lg border border-border bg-panel p-4 text-xs sm:grid-cols-2 lg:grid-cols-3"
        >
          <label htmlFor={`${idPrefix}-scenario`} className="flex flex-col gap-1">
            <span className="text-muted">Scenario</span>
            <select
              id={`${idPrefix}-scenario`}
              value={scenario}
              onChange={(e) => setScenario(e.target.value as TwinScenarioKind)}
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
              step={10}
              value={horizonMinutes}
              onChange={(e) => setHorizonMinutes(Number(e.target.value))}
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
              step={0.05}
              value={leakRate}
              onChange={(e) => setLeakRate(Number(e.target.value))}
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
              step={0.1}
              value={demandFactor}
              onChange={(e) => setDemandFactor(Number(e.target.value))}
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
              step={1}
              value={repairAtMinute}
              onChange={(e) => setRepairAtMinute(Number(e.target.value))}
              className="rounded border border-border bg-surface px-2 py-1.5 text-ink"
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
            <p className="text-muted">
              Model: <span className="font-mono text-ink">{modelData.model_version}</span> · Split:{' '}
              {modelData.validation.split_policy}
            </p>
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
