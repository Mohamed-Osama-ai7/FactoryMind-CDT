import { type FormEvent, useId, useState } from 'react'
import { describeApiError } from '../../api/errors'
import { useMe } from '../../api/generated/incidents/incidents'
import {
  useGetAi4iBenchmark,
  useListMlCatalog,
  usePredictAi4iSample,
} from '../../api/generated/ml/ml'
import type { Ai4iPredictRequest, Ai4iPredictRequestQualityStatus } from '../../api/generated/model'

const GOV_BADGE = {
  active: 'border-ok/60 bg-ok/10 text-ok',
  candidate: 'border-warn/60 bg-warn/10 text-warn',
  not_calibrated: 'border-critical/60 bg-critical/10 text-critical',
} as const

type PresetKey = 'nominal' | 'ood_shift' | 'degraded_quality' | 'cross_asset'

const DEFAULT_PRESET: Ai4iPredictRequest = {
  asset_id: 'ai4i-milling-bench',
  product_type: 'M',
  air_temperature_k: 300.0,
  process_temperature_k: 310.0,
  rotational_speed_rpm: 1520,
  torque_nm: 38.0,
  tool_wear_min: 60,
  quality_status: 'healthy',
}

const PRESETS: Record<PresetKey, Ai4iPredictRequest> = {
  nominal: DEFAULT_PRESET,
  ood_shift: {
    ...DEFAULT_PRESET,
    air_temperature_k: 340.0,
    process_temperature_k: 280.0,
    rotational_speed_rpm: 1550,
    torque_nm: 42.0,
    tool_wear_min: 90,
  },
  degraded_quality: {
    ...DEFAULT_PRESET,
    quality_status: 'degraded',
  },
  cross_asset: {
    ...DEFAULT_PRESET,
    asset_id: 'metropt3-apu-1',
  },
}

/** Multi-model governance catalog, AI4I 2020 calibration benchmark & OOD abstention console. */
export function ModelsWorkspace() {
  const idPrefix = useId()
  const me = useMe()
  const canRead = me.data?.data.permissions.includes('anomaly:read') ?? false

  const catalogQuery = useListMlCatalog({ query: { enabled: canRead } })
  const benchmarkQuery = useGetAi4iBenchmark({ query: { enabled: canRead, retry: false } })
  const predict = usePredictAi4iSample()

  const [form, setForm] = useState<Ai4iPredictRequest>(DEFAULT_PRESET)

  if (me.isPending) return <p role="status">Checking model access permissions…</p>
  if (me.isError) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Multi-Model ML &amp; Uncertainty Engine
        </h1>
        <p role="alert" className="mt-2 rounded border border-critical p-3 text-sm text-critical">
          Could not verify model permissions: {describeApiError(me.error)}
        </p>
      </section>
    )
  }
  if (!canRead) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Multi-Model ML &amp; Uncertainty Engine
        </h1>
        <p
          role="status"
          className="mt-2 rounded border border-border bg-panel p-4 text-sm text-muted"
        >
          You do not hold the <span className="font-mono">anomaly:read</span> permission required to
          inspect model calibration benchmarks or run uncertainty diagnostics.
        </p>
      </section>
    )
  }

  const catalog = catalogQuery.data?.status === 200 ? catalogQuery.data.data : undefined
  const benchmark = benchmarkQuery.data?.status === 200 ? benchmarkQuery.data.data : undefined
  const prediction = predict.data?.status === 200 ? predict.data.data : undefined

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    predict.mutate({ data: form })
  }

  return (
    <section aria-labelledby={`${idPrefix}-title`} className="flex max-w-5xl flex-col gap-6">
      <header>
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Multi-Model ML, Calibration &amp; OOD Abstention
        </h1>
        <p className="mt-1 text-sm text-muted">
          Specialized model governance separated across real operational compressor telemetry and
          the controlled UCI AI4I 2020 milling benchmark. Outputs are advisory predictions only.
        </p>
      </header>

      {catalog && (
        <div className="rounded-lg border border-warn/60 bg-panel p-3 text-xs text-ink">
          <strong>Domain separation guarantee:</strong> {catalog.separation_disclaimer}
        </div>
      )}

      <section aria-labelledby={`${idPrefix}-catalog`} className="flex flex-col gap-3">
        <h2 id={`${idPrefix}-catalog`} className="text-lg font-medium">
          Specialized model registry
        </h2>
        {catalogQuery.isPending && <p role="status">Loading specialist catalog…</p>}
        {catalogQuery.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-sm text-critical">
            Could not load model catalog: {describeApiError(catalogQuery.error)}
          </p>
        )}
        {catalog && (
          <div className="overflow-x-auto rounded-lg border border-border bg-panel p-3">
            <table className="w-full text-xs">
              <caption className="sr-only">
                Registered specialized models by asset and domain
              </caption>
              <thead>
                <tr className="text-left text-muted">
                  {[
                    'Model ID',
                    'Asset & Dataset',
                    'Domain Track',
                    'Status',
                    'Uncertainty & Scope',
                  ].map((h) => (
                    <th key={h} scope="col" className="pr-3 font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {catalog.specialists.map((s) => (
                  <tr key={s.model_id} className="border-t border-border align-top">
                    <th
                      scope="row"
                      className="py-1.5 pr-3 text-left font-mono font-medium text-ink"
                    >
                      {s.model_id}
                      <span className="block font-sans text-[11px] font-normal text-muted">
                        {s.model_family}
                      </span>
                    </th>
                    <td className="py-1.5 pr-3 font-mono">
                      {s.asset_id}
                      <span className="block text-muted">{s.dataset_id}</span>
                    </td>
                    <td className="py-1.5 pr-3 font-mono text-muted">{s.domain_kind}</td>
                    <td className="py-1.5 pr-3">
                      <span
                        className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${GOV_BADGE[s.governance_status]}`}
                      >
                        {s.governance_status}
                      </span>
                    </td>
                    <td className="py-1.5">
                      <span className="text-ink">{s.scope_note}</span>
                      <span className="block text-muted">{s.uncertainty_mechanism}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section aria-labelledby={`${idPrefix}-bench`} className="flex flex-col gap-3">
        <h2 id={`${idPrefix}-bench`} className="text-lg font-medium">
          AI4I 2020 Calibration &amp; Promotion Benchmark
        </h2>
        {benchmarkQuery.isPending && <p role="status">Loading benchmark metrics…</p>}
        {benchmarkQuery.isError && (
          <p role="status" className="rounded border border-border bg-panel p-3 text-xs text-muted">
            Benchmark unavailable: {describeApiError(benchmarkQuery.error)}
          </p>
        )}
        {benchmark && (
          <div className="flex flex-col gap-4 rounded-lg border border-border bg-panel p-4 text-xs">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-muted">{benchmark.dataset_scope_note}</p>
              <span
                className={`rounded border px-2 py-0.5 font-mono font-medium ${
                  benchmark.promotion_contract.promoted
                    ? 'border-ok/60 bg-ok/10 text-ok'
                    : 'border-critical/60 text-critical'
                }`}
              >
                {benchmark.promotion_contract.promoted ? 'Promoted' : 'Blocked'} ·{' '}
                {benchmark.selected_model_id}
              </span>
            </div>

            <div className="grid grid-cols-1 gap-2 rounded border border-border p-2.5 sm:grid-cols-2">
              <div>
                <p className="text-muted">
                  Leakage-free features ({benchmark.rows_total.toLocaleString()} rows)
                </p>
                <p className="font-mono text-ink">{benchmark.leakage_free_features.join(', ')}</p>
              </div>
              <div>
                <p className="text-muted">Excluded targets &amp; identifiers</p>
                <p className="font-mono text-ink">
                  {benchmark.excluded_target_and_id_columns.join(', ')}
                </p>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <caption className="mb-1 text-left text-muted">
                  Held-out test split performance (train {benchmark.split_counts.train} / cal{' '}
                  {benchmark.split_counts.cal} / test {benchmark.split_counts.test})
                </caption>
                <thead>
                  <tr className="text-left text-muted">
                    {[
                      'Model',
                      'ROC-AUC',
                      'PR-AUC',
                      'Brier',
                      'ECE',
                      'F1',
                      'Selective Acc',
                      'OOD Thr',
                    ].map((h) => (
                      <th key={h} scope="col" className="pr-3 font-normal">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {benchmark.models.map((row) => (
                    <tr key={row.model_id} className="border-t border-border">
                      <th
                        scope="row"
                        className="py-1.5 pr-3 text-left font-mono font-medium text-ink"
                      >
                        {row.model_id}
                      </th>
                      <td className="py-1.5 pr-3 font-mono">
                        {row.test_metrics.roc_auc.toFixed(4)}
                      </td>
                      <td className="py-1.5 pr-3 font-mono">
                        {row.test_metrics.pr_auc.toFixed(4)}
                      </td>
                      <td className="py-1.5 pr-3 font-mono">
                        {row.test_metrics.brier_score.toFixed(4)}
                      </td>
                      <td className="py-1.5 pr-3 font-mono">
                        {row.test_metrics.ece_10_bins.toFixed(4)}
                      </td>
                      <td className="py-1.5 pr-3 font-mono">
                        {row.test_metrics.f1_at_threshold.toFixed(4)}
                      </td>
                      <td className="py-1.5 pr-3 font-mono">
                        {row.test_metrics.selective_accuracy_at_margin?.toFixed(4) ?? 'n/a'}
                      </td>
                      <td className="py-1.5 font-mono">{row.ood_threshold.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {benchmark.promotion_contract.gates.map((gate) => (
                <li key={gate.gate_id} className="rounded border border-border p-2">
                  <span
                    className={`mr-2 font-mono font-medium ${
                      gate.passed ? 'text-ok' : 'text-critical'
                    }`}
                  >
                    [{gate.passed ? 'PASS' : 'FAIL'}]
                  </span>
                  <span className="font-mono text-ink">{gate.gate_id}</span>
                  <span className="mt-0.5 block text-muted">{gate.detail}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      <section aria-labelledby={`${idPrefix}-probe`} className="flex flex-col gap-3">
        <h2 id={`${idPrefix}-probe`} className="text-lg font-medium">
          OOD, Quality &amp; Uncertainty Abstention Sandbox
        </h2>
        <div className="rounded-lg border border-border bg-panel p-4 text-xs">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <span className="text-muted">Presets:</span>
            {(['nominal', 'ood_shift', 'degraded_quality', 'cross_asset'] as const).map((key) => (
              <button
                key={key}
                type="button"
                onClick={() => {
                  const preset = PRESETS[key]
                  if (preset) setForm(preset)
                  predict.reset()
                }}
                className="rounded border border-border px-2 py-1 hover:bg-border"
              >
                {key}
              </button>
            ))}
          </div>

          <form onSubmit={onSubmit} className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <label htmlFor={`${idPrefix}-asset`} className="flex flex-col gap-1">
              <span className="text-muted">Target Asset ID</span>
              <input
                id={`${idPrefix}-asset`}
                type="text"
                value={form.asset_id ?? 'ai4i-milling-bench'}
                onChange={(e) => setForm({ ...form, asset_id: e.target.value })}
                className="rounded border border-border bg-surface px-2 py-1 font-mono text-ink"
              />
            </label>
            <label htmlFor={`${idPrefix}-type`} className="flex flex-col gap-1">
              <span className="text-muted">Product Type</span>
              <select
                id={`${idPrefix}-type`}
                value={form.product_type}
                onChange={(e) => setForm({ ...form, product_type: e.target.value })}
                className="rounded border border-border bg-surface px-2 py-1 text-ink"
              >
                {['L', 'M', 'H'].map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <label htmlFor={`${idPrefix}-air`} className="flex flex-col gap-1">
              <span className="text-muted">Air Temp [K]</span>
              <input
                id={`${idPrefix}-air`}
                type="number"
                step="0.1"
                value={form.air_temperature_k}
                onChange={(e) => setForm({ ...form, air_temperature_k: Number(e.target.value) })}
                className="rounded border border-border bg-surface px-2 py-1 font-mono text-ink"
              />
            </label>
            <label htmlFor={`${idPrefix}-proc`} className="flex flex-col gap-1">
              <span className="text-muted">Process Temp [K]</span>
              <input
                id={`${idPrefix}-proc`}
                type="number"
                step="0.1"
                value={form.process_temperature_k}
                onChange={(e) =>
                  setForm({ ...form, process_temperature_k: Number(e.target.value) })
                }
                className="rounded border border-border bg-surface px-2 py-1 font-mono text-ink"
              />
            </label>
            <label htmlFor={`${idPrefix}-rpm`} className="flex flex-col gap-1">
              <span className="text-muted">Speed [rpm]</span>
              <input
                id={`${idPrefix}-rpm`}
                type="number"
                step="1"
                value={form.rotational_speed_rpm}
                onChange={(e) => setForm({ ...form, rotational_speed_rpm: Number(e.target.value) })}
                className="rounded border border-border bg-surface px-2 py-1 font-mono text-ink"
              />
            </label>
            <label htmlFor={`${idPrefix}-torque`} className="flex flex-col gap-1">
              <span className="text-muted">Torque [Nm]</span>
              <input
                id={`${idPrefix}-torque`}
                type="number"
                step="0.1"
                value={form.torque_nm}
                onChange={(e) => setForm({ ...form, torque_nm: Number(e.target.value) })}
                className="rounded border border-border bg-surface px-2 py-1 font-mono text-ink"
              />
            </label>
            <label htmlFor={`${idPrefix}-wear`} className="flex flex-col gap-1">
              <span className="text-muted">Tool wear [min]</span>
              <input
                id={`${idPrefix}-wear`}
                type="number"
                step="1"
                value={form.tool_wear_min}
                onChange={(e) => setForm({ ...form, tool_wear_min: Number(e.target.value) })}
                className="rounded border border-border bg-surface px-2 py-1 font-mono text-ink"
              />
            </label>
            <label htmlFor={`${idPrefix}-qual`} className="flex flex-col gap-1">
              <span className="text-muted">Quality Gate Input</span>
              <select
                id={`${idPrefix}-qual`}
                value={form.quality_status ?? 'healthy'}
                onChange={(e) =>
                  setForm({
                    ...form,
                    quality_status: e.target.value as Ai4iPredictRequestQualityStatus,
                  })
                }
                className="rounded border border-border bg-surface px-2 py-1 text-ink"
              >
                <option value="healthy">healthy</option>
                <option value="degraded">degraded</option>
                <option value="insufficient">insufficient</option>
              </select>
            </label>
            <div className="col-span-2 flex items-center gap-3 sm:col-span-4">
              <button
                type="submit"
                disabled={predict.isPending}
                className="rounded bg-accent px-3 py-1.5 font-medium text-white disabled:opacity-50"
              >
                {predict.isPending ? 'Evaluating…' : 'Evaluate routed sample'}
              </button>
              {predict.isError && (
                <span role="alert" className="text-critical">
                  {describeApiError(predict.error)}
                </span>
              )}
            </div>
          </form>

          {prediction && (
            <div
              role="status"
              className={`mt-4 rounded border p-3 ${
                prediction.abstained
                  ? 'border-warn/60 bg-warn/10 text-ink'
                  : 'border-ok/60 bg-ok/10 text-ink'
              }`}
            >
              <p className="font-medium">
                {prediction.abstained
                  ? prediction.abstention_message
                  : `Prediction: ${prediction.prediction_label} (calibrated failure p=${prediction.calibrated_failure_probability?.toFixed(4) ?? 'n/a'})`}
              </p>
              <p className="mt-1 font-mono text-[11px] text-muted">
                Model={prediction.selected_model_id} · Uncertainty=
                {prediction.uncertainty.uncertainty_band} · Mahalanobis=
                {prediction.uncertainty.mahalanobis_distance?.toFixed(2) ?? 'n/a'} (thr{' '}
                {prediction.uncertainty.ood_threshold.toFixed(2)}) · Origin=
                {prediction.provenance_origin}
                {prediction.abstention_reasons?.length
                  ? ` · Reasons=${prediction.abstention_reasons.join(', ')}`
                  : ''}
              </p>
            </div>
          )}
        </div>
      </section>
    </section>
  )
}
