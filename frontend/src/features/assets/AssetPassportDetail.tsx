import { type ReactNode, useId, useState } from 'react'
import { describeApiError } from '../../api/errors'
import { useGetAsset } from '../../api/generated/assets/assets'
import { useMe } from '../../api/generated/incidents/incidents'
import { SourceDatasetCard } from './SourceDatasetCard'

export interface AssetPassportDetailProps {
  assetId: string
  backLink?: ReactNode
}

const MODEL_BADGE = {
  active: 'border-ok/60 bg-ok/10 text-ok',
  experimental: 'border-warn/60 bg-warn/10 text-warn',
  not_calibrated: 'border-critical/60 bg-critical/10 text-critical',
} as const

/** Full Industrial Asset Digital Passport inspector for one scoped asset. */
export function AssetPassportDetail({ assetId, backLink }: AssetPassportDetailProps) {
  const idPrefix = useId()
  const me = useMe()
  const canRead = me.data?.data.permissions.includes('asset:read') ?? false
  const assetQuery = useGetAsset(assetId, { query: { enabled: canRead } })
  const [sensorFilter, setSensorFilter] = useState<'all' | 'analog' | 'digital'>('all')

  if (me.isPending) return <p role="status">Checking asset permissions…</p>
  if (me.isError) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Asset passport
        </h1>
        <p role="alert" className="mt-2 rounded border border-critical p-3 text-sm text-critical">
          Could not verify asset permissions: {describeApiError(me.error)}
        </p>
      </section>
    )
  }
  if (!canRead) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Asset passport
        </h1>
        <p
          role="status"
          className="mt-2 rounded border border-border bg-panel p-4 text-sm text-muted"
        >
          You do not hold the <span className="font-mono">asset:read</span> permission required to
          view this asset passport.
        </p>
      </section>
    )
  }

  const detail = assetQuery.data?.status === 200 ? assetQuery.data.data : undefined
  const passport = detail?.passport

  return (
    <section aria-labelledby={`${idPrefix}-title`} className="flex max-w-5xl flex-col gap-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div>
          {backLink && <p className="text-xs text-muted">{backLink}</p>}
          <h1 id={`${idPrefix}-title`} className="mt-1 text-2xl font-semibold">
            {passport?.name ?? assetId}
          </h1>
          {passport && (
            <p className="mt-1 text-sm text-muted">
              <span className="font-mono text-ink">{passport.asset_id}</span> ·{' '}
              {passport.asset_type} · {passport.site_or_domain}
            </p>
          )}
        </div>
        {passport && (
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <span className="rounded border border-border px-2 py-0.5 font-mono text-ink">
              {passport.safety_boundary}
            </span>
            <span className="rounded border border-border px-2 py-0.5 font-mono text-muted">
              Model: {detail.runtime_model_version ?? 'not loaded'}
            </span>
          </div>
        )}
      </header>

      {assetQuery.isPending && <p role="status">Loading asset passport…</p>}
      {assetQuery.isError && (
        <p role="alert" className="rounded border border-critical p-3 text-sm text-critical">
          Could not load asset passport: {describeApiError(assetQuery.error)}
        </p>
      )}

      {passport && (
        <>
          <div className="rounded-lg border border-border bg-panel p-4 text-sm">
            <p className="text-ink">{passport.system_context}</p>
            <p className="mt-2 text-xs text-muted">
              {passport.deployment_scope_note} · Nominal cadence{' '}
              <span className="font-mono text-ink">{passport.nominal_sampling_period_s}s</span>
            </p>
          </div>

          <section
            aria-labelledby={`${idPrefix}-unobs`}
            className="rounded-lg border border-warn/60 bg-panel p-4 text-sm"
          >
            <h2 id={`${idPrefix}-unobs`} className="font-medium text-ink">
              Explicitly unobserved quantities &amp; boundaries
            </h2>
            <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-muted">
              {passport.unobserved_quantities.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </section>

          <section aria-labelledby={`${idPrefix}-regimes`} className="flex flex-col gap-3">
            <h2 id={`${idPrefix}-regimes`} className="text-lg font-medium">
              Operating regimes &amp; physical envelopes
            </h2>
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {passport.operating_regimes.map((reg) => (
                <div key={reg} className="rounded border border-border bg-panel p-3 text-xs">
                  <p className="font-mono font-medium text-ink">{reg}</p>
                  <p className="mt-1 text-muted">
                    {passport.operating_envelopes[reg] ?? 'No envelope note.'}
                  </p>
                </div>
              ))}
            </div>
          </section>

          <section aria-labelledby={`${idPrefix}-sensors`} className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 id={`${idPrefix}-sensors`} className="text-lg font-medium">
                Sensor catalog ({passport.sensors.length})
              </h2>
              <div className="flex items-center gap-2 text-xs">
                {(['all', 'analog', 'digital'] as const).map((kind) => (
                  <button
                    key={kind}
                    type="button"
                    aria-pressed={sensorFilter === kind}
                    onClick={() => setSensorFilter(kind)}
                    className={`rounded border px-2 py-0.5 ${
                      sensorFilter === kind
                        ? 'border-accent bg-accent/15 font-medium text-ink'
                        : 'border-border text-muted hover:bg-border'
                    }`}
                  >
                    {kind}
                  </button>
                ))}
              </div>
            </div>
            <div className="overflow-x-auto rounded-lg border border-border bg-panel p-3">
              <table className="w-full text-xs">
                <caption className="sr-only">
                  Sensor specifications, units, valid bounds, and observability notes
                </caption>
                <thead>
                  <tr className="text-left text-muted">
                    {['Sensor', 'Kind', 'Unit', 'Valid range', 'Role & nominal envelope'].map(
                      (h) => (
                        <th key={h} scope="col" className="pr-3 font-normal">
                          {h}
                        </th>
                      ),
                    )}
                  </tr>
                </thead>
                <tbody>
                  {passport.sensors
                    .filter((s) => sensorFilter === 'all' || s.kind === sensorFilter)
                    .map((s) => (
                      <tr key={s.sensor_id} className="border-t border-border align-top">
                        <td className="py-1.5 pr-3">
                          <span className="font-mono font-medium text-ink">{s.sensor_id}</span>
                          <span className="block text-muted">{s.display_name}</span>
                        </td>
                        <td className="py-1.5 pr-3 font-mono">{s.kind}</td>
                        <td className="py-1.5 pr-3 font-mono">{s.unit}</td>
                        <td className="py-1.5 pr-3 font-mono">
                          [{s.valid_range[0]}, {s.valid_range[1]}]
                        </td>
                        <td className="py-1.5">
                          <span className="text-ink">{s.physical_role}</span>
                          <span className="block text-muted">{s.nominal_envelope}</span>
                          {s.unobservability_note && (
                            <span className="block text-warn">Note: {s.unobservability_note}</span>
                          )}
                        </td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </section>

          <section aria-labelledby={`${idPrefix}-modes`} className="flex flex-col gap-3">
            <h2 id={`${idPrefix}-modes`} className="text-lg font-medium">
              Failure &amp; instrumentation modes ({passport.failure_modes.length})
            </h2>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {passport.failure_modes.map((fm) => (
                <article
                  key={fm.mode_id}
                  className="flex flex-col gap-2 rounded-lg border border-border bg-panel p-3 text-xs"
                >
                  <header>
                    <h3 className="font-medium text-ink">{fm.display_name}</h3>
                    <p className="font-mono text-muted">{fm.mode_id}</p>
                  </header>
                  <p className="text-muted">{fm.description}</p>
                  <div>
                    <p className="font-medium text-ink">Primary signatures</p>
                    <ul className="list-disc pl-4 text-muted">
                      {fm.primary_signatures.map((sig) => (
                        <li key={sig}>{sig}</li>
                      ))}
                    </ul>
                  </div>
                  <p className="text-muted">
                    <strong className="text-ink">Competing modes:</strong>{' '}
                    {fm.competing_explanations.join(', ')}
                  </p>
                </article>
              ))}
            </div>
          </section>

          <section aria-labelledby={`${idPrefix}-models`} className="flex flex-col gap-3">
            <h2 id={`${idPrefix}-models`} className="text-lg font-medium">
              Model applicability &amp; boundaries
            </h2>
            <div className="space-y-2">
              {passport.model_applicability.map((m) => (
                <div
                  key={m.model_id}
                  className="flex flex-col gap-1.5 rounded border border-border bg-panel p-3 text-xs"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-mono font-medium text-ink">{m.model_id}</span>
                    <span
                      className={`rounded border px-2 py-0.5 font-mono text-[11px] ${MODEL_BADGE[m.status]}`}
                    >
                      {m.status}
                    </span>
                  </div>
                  <p className="text-ink">{m.scope_note}</p>
                  <ul className="list-disc pl-4 text-muted">
                    {m.limitations.map((lim) => (
                      <li key={lim}>{lim}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </section>

          <section aria-labelledby={`${idPrefix}-ds`} className="flex flex-col gap-3">
            <h2 id={`${idPrefix}-ds`} className="text-lg font-medium">
              Connected data foundation ({detail.datasets.length})
            </h2>
            {detail.datasets.map((ds) => (
              <SourceDatasetCard key={ds.dataset_id} dataset={ds} />
            ))}
          </section>
        </>
      )}
    </section>
  )
}
