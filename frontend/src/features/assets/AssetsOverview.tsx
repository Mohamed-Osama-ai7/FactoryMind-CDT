import { type ReactNode, useId } from 'react'
import { describeApiError } from '../../api/errors'
import { useListAssets, useListDatasets } from '../../api/generated/assets/assets'
import { useMe } from '../../api/generated/incidents/incidents'
import { SourceDatasetCard } from './SourceDatasetCard'

export interface AssetsOverviewProps {
  renderOpenLink?: (assetId: string) => ReactNode
}

/** Asset Digital Passport & Data Foundation overview scoped to caller grants. */
export function AssetsOverview({ renderOpenLink }: AssetsOverviewProps) {
  const idPrefix = useId()
  const me = useMe()
  const canRead = me.data?.data.permissions.includes('asset:read') ?? false

  const assetsQuery = useListAssets({ query: { enabled: canRead } })
  const datasetsQuery = useListDatasets(undefined, { query: { enabled: canRead } })

  if (me.isPending) return <p role="status">Checking asset permissions…</p>
  if (me.isError) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Assets &amp; Data Foundation
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
          Assets &amp; Data Foundation
        </h1>
        <p
          role="status"
          className="mt-2 rounded border border-border bg-panel p-4 text-sm text-muted"
        >
          You do not hold the <span className="font-mono">asset:read</span> permission required to
          inspect asset digital passports and data lineage.
        </p>
      </section>
    )
  }

  const assets = assetsQuery.data?.status === 200 ? assetsQuery.data.data.items : []
  const datasets = datasetsQuery.data?.status === 200 ? datasetsQuery.data.data.items : []

  return (
    <section aria-labelledby={`${idPrefix}-title`} className="flex max-w-5xl flex-col gap-6">
      <header>
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Assets &amp; Data Foundation
        </h1>
        <p className="mt-1 text-sm text-muted">
          Physical asset digital passports, sensor envelopes, failure mode specifications, and
          non-destructive dataset provenance. Advisory monitoring only; never actuates equipment.
        </p>
      </header>

      <section aria-labelledby={`${idPrefix}-registry`} className="flex flex-col gap-3">
        <h2 id={`${idPrefix}-registry`} className="text-lg font-medium">
          Registered asset passports ({assets.length})
        </h2>
        {assetsQuery.isPending && <p role="status">Loading asset passports…</p>}
        {assetsQuery.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-sm text-critical">
            Could not load asset registry: {describeApiError(assetsQuery.error)}
          </p>
        )}
        {!assetsQuery.isPending && !assetsQuery.isError && assets.length === 0 && (
          <p className="text-sm text-muted">No asset passports granted for this principal.</p>
        )}
        {assets.length > 0 && (
          <div className="grid grid-cols-1 gap-4">
            {assets.map((asset) => (
              <article
                key={asset.asset_id}
                className="flex flex-col gap-3 rounded-lg border border-border bg-panel p-4 text-sm"
              >
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <h3 className="font-medium text-ink">{asset.name}</h3>
                    <p className="text-xs text-muted">
                      <span className="font-mono text-ink">{asset.asset_id}</span> ·{' '}
                      {asset.asset_type}
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span
                      className={`rounded border px-2 py-0.5 ${
                        asset.telemetry_available
                          ? 'border-ok/60 bg-ok/10 text-ok'
                          : 'border-warn/60 text-warn'
                      }`}
                    >
                      {asset.telemetry_available ? 'Telemetry online' : 'Telemetry offline'}
                    </span>
                    <span className="rounded border border-border px-2 py-0.5 font-mono text-muted">
                      {asset.safety_boundary}
                    </span>
                  </div>
                </div>
                <dl className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
                  <div>
                    <dt className="text-muted">Sensors</dt>
                    <dd className="font-mono text-ink">{asset.sensor_count}</dd>
                  </div>
                  <div>
                    <dt className="text-muted">Failure modes</dt>
                    <dd className="font-mono text-ink">{asset.failure_mode_count}</dd>
                  </div>
                  <div>
                    <dt className="text-muted">Active models</dt>
                    <dd className="font-mono text-ink">{asset.active_models.join(', ')}</dd>
                  </div>
                  <div>
                    <dt className="text-muted">Runtime version</dt>
                    <dd className="font-mono text-ink">
                      {asset.runtime_model_version ?? 'not loaded'}
                    </dd>
                  </div>
                </dl>
                {renderOpenLink && <div>{renderOpenLink(asset.asset_id)}</div>}
              </article>
            ))}
          </div>
        )}
      </section>

      <section aria-labelledby={`${idPrefix}-datasets`} className="flex flex-col gap-3">
        <h2 id={`${idPrefix}-datasets`} className="text-lg font-medium">
          Connected source datasets ({datasets.length})
        </h2>
        {datasetsQuery.isPending && <p role="status">Loading dataset provenance…</p>}
        {datasetsQuery.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-sm text-critical">
            Could not load datasets: {describeApiError(datasetsQuery.error)}
          </p>
        )}
        {!datasetsQuery.isPending && !datasetsQuery.isError && datasets.length === 0 && (
          <p className="text-sm text-muted">No connected datasets registered for this scope.</p>
        )}
        {datasets.map((ds) => (
          <SourceDatasetCard key={ds.dataset_id} dataset={ds} />
        ))}
      </section>
    </section>
  )
}
