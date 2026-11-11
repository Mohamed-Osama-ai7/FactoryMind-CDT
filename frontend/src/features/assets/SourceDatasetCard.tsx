import type { SourceDatasetRecord } from '../../api/generated/model'

const STATUS_BADGE = {
  verified: 'border-ok/60 bg-ok/10 text-ok',
  partial: 'border-warn/60 bg-warn/10 text-warn',
  catalog_only: 'border-border text-muted',
  pass: 'border-ok/60 text-ok',
  warn: 'border-warn/60 text-warn',
  unavailable: 'border-border text-muted',
} as const

/** Displays dataset provenance, SHA-256 checksum, quality checks, and non-destructive lineage. */
export function SourceDatasetCard({ dataset }: { dataset: SourceDatasetRecord }) {
  return (
    <article className="flex flex-col gap-3 rounded-lg border border-border bg-panel p-4 text-sm">
      <header className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 className="font-medium text-ink">{dataset.title}</h3>
          <p className="text-xs text-muted">
            Dataset <span className="font-mono text-ink">{dataset.dataset_id}</span> · License:{' '}
            {dataset.license} · Origin:{' '}
            <span className="font-mono">{dataset.provenance_origin}</span>
          </p>
        </div>
        <span
          className={`rounded border px-2 py-0.5 text-xs font-medium ${STATUS_BADGE[dataset.artifact_status]}`}
        >
          {dataset.artifact_status}
        </span>
      </header>

      <dl className="grid grid-cols-1 gap-2 rounded border border-border p-2.5 text-xs sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <dt className="text-muted">Raw archive SHA-256</dt>
          <dd className="font-mono text-ink" title={dataset.archive_sha256}>
            {dataset.archive_sha256.slice(0, 16)}…
          </dd>
        </div>
        <div>
          <dt className="text-muted">Raw source preserved</dt>
          <dd className="text-ink">{dataset.raw_preserved ? 'Yes (untouched)' : 'Not on disk'}</dd>
        </div>
        <div>
          <dt className="text-muted">Failure report labels</dt>
          <dd className="text-ink">{dataset.failure_report_count} operator windows</dd>
        </div>
        <div>
          <dt className="text-muted">Artifacts present</dt>
          <dd className="text-ink">
            manifest={dataset.manifest_present ? 'yes' : 'no'} · parquet=
            {dataset.telemetry_present ? 'yes' : 'no'}
          </dd>
        </div>
      </dl>

      {dataset.quality && (
        <dl className="grid grid-cols-2 gap-2 rounded border border-border p-2.5 text-xs sm:grid-cols-4">
          <div>
            <dt className="text-muted">Rows (raw / cleaned)</dt>
            <dd className="font-mono text-ink">
              {dataset.quality.rows_raw.toLocaleString()} /{' '}
              {dataset.quality.rows_cleaned.toLocaleString()}
            </dd>
          </div>
          <div>
            <dt className="text-muted">Span (local train time)</dt>
            <dd className="font-mono text-ink">
              {dataset.quality.start} → {dataset.quality.end}
            </dd>
          </div>
          <div>
            <dt className="text-muted">Gaps &gt; 60s</dt>
            <dd className="font-mono text-ink">
              {dataset.quality.gaps_over_60s} ({dataset.quality.gap_hours} h)
            </dd>
          </div>
          <div>
            <dt className="text-muted">Gap policy</dt>
            <dd className="font-mono text-ink">{dataset.quality.gap_policy}</dd>
          </div>
        </dl>
      )}

      <div>
        <h4 className="mb-1.5 text-xs font-medium text-muted">
          Data quality &amp; provenance checks
        </h4>
        <ul className="space-y-1 text-xs">
          {dataset.quality_checks.map((chk) => (
            <li key={chk.check_id} className="flex flex-wrap items-baseline gap-2">
              <span
                className={`rounded border px-1.5 py-0.5 font-mono text-[11px] uppercase ${STATUS_BADGE[chk.status]}`}
              >
                {chk.status}
              </span>
              <span className="font-mono text-ink">{chk.check_id}</span>
              <span className="text-muted">{chk.summary}</span>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h4 className="mb-1.5 text-xs font-medium text-muted">
          Non-destructive normalization lineage
        </h4>
        <ol className="space-y-1 text-xs">
          {dataset.lineage.map((step) => (
            <li key={step.step} className="rounded border border-border px-2.5 py-1.5">
              <span className="font-mono font-medium text-ink">
                {step.step}. {step.operation}
              </span>{' '}
              <span className="text-muted">— {step.description}</span>
            </li>
          ))}
        </ol>
      </div>
    </article>
  )
}
