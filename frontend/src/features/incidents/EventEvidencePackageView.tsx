import { useId, useMemo, useState } from 'react'
import type { EventEvidencePackage } from '../../api/generated/model'
import { formatNaive } from '../../lib/naiveTime'

type EventPhaseName = EventEvidencePackage['phases'][number]['phase']

export interface EventEvidencePackageViewProps {
  pkg: EventEvidencePackage
}

const ORIGIN_LABELS: Record<string, string> = {
  observed: 'Observed',
  derived: 'Derived feature',
  statistical_inference: 'Statistical inference',
  model_prediction: 'Model prediction',
  human_observation: 'Human observation',
}

const QUALITY_STYLES: Record<string, string> = {
  sufficient: 'border-ok/60 text-ok',
  degraded: 'border-warn/60 text-warn',
  insufficient: 'border-critical/60 text-critical',
}

const TIMELINE_CATEGORIES = [
  'all',
  'alarm_episode',
  'consistency_warning',
  'regime_change',
  'model_output',
  'evidence_group',
  'human_observation',
  'operator_report',
  'telemetry_window',
] as const

function pct(value: number): string {
  return `${(value * 100).toFixed(1)} %`
}

function fmtNum(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined ? 'n/a' : value.toFixed(digits)
}

/** Accessible, origin-tagged multi-phase event evidence explorer and timeline. */
export function EventEvidencePackageView({ pkg }: EventEvidencePackageViewProps) {
  const headingPrefix = useId()
  const [selectedPhase, setSelectedPhase] = useState<EventPhaseName>('peak')
  const [timelineFilter, setTimelineFilter] = useState<string>('all')

  const activePhase = useMemo(
    () => pkg.phases.find((p) => p.phase === selectedPhase) ?? pkg.phases[0],
    [pkg.phases, selectedPhase],
  )

  const filteredTimeline = useMemo(() => {
    const items = pkg.timeline ?? []
    if (timelineFilter === 'all') return items
    return items.filter((entry) => entry.category === timelineFilter)
  }, [pkg.timeline, timelineFilter])

  const qualityStyle = QUALITY_STYLES[pkg.quality.overall_status] ?? 'border-border text-ink'

  return (
    <div className="flex flex-col gap-5 text-sm">
      <section
        aria-labelledby={`${headingPrefix}-quality`}
        className="rounded-lg border border-border bg-panel p-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 id={`${headingPrefix}-quality`} className="font-medium">
            Evidence quality &amp; provenance
          </h3>
          <span className={`rounded border px-2 py-0.5 text-xs font-medium ${qualityStyle}`}>
            Quality status: {pkg.quality.overall_status}
          </span>
        </div>

        <dl className="mt-3 grid grid-cols-1 gap-x-6 gap-y-2 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <dt className="text-xs text-muted">Completeness score</dt>
            <dd className="font-medium">{pct(pkg.quality.completeness_score)}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Event window coverage</dt>
            <dd className="font-medium">
              {pkg.quality.valid_event_windows}/{pkg.quality.expected_event_windows} windows (
              {pct(pkg.quality.coverage_ratio)})
            </dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Confidence penalty factor</dt>
            <dd className="font-medium">{pkg.quality.confidence_penalty_factor.toFixed(2)}x</dd>
          </div>
          <div>
            <dt className="text-xs text-muted">Recovery tail status</dt>
            <dd className="font-medium">{pkg.quality.recovery_status}</dd>
          </div>
        </dl>

        <dl className="mt-3 grid grid-cols-1 gap-x-6 gap-y-1 border-t border-border pt-3 text-xs sm:grid-cols-2">
          <div>
            <dt className="text-muted">Dataset &amp; preprocessing</dt>
            <dd>
              {pkg.provenance.dataset_id} ({pkg.provenance.dataset_license}) ·{' '}
              <span className="font-mono">{pkg.provenance.preprocessing_version}</span> ·{' '}
              {pkg.provenance.sampling_contract}
            </dd>
          </div>
          <div>
            <dt className="text-muted">Detector lineage</dt>
            <dd>
              Primary:{' '}
              <span className="font-mono">
                {pkg.provenance.primary_detector} ({pkg.provenance.primary_model_version})
              </span>
              {pkg.provenance.warning_detector && (
                <>
                  {' '}
                  · Warning:{' '}
                  <span className="font-mono">
                    {pkg.provenance.warning_detector} ({pkg.provenance.warning_model_version})
                  </span>
                </>
              )}
            </dd>
          </div>
        </dl>

        {((pkg.missing_measurements?.length ?? 0) > 0 ||
          (pkg.conflicting_measurements?.length ?? 0) > 0 ||
          (pkg.quality.quality_warnings?.length ?? 0) > 0) && (
          <div className="mt-3 flex flex-col gap-2 border-t border-border pt-3 text-xs">
            {(pkg.quality.frozen_sensors?.length ?? 0) > 0 && (
              <p className="text-warn">
                Constant / frozen channels detected:{' '}
                <span className="font-mono">{pkg.quality.frozen_sensors?.join(', ')}</span>
              </p>
            )}
            {(pkg.missing_measurements?.length ?? 0) > 0 && (
              <div>
                <p className="font-medium text-warn">Missing measurements (not interpolated):</p>
                <ul className="mt-1 list-disc pl-5 text-muted">
                  {pkg.missing_measurements.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
            {(pkg.conflicting_measurements?.length ?? 0) > 0 && (
              <div>
                <p className="font-medium text-critical">Conflicting telemetry / signals:</p>
                <ul className="mt-1 list-disc pl-5 text-muted">
                  {pkg.conflicting_measurements.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
            {(pkg.quality.quality_warnings?.length ?? 0) > 0 && (
              <div>
                <p className="font-medium text-ink">Quality caveats:</p>
                <ul className="mt-1 list-disc pl-5 text-muted">
                  {pkg.quality.quality_warnings?.map((w) => (
                    <li key={w}>{w}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </section>

      <section
        aria-labelledby={`${headingPrefix}-phases`}
        className="rounded-lg border border-border bg-panel p-4"
      >
        <h3 id={`${headingPrefix}-phases`} className="font-medium">
          Event phases (observed telemetry vs derived &amp; inferred signals)
        </h3>
        <div className="mt-3 flex flex-wrap gap-2">
          {pkg.phases.map((p) => (
            <button
              key={p.phase}
              type="button"
              aria-pressed={activePhase?.phase === p.phase}
              onClick={() => setSelectedPhase(p.phase)}
              className={`rounded border px-2.5 py-1 text-xs ${
                activePhase?.phase === p.phase
                  ? 'border-accent bg-accent/15 text-ink font-medium'
                  : 'border-border text-muted hover:bg-border'
              }`}
            >
              {p.label} ({p.available ? `${p.valid_windows}w` : 'unavailable'})
            </button>
          ))}
        </div>

        {activePhase && (
          <div className="mt-4 flex flex-col gap-4">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border pb-2 text-xs text-muted">
              <span>
                {formatNaive(activePhase.start)} to {formatNaive(activePhase.end)} ·{' '}
                {activePhase.sample_count} samples · {activePhase.valid_windows}/
                {activePhase.expected_windows} valid windows
              </span>
              <span>
                Regime:{' '}
                <strong className="text-ink">{activePhase.dominant_regime ?? 'unavailable'}</strong>
              </span>
            </div>

            {!activePhase.available && activePhase.reason_unavailable && (
              <p role="status" className="rounded border border-warn/50 p-2 text-xs text-warn">
                Phase unavailable: {activePhase.reason_unavailable}
              </p>
            )}

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div className="rounded border border-border p-3 text-xs">
                <p className="font-medium text-ink">
                  Model prediction &amp; statistical exceedance{' '}
                  <span className="ml-1 rounded border border-border px-1.5 py-0.5 text-[11px] font-normal text-muted">
                    {ORIGIN_LABELS.model_prediction} / {ORIGIN_LABELS.statistical_inference}
                  </span>
                </p>
                <p className="mt-1 text-muted">
                  Primary peak score: {fmtNum(activePhase.primary_peak_score)} · Min exceedance:{' '}
                  {fmtNum(activePhase.primary_min_exceedance, 4)}
                </p>
                <p className="mt-0.5 text-muted">
                  Warning peak residual: {fmtNum(activePhase.warning_peak_score, 2)} · Active
                  warning windows: {activePhase.warning_active_windows ?? 0}
                </p>
              </div>
              <div className="rounded border border-border p-3 text-xs">
                <p className="font-medium text-ink">
                  Derived compressor features{' '}
                  <span className="ml-1 rounded border border-border px-1.5 py-0.5 text-[11px] font-normal text-muted">
                    {ORIGIN_LABELS.derived}
                  </span>
                </p>
                <ul className="mt-1 space-y-0.5 text-muted">
                  {(activePhase.derived_features ?? []).map((f) => (
                    <li key={f.name}>
                      <span className="font-mono text-ink">{f.name}</span>:{' '}
                      {f.observable && f.value !== null && f.value !== undefined
                        ? `${f.value} ${f.unit}`
                        : `Unobservable (${f.note ?? 'no active segment'})`}
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <caption className="mb-1 text-left text-muted">
                  Direct sensor summaries in {activePhase.label} ({ORIGIN_LABELS.observed})
                </caption>
                <thead>
                  <tr className="text-left text-muted">
                    <th scope="col" className="pr-3 font-normal">
                      Sensor
                    </th>
                    <th scope="col" className="pr-3 font-normal">
                      Origin
                    </th>
                    <th scope="col" className="pr-3 font-normal">
                      Unit
                    </th>
                    <th scope="col" className="pr-3 font-normal">
                      Samples
                    </th>
                    <th scope="col" className="pr-3 font-normal">
                      Min / Mean / Max
                    </th>
                    <th scope="col" className="pr-3 font-normal">
                      Active frac / Last
                    </th>
                    <th scope="col" className="font-normal">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {(activePhase.observed_signals ?? []).map((sig) => (
                    <tr key={sig.signal} className="border-t border-border">
                      <th scope="row" className="py-1 pr-3 text-left font-mono font-normal">
                        {sig.signal}
                      </th>
                      <td className="py-1 pr-3 text-muted">
                        {ORIGIN_LABELS[sig.origin ?? 'observed'] ?? sig.origin ?? 'observed'}
                      </td>
                      <td className="py-1 pr-3">{sig.unit}</td>
                      <td className="py-1 pr-3">
                        {sig.sample_count}
                        {(sig.missing_count ?? 0) > 0 && ` (${sig.missing_count} missing)`}
                      </td>
                      <td className="py-1 pr-3">
                        {sig.mean !== null && sig.mean !== undefined
                          ? `${fmtNum(sig.min, 2)} / ${fmtNum(sig.mean, 2)} / ${fmtNum(sig.max, 2)}`
                          : 'n/a'}
                      </td>
                      <td className="py-1 pr-3">
                        {sig.active_fraction !== null && sig.active_fraction !== undefined
                          ? `${(sig.active_fraction * 100).toFixed(0)} % active`
                          : fmtNum(sig.last, 2)}
                      </td>
                      <td className="py-1">
                        {sig.frozen
                          ? 'frozen (const)'
                          : (sig.out_of_range_count ?? 0) > 0
                            ? `${sig.out_of_range_count} out of range`
                            : sig.sample_count === 0
                              ? 'missing'
                              : 'ok'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {(pkg.neighboring_events.length > 0 || (pkg.quality.contradictions?.length ?? 0) > 0) && (
        <section
          aria-labelledby={`${headingPrefix}-neighbors`}
          className="rounded-lg border border-border bg-panel p-4"
        >
          <h3 id={`${headingPrefix}-neighbors`} className="font-medium">
            Neighboring events &amp; signal contradictions
          </h3>
          {pkg.neighboring_events.length > 0 && (
            <ul className="mt-2 space-y-1 text-xs text-muted">
              {pkg.neighboring_events.map((n) => (
                <li key={`${n.kind}-${n.start}-${n.title}`}>
                  <span className="font-medium text-ink">[{n.relationship}]</span> {n.title} (
                  {formatNaive(n.start)} to {formatNaive(n.end)} · origin:{' '}
                  {ORIGIN_LABELS[n.origin] ?? n.origin})
                </li>
              ))}
            </ul>
          )}
          {(pkg.quality.contradictions?.length ?? 0) > 0 && (
            <ul className="mt-2 space-y-1 text-xs text-warn">
              {pkg.quality.contradictions?.map((c) => (
                <li key={`${c.kind}-${c.timestamp}-${c.signals.join('-')}`}>
                  [{c.severity}] {formatNaive(c.timestamp)} · {c.kind} ({c.signals.join(', ')}):{' '}
                  {c.description}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      <section
        aria-labelledby={`${headingPrefix}-timeline`}
        className="rounded-lg border border-border bg-panel p-4"
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h3 id={`${headingPrefix}-timeline`} className="font-medium">
            Chronological event timeline ({filteredTimeline.length})
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {TIMELINE_CATEGORIES.map((cat) => (
              <button
                key={cat}
                type="button"
                aria-pressed={timelineFilter === cat}
                onClick={() => setTimelineFilter(cat)}
                className={`rounded border px-2 py-0.5 text-xs ${
                  timelineFilter === cat
                    ? 'border-accent bg-accent/15 text-ink font-medium'
                    : 'border-border text-muted hover:bg-border'
                }`}
              >
                {cat === 'all' ? 'All' : cat.replaceAll('_', ' ')}
              </button>
            ))}
          </div>
        </div>

        <ol className="mt-3 space-y-2">
          {filteredTimeline.map((entry) => (
            <li
              key={`${entry.timestamp}-${entry.category}-${entry.phase ?? 'none'}-${entry.source_id}-${entry.title}`}
              className="rounded border border-border p-2.5 text-xs"
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-medium text-ink">
                  {formatNaive(entry.timestamp)} — {entry.title}
                </span>
                <span className="rounded border border-border px-1.5 py-0.5 text-[11px] text-muted">
                  {ORIGIN_LABELS[entry.origin] ?? entry.origin}
                </span>
              </div>
              <p className="mt-1 text-muted">{entry.summary}</p>
              <p className="mt-0.5 text-[11px] text-muted">
                Source: <span className="font-mono">{entry.source_id}</span>
                {entry.phase && ` · Phase: ${entry.phase}`}
                {entry.regime && ` · Regime: ${entry.regime}`}
              </p>
            </li>
          ))}
        </ol>
      </section>

      <p className="text-xs text-muted">{pkg.advisory}</p>
    </div>
  )
}
