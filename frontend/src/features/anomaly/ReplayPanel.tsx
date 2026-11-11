import { Fragment, useMemo, useState } from 'react'
import { describeApiError } from '../../api/errors'
import {
  useAnomalyEventEvidence,
  useAnomalyRca,
  useAnomalyReplay,
} from '../../api/generated/anomaly/anomaly'
import type {
  AlarmEpisode,
  Evidence,
  ReplayResponse,
  WarningEpisode,
  WindowResult,
} from '../../api/generated/model'
import { addHours, formatNaive, toEpochMs } from '../../lib/naiveTime'
import { EventEvidencePackageView } from '../incidents/EventEvidencePackageView'
import { RcaPanel } from '../incidents/RcaPanel'
import { CreateIncidentButton } from './CreateIncidentButton'
import { ScoreChart } from './ScoreChart'

export interface ReplayPanelProps {
  start: string
  hours: number
}

const REGIME_LABELS: Record<string, string> = {
  loaded: 'Loaded',
  unloaded: 'Unloaded',
  starting: 'Starting (transitional)',
  stopped: 'Stopped',
  insufficient_evidence: 'Insufficient evidence',
}

const WARNING_KIND_LABELS: Record<string, string> = {
  stale_telemetry: 'Frozen / stale telemetry feed',
  signal_inconsistency: 'Valve / motor-current contradiction',
  multivariate_sensor_drift: 'Multivariate PCA SPE residual exceedance',
  mixed_consistency_warning: 'PCA SPE + telemetry integrity fault',
}

function formatRegimeLabel(regime: string | undefined): string {
  if (!regime) return 'Insufficient evidence'
  return REGIME_LABELS[regime] ?? regime
}

function formatBreakdown(breakdown: Record<string, number> | undefined): string {
  if (!breakdown) return ''
  const entries = Object.entries(breakdown).filter(([, n]) => n > 0)
  return entries.map(([k, n]) => `${formatRegimeLabel(k)}: ${n}`).join(', ')
}

/** Highest-scoring window inside an alarm episode, which carries the feature evidence. */
function peakWindow(windows: WindowResult[], episode: AlarmEpisode): WindowResult | undefined {
  const lo = toEpochMs(episode.start)
  const hi = toEpochMs(episode.end)
  return windows
    .filter((w) => {
      const t = toEpochMs(w.window_start)
      return t >= lo && t < hi && w.score !== null
    })
    .reduce<WindowResult | undefined>(
      (best, w) => (best === undefined || (w.score ?? 0) > (best.score ?? 0) ? w : best),
      undefined,
    )
}

/** Highest-scoring (or first integrity-flagged) window inside a warning episode. */
function peakWarningWindow(
  windows: WindowResult[],
  warning: WarningEpisode,
): WindowResult | undefined {
  const lo = toEpochMs(warning.start)
  const hi = toEpochMs(warning.end)
  const matching = windows.filter((w) => {
    const t = toEpochMs(w.window_start)
    return t >= lo && t < hi && w.warning_active
  })
  return matching.reduce<WindowResult | undefined>(
    (best, w) =>
      best === undefined || (w.warning_score ?? -1) > (best.warning_score ?? -1) ? w : best,
    undefined,
  )
}

function Summary({ data }: { data: ReplayResponse }) {
  const counts = data.windows.reduce(
    (acc, w) => {
      acc[w.status] += 1
      return acc
    },
    { normal: 0, anomalous: 0, stopped: 0, insufficient_data: 0 },
  )
  const regimeCounts = data.windows.reduce<Record<string, number>>((acc, w) => {
    const r = w.regime.regime
    acc[r] = (acc[r] ?? 0) + 1
    return acc
  }, {})
  const insufficientReasons = Array.from(
    new Set(
      data.windows
        .filter((w) => w.regime.regime === 'insufficient_evidence')
        .map((w) => w.regime.reason),
    ),
  )
  const warnings = data.warnings ?? []
  const warningWindows = data.windows.filter((w) => w.warning_active).length

  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm text-muted">
        {data.windows.length} windows: {counts.normal} normal, {counts.anomalous} above threshold,{' '}
        {counts.stopped} machine stopped (explained), {counts.insufficient_data} with insufficient
        data · {data.episodes.length} air-leak alarm episode
        {data.episodes.length === 1 ? '' : 's'} (≥{data.persistence} consecutive windows) ·{' '}
        {warnings.length} consistency warning episode{warnings.length === 1 ? '' : 's'} (
        {warningWindows} warning window{warningWindows === 1 ? '' : 's'})
      </p>
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-muted">Operating regimes:</span>
        {(['loaded', 'unloaded', 'starting', 'stopped', 'insufficient_evidence'] as const).map(
          (key) => (
            <span key={key} className="rounded border border-border bg-panel px-2 py-0.5 text-ink">
              {formatRegimeLabel(key)}: {regimeCounts[key] ?? 0}
            </span>
          ),
        )}
        {insufficientReasons.length > 0 && (
          <span className="text-muted">
            (insufficient-evidence causes: {insufficientReasons.join(', ')})
          </span>
        )}
      </div>
    </div>
  )
}

function EvidenceTable({
  windowStart,
  evidence,
  captionNote,
}: {
  windowStart: string
  evidence: Evidence[]
  captionNote: string
}) {
  return (
    <table className="mt-2 w-full text-sm">
      <caption className="text-left text-muted">
        Evidence in the peak window ({formatNaive(windowStart)}), one item per physical signal
        group. {captionNote}
      </caption>
      <thead>
        <tr className="text-left text-muted">
          <th scope="col" className="pr-4 font-normal">
            Signal group
          </th>
          <th scope="col" className="pr-4 font-normal">
            Feature
          </th>
          <th scope="col" className="pr-4 font-normal">
            Value
          </th>
          <th scope="col" className="pr-4 font-normal">
            Training median
          </th>
          <th scope="col" className="font-normal">
            Deviation (robust z)
          </th>
        </tr>
      </thead>
      <tbody>
        {evidence.map((e) => (
          <tr key={e.group} className="border-t border-border align-top">
            <td className="py-1 pr-4">
              <span className="font-medium">{e.group_label}</span>
              <span className="block text-xs text-muted">{e.meaning}</span>
            </td>
            <td className="py-1 pr-4 font-mono">{e.feature}</td>
            <td className="py-1 pr-4">{e.value.toFixed(3)}</td>
            <td className="py-1 pr-4">{e.training_median.toFixed(3)}</td>
            <td className="py-1">
              {e.novel
                ? 'never varied in training'
                : e.robust_z === null
                  ? 'n/a'
                  : e.robust_z.toFixed(1)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function InlineEventEvidence({
  eventStart,
  eventEnd,
  raisedAt,
}: {
  eventStart: string
  eventEnd: string
  raisedAt?: string
}) {
  const [showRca, setShowRca] = useState(false)
  const params = {
    start: eventStart,
    end: eventEnd,
    ...(raisedAt ? { raised_at: raisedAt } : {}),
  }
  const pkgQuery = useAnomalyEventEvidence(params)
  const rcaQuery = useAnomalyRca(params, { query: { enabled: showRca } })

  if (pkgQuery.isPending) return <p role="status">Building event evidence package…</p>
  if (pkgQuery.isError) {
    return (
      <p role="alert" className="rounded border border-critical p-2 text-xs">
        Could not load event package: {describeApiError(pkgQuery.error)}
      </p>
    )
  }
  const pkg = pkgQuery.data?.status === 200 ? pkgQuery.data.data : undefined
  const rca = rcaQuery.data?.status === 200 ? rcaQuery.data.data : undefined
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <button
          type="button"
          aria-pressed={showRca}
          onClick={() => setShowRca((v) => !v)}
          className="rounded border border-border px-2.5 py-1 text-xs text-accent hover:bg-border"
        >
          {showRca ? 'Hide RCA hypotheses' : 'Explain root causes (RCA)'}
        </button>
      </div>
      {showRca && rcaQuery.isPending && <p role="status">Evaluating RCA hypotheses…</p>}
      {showRca && rcaQuery.isError && (
        <p role="alert" className="rounded border border-critical p-2 text-xs">
          Could not load RCA: {describeApiError(rcaQuery.error)}
        </p>
      )}
      {showRca && rca && <RcaPanel key={rca.rca_id} rca={rca} headingLevel="h4" />}
      {pkg && <EventEvidencePackageView key={pkg.package_id} pkg={pkg} headingLevel="h4" />}
    </div>
  )
}

/** Replays recorded telemetry through primary and warning detectors with regime context. */
export function ReplayPanel({ start, hours }: ReplayPanelProps) {
  const end = addHours(start, hours)
  const { data, isPending, isError, error, isFetching } = useAnomalyReplay({ start, end })
  const [selected, setSelected] = useState<number | null>(null)
  const [selectedWarning, setSelectedWarning] = useState<number | null>(null)
  const [inspectedEvent, setInspectedEvent] = useState<{
    start: string
    end: string
    raisedAt?: string
  } | null>(null)
  const rangeKey = `${start}-${hours}`
  const [prevRangeKey, setPrevRangeKey] = useState(rangeKey)
  if (rangeKey !== prevRangeKey) {
    setPrevRangeKey(rangeKey)
    setSelected(null)
    setSelectedWarning(null)
    setInspectedEvent(null)
  }

  // Non-2xx responses throw (see api/http.ts), so a resolved response is always the 200 shape.
  const replay = data?.status === 200 ? data.data : undefined
  const warnings = replay?.warnings ?? []
  const selectedEpisode =
    replay && selected !== null ? (replay.episodes[selected] ?? undefined) : undefined
  const evidenceWindow = useMemo(
    () => (replay && selectedEpisode ? peakWindow(replay.windows, selectedEpisode) : undefined),
    [replay, selectedEpisode],
  )
  const activeWarning =
    replay && selectedWarning !== null ? (warnings[selectedWarning] ?? undefined) : undefined
  const warningWindow = useMemo(
    () => (replay && activeWarning ? peakWarningWindow(replay.windows, activeWarning) : undefined),
    [replay, activeWarning],
  )

  if (isPending) {
    return <p role="status">Scoring telemetry…</p>
  }
  if (isError) {
    return (
      <p role="alert" className="rounded border border-critical p-3 text-sm">
        Replay failed: {describeApiError(error)}
      </p>
    )
  }
  if (!replay) return null
  if (replay.windows.length === 0) {
    return (
      <p role="status" className="text-sm text-muted">
        No telemetry recorded between {formatNaive(start)} and {formatNaive(end)}.
      </p>
    )
  }

  return (
    <div className="flex flex-col gap-5" aria-busy={isFetching}>
      <Summary data={replay} />
      <ScoreChart
        windows={replay.windows}
        threshold={replay.threshold}
        episodes={replay.episodes}
        warnings={warnings}
        reports={replay.operator_reports}
      />
      {replay.operator_reports.length > 0 && (
        <p className="text-sm">
          Operator reports in range:{' '}
          {replay.operator_reports
            .map(
              (r) =>
                `${r.report_id} ${r.failure} (${formatNaive(r.start)} to ${formatNaive(r.end)})`,
            )
            .join('; ')}
        </p>
      )}
      <section aria-labelledby="episodes-title" className="rounded border border-critical/50 p-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 id="episodes-title" className="font-medium text-critical">
            Air-leak alarm episodes
          </h3>
          <span className="rounded border border-critical/50 px-2 py-0.5 text-xs text-critical">
            air_leak_alarm hypothesis
          </span>
        </div>
        {replay.episodes.length === 0 ? (
          <p className="mt-1 text-sm text-muted">No air-leak alarms raised in this range.</p>
        ) : (
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-muted">
                  <th scope="col" className="pr-4 font-normal">
                    Raised at
                  </th>
                  <th scope="col" className="pr-4 font-normal">
                    Anomalous from
                  </th>
                  <th scope="col" className="pr-4 font-normal">
                    To
                  </th>
                  <th scope="col" className="pr-4 font-normal">
                    Regime &amp; hypothesis
                  </th>
                  <th scope="col" className="pr-4 font-normal">
                    Peak score
                  </th>
                  <th scope="col" className="font-normal">
                    <span className="sr-only">Evidence</span>
                  </th>
                  <th scope="col" className="pl-4 font-normal">
                    <span className="sr-only">Incident</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {replay.episodes.map((e, i) => (
                  <Fragment key={e.start}>
                    <tr className="border-t border-border align-top">
                      <td className="py-1.5 pr-4">
                        {formatNaive(e.raised_at)}
                        {e.ongoing && <span className="block text-xs text-muted">ongoing</span>}
                      </td>
                      <td className="py-1.5 pr-4">
                        {formatNaive(e.start)}
                        {e.truncated && (
                          <span className="block text-xs text-muted">
                            started earlier (look-back limit)
                          </span>
                        )}
                      </td>
                      <td className="py-1.5 pr-4">{formatNaive(e.end)}</td>
                      <td className="py-1.5 pr-4">
                        <span className="font-medium">{formatRegimeLabel(e.dominant_regime)}</span>
                        {e.regime_breakdown && Object.keys(e.regime_breakdown).length > 0 && (
                          <span className="block text-xs text-muted">
                            {formatBreakdown(e.regime_breakdown)}
                          </span>
                        )}
                        {e.hypothesis && (
                          <span className="mt-1 block text-xs text-ink">{e.hypothesis}</span>
                        )}
                        {e.uncertainty_note && (
                          <span className="block text-xs text-muted">{e.uncertainty_note}</span>
                        )}
                      </td>
                      <td className="py-1.5 pr-4">{e.peak_score.toFixed(3)}</td>
                      <td className="py-1.5">
                        {peakWindow(replay.windows, e)?.evidence.length ? (
                          <button
                            type="button"
                            className="text-accent underline"
                            aria-pressed={selected === i}
                            onClick={() => setSelected(selected === i ? null : i)}
                          >
                            {selected === i ? 'Hide evidence' : 'Show evidence'}
                          </button>
                        ) : (
                          <span className="text-xs text-muted">outside view</span>
                        )}
                      </td>
                      <td className="py-1.5 pl-4">
                        <div className="flex flex-col items-start gap-1">
                          <CreateIncidentButton
                            rangeStart={start}
                            rangeEnd={end}
                            episodeStart={e.start}
                          />
                          <button
                            type="button"
                            className="text-xs text-accent underline"
                            aria-pressed={inspectedEvent?.start === e.start}
                            onClick={() =>
                              setInspectedEvent(
                                inspectedEvent?.start === e.start
                                  ? null
                                  : { start: e.start, end: e.end, raisedAt: e.raised_at },
                              )
                            }
                          >
                            {inspectedEvent?.start === e.start
                              ? 'Hide event package'
                              : 'Inspect event package'}
                          </button>
                        </div>
                      </td>
                    </tr>
                    {selected === i && evidenceWindow && (
                      <tr>
                        <td colSpan={7} className="pb-3">
                          <EvidenceTable
                            windowStart={evidenceWindow.window_start}
                            evidence={evidenceWindow.evidence}
                            captionNote="A deviation from the training distribution shows an unusual state, not a cause."
                          />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section aria-labelledby="warnings-title" className="rounded border border-warn/50 p-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 id="warnings-title" className="font-medium text-warn">
            Sensor-consistency &amp; telemetry integrity warnings
          </h3>
          <span className="rounded border border-warn/50 px-2 py-0.5 text-xs text-warn">
            sensor_consistency_warning (not an air leak)
          </span>
        </div>
        {warnings.length === 0 ? (
          <p className="mt-1 text-sm text-muted">
            No sensor-consistency or telemetry-integrity warnings in this range.
          </p>
        ) : (
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-muted">
                  <th scope="col" className="pr-4 font-normal">
                    Raised at
                  </th>
                  <th scope="col" className="pr-4 font-normal">
                    From
                  </th>
                  <th scope="col" className="pr-4 font-normal">
                    To
                  </th>
                  <th scope="col" className="pr-4 font-normal">
                    Classification &amp; caveats
                  </th>
                  <th scope="col" className="pr-4 font-normal">
                    Peak residual
                  </th>
                  <th scope="col" className="font-normal">
                    <span className="sr-only">Warning evidence</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {warnings.map((w, i) => (
                  <Fragment key={`${w.start}-${w.detector_name}`}>
                    <tr className="border-t border-border align-top">
                      <td className="py-1.5 pr-4">
                        {formatNaive(w.raised_at)}
                        {w.ongoing && <span className="block text-xs text-muted">ongoing</span>}
                      </td>
                      <td className="py-1.5 pr-4">{formatNaive(w.start)}</td>
                      <td className="py-1.5 pr-4">{formatNaive(w.end)}</td>
                      <td className="py-1.5 pr-4">
                        <span className="font-medium">
                          {WARNING_KIND_LABELS[w.warning_kind] ?? w.warning_kind}
                        </span>
                        <span className="block text-xs text-muted">
                          Detector: {w.detector_name} · Regime:{' '}
                          {formatRegimeLabel(w.dominant_regime)} ({w.sources.join(', ')})
                        </span>
                        <span className="mt-1 block text-xs text-ink">{w.hypothesis}</span>
                        <span className="block text-xs text-muted">{w.uncertainty_note}</span>
                      </td>
                      <td className="py-1.5 pr-4">
                        {w.peak_score === null || w.peak_score === undefined
                          ? 'integrity check'
                          : w.peak_score.toFixed(2)}
                      </td>
                      <td className="py-1.5">
                        {(peakWarningWindow(replay.windows, w)?.warning_evidence?.length ?? 0) >
                        0 ? (
                          <button
                            type="button"
                            className="text-accent underline"
                            aria-pressed={selectedWarning === i}
                            onClick={() => setSelectedWarning(selectedWarning === i ? null : i)}
                          >
                            {selectedWarning === i ? 'Hide evidence' : 'Show warning evidence'}
                          </button>
                        ) : (
                          <span className="text-xs text-muted">n/a</span>
                        )}
                      </td>
                    </tr>
                    {selectedWarning === i && warningWindow && (
                      <tr>
                        <td colSpan={6} className="pb-3">
                          <EvidenceTable
                            windowStart={warningWindow.window_start}
                            evidence={warningWindow.warning_evidence ?? []}
                            captionNote="Warning evidence indicates sensor/telemetry decoupling or frozen signals, not a confirmed air leak."
                          />
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {inspectedEvent && (
        <section aria-label="Selected event evidence package" className="flex flex-col gap-2">
          <div className="flex items-center justify-between">
            <h3 className="font-medium">
              Event evidence package ({formatNaive(inspectedEvent.start)} to{' '}
              {formatNaive(inspectedEvent.end)})
            </h3>
            <button
              type="button"
              onClick={() => setInspectedEvent(null)}
              className="text-xs text-accent underline"
            >
              Close
            </button>
          </div>
          <InlineEventEvidence
            key={`${inspectedEvent.start}-${inspectedEvent.end}`}
            eventStart={inspectedEvent.start}
            eventEnd={inspectedEvent.end}
            raisedAt={inspectedEvent.raisedAt}
          />
        </section>
      )}
    </div>
  )
}
