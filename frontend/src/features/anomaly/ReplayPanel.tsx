import { Fragment, useMemo, useState } from 'react'
import { describeApiError } from '../../api/errors'
import { useAnomalyReplay } from '../../api/generated/anomaly/anomaly'
import type { AlarmEpisode, ReplayResponse, WindowResult } from '../../api/generated/model'
import { addHours, formatNaive, toEpochMs } from '../../lib/naiveTime'
import { ScoreChart } from './ScoreChart'

export interface ReplayPanelProps {
  start: string
  hours: number
}

/** Highest-scoring window inside an episode, which carries the feature evidence. */
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

function Summary({ data }: { data: ReplayResponse }) {
  const counts = data.windows.reduce(
    (acc, w) => {
      acc[w.status] += 1
      return acc
    },
    { normal: 0, anomalous: 0, insufficient_data: 0 },
  )
  return (
    <p className="text-sm text-muted">
      {data.windows.length} windows: {counts.normal} normal, {counts.anomalous} above threshold,{' '}
      {counts.insufficient_data} with insufficient data · {data.episodes.length} alarm episode
      {data.episodes.length === 1 ? '' : 's'} (≥{data.persistence} consecutive windows)
    </p>
  )
}

function EvidenceTable({ window }: { window: WindowResult }) {
  return (
    <table className="mt-2 w-full text-sm">
      <caption className="text-left text-muted">
        Largest deviations from the training distribution in the peak window (
        {formatNaive(window.window_start)}). Deviation is evidence for review, not a root cause.
      </caption>
      <thead>
        <tr className="text-left text-muted">
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
            Robust z
          </th>
        </tr>
      </thead>
      <tbody>
        {window.evidence.map((e) => (
          <tr key={e.feature} className="border-t border-border">
            <td className="py-1 pr-4 font-mono">{e.feature}</td>
            <td className="py-1 pr-4">{e.value === null ? 'n/a' : e.value.toFixed(3)}</td>
            <td className="py-1 pr-4">{e.training_median.toFixed(3)}</td>
            <td className="py-1">{e.robust_z === null ? 'n/a' : e.robust_z.toFixed(1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

/** Replays recorded telemetry through the deployed model and shows scores, alarms and evidence. */
export function ReplayPanel({ start, hours }: ReplayPanelProps) {
  const end = addHours(start, hours)
  const { data, isPending, isError, error, isFetching } = useAnomalyReplay({ start, end })
  const [selected, setSelected] = useState<number | null>(null)

  // Non-2xx responses throw (see api/http.ts), so a resolved response is always the 200 shape.
  const replay = data?.status === 200 ? data.data : undefined
  const selectedEpisode =
    replay && selected !== null ? (replay.episodes[selected] ?? undefined) : undefined
  const evidenceWindow = useMemo(
    () => (replay && selectedEpisode ? peakWindow(replay.windows, selectedEpisode) : undefined),
    [replay, selectedEpisode],
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
    <div className="flex flex-col gap-4" aria-busy={isFetching}>
      <Summary data={replay} />
      <ScoreChart
        windows={replay.windows}
        threshold={replay.threshold}
        episodes={replay.episodes}
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
      <section aria-labelledby="episodes-title">
        <h3 id="episodes-title" className="font-medium">
          Alarm episodes
        </h3>
        {replay.episodes.length === 0 ? (
          <p className="text-sm text-muted">No alarms raised in this range.</p>
        ) : (
          <table className="mt-2 w-full text-sm">
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
                  Peak score
                </th>
                <th scope="col" className="font-normal">
                  <span className="sr-only">Evidence</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {replay.episodes.map((e, i) => (
                <Fragment key={e.start}>
                  <tr className="border-t border-border">
                    <td className="py-1 pr-4">{formatNaive(e.raised_at)}</td>
                    <td className="py-1 pr-4">{formatNaive(e.start)}</td>
                    <td className="py-1 pr-4">{formatNaive(e.end)}</td>
                    <td className="py-1 pr-4">{e.peak_score.toFixed(3)}</td>
                    <td className="py-1">
                      <button
                        type="button"
                        className="text-accent underline"
                        aria-pressed={selected === i}
                        onClick={() => setSelected(selected === i ? null : i)}
                      >
                        {selected === i ? 'Hide evidence' : 'Show evidence'}
                      </button>
                    </td>
                  </tr>
                  {selected === i && evidenceWindow && (
                    <tr>
                      <td colSpan={5} className="pb-3">
                        <EvidenceTable window={evidenceWindow} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        )}
      </section>
    </div>
  )
}
