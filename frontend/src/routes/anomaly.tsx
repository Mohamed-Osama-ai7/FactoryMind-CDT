import { createFileRoute, useNavigate } from '@tanstack/react-router'
import { type FormEvent, useState } from 'react'
import { describeApiError } from '../api/errors'
import { useAnomalyModel } from '../api/generated/anomaly/anomaly'
import { ModelCard } from '../features/anomaly/ModelCard'
import { ReplayPanel } from '../features/anomaly/ReplayPanel'
import { addHours, formatNaive, fromEpochMs, isNaiveTimestamp, toEpochMs } from '../lib/naiveTime'

const DURATIONS = [6, 24, 72, 168] as const
const DEFAULT_START = '2020-06-05T00:00:00'
const DEFAULT_HOURS = 24

// Optional so plain links to /anomaly work; defaults are applied in the component.
interface AnomalySearch {
  start?: string
  hours?: number
}

export const Route = createFileRoute('/anomaly')({
  validateSearch: (search: Record<string, unknown>): AnomalySearch => {
    const start = typeof search.start === 'string' ? search.start : ''
    const hours = Number(search.hours)
    return {
      start: isNaiveTimestamp(start) ? fromEpochMs(toEpochMs(start)) : undefined,
      hours: (DURATIONS as readonly number[]).includes(hours) ? hours : undefined,
    }
  },
  component: AnomalyPage,
})

function AnomalyPage() {
  const search = Route.useSearch()
  const start = search.start ?? DEFAULT_START
  const hours = search.hours ?? DEFAULT_HOURS
  const navigate = useNavigate({ from: Route.fullPath })
  const model = useAnomalyModel()
  const [draft, setDraft] = useState(start.slice(0, 16))

  const go = (nextStart: string, nextHours: number) => {
    void navigate({ search: { start: nextStart, hours: nextHours } })
  }

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    if (isNaiveTimestamp(draft)) go(`${draft.slice(0, 16)}:00`, hours)
  }

  return (
    <section aria-labelledby="anomaly-title" className="flex max-w-5xl flex-col gap-6">
      <header>
        <h1 id="anomaly-title" className="text-2xl font-semibold">
          Compressor anomaly monitor
        </h1>
        <p className="mt-1 text-sm text-muted">
          Replays recorded MetroPT-3 air-compressor telemetry through the deployed detector. Alarms
          are advisory and require human review; FactoryMind never controls equipment.
        </p>
      </header>

      {model.isPending && <p role="status">Loading model…</p>}
      {model.isError && (
        <p role="alert" className="rounded border border-critical p-3 text-sm">
          Model unavailable: {describeApiError(model.error)}
        </p>
      )}
      {model.data && <ModelCard model={model.data.data} />}

      {model.data && (
        <section
          aria-labelledby="replay-title"
          className="rounded-lg border border-border bg-panel p-4"
        >
          <h2 id="replay-title" className="font-medium">
            Replay
          </h2>
          <form onSubmit={onSubmit} className="mt-3 flex flex-wrap items-end gap-3 text-sm">
            <label className="flex flex-col gap-1">
              <span className="text-muted">Start (train local time)</span>
              <input
                type="datetime-local"
                value={draft}
                min={model.data.data.replay_available_from?.slice(0, 16)}
                max={model.data.data.replay_available_to?.slice(0, 16)}
                onChange={(e) => setDraft(e.target.value)}
                className="rounded border border-border bg-surface px-2 py-1"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-muted">Duration</span>
              <select
                value={hours}
                onChange={(e) => go(start, Number(e.target.value))}
                className="rounded border border-border bg-surface px-2 py-1"
              >
                {DURATIONS.map((h) => (
                  <option key={h} value={h}>
                    {h < 24 ? `${h} h` : `${h / 24} d`}
                  </option>
                ))}
              </select>
            </label>
            <button type="submit" className="rounded bg-accent px-3 py-1.5 font-medium text-white">
              Replay
            </button>
          </form>
          <fieldset className="mt-3 flex flex-wrap gap-2 text-sm">
            <legend className="sr-only">Operator report presets</legend>
            {model.data.data.operator_reports.map((r) => {
              const presetStart = addHours(r.start.slice(0, 13).concat(':00:00'), -12)
              return (
                <button
                  key={r.report_id}
                  type="button"
                  className="rounded border border-border px-2 py-1 hover:bg-border"
                  onClick={() => {
                    setDraft(presetStart.slice(0, 16))
                    go(presetStart, 72)
                  }}
                >
                  Around {r.report_id} ({formatNaive(r.start)})
                </button>
              )
            })}
          </fieldset>
          <div className="mt-4">
            <ReplayPanel start={start} hours={hours} />
          </div>
        </section>
      )}
    </section>
  )
}
