import { useQueryClient } from '@tanstack/react-query'
import { type FormEvent, useId, useState } from 'react'
import { describeApiError } from '../../api/errors'
import {
  getAuditEventsQueryKey,
  getGetIncidentQueryKey,
  useCreateOutcomeVerification,
  useMe,
} from '../../api/generated/incidents/incidents'
import type { IncidentDetail, OutcomeVerificationOutOutcome } from '../../api/generated/model'
import { formatNaive } from '../../lib/naiveTime'

export interface OutcomeVerificationPanelProps {
  incident: IncidentDetail
}

const OUTCOMES: { value: OutcomeVerificationOutOutcome; label: string }[] = [
  { value: 'confirmed_leak', label: 'Confirmed air leak repaired/found' },
  { value: 'sensor_fault_confirmed', label: 'Confirmed sensor / instrumentation fault' },
  { value: 'high_demand_confirmed', label: 'Confirmed heavy operational demand (no leak)' },
  { value: 'false_positive', label: 'False positive / spurious alert' },
  { value: 'inconclusive', label: 'Inconclusive field check' },
]

const OUTCOME_LABELS = new Map(OUTCOMES.map((o) => [o.value, o.label]))

/** Append-only post-review maintenance and physical outcome verification workspace. */
export function OutcomeVerificationPanel({ incident }: OutcomeVerificationPanelProps) {
  const idPrefix = useId()
  const me = useMe()
  const queryClient = useQueryClient()
  const [outcome, setOutcome] = useState<OutcomeVerificationOutOutcome>('confirmed_leak')
  const [action, setAction] = useState('')
  const [verifiedAt, setVerifiedAt] = useState(incident.episode_end.slice(0, 16))
  const [notes, setNotes] = useState('')
  const [validationError, setValidationError] = useState<string | null>(null)

  const mutation = useCreateOutcomeVerification({
    mutation: {
      onSuccess: () => {
        setAction('')
        setNotes('')
        setValidationError(null)
        void queryClient.invalidateQueries({ queryKey: getGetIncidentQueryKey(incident.id) })
        void queryClient.invalidateQueries({ queryKey: getAuditEventsQueryKey() })
      },
    },
  })

  const canCreate = me.data?.data.permissions.includes('outcome:create') ?? false
  const outcomes = incident.outcomes ?? []

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    const cleanAction = action.trim()
    const cleanNotes = notes.trim()
    if (!cleanAction || !cleanNotes) {
      setValidationError('Please enter both a maintenance summary and verification notes.')
      return
    }
    setValidationError(null)
    mutation.mutate({
      incidentId: incident.id,
      data: {
        outcome,
        maintenance_action: cleanAction,
        verified_at: verifiedAt ? `${verifiedAt.slice(0, 16)}:00` : null,
        notes: cleanNotes,
      },
    })
  }

  return (
    <section
      aria-labelledby={`${idPrefix}-title`}
      className="flex flex-col gap-3 rounded-lg border border-border bg-panel p-4 text-sm"
    >
      <h2 id={`${idPrefix}-title`} className="font-medium">
        Post-review outcome verification ({outcomes.length})
      </h2>
      <p className="text-xs text-muted">
        Append-only human verification records comparing incident diagnoses against maintenance
        checks (separate from observed telemetry; FactoryMind never actuates equipment).
      </p>

      {outcomes.length === 0 ? (
        <p className="text-xs text-muted">No physical outcome verification logged yet.</p>
      ) : (
        <ol className="space-y-2 text-xs">
          {outcomes.map((o) => (
            <li key={o.id} className="rounded border border-border p-2.5">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-medium text-ink">
                  {OUTCOME_LABELS.get(o.outcome) ?? o.outcome} · {o.actor_id}
                </span>
                <span className="text-muted">
                  {o.verified_at
                    ? `${formatNaive(o.verified_at)} (train local)`
                    : `${formatNaive(o.created_at)} (recorded UTC)`}{' '}
                  · Human observation
                </span>
              </div>
              <p className="mt-1 font-medium text-ink">{o.maintenance_action}</p>
              <p className="mt-0.5 text-muted">{o.notes}</p>
            </li>
          ))}
        </ol>
      )}

      {canCreate && (
        <form
          onSubmit={onSubmit}
          className="flex flex-col gap-3 rounded border border-border p-3 text-xs"
        >
          <p className="font-medium text-ink">Log maintenance / inspection outcome (append-only)</p>
          <div className="flex flex-wrap gap-3">
            <label htmlFor={`${idPrefix}-outcome`} className="flex flex-col gap-1">
              <span className="text-muted">Verified outcome</span>
              <select
                id={`${idPrefix}-outcome`}
                value={outcome}
                onChange={(e) => {
                  setOutcome(e.target.value as OutcomeVerificationOutOutcome)
                  if (mutation.isSuccess || mutation.isError) mutation.reset()
                }}
                className="rounded border border-border bg-surface px-2 py-1 text-ink"
              >
                {OUTCOMES.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </label>
            <label htmlFor={`${idPrefix}-ts`} className="flex flex-col gap-1">
              <span className="text-muted">Verified at (train local time)</span>
              <input
                id={`${idPrefix}-ts`}
                type="datetime-local"
                value={verifiedAt}
                onChange={(e) => {
                  setVerifiedAt(e.target.value)
                  if (mutation.isSuccess || mutation.isError) mutation.reset()
                }}
                className="rounded border border-border bg-surface px-2 py-1 text-ink"
              />
            </label>
          </div>
          <label htmlFor={`${idPrefix}-action`} className="flex flex-col gap-1">
            <span className="text-muted">Physical check / maintenance summary</span>
            <input
              id={`${idPrefix}-action`}
              type="text"
              required
              maxLength={256}
              value={action}
              onChange={(e) => {
                setAction(e.target.value)
                if (validationError) setValidationError(null)
                if (mutation.isSuccess || mutation.isError) mutation.reset()
              }}
              placeholder="e.g., Inspected purge hose fitting and verified stationary pressure hold"
              className="rounded border border-border bg-surface px-2 py-1 text-ink"
            />
          </label>
          <label htmlFor={`${idPrefix}-notes`} className="flex flex-col gap-1">
            <span className="text-muted">Verification notes</span>
            <textarea
              id={`${idPrefix}-notes`}
              required
              rows={2}
              maxLength={2000}
              value={notes}
              onChange={(e) => {
                setNotes(e.target.value)
                if (validationError) setValidationError(null)
                if (mutation.isSuccess || mutation.isError) mutation.reset()
              }}
              placeholder="Record physical evidence found, parts replaced, or why alert differed..."
              className="rounded border border-border bg-surface p-2 text-ink"
            />
          </label>
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={mutation.isPending}
              className="rounded bg-accent px-3 py-1.5 font-medium text-white disabled:opacity-50"
            >
              {mutation.isPending ? 'Saving…' : 'Record outcome'}
            </button>
            {validationError && (
              <span role="alert" className="text-critical">
                {validationError}
              </span>
            )}
            {mutation.isSuccess && (
              <span role="status" className="text-ok">
                Outcome recorded and audited.
              </span>
            )}
            {mutation.isError && (
              <span role="alert" className="text-critical">
                {describeApiError(mutation.error)}
              </span>
            )}
          </div>
        </form>
      )}
    </section>
  )
}
