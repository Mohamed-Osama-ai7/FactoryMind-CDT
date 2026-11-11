import { useQueryClient } from '@tanstack/react-query'
import { type FormEvent, useId, useState } from 'react'
import { describeApiError } from '../../api/errors'
import {
  getGetIncidentQueryKey,
  getListIncidentsQueryKey,
  useCreateDecision,
  useMe,
} from '../../api/generated/incidents/incidents'
import type { DecisionCreateDecision, IncidentDetail } from '../../api/generated/model'
import { formatNaive } from '../../lib/naiveTime'

export interface DecisionPanelProps {
  incident: IncidentDetail
}

const DECISIONS: { value: DecisionCreateDecision; label: string; hint: string }[] = [
  {
    value: 'confirm',
    label: 'Confirm fault',
    hint: 'Close as confirmed (immutable status; post-review outcomes may still be appended)',
  },
  {
    value: 'dismiss',
    label: 'Dismiss alarm',
    hint: 'Close as dismissed (false alarm or explained non-fault behaviour)',
  },
  {
    value: 'defer',
    label: 'Defer pending check',
    hint: 'Keep open and bump version while awaiting physical inspection',
  },
]

/** Human review decision panel enforcing separation of duties and optimistic concurrency. */
export function DecisionPanel({ incident }: DecisionPanelProps) {
  const idPrefix = useId()
  const me = useMe()
  const queryClient = useQueryClient()
  const [decision, setDecision] = useState<DecisionCreateDecision>('confirm')
  const [rationale, setRationale] = useState('')

  const mutation = useCreateDecision({
    mutation: {
      onSuccess: () => {
        setRationale('')
        void queryClient.invalidateQueries({ queryKey: getGetIncidentQueryKey(incident.id) })
        void queryClient.invalidateQueries({ queryKey: getListIncidentsQueryKey() })
      },
    },
  })

  const profile = me.data?.data
  const canDecide = profile?.permissions.includes('decision:create') ?? false
  const isCreator = profile?.actor_id === incident.created_by
  const canSubmit = incident.status === 'open' && canDecide && !isCreator

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    const trimmed = rationale.trim()
    if (!trimmed || !canSubmit) return
    mutation.mutate({
      incidentId: incident.id,
      data: {
        decision,
        rationale: trimmed,
        expected_version: incident.version,
      },
    })
  }

  return (
    <section
      aria-labelledby={`${idPrefix}-title`}
      className="flex flex-col gap-3 rounded-lg border border-border bg-panel p-4 text-sm"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 id={`${idPrefix}-title`} className="font-medium">
          Human review decisions ({incident.decisions.length})
        </h2>
        <span className="rounded border border-border px-2 py-0.5 text-xs text-muted">
          Status: <strong className="text-ink">{incident.status}</strong> · v{incident.version}
        </span>
      </div>

      {incident.decisions.length === 0 ? (
        <p className="text-xs text-muted">No review decisions recorded yet.</p>
      ) : (
        <ol className="space-y-2 text-xs">
          {incident.decisions.map((d) => (
            <li key={d.id} className="rounded border border-border p-2.5">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="font-medium text-ink">
                  {d.decision.toUpperCase()} by {d.actor_id} (saw v{d.incident_version})
                </span>
                <span className="text-muted">{formatNaive(d.created_at)}</span>
              </div>
              <p className="mt-1 text-ink">{d.rationale}</p>
            </li>
          ))}
        </ol>
      )}

      {incident.status === 'open' && canDecide && isCreator && (
        <p role="status" className="rounded border border-warn/60 p-2.5 text-xs text-warn">
          Separation of duties active: you created this incident ({incident.created_by}), so a
          different approver must confirm, dismiss, or defer it.
        </p>
      )}

      {canSubmit && (
        <form
          onSubmit={onSubmit}
          className="flex flex-col gap-3 rounded border border-border p-3 text-xs"
        >
          <p className="font-medium text-ink">
            Record review decision (append-only, human-in-loop)
          </p>
          <fieldset className="flex flex-col gap-1.5">
            <legend className="text-muted">Decision action</legend>
            {DECISIONS.map((opt) => (
              <label key={opt.value} className="flex items-start gap-2">
                <input
                  type="radio"
                  name={`${idPrefix}-decision`}
                  value={opt.value}
                  checked={decision === opt.value}
                  onChange={() => setDecision(opt.value)}
                  className="mt-0.5"
                />
                <span>
                  <strong className="text-ink">{opt.label}</strong> —{' '}
                  <span className="text-muted">{opt.hint}</span>
                </span>
              </label>
            ))}
          </fieldset>
          <label htmlFor={`${idPrefix}-rationale`} className="flex flex-col gap-1">
            <span className="text-muted">Engineering rationale</span>
            <textarea
              id={`${idPrefix}-rationale`}
              required
              rows={3}
              maxLength={2000}
              value={rationale}
              onChange={(e) => setRationale(e.target.value)}
              placeholder="Document supporting physical evidence, sensor checks, or defer reason..."
              className="rounded border border-border bg-surface p-2 text-ink"
            />
          </label>
          <div className="flex items-center gap-3">
            <button
              type="submit"
              disabled={mutation.isPending}
              className="rounded bg-accent px-3 py-1.5 font-medium text-white disabled:opacity-50"
            >
              {mutation.isPending ? 'Recording…' : 'Submit review decision'}
            </button>
            {mutation.isSuccess && (
              <span role="status" className="text-ok">
                Decision recorded and audited.
              </span>
            )}
            {mutation.isError && (
              <span role="alert" className="text-critical">
                {describeApiError(mutation.error)}
              </span>
            )}
          </div>
          <p className="text-[11px] text-muted">
            Advisory governance record only; never issues automated commands to plant equipment.
          </p>
        </form>
      )}
    </section>
  )
}
