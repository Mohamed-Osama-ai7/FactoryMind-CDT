import { useQueryClient } from '@tanstack/react-query'
import { type FormEvent, useId, useState } from 'react'
import { describeApiError } from '../../api/errors'
import {
  getGetIncidentEvidencePackageQueryKey,
  getGetIncidentQueryKey,
  useCreateObservation,
  useMe,
} from '../../api/generated/incidents/incidents'
import type { HumanObservationOutCategory } from '../../api/generated/model'

export interface ObservationFormProps {
  incidentId: string
  defaultTimestamp: string
}

const CATEGORIES: { value: HumanObservationOutCategory; label: string }[] = [
  { value: 'field_inspection', label: 'Field inspection' },
  { value: 'sensor_check', label: 'Sensor / calibration check' },
  { value: 'maintenance_note', label: 'Maintenance note' },
  { value: 'process_context', label: 'Process / train duty context' },
]

/** Append-only form for engineers/approvers to log human field observations on an incident. */
export function ObservationForm({ incidentId, defaultTimestamp }: ObservationFormProps) {
  const idPrefix = useId()
  const me = useMe()
  const queryClient = useQueryClient()
  const [category, setCategory] = useState<HumanObservationOutCategory>('field_inspection')
  const [observedAt, setObservedAt] = useState(defaultTimestamp.slice(0, 16))
  const [note, setNote] = useState('')

  const mutation = useCreateObservation({
    mutation: {
      onSuccess: () => {
        setNote('')
        void queryClient.invalidateQueries({ queryKey: getGetIncidentQueryKey(incidentId) })
        void queryClient.invalidateQueries({
          queryKey: getGetIncidentEvidencePackageQueryKey(incidentId),
        })
      },
    },
  })

  const canCreate = me.data?.data.permissions.includes('observation:create') ?? false
  if (!canCreate) return null

  const onSubmit = (event: FormEvent) => {
    event.preventDefault()
    const trimmed = note.trim()
    if (!trimmed) return
    mutation.mutate({
      incidentId,
      data: {
        category,
        note: trimmed,
        observed_at: observedAt ? `${observedAt.slice(0, 16)}:00` : null,
      },
    })
  }

  return (
    <form
      onSubmit={onSubmit}
      className="mt-3 flex flex-col gap-3 rounded border border-border p-3 text-sm"
    >
      <h4 className="font-medium">Record human field observation (append-only)</h4>
      <div className="flex flex-wrap gap-3 text-xs">
        <label htmlFor={`${idPrefix}-cat`} className="flex flex-col gap-1">
          <span className="text-muted">Category</span>
          <select
            id={`${idPrefix}-cat`}
            value={category}
            onChange={(e) => setCategory(e.target.value as HumanObservationOutCategory)}
            className="rounded border border-border bg-surface px-2 py-1 text-ink"
          >
            {CATEGORIES.map((c) => (
              <option key={c.value} value={c.value}>
                {c.label}
              </option>
            ))}
          </select>
        </label>
        <label htmlFor={`${idPrefix}-ts`} className="flex flex-col gap-1">
          <span className="text-muted">Observed at (train local time)</span>
          <input
            id={`${idPrefix}-ts`}
            type="datetime-local"
            value={observedAt}
            onChange={(e) => setObservedAt(e.target.value)}
            className="rounded border border-border bg-surface px-2 py-1 text-ink"
          />
        </label>
      </div>
      <label htmlFor={`${idPrefix}-note`} className="flex flex-col gap-1 text-xs">
        <span className="text-muted">Observation note</span>
        <textarea
          id={`${idPrefix}-note`}
          required
          rows={3}
          maxLength={2000}
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="Document inspected seals, gauge checks, valve behaviour, or operating load context..."
          className="rounded border border-border bg-surface p-2 text-ink"
        />
      </label>
      <div className="flex items-center gap-3">
        <button
          type="submit"
          disabled={mutation.isPending}
          className="rounded bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
        >
          {mutation.isPending ? 'Recording…' : 'Record observation'}
        </button>
        {mutation.isSuccess && (
          <span role="status" className="text-xs text-ok">
            Observation recorded and audited.
          </span>
        )}
        {mutation.isError && (
          <span role="alert" className="text-xs text-critical">
            {describeApiError(mutation.error)}
          </span>
        )}
      </div>
    </form>
  )
}
