import { useId, useState } from 'react'
import { describeApiError } from '../../api/errors'
import { useAuditEvents, useAuditVerify, useMe } from '../../api/generated/incidents/incidents'

const ACTIONS = [
  'all',
  'incident.create',
  'decision.create',
  'observation.create',
  'outcome.create',
  'authz.denied',
  'authn.failure',
] as const

/** Tenant-scoped tamper-evident audit trail viewer and SHA-256 chain verifier. */
export function AuditViewer() {
  const idPrefix = useId()
  const me = useMe()
  const [actionFilter, setActionFilter] = useState<string>('all')
  const [offset, setOffset] = useState(0)
  const limit = 25

  const canRead = me.data?.data.permissions.includes('audit:read') ?? false
  const eventsQuery = useAuditEvents(
    {
      ...(actionFilter !== 'all' ? { action: actionFilter } : {}),
      limit,
      offset,
    },
    { query: { enabled: canRead } },
  )
  const verifyQuery = useAuditVerify({ query: { enabled: false } })

  if (me.isPending) return <p role="status">Checking audit permissions…</p>
  if (!canRead) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-4xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Audit trail
        </h1>
        <p
          role="status"
          className="mt-2 rounded border border-border bg-panel p-4 text-sm text-muted"
        >
          You do not hold the <span className="font-mono">audit:read</span> permission required to
          view or verify the tenant SHA-256 audit log.
        </p>
      </section>
    )
  }

  const events = eventsQuery.data?.status === 200 ? eventsQuery.data.data.items : []
  const verify = verifyQuery.data?.status === 200 ? verifyQuery.data.data : undefined

  return (
    <section aria-labelledby={`${idPrefix}-title`} className="flex max-w-5xl flex-col gap-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
            Audit trail &amp; hash-chain verification
          </h1>
          <p className="mt-1 text-sm text-muted">
            Append-only per-tenant SHA-256 audit records scoped cleanly by authenticated caller.
          </p>
        </div>
        <button
          type="button"
          disabled={verifyQuery.isFetching}
          onClick={() => void verifyQuery.refetch()}
          className="rounded bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
        >
          {verifyQuery.isFetching ? 'Verifying chain…' : 'Verify SHA-256 hash chain'}
        </button>
      </header>

      {verify && (
        <p
          role="status"
          className={`rounded border p-3 text-xs ${
            verify.valid
              ? 'border-ok/60 bg-ok/10 text-ok'
              : 'border-critical/60 bg-critical/10 text-critical'
          }`}
        >
          {verify.valid
            ? `Hash chain intact across ${verify.events} audit event(s).`
            : `Hash chain break detected at seq ${verify.first_invalid_seq ?? 'unknown'} (${verify.events} total).`}
        </p>
      )}
      {verifyQuery.isError && (
        <p role="alert" className="rounded border border-critical p-2 text-xs text-critical">
          Verification failed: {describeApiError(verifyQuery.error)}
        </p>
      )}

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-muted">Filter action:</span>
        {ACTIONS.map((act) => (
          <button
            key={act}
            type="button"
            aria-pressed={actionFilter === act}
            onClick={() => {
              setActionFilter(act)
              setOffset(0)
            }}
            className={`rounded border px-2 py-0.5 ${
              actionFilter === act
                ? 'border-accent bg-accent/15 font-medium text-ink'
                : 'border-border text-muted hover:bg-border'
            }`}
          >
            {act}
          </button>
        ))}
      </div>

      {eventsQuery.isPending && <p role="status">Loading audit events…</p>}
      {eventsQuery.isError && (
        <p role="alert" className="rounded border border-critical p-3 text-xs text-critical">
          Could not load audit events: {describeApiError(eventsQuery.error)}
        </p>
      )}

      {!eventsQuery.isPending && !eventsQuery.isError && events.length === 0 && (
        <p className="text-sm text-muted">No matching audit events recorded.</p>
      )}

      {events.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-border bg-panel p-3">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-muted">
                <th scope="col" className="pr-3 font-normal">
                  Seq
                </th>
                <th scope="col" className="pr-3 font-normal">
                  Timestamp (UTC)
                </th>
                <th scope="col" className="pr-3 font-normal">
                  Actor &amp; roles
                </th>
                <th scope="col" className="pr-3 font-normal">
                  Action
                </th>
                <th scope="col" className="pr-3 font-normal">
                  Object &amp; outcome
                </th>
                <th scope="col" className="font-normal">
                  Hash &amp; details
                </th>
              </tr>
            </thead>
            <tbody>
              {events.map((ev) => (
                <tr key={ev.seq} className="border-t border-border align-top">
                  <td className="py-1.5 pr-3 font-mono">{ev.seq}</td>
                  <td className="py-1.5 pr-3 font-mono text-muted">{ev.ts}</td>
                  <td className="py-1.5 pr-3">
                    <span className="font-medium text-ink">{ev.actor_id}</span>
                    <span className="block text-muted">{ev.roles}</span>
                  </td>
                  <td className="py-1.5 pr-3 font-mono">{ev.action}</td>
                  <td className="py-1.5 pr-3">
                    <span className="font-mono text-ink">
                      {ev.object_type}:{ev.object_id}
                    </span>
                    <span className={`block ${ev.outcome === 'allowed' ? 'text-ok' : 'text-warn'}`}>
                      {ev.outcome}
                    </span>
                  </td>
                  <td className="py-1.5 font-mono text-[11px] text-muted">
                    {ev.hash.slice(0, 12)}… · {JSON.stringify(ev.details)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="flex items-center gap-3 text-xs">
        <button
          type="button"
          disabled={offset === 0}
          onClick={() => setOffset((o) => Math.max(0, o - limit))}
          className="rounded border border-border px-2.5 py-1 disabled:opacity-40"
        >
          Previous
        </button>
        <span className="text-muted">Offset: {offset}</span>
        <button
          type="button"
          disabled={events.length < limit}
          onClick={() => setOffset((o) => o + limit)}
          className="rounded border border-border px-2.5 py-1 disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </section>
  )
}
