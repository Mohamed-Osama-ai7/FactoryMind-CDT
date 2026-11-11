import { type FormEvent, useId, useState } from 'react'
import { describeApiError } from '../../api/errors'
import { useListCopilotTools, useQueryCopilot } from '../../api/generated/copilot/copilot'
import { useMe } from '../../api/generated/incidents/incidents'

const INCIDENT_HEX_RE = /^[a-f0-9]{32}$/
type ValidationField = 'question' | 'incidentId' | 'leakRate' | 'horizonHours'

function parseFiniteBounded(
  raw: string,
  label: string,
  lo: number,
  hi: number,
): { value: number; error: null } | { value: null; error: string } {
  const trimmed = raw.trim()
  if (!trimmed) {
    return { value: null, error: `${label} is required.` }
  }
  const n = Number(trimmed)
  if (!Number.isFinite(n) || n < lo || n > hi) {
    return { value: null, error: `${label} must be between ${lo} and ${hi}.` }
  }
  return { value: n, error: null }
}

/** Permission-scoped, red-team-guarded, citation-grounded Industrial Copilot workspace. */
export function CopilotWorkspace() {
  const idPrefix = useId()
  const me = useMe()
  const canRead = me.data?.data.permissions.includes('asset:read') ?? false

  const [question, setQuestion] = useState(
    'Summarize unloaded TP3 air leakage risk and compare repair options.',
  )
  const [incidentId, setIncidentId] = useState('')
  const [leakRate, setLeakRate] = useState('0.35')
  const [horizonHours, setHorizonHours] = useState('8')
  const [validation, setValidation] = useState<{
    field: ValidationField
    message: string
  } | null>(null)

  const toolsQuery = useListCopilotTools({ query: { enabled: canRead } })
  const mutation = useQueryCopilot()

  if (me.isPending) return <p role="status">Checking copilot permissions…</p>
  if (me.isError) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Controlled Industrial Copilot (Read-Only Advisory)
        </h1>
        <p role="alert" className="mt-2 rounded border border-critical p-3 text-sm text-critical">
          Could not verify copilot permissions: {describeApiError(me.error)}
        </p>
      </section>
    )
  }
  if (!canRead) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Controlled Industrial Copilot (Read-Only Advisory)
        </h1>
        <p
          role="status"
          className="mt-2 rounded border border-border bg-panel p-4 text-sm text-muted"
        >
          You do not hold the <span className="font-mono">asset:read</span> permission required to
          query the Industrial Copilot orchestrator.
        </p>
      </section>
    )
  }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    const cleanQ = question.trim()
    if (!cleanQ) {
      setValidation({ field: 'question', message: 'Enter a non-empty engineering question.' })
      return
    }
    const cleanInc = incidentId.trim()
    if (cleanInc && !INCIDENT_HEX_RE.test(cleanInc)) {
      setValidation({
        field: 'incidentId',
        message: 'Incident ID must be blank or a 32-character lowercase hex identifier.',
      })
      return
    }
    const leak = parseFiniteBounded(leakRate, 'Leak loss rate (bar/min)', 0, 1.5)
    if (leak.value === null) {
      setValidation({ field: 'leakRate', message: leak.error })
      return
    }
    const horizon = parseFiniteBounded(horizonHours, 'Horizon (hours)', 1, 72)
    if (horizon.value === null) {
      setValidation({ field: 'horizonHours', message: horizon.error })
      return
    }

    setValidation(null)
    mutation.mutate({
      data: {
        question: cleanQ,
        asset_id: 'metropt3-apu-1',
        ...(cleanInc ? { incident_id: cleanInc } : {}),
        leak_rate_bar_per_min: leak.value,
        planning_horizon_hours: horizon.value,
        use_llm_adapter: false,
      },
    })
  }

  const toolsData = toolsQuery.data?.status === 200 ? toolsQuery.data.data : undefined
  const reply = mutation.data?.status === 200 ? mutation.data.data : undefined
  const errorAlertId = `${idPrefix}-validation`

  return (
    <section aria-labelledby={`${idPrefix}-title`} className="flex max-w-5xl flex-col gap-6">
      <header>
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Controlled Industrial Copilot (Read-Only Advisory)
        </h1>
        <p className="mt-1 text-sm text-muted">
          Synthesizes briefings strictly from authorized incident evidence, Hybrid RAG docs, typed
          KG edges, digital-twin projections, and option rankings. Possesses zero actuating tools.
        </p>
      </header>

      <section aria-labelledby={`${idPrefix}-form`} className="flex flex-col gap-3">
        <h2 id={`${idPrefix}-form`} className="text-lg font-medium">
          Ask grounded advisory question
        </h2>
        <form
          noValidate
          onSubmit={onSubmit}
          aria-describedby={validation ? errorAlertId : undefined}
          className="flex flex-col gap-3 rounded-lg border border-border bg-panel p-4 text-xs"
        >
          <label htmlFor={`${idPrefix}-q`} className="flex flex-col gap-1">
            <span className="text-muted">Engineering question</span>
            <textarea
              id={`${idPrefix}-q`}
              rows={2}
              maxLength={1000}
              value={question}
              aria-invalid={validation?.field === 'question' ? true : undefined}
              onChange={(e) => {
                setQuestion(e.target.value)
                if (validation) setValidation(null)
              }}
              className="rounded border border-border bg-surface p-2 text-ink"
            />
          </label>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
            <label htmlFor={`${idPrefix}-inc`} className="flex flex-col gap-1">
              <span className="text-muted">Optional incident ID (32-char hex)</span>
              <input
                id={`${idPrefix}-inc`}
                type="text"
                value={incidentId}
                aria-invalid={validation?.field === 'incidentId' ? true : undefined}
                onChange={(e) => {
                  setIncidentId(e.target.value)
                  if (validation) setValidation(null)
                }}
                placeholder="Leave blank for asset overview"
                className="rounded border border-border bg-surface px-2 py-1.5 font-mono text-ink"
              />
            </label>
            <label htmlFor={`${idPrefix}-leak`} className="flex flex-col gap-1">
              <span className="text-muted">Simulated leak rate (bar/min)</span>
              <input
                id={`${idPrefix}-leak`}
                type="number"
                step="any"
                value={leakRate}
                aria-invalid={validation?.field === 'leakRate' ? true : undefined}
                onChange={(e) => {
                  setLeakRate(e.target.value)
                  if (validation) setValidation(null)
                }}
                className="rounded border border-border bg-surface px-2 py-1.5 text-ink"
              />
            </label>
            <label htmlFor={`${idPrefix}-horizon`} className="flex flex-col gap-1">
              <span className="text-muted">Planning horizon (hours)</span>
              <input
                id={`${idPrefix}-horizon`}
                type="number"
                step="any"
                value={horizonHours}
                aria-invalid={validation?.field === 'horizonHours' ? true : undefined}
                onChange={(e) => {
                  setHorizonHours(e.target.value)
                  if (validation) setValidation(null)
                }}
                className="rounded border border-border bg-surface px-2 py-1.5 text-ink"
              />
            </label>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-2">
            <button
              type="submit"
              disabled={mutation.isPending}
              className="rounded bg-accent px-3 py-1.5 font-medium text-white disabled:opacity-50"
            >
              {mutation.isPending ? 'Synthesizing…' : 'Ask Copilot'}
            </button>
            <span className="text-muted">
              Mode: offline deterministic verified grounding · zero PLC writes
            </span>
          </div>

          {validation && (
            <p id={errorAlertId} role="alert" className="text-critical">
              {validation.message}
            </p>
          )}
        </form>

        {mutation.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-xs text-critical">
            Copilot query failed: {describeApiError(mutation.error)}
          </p>
        )}

        {reply && (
          <div className="flex flex-col gap-4 rounded-lg border border-border bg-panel p-4 text-xs">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap gap-2 font-mono text-[11px]">
                <span className="rounded border border-border px-2 py-0.5 text-ink">
                  {reply.synthesis_mode}
                </span>
                {reply.safety_refusal_triggered && (
                  <span className="rounded border border-critical/60 bg-critical/10 px-2 py-0.5 text-critical">
                    SAFETY REFUSAL ({(reply.red_team_flags ?? []).join(', ')})
                  </span>
                )}
                {reply.abstained && !reply.safety_refusal_triggered && (
                  <span className="rounded border border-warn/60 bg-warn/10 px-2 py-0.5 text-warn">
                    ABSTAINED ({reply.abstention_reason ?? 'insufficient_evidence'})
                  </span>
                )}
                {!reply.safety_refusal_triggered && (reply.red_team_flags ?? []).length > 0 && (
                  <span className="rounded border border-warn/60 bg-warn/10 px-2 py-0.5 text-warn">
                    FLAGS: {(reply.red_team_flags ?? []).join(', ')}
                  </span>
                )}
              </div>
              <span className="text-muted">{reply.advisory_disclaimer}</span>
            </div>

            <pre className="whitespace-pre-wrap rounded border border-border bg-surface p-3 font-sans text-xs text-ink">
              {reply.answer}
            </pre>

            <p className="text-muted">{reply.causal_vs_correlation_note}</p>

            {reply.citations.length > 0 && (
              <div>
                <h3 className="mb-1.5 font-medium text-muted">
                  Grounded citations ({reply.citations.length})
                </h3>
                <ul className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {reply.citations.map((c) => (
                    <li key={c.citation_id} className="rounded border border-border p-2.5">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-mono font-medium text-ink">
                          [{c.citation_id}] {c.title}
                        </span>
                        <span className="rounded border border-border px-1 py-0.5 font-mono text-[10px] text-muted">
                          {c.provenance_origin}
                        </span>
                      </div>
                      <p className="mt-1 text-muted">{c.detail}</p>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            <div>
              <h3 className="mb-1.5 font-medium text-muted">Tool execution audit</h3>
              <ul className="grid grid-cols-1 gap-1.5 sm:grid-cols-3">
                {reply.tools_executed.map((t) => (
                  <li key={t.tool} className="rounded border border-border p-2">
                    <div className="flex items-center justify-between font-mono">
                      <span className="text-ink">{t.tool}</span>
                      <span className={t.status === 'executed' ? 'text-ok' : 'text-muted'}>
                        {t.status}
                      </span>
                    </div>
                    <p className="mt-0.5 text-[11px] text-muted">{t.summary}</p>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </section>

      <section aria-labelledby={`${idPrefix}-tools`} className="flex flex-col gap-2">
        <h2 id={`${idPrefix}-tools`} className="text-lg font-medium">
          Authorized read-only tool scope
        </h2>
        {toolsQuery.isPending && <p role="status">Loading tool permissions…</p>}
        {toolsQuery.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-xs text-critical">
            Could not load Copilot tool catalog: {describeApiError(toolsQuery.error)}
          </p>
        )}
        {toolsData && (
          <ul className="grid grid-cols-1 gap-2 text-xs sm:grid-cols-2 lg:grid-cols-3">
            {toolsData.tools.map((t) => (
              <li key={t.tool} className="rounded border border-border bg-panel p-3">
                <div className="flex items-center justify-between font-mono">
                  <span className="font-medium text-ink">{t.tool}</span>
                  <span className={t.permitted_for_caller ? 'text-ok' : 'text-muted'}>
                    {t.permitted_for_caller ? 'allowed' : 'denied'}
                  </span>
                </div>
                <p className="mt-1 text-muted">{t.description}</p>
              </li>
            ))}
          </ul>
        )}
      </section>
    </section>
  )
}
