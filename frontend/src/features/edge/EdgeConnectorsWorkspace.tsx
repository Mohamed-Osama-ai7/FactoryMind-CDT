import { type FormEvent, useId, useState } from 'react'
import { describeApiError } from '../../api/errors'
import {
  useGetEdgeSources,
  useGetSupplyChainHardening,
  useValidateEdgeSample,
} from '../../api/generated/edge/edge'
import { useMe } from '../../api/generated/incidents/incidents'

type ValidationField = 'intervalSeconds' | 'disconnects' | 'bufferCapacity'

function parseBoundedInt(
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
  if (!Number.isInteger(n) || n < lo || n > hi) {
    return { value: null, error: `${label} must be an integer between ${lo} and ${hi}.` }
  }
  return { value: n, error: null }
}

function statusBadgeTone(status: string): string {
  if (
    status === 'tamper_or_hash_mismatch' ||
    status === 'integrity_failed' ||
    status === 'manifest_invalid' ||
    status === 'invalid_pointer' ||
    status === 'rejected'
  ) {
    return 'rounded border border-critical px-1.5 py-0.5 font-mono text-[11px] text-critical'
  }
  if (
    status === 'missing' ||
    status === 'no_current_model' ||
    status === 'manifest_missing' ||
    status === 'unattested' ||
    status === 'self_consistent_unsigned' ||
    status === 'not_checked' ||
    status === 'unknown'
  ) {
    return 'rounded border border-border bg-surface px-1.5 py-0.5 font-mono text-[11px] text-muted'
  }
  return 'rounded bg-border px-1.5 py-0.5 font-mono text-[11px]'
}

/** Source descriptors, stateless batch preflight checks, and honest supply-chain check report. */
export function EdgeConnectorsWorkspace() {
  const idPrefix = useId()
  const me = useMe()
  const canRead = me.data?.data.permissions.includes('asset:read') ?? false

  const [sourceId, setSourceId] = useState('metropt3-opcua-ro')
  const [intervalSeconds, setIntervalSeconds] = useState('10')
  const [disconnects, setDisconnects] = useState('0')
  const [bufferCapacity, setBufferCapacity] = useState('4')
  const [includeGap, setIncludeGap] = useState(false)
  const [includeUnallowlistedTag, setIncludeUnallowlistedTag] = useState(false)
  const [validation, setValidation] = useState<{
    field: ValidationField
    message: string
  } | null>(null)

  const sourcesQuery = useGetEdgeSources(undefined, { query: { enabled: canRead } })
  const hardeningQuery = useGetSupplyChainHardening({ query: { enabled: canRead } })
  const validateMutation = useValidateEdgeSample()

  if (me.isPending) return <p role="status">Checking edge preflight permissions…</p>
  if (me.isError) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          OT &amp; Edge Source Descriptors + Supply-Chain Checks
        </h1>
        <p role="alert" className="mt-2 rounded border border-critical p-3 text-sm text-critical">
          Could not verify edge preflight permissions: {describeApiError(me.error)}
        </p>
      </section>
    )
  }
  if (!canRead) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          OT &amp; Edge Source Descriptors + Supply-Chain Checks
        </h1>
        <p
          role="status"
          className="mt-2 rounded border border-border bg-panel p-4 text-sm text-muted"
        >
          You do not hold the <span className="font-mono">asset:read</span> permission required to
          inspect edge source descriptors and supply-chain checks.
        </p>
      </section>
    )
  }

  const sourcesData = sourcesQuery.data?.status === 200 ? sourcesQuery.data.data : null
  const hardeningData = hardeningQuery.data?.status === 200 ? hardeningQuery.data.data : null
  const validationResult = validateMutation.data?.status === 200 ? validateMutation.data.data : null
  const sourceOptions = sourcesData
    ? sourcesData.sources.map((src) => ({
        source_id: src.source_id,
        asset_id: src.asset_id,
      }))
    : []
  const activeSource =
    sourceOptions.find((item) => item.source_id === sourceId) ?? sourceOptions[0] ?? null
  const validationId = `${idPrefix}-validation-error`

  function handleValidateSubmit(event: FormEvent) {
    event.preventDefault()
    const selected = activeSource
    if (!selected) {
      return
    }
    const parsedStep = parseBoundedInt(intervalSeconds, 'Nominal step (s)', 1, 300)
    if (parsedStep.error !== null) {
      setValidation({ field: 'intervalSeconds', message: parsedStep.error })
      return
    }
    const parsedDisc = parseBoundedInt(disconnects, 'Simulated disconnects', 0, 20)
    if (parsedDisc.error !== null) {
      setValidation({ field: 'disconnects', message: parsedDisc.error })
      return
    }
    const parsedCap = parseBoundedInt(bufferCapacity, 'Buffer capacity', 4, 10000)
    if (parsedCap.error !== null) {
      setValidation({ field: 'bufferCapacity', message: parsedCap.error })
      return
    }
    setValidation(null)

    const primarySensor = selected.source_id === 'metropt3-mqtt-ro' ? 'COMP' : 'TP2'
    const baseMs = Date.UTC(2020, 3, 15, 10, 0, 0)
    let cursorMs = baseMs
    const samples = Array.from({ length: 6 }, (_, idx) => {
      const ts = new Date(cursorMs).toISOString()
      const gapStep = includeGap && idx === 2 ? parsedStep.value * 4 : parsedStep.value
      cursorMs += gapStep * 1000
      const sensorValues: Record<string, number> = {
        [primarySensor]: selected.source_id === 'metropt3-mqtt-ro' ? 1.0 : 8.4 + idx * 0.05,
      }
      if (includeUnallowlistedTag) {
        sensorValues.UNALLOWLISTED_SETPOINT_WRITE = 9.9
      }
      return {
        timestamp: ts,
        source_id: selected.source_id,
        sensor_values: sensorValues,
      }
    })

    validateMutation.mutate({
      data: {
        source_id: selected.source_id,
        asset_id: selected.asset_id,
        samples,
        simulate_disconnects: parsedDisc.value,
        buffer_capacity_override: parsedCap.value,
      },
    })
  }

  return (
    <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl space-y-6">
      <header>
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          OT &amp; Edge Source Descriptors + Supply-Chain Checks
        </h1>
        <p className="mt-1 text-sm text-muted">
          Descriptors and stateless batch preflight checks only. No OPC-UA, MQTT, or historian
          client I/O is implemented; endpoint checks inspect URL text only without DNS resolution or
          live plant connections.
        </p>
      </header>

      <div className="rounded border border-border bg-panel p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-semibold">Registered Source Descriptors (Not Connected)</h2>
          {sourcesData && (
            <div className="flex flex-wrap gap-2 text-xs">
              <span className="rounded bg-border px-2 py-0.5 font-mono">
                connector I/O: {sourcesData.connector_io_status ?? 'not_implemented'}
              </span>
              <span className="rounded bg-border px-2 py-0.5 font-mono">
                write capabilities:{' '}
                {sourcesData.write_capabilities ?? 'none_no_connector_io_exists'}
              </span>
            </div>
          )}
        </div>
        {sourcesData?.capability_note && (
          <p className="mt-2 rounded border border-border bg-surface px-3 py-2 text-xs text-muted">
            {sourcesData.capability_note}
          </p>
        )}
        {sourcesQuery.isPending && (
          <p role="status" className="mt-2 text-xs text-muted">
            Loading edge source descriptors…
          </p>
        )}
        {sourcesQuery.isError && (
          <p role="alert" className="mt-2 text-xs text-critical">
            Could not load edge source descriptors: {describeApiError(sourcesQuery.error)}
          </p>
        )}
        {sourcesData && sourcesData.sources.length === 0 && (
          <p role="status" className="mt-2 text-xs text-muted">
            No registered source descriptors available for your asset permissions; preflight
            submission is disabled.
          </p>
        )}
        {sourcesData && sourcesData.sources.length > 0 && (
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="border-b border-border text-muted">
                  <th className="py-1.5 pr-3">Source ID</th>
                  <th className="py-1.5 pr-3">Protocol</th>
                  <th className="py-1.5 pr-3">Endpoint</th>
                  <th className="py-1.5 pr-3">Descriptor State</th>
                  <th className="py-1.5 pr-3">URL Check (No DNS)</th>
                  <th className="py-1.5">Tags / Buffer</th>
                </tr>
              </thead>
              <tbody>
                {sourcesData.sources.map((src) => (
                  <tr key={src.source_id} className="border-b border-border/60">
                    <td className="py-1.5 pr-3 font-mono font-medium">{src.source_id}</td>
                    <td className="py-1.5 pr-3">{src.protocol}</td>
                    <td className="py-1.5 pr-3 font-mono">{src.endpoint_url}</td>
                    <td className="py-1.5 pr-3 font-mono">
                      {src.connection_state ?? 'descriptor_only_not_connected'} (
                      {src.connector_io ?? 'not_implemented'})
                    </td>
                    <td className="py-1.5 pr-3">
                      <span className={statusBadgeTone(src.endpoint_url_policy)}>
                        {src.endpoint_url_policy}
                      </span>{' '}
                      <span className="text-[11px] text-muted">
                        ({src.endpoint_check_scope ?? 'url_string_only_no_dns_resolution'})
                      </span>
                    </td>
                    <td className="py-1.5">
                      {src.node_allowlist_count} tags · {src.max_buffer_records} cap (
                      {src.nominal_cadence_s}s)
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="rounded border border-border bg-panel p-4">
        <h2 className="text-sm font-semibold">Supply-Chain &amp; Runtime Artifact Checks</h2>
        {hardeningQuery.isPending && (
          <p role="status" className="mt-2 text-xs text-muted">
            Running repository and artifact checks…
          </p>
        )}
        {hardeningQuery.isError && (
          <p role="alert" className="mt-2 text-xs text-critical">
            Could not load supply-chain check report: {describeApiError(hardeningQuery.error)}
          </p>
        )}
        {hardeningData && (
          <div className="mt-3 space-y-4">
            <div className="flex flex-wrap gap-2 text-xs">
              <span
                className={statusBadgeTone(hardeningData.license_present ? 'present' : 'missing')}
              >
                LICENSE: {hardeningData.license_present ? 'MIT phrases present' : 'missing/invalid'}
              </span>
              <span
                className={statusBadgeTone(
                  hardeningData.ci_actions_sha_pinned ? 'sha_pinned' : 'tamper_or_hash_mismatch',
                )}
              >
                CI uses syntax:{' '}
                {hardeningData.ci_actions_sha_pinned ? '40-hex commit SHAs' : 'unpinned'}
              </span>
              <span className={statusBadgeTone(hardeningData.runtime_model_attestation)}>
                runtime model: {hardeningData.runtime_model_attestation}
              </span>
              <span className="rounded bg-border px-2 py-0.5 font-mono text-[11px]">
                format: {hardeningData.runtime_model_format}
              </span>
              <span className="rounded border border-border bg-surface px-2 py-0.5 font-mono text-[11px] text-muted">
                loader consults manifest:{' '}
                {hardeningData.runtime_loader_consults_manifest ? 'yes' : 'no'}
              </span>
              <span className="rounded border border-border bg-surface px-2 py-0.5 font-mono text-[11px] text-muted">
                HMAC key configured:{' '}
                {hardeningData.manifest_signature_key_configured ? 'yes' : 'no'}
              </span>
            </div>

            {hardeningData.unpinned_workflow_uses &&
              hardeningData.unpinned_workflow_uses.length > 0 && (
                <p role="alert" className="text-xs text-critical">
                  Unpinned CI actions: {hardeningData.unpinned_workflow_uses.join(', ')}
                </p>
              )}

            <ul className="space-y-1.5 text-xs">
              {hardeningData.attestations.map((entry) => (
                <li
                  key={`${entry.name}-${entry.relative_path}`}
                  className="rounded border border-border px-3 py-2"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="font-medium">{entry.name}</span>
                    <span className={statusBadgeTone(entry.status)}>{entry.status}</span>
                  </div>
                  <p className="mt-0.5 text-muted">
                    <span className="font-mono">{entry.relative_path}</span> — {entry.detail}
                  </p>
                </li>
              ))}
            </ul>

            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">
                Check Scope &amp; Limitations
              </h3>
              <ul className="mt-2 list-disc space-y-1 pl-5 text-xs text-muted">
                {hardeningData.limitations.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </div>

      <form
        noValidate
        onSubmit={handleValidateSubmit}
        className="space-y-4 rounded border border-border bg-panel p-4"
      >
        <div>
          <h2 className="text-sm font-semibold">
            Stateless Batch Preflight Check (No Network I/O)
          </h2>
          <p className="mt-0.5 text-xs text-muted">
            Validates caller-submitted samples against tag allowlists, sampling cadence, and
            in-memory ring-buffer capacity. Does not connect to or read from any source.
          </p>
        </div>
        <div className="grid gap-3 sm:grid-cols-4">
          <div>
            <label htmlFor={`${idPrefix}-source`} className="block text-xs font-medium">
              Source Descriptor
            </label>
            <select
              id={`${idPrefix}-source`}
              value={activeSource?.source_id ?? ''}
              disabled={!activeSource}
              onChange={(e) => {
                setSourceId(e.target.value)
                setValidation(null)
              }}
              className="mt-1 w-full rounded border border-border bg-surface px-2 py-1.5 text-sm disabled:opacity-50"
            >
              {sourceOptions.length === 0 ? (
                <option value="">No valid descriptors</option>
              ) : (
                sourceOptions.map((item) => (
                  <option key={item.source_id} value={item.source_id}>
                    {item.source_id}
                  </option>
                ))
              )}
            </select>
          </div>
          <div>
            <label htmlFor={`${idPrefix}-step`} className="block text-xs font-medium">
              Nominal step (s)
            </label>
            <input
              id={`${idPrefix}-step`}
              type="text"
              inputMode="numeric"
              value={intervalSeconds}
              onChange={(e) => {
                setIntervalSeconds(e.target.value)
                setValidation(null)
              }}
              aria-invalid={validation?.field === 'intervalSeconds'}
              aria-describedby={validation?.field === 'intervalSeconds' ? validationId : undefined}
              className="mt-1 w-full rounded border border-border bg-surface px-2 py-1.5 text-sm font-mono"
            />
          </div>
          <div>
            <label htmlFor={`${idPrefix}-disc`} className="block text-xs font-medium">
              Simulated disconnects
            </label>
            <input
              id={`${idPrefix}-disc`}
              type="text"
              inputMode="numeric"
              value={disconnects}
              onChange={(e) => {
                setDisconnects(e.target.value)
                setValidation(null)
              }}
              aria-invalid={validation?.field === 'disconnects'}
              aria-describedby={validation?.field === 'disconnects' ? validationId : undefined}
              className="mt-1 w-full rounded border border-border bg-surface px-2 py-1.5 text-sm font-mono"
            />
          </div>
          <div>
            <label htmlFor={`${idPrefix}-cap`} className="block text-xs font-medium">
              Buffer capacity
            </label>
            <input
              id={`${idPrefix}-cap`}
              type="text"
              inputMode="numeric"
              value={bufferCapacity}
              onChange={(e) => {
                setBufferCapacity(e.target.value)
                setValidation(null)
              }}
              aria-invalid={validation?.field === 'bufferCapacity'}
              aria-describedby={validation?.field === 'bufferCapacity' ? validationId : undefined}
              className="mt-1 w-full rounded border border-border bg-surface px-2 py-1.5 text-sm font-mono"
            />
          </div>
        </div>

        <div className="flex flex-wrap gap-4 text-xs">
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={includeGap}
              onChange={(e) => setIncludeGap(e.target.checked)}
            />
            Inject uninterpolated sampling gap (&gt;2x nominal)
          </label>
          <label className="flex items-center gap-1.5">
            <input
              type="checkbox"
              checked={includeUnallowlistedTag}
              onChange={(e) => setIncludeUnallowlistedTag(e.target.checked)}
            />
            Inject unallowlisted write tag
          </label>
        </div>

        {validation && (
          <p id={validationId} role="alert" className="text-xs text-critical">
            {validation.message}
          </p>
        )}
        {validateMutation.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-sm text-critical">
            {describeApiError(validateMutation.error)}
          </p>
        )}

        <button
          type="submit"
          disabled={!activeSource || validateMutation.isPending}
          className="rounded border border-border bg-border px-4 py-1.5 text-sm font-medium hover:opacity-90 disabled:opacity-50"
        >
          {validateMutation.isPending ? 'Validating batch…' : 'Run Preflight Batch Check'}
        </button>

        {validationResult && (
          <div className="mt-3 rounded border border-border bg-surface p-3 text-xs space-y-2">
            <div className="flex flex-wrap gap-2">
              <span className="rounded bg-border px-2 py-0.5 font-mono">
                mode: {validationResult.mode ?? 'preflight_validation_only'}
              </span>
              <span className="rounded bg-border px-2 py-0.5 font-mono">
                connector I/O: {String(validationResult.real_connector_io ?? false)}
              </span>
              <span className="rounded bg-border px-2 py-0.5 font-mono">
                cadence: {validationResult.cadence.cadence_status}
              </span>
              <span className="rounded bg-border px-2 py-0.5 font-mono">
                buffer state: {validationResult.buffer.state}
              </span>
              <span className="rounded bg-border px-2 py-0.5 font-mono">
                retained {validationResult.buffer.retained_count}/{validationResult.buffer.capacity}{' '}
                · evicted {validationResult.buffer.evicted_count} · sim reconnects{' '}
                {validationResult.buffer.simulated_reconnect_attempts}
              </span>
            </div>
            {validationResult.warnings && validationResult.warnings.length > 0 && (
              <ul className="list-disc pl-5 text-muted">
                {validationResult.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            )}
            <p className="text-muted">{validationResult.safety_note}</p>
          </div>
        )}
      </form>
    </section>
  )
}
