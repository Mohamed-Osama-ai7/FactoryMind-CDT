import { useHealth } from '../../api/generated/system/system'

/** Backend connectivity panel: shows API version, environment and the operating mode. */
export function SystemStatus() {
  const { data, isPending, isError } = useHealth()

  if (isPending) {
    return <StatusCard label="Checking backend…" tone="muted" />
  }
  if (isError) {
    return <StatusCard label="Backend unreachable" tone="critical" />
  }

  const health = data.data
  return (
    <StatusCard label="Backend online" tone="ok">
      <dl className="mt-3 grid grid-cols-[max-content_1fr] gap-x-6 gap-y-1 text-sm">
        <dt className="text-muted">API version</dt>
        <dd className="font-mono">{health.version}</dd>
        <dt className="text-muted">Environment</dt>
        <dd>{health.environment}</dd>
        <dt className="text-muted">Operating mode</dt>
        <dd>{health.operating_mode}</dd>
      </dl>
    </StatusCard>
  )
}

const TONE_CLASS = {
  ok: 'bg-ok',
  critical: 'bg-critical',
  muted: 'bg-muted',
} as const

function StatusCard({
  label,
  tone,
  children,
}: {
  label: string
  tone: keyof typeof TONE_CLASS
  children?: React.ReactNode
}) {
  return (
    <div role="status" className="rounded-lg border border-border bg-panel p-4">
      <p className="flex items-center gap-2 font-medium">
        <span
          aria-hidden="true"
          className={`inline-block size-2.5 rounded-full ${TONE_CLASS[tone]}`}
        />
        {label}
      </p>
      {children}
    </div>
  )
}
