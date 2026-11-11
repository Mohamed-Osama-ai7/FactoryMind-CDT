import type { QueryClient } from '@tanstack/react-query'
import { createRootRouteWithContext, Link, Outlet } from '@tanstack/react-router'
import { useMe } from '../api/generated/incidents/incidents'
import { AuthGate } from '../auth/AuthGate'
import { useAuth } from '../auth/AuthProvider'

export interface RouterContext {
  queryClient: QueryClient
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
})

const LINK = 'rounded px-2 py-1.5 text-muted hover:bg-border hover:text-ink'
const ACTIVE = { className: 'bg-border text-ink' }

function Shell() {
  const { signOut } = useAuth()
  const me = useMe()
  const permissions = new Set(me.data?.data.permissions ?? [])

  return (
    <div className="flex min-h-full">
      <aside className="flex w-56 shrink-0 flex-col border-r border-border bg-panel p-4">
        <p className="text-lg font-semibold tracking-tight">FactoryMind CDT</p>
        <p className="mt-1 text-xs text-muted">Advisory decision support</p>
        <nav aria-label="Primary" className="mt-6 flex flex-col gap-1 text-sm">
          <Link to="/" className={LINK} activeProps={ACTIVE}>
            Overview
          </Link>
          {permissions.has('asset:read') && (
            <Link to="/assets" className={LINK} activeProps={ACTIVE}>
              Assets &amp; Data
            </Link>
          )}
          {permissions.has('asset:read') && (
            <Link to="/knowledge" className={LINK} activeProps={ACTIVE}>
              Knowledge &amp; KG
            </Link>
          )}
          {permissions.has('asset:read') && (
            <Link to="/twin" className={LINK} activeProps={ACTIVE}>
              Digital Twin
            </Link>
          )}
          {permissions.has('asset:read') && (
            <Link to="/copilot" className={LINK} activeProps={ACTIVE}>
              Copilot
            </Link>
          )}
          {permissions.has('asset:read') && (
            <Link to="/edge" className={LINK} activeProps={ACTIVE}>
              Edge Preflight
            </Link>
          )}
          {permissions.has('anomaly:read') && (
            <Link to="/anomaly" className={LINK} activeProps={ACTIVE}>
              Anomaly monitor
            </Link>
          )}
          {permissions.has('anomaly:read') && (
            <Link to="/models" className={LINK} activeProps={ACTIVE}>
              ML Models &amp; OOD
            </Link>
          )}
          {permissions.has('incident:read') && (
            <Link to="/incidents" className={LINK} activeProps={ACTIVE}>
              Incidents
            </Link>
          )}
          {permissions.has('audit:read') && (
            <Link to="/audit" className={LINK} activeProps={ACTIVE}>
              Audit trail
            </Link>
          )}
        </nav>
        <div className="mt-auto pt-6 text-xs text-muted">
          {me.isPending && <p role="status">Loading identity…</p>}
          {me.isError && <p role="alert">Access denied for this account.</p>}
          {me.data && (
            <dl className="space-y-0.5" aria-label="Signed-in user">
              <dt className="sr-only">User</dt>
              <dd className="font-medium text-ink">{me.data.data.actor_id}</dd>
              <dt className="sr-only">Tenant</dt>
              <dd>{me.data.data.tenant_id}</dd>
              <dt className="sr-only">Roles</dt>
              <dd>{me.data.data.roles.length ? me.data.data.roles.join(', ') : 'no roles'}</dd>
            </dl>
          )}
          <button type="button" onClick={signOut} className="mt-3 underline">
            Sign out
          </button>
        </div>
      </aside>
      <main className="flex-1 p-6">
        <Outlet />
      </main>
    </div>
  )
}

function RootLayout() {
  return (
    <AuthGate>
      <Shell />
    </AuthGate>
  )
}
