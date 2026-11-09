import type { QueryClient } from '@tanstack/react-query'
import { createRootRouteWithContext, Link, Outlet } from '@tanstack/react-router'

export interface RouterContext {
  queryClient: QueryClient
}

export const Route = createRootRouteWithContext<RouterContext>()({
  component: RootLayout,
})

function RootLayout() {
  return (
    <div className="flex min-h-full">
      <aside className="w-56 shrink-0 border-r border-border bg-panel p-4">
        <p className="text-lg font-semibold tracking-tight">FactoryMind CDT</p>
        <p className="mt-1 text-xs text-muted">Advisory decision support</p>
        <nav aria-label="Primary" className="mt-6 flex flex-col gap-1 text-sm">
          <Link
            to="/"
            className="rounded px-2 py-1.5 text-muted hover:bg-border hover:text-ink"
            activeProps={{ className: 'bg-border text-ink' }}
          >
            Overview
          </Link>
        </nav>
      </aside>
      <main className="flex-1 p-6">
        <Outlet />
      </main>
    </div>
  )
}
