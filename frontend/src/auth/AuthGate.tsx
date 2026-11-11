import { useQueryClient } from '@tanstack/react-query'
import { type ReactNode, useEffect } from 'react'
import { useAuth } from './AuthProvider'

function Centered({ children }: { children: ReactNode }) {
  return (
    <main className="mx-auto flex min-h-full max-w-md flex-col justify-center gap-4 p-6">
      {children}
    </main>
  )
}

/** Renders the application only for a signed-in user; clears cached data on sign-out. */
export function AuthGate({ children }: { children: ReactNode }) {
  const { status, error, signIn } = useAuth()
  const queryClient = useQueryClient()

  useEffect(() => {
    if (status === 'signed-out' || status === 'error') queryClient.clear()
  }, [status, queryClient])

  if (status === 'signed-in') return <>{children}</>

  if (status === 'loading') {
    return (
      <Centered>
        <p role="status">Signing in…</p>
      </Centered>
    )
  }

  if (status === 'unconfigured') {
    return (
      <Centered>
        <h1 className="text-xl font-semibold">Authentication is not configured</h1>
        <p role="alert" className="text-sm text-muted">
          This build has no identity provider. Set VITE_OIDC_AUTHORITY and VITE_OIDC_CLIENT_ID and
          rebuild. FactoryMind does not serve data to unauthenticated users.
        </p>
      </Centered>
    )
  }

  return (
    <Centered>
      <h1 className="text-xl font-semibold">FactoryMind CDT</h1>
      <p className="text-sm text-muted">
        Advisory decision support. Sign in with your organisation account to continue.
      </p>
      {error && (
        <p role="alert" className="rounded border border-critical p-3 text-sm">
          {error}
        </p>
      )}
      <button
        type="button"
        onClick={signIn}
        className="rounded bg-accent px-3 py-2 font-medium text-white"
      >
        Sign in
      </button>
    </Centered>
  )
}
