import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { UserManager } from 'oidc-client-ts'
import { describe, expect, it, vi } from 'vitest'
import { AuthGate } from './AuthGate'
import { AuthProvider, type OidcConfig, readOidcConfig } from './AuthProvider'
import { getAccessToken } from './token'

const CONFIG: OidcConfig = {
  authority: 'https://idp.example.test',
  clientId: 'factorymind-spa',
  redirectUri: 'http://localhost:5173/auth/callback',
  scope: 'openid profile',
}

function fakeManager(user: { access_token: string; expired: boolean } | null) {
  const events = { addAccessTokenExpired: vi.fn(), removeAccessTokenExpired: vi.fn() }
  const manager = {
    getUser: vi.fn().mockResolvedValue(user),
    signinRedirect: vi.fn().mockResolvedValue(undefined),
    signoutRedirect: vi.fn().mockResolvedValue(undefined),
    signinRedirectCallback: vi.fn(),
    events,
  }
  return { manager, typed: manager as unknown as UserManager }
}

function renderGate(props: { config: OidcConfig | null; manager?: UserManager }) {
  const qc = new QueryClient()
  return render(
    <QueryClientProvider client={qc}>
      <AuthProvider {...props}>
        <AuthGate>
          <p>app content</p>
        </AuthGate>
      </AuthProvider>
    </QueryClientProvider>,
  )
}

describe('readOidcConfig', () => {
  it('is null unless authority and client id are both set', () => {
    expect(readOidcConfig({}, 'http://x')).toBeNull()
    expect(readOidcConfig({ VITE_OIDC_AUTHORITY: 'https://a' }, 'http://x')).toBeNull()
  })

  it('defaults the redirect URI to the callback path and scope to openid profile', () => {
    const c = readOidcConfig(
      { VITE_OIDC_AUTHORITY: 'https://a', VITE_OIDC_CLIENT_ID: 'c' },
      'http://x',
    )
    expect(c?.redirectUri).toBe('http://x/auth/callback')
    expect(c?.scope).toBe('openid profile')
  })
})

describe('AuthGate', () => {
  it('refuses to render the app when authentication is not configured', () => {
    renderGate({ config: null })
    expect(screen.getByText(/authentication is not configured/i)).toBeInTheDocument()
    expect(screen.queryByText('app content')).not.toBeInTheDocument()
  })

  it('shows a sign-in prompt, not the app, when signed out', async () => {
    const { manager, typed } = fakeManager(null)
    renderGate({ config: CONFIG, manager: typed })
    const button = await screen.findByRole('button', { name: 'Sign in' })
    expect(screen.queryByText('app content')).not.toBeInTheDocument()
    expect(getAccessToken()).toBeNull()
    await userEvent.click(button)
    expect(manager.signinRedirect).toHaveBeenCalledOnce()
  })

  it('treats an expired stored user as signed out', async () => {
    const { typed } = fakeManager({ access_token: 'old', expired: true })
    renderGate({ config: CONFIG, manager: typed })
    await screen.findByRole('button', { name: 'Sign in' })
    expect(getAccessToken()).toBeNull()
  })

  it('renders the app and exposes the token when signed in', async () => {
    const { typed } = fakeManager({ access_token: 'tok-123', expired: false })
    renderGate({ config: CONFIG, manager: typed })
    expect(await screen.findByText('app content')).toBeInTheDocument()
    expect(getAccessToken()).toBe('tok-123')
  })

  it('shows a generic error without provider details when sign-in fails', async () => {
    const { manager, typed } = fakeManager(null)
    manager.getUser.mockRejectedValue(new Error('secret-detail code=abc'))
    renderGate({ config: CONFIG, manager: typed })
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(screen.getByRole('alert')).not.toHaveTextContent('secret-detail')
    expect(screen.queryByText('app content')).not.toBeInTheDocument()
  })
})
