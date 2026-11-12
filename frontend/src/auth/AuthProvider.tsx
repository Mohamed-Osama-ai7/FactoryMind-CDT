import {
  InMemoryWebStorage,
  type User,
  UserManager,
  type UserManagerSettings,
  WebStorageStateStore,
} from 'oidc-client-ts'
import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react'
import { setTokenProvider, setUnauthorizedHandler } from './token'

export const CALLBACK_PATH = '/auth/callback'
const RETURN_KEY = 'factorymind.returnTo'

export type AuthStatus = 'loading' | 'unconfigured' | 'signed-out' | 'signed-in' | 'error'

export interface AuthContextValue {
  status: AuthStatus
  error: string | null
  signIn: () => void
  signOut: () => void
}

const AuthContext = createContext<AuthContextValue | null>(null)

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext)
  if (value === null) throw new Error('useAuth must be used inside <AuthProvider>')
  return value
}

export interface OidcConfig {
  authority: string
  clientId: string
  redirectUri: string
  scope: string
}

/** Read from Vite env; null when authentication is not configured for this build. */
export function readOidcConfig(
  env: Record<string, string | undefined>,
  origin: string,
): OidcConfig | null {
  const authority = env.VITE_OIDC_AUTHORITY
  const clientId = env.VITE_OIDC_CLIENT_ID
  if (!authority || !clientId) return null
  return {
    authority,
    clientId,
    redirectUri: env.VITE_OIDC_REDIRECT_URI ?? `${origin}${CALLBACK_PATH}`,
    scope: env.VITE_OIDC_SCOPE ?? 'openid profile',
  }
}

export function createUserManager(
  config: OidcConfig,
  overrides: Partial<UserManagerSettings> = {},
): UserManager {
  const settings: UserManagerSettings = {
    authority: config.authority,
    client_id: config.clientId,
    redirect_uri: config.redirectUri,
    post_logout_redirect_uri: new URL(config.redirectUri).origin,
    response_type: 'code', // authorization code flow; oidc-client-ts adds PKCE (S256)
    scope: config.scope,
    automaticSilentRenew: false,
    // access tokens are kept in memory only; the transient PKCE state (code verifier, state) is
    // pinned to sessionStorage because the library would otherwise default to localStorage
    userStore: new WebStorageStateStore({ store: new InMemoryWebStorage() }),
    stateStore: new WebStorageStateStore({ store: window.sessionStorage }),
    ...overrides,
  }
  return new UserManager(settings)
}

interface CompletedSignin {
  user: User
  returnTo: string
}

// An authorization code can be exchanged exactly once. React StrictMode (development) runs the
// provider effect twice, so both runs must share one in-flight exchange instead of replaying the
// code (the second exchange would be rejected and show a spurious sign-in error).
let signinInFlight: Promise<CompletedSignin> | null = null

function completeSignin(manager: UserManager): Promise<CompletedSignin> {
  if (signinInFlight === null) {
    signinInFlight = manager
      .signinRedirectCallback()
      .then((user) => {
        const returnTo = sessionStorage.getItem(RETURN_KEY) ?? '/'
        sessionStorage.removeItem(RETURN_KEY)
        return { user, returnTo }
      })
      .finally(() => {
        // keep the settled promise only briefly so a later, genuine callback is processed anew
        setTimeout(() => {
          signinInFlight = null
        }, 0)
      })
  }
  return signinInFlight
}

export interface AuthProviderProps {
  children: ReactNode
  /** Injected for tests; defaults to the Vite environment. */
  config?: OidcConfig | null
  manager?: UserManager
}

export function AuthProvider({ children, config, manager }: AuthProviderProps) {
  const resolved = useMemo(
    () =>
      // an explicit `null` means "unconfigured" (tests); only `undefined` falls back to the env
      config !== undefined
        ? config
        : readOidcConfig(
            import.meta.env as Record<string, string | undefined>,
            window.location.origin,
          ),
    [config],
  )
  const userManager = useMemo(
    () => manager ?? (resolved ? createUserManager(resolved) : null),
    [manager, resolved],
  )
  const [status, setStatus] = useState<AuthStatus>(userManager ? 'loading' : 'unconfigured')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!userManager) return
    let cancelled = false
    const finish = (next: AuthStatus, message: string | null = null) => {
      if (cancelled) return
      setError(message)
      setStatus(next)
    }

    setTokenProvider(() => null) // until a user is known
    const syncToken = (accessToken: string | null, expired: boolean) => {
      setTokenProvider(() => (expired ? null : accessToken))
    }

    async function init(manager: UserManager) {
      try {
        if (
          window.location.pathname === CALLBACK_PATH &&
          window.location.search.includes('code=')
        ) {
          const { user, returnTo } = await completeSignin(manager)
          syncToken(user.access_token, user.expired === true)
          window.history.replaceState({}, '', returnTo)
          finish('signed-in')
          return
        }
        const user = await manager.getUser()
        if (user && !user.expired) {
          syncToken(user.access_token, false)
          finish('signed-in')
        } else {
          finish('signed-out')
        }
      } catch {
        // never show provider error details (they can echo request parameters)
        finish('error', 'Sign-in could not be completed. Please try again.')
      }
    }
    void init(userManager)

    const drop = () => {
      syncToken(null, true)
      finish('signed-out')
    }
    setUnauthorizedHandler(drop)
    userManager.events.addAccessTokenExpired(drop)
    return () => {
      cancelled = true
      userManager.events.removeAccessTokenExpired(drop)
    }
  }, [userManager])

  const signIn = useCallback(() => {
    if (!userManager) return
    sessionStorage.setItem(RETURN_KEY, window.location.pathname + window.location.search)
    void userManager.signinRedirect()
  }, [userManager])

  const signOut = useCallback(() => {
    setTokenProvider(() => null)
    setStatus('signed-out')
    if (userManager) void userManager.signoutRedirect().catch(() => {})
  }, [userManager])

  const value = useMemo(
    () => ({ status, error, signIn, signOut }),
    [status, error, signIn, signOut],
  )
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}
