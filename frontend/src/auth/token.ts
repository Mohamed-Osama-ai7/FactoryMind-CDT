/**
 * Module-level bridge between the auth provider and the API client.
 *
 * Tokens live only in memory (never in localStorage/sessionStorage). The generated API client calls
 * `getAccessToken()`; when there is no token it refuses to touch the network (see `apiFetch`).
 */

type TokenProvider = () => string | null
type UnauthorizedHandler = () => void

let provider: TokenProvider = () => null
let onUnauthorized: UnauthorizedHandler = () => {}

export function setTokenProvider(next: TokenProvider): void {
  provider = next
}

export function setUnauthorizedHandler(next: UnauthorizedHandler): void {
  onUnauthorized = next
}

export function getAccessToken(): string | null {
  return provider()
}

export function notifyUnauthorized(): void {
  onUnauthorized()
}

/** Endpoints that are public by design (no token required, none sent). */
export const PUBLIC_PATHS: readonly string[] = ['/api/v1/health']
