import { createFileRoute } from '@tanstack/react-router'

// The AuthProvider completes the authorization-code exchange before the app renders; this route only
// exists so the router has a match while that happens.
export const Route = createFileRoute('/auth/callback')({
  component: () => <p role="status">Completing sign-in…</p>,
})
