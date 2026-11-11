import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, beforeEach } from 'vitest'
import { setTokenProvider, setUnauthorizedHandler } from '../auth/token'

// Tests act as a signed-in user unless they override the provider explicitly.
beforeEach(() => {
  setTokenProvider(() => 'test-token')
  setUnauthorizedHandler(() => {})
})

afterEach(() => {
  cleanup()
})
