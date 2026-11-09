import { defineConfig } from 'orval'

// Generates the typed API client + TanStack Query hooks from the backend OpenAPI contract.
// Regenerate with `npm run generate:api` after exporting frontend/openapi.json from the backend.
export default defineConfig({
  factorymind: {
    input: { target: './openapi.json' },
    output: {
      mode: 'tags-split',
      target: './src/api/generated',
      schemas: './src/api/generated/model',
      client: 'react-query',
      httpClient: 'fetch',
      clean: true,
      override: {
        mutator: { path: './src/api/http.ts', name: 'apiFetch' },
      },
    },
  },
})
