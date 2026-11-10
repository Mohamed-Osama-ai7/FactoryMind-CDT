/// <reference types="vitest/config" />
import tailwindcss from '@tailwindcss/vite'
import { tanstackRouter } from '@tanstack/router-plugin/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

const API_TARGET = process.env.FACTORYMIND_API_URL ?? 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [tanstackRouter({ target: 'react', autoCodeSplitting: true }), react(), tailwindcss()],
  build: {
    // ECharts (~190 kB gzip) is only loaded on the anomaly route via automatic code splitting;
    // the initial bundle stays small, so allow that route chunk without a warning.
    chunkSizeWarningLimit: 600,
  },
  server: {
    proxy: { '/api': { target: API_TARGET, changeOrigin: true } },
  },
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    restoreMocks: true,
  },
})
