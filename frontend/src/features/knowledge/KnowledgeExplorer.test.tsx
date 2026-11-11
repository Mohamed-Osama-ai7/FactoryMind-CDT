import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { KnowledgeExplorer } from './KnowledgeExplorer'

describe('KnowledgeExplorer', () => {
  it('blocks unauthorized users lacking asset:read', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        new Response(
          JSON.stringify({
            actor_id: 'aud-1',
            tenant_id: 'acme',
            roles: ['auditor'],
            permissions: ['incident:read', 'audit:read'],
            asset_ids: ['metropt3-apu-1'],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } },
        ),
      ),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <KnowledgeExplorer />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/do not hold the/)).toBeInTheDocument()
  })

  it('renders hybrid RAG results, omits null params on KG traversal, and displays eval metrics', async () => {
    const seenUrls: string[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((url: string) => {
        const s = String(url)
        seenUrls.push(s)
        if (s.startsWith('/api/v1/me')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                actor_id: 'eng-1',
                tenant_id: 'acme',
                roles: ['engineer'],
                permissions: ['asset:read'],
                asset_ids: ['metropt3-apu-1'],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        if (s.startsWith('/api/v1/knowledge/search')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                query: 'unloaded TP3 air leakage',
                total_indexed_chunks: 51,
                accessible_chunks: 48,
                safety_note: 'Advisory reference only.',
                items: [
                  {
                    rank: 1,
                    chunk_id: 'passport:mode-air_leakage',
                    doc_id: 'passport-metropt3-apu-1',
                    title: 'MetroPT-3 APU Failure Mode Catalog',
                    section: 'Pneumatic Air Leakage',
                    asset_id: 'metropt3-apu-1',
                    source_path: 'factorymind/assets/passport.py',
                    license: 'MIT (Repository Contract)',
                    provenance_origin: 'derived',
                    snippet: 'Negative unloaded TP3 slope indicates possible air leak.',
                    bm25_score: 4.12,
                    semantic_score: 0.782,
                    rrf_score: 0.032787,
                    linked_node_ids: ['mode:air_leakage', 'sensor:TP3'],
                  },
                ],
                related_kg_edges: [],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        if (s.startsWith('/api/v1/knowledge/kg')) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                center_node_id: null,
                depth: 2,
                epistemic_legend: {
                  causal_hypothesis: 'Requires field check.',
                  correlational_indicator: 'Correlation only.',
                },
                nodes: [
                  {
                    node_id: 'mode:air_leakage',
                    kind: 'failure_mode',
                    label: 'Pneumatic Air Leakage',
                    asset_id: 'metropt3-apu-1',
                    summary: 'Leak hypotheses.',
                    attributes: {},
                  },
                ],
                edges: [
                  {
                    source_id: 'mode:air_leakage',
                    target_id: 'sensor:Oil_temperature',
                    relation: 'co_occurs_with_thermal_rise',
                    epistemic_type: 'correlational_indicator',
                    rationale: 'Thermal rise is non-specific correlation.',
                  },
                ],
              }),
              { status: 200, headers: { 'Content-Type': 'application/json' } },
            ),
          )
        }
        return Promise.resolve(
          new Response(
            JSON.stringify({
              corpus_chunks: 48,
              notes: 'Scoped ablation.',
              modes: [
                {
                  mode: 'hybrid_rrf',
                  query_count: 7,
                  recall_at_5: 1.0,
                  mrr_at_10: 1.0,
                  ndcg_at_5: 0.7683,
                },
              ],
            }),
            { status: 200, headers: { 'Content-Type': 'application/json' } },
          ),
        )
      }),
    )

    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <KnowledgeExplorer />
      </QueryClientProvider>,
    )

    expect(await screen.findByText(/Pneumatic Air Leakage/)).toBeInTheDocument()
    expect(screen.getByRole('cell', { name: 'correlational_indicator' })).toBeInTheDocument()
    expect(screen.getByText('0.7683')).toBeInTheDocument()
    expect(seenUrls.some((u) => u.includes('node_id=null'))).toBe(false)

    // Whitespace query triggers inline validation alert
    const qInput = screen.getByLabelText('Technical query')
    await userEvent.clear(qInput)
    await userEvent.click(screen.getByRole('button', { name: 'Search corpus' }))
    expect(screen.getByRole('alert')).toHaveTextContent(/non-empty technical query/)

    // Clicking KG entity node sets centerNodeId on next KG query
    await userEvent.click(screen.getByRole('button', { name: 'sensor:TP3' }))
    expect(seenUrls.some((u) => u.includes('node_id=sensor%3ATP3'))).toBe(true)
  })
})
