import { type FormEvent, useId, useState } from 'react'
import { describeApiError } from '../../api/errors'
import { useMe } from '../../api/generated/incidents/incidents'
import {
  useEvaluateKnowledgeRetrieval,
  useQueryKnowledgeGraph,
  useSearchKnowledge,
} from '../../api/generated/knowledge/knowledge'
import type {
  QueryKnowledgeGraphEpistemicType,
  SearchKnowledgeMode,
} from '../../api/generated/model'

const EPISTEMIC_BADGE = {
  physical_mechanism: 'border-ok/60 bg-ok/10 text-ok',
  causal_hypothesis: 'border-accent/60 bg-accent/15 text-ink',
  correlational_indicator: 'border-warn/60 bg-warn/10 text-warn',
  structural_membership: 'border-border text-muted',
} as const

type EpistemicFilter = 'all' | NonNullable<QueryKnowledgeGraphEpistemicType>

/** Hybrid RAG + typed Knowledge Graph explorer gated cleanly by asset:read grants. */
export function KnowledgeExplorer() {
  const idPrefix = useId()
  const me = useMe()
  const canRead = me.data?.data.permissions.includes('asset:read') ?? false

  const [draftQuery, setDraftQuery] = useState('unloaded TP3 air leakage')
  const [submittedQuery, setSubmittedQuery] = useState('unloaded TP3 air leakage')
  const [mode, setMode] = useState<SearchKnowledgeMode>('hybrid_rrf')
  const [validationError, setValidationError] = useState<string | null>(null)

  const [centerNodeId, setCenterNodeId] = useState<string | undefined>(undefined)
  const [depth, setDepth] = useState<number>(2)
  const [epistemicFilter, setEpistemicFilter] = useState<EpistemicFilter>('all')

  const searchQuery = useSearchKnowledge(
    {
      q: submittedQuery,
      top_k: 5,
      mode,
    },
    { query: { enabled: canRead && submittedQuery.length > 0 } },
  )

  const kgQuery = useQueryKnowledgeGraph(
    {
      ...(centerNodeId ? { node_id: centerNodeId } : {}),
      depth,
      ...(epistemicFilter !== 'all' ? { epistemic_type: epistemicFilter } : {}),
    },
    { query: { enabled: canRead } },
  )

  const evalQuery = useEvaluateKnowledgeRetrieval({ query: { enabled: canRead } })

  if (me.isPending) return <p role="status">Checking knowledge permissions…</p>
  if (me.isError) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Knowledge Base &amp; Causal-vs-Correlation Graph
        </h1>
        <p role="alert" className="mt-2 rounded border border-critical p-3 text-sm text-critical">
          Could not verify knowledge permissions: {describeApiError(me.error)}
        </p>
      </section>
    )
  }
  if (!canRead) {
    return (
      <section aria-labelledby={`${idPrefix}-title`} className="max-w-5xl">
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Knowledge Base &amp; Causal-vs-Correlation Graph
        </h1>
        <p
          role="status"
          className="mt-2 rounded border border-border bg-panel p-4 text-sm text-muted"
        >
          You do not hold the <span className="font-mono">asset:read</span> permission required to
          query verified technical documentation or the knowledge graph.
        </p>
      </section>
    )
  }

  const onSearchSubmit = (event: FormEvent) => {
    event.preventDefault()
    const clean = draftQuery.trim()
    if (!clean) {
      setValidationError('Enter a non-empty technical query.')
      return
    }
    setValidationError(null)
    setSubmittedQuery(clean)
  }

  const searchData = searchQuery.data?.status === 200 ? searchQuery.data.data : undefined
  const kgData = kgQuery.data?.status === 200 ? kgQuery.data.data : undefined
  const evalData = evalQuery.data?.status === 200 ? evalQuery.data.data : undefined

  const queryInputId = `${idPrefix}-q`
  const validationId = `${idPrefix}-q-error`

  return (
    <section aria-labelledby={`${idPrefix}-title`} className="flex max-w-5xl flex-col gap-6">
      <header>
        <h1 id={`${idPrefix}-title`} className="text-2xl font-semibold">
          Knowledge Base &amp; Causal-vs-Correlation Graph
        </h1>
        <p className="mt-1 text-sm text-muted">
          Hybrid BM25 + TF-IDF/SVD Reciprocal Rank Fusion over verified documents paired with an
          epistemically typed Knowledge Graph. Advisory reference only.
        </p>
      </header>

      <section aria-labelledby={`${idPrefix}-rag`} className="flex flex-col gap-3">
        <h2 id={`${idPrefix}-rag`} className="text-lg font-medium">
          Hybrid RAG documentation search
        </h2>
        <form
          onSubmit={onSearchSubmit}
          className="flex flex-wrap items-end gap-3 rounded-lg border border-border bg-panel p-4 text-xs"
        >
          <label htmlFor={queryInputId} className="flex min-w-60 flex-1 flex-col gap-1">
            <span className="text-muted">Technical query</span>
            <input
              id={queryInputId}
              type="text"
              maxLength={256}
              value={draftQuery}
              aria-invalid={validationError ? true : undefined}
              aria-describedby={validationError ? validationId : undefined}
              onChange={(e) => {
                setDraftQuery(e.target.value)
                if (validationError) setValidationError(null)
              }}
              placeholder="e.g., unloaded TP3 pressure decay or COMP DV_eletric contradiction"
              className="rounded border border-border bg-surface px-2 py-1.5 text-ink"
            />
          </label>
          <label htmlFor={`${idPrefix}-mode`} className="flex flex-col gap-1">
            <span className="text-muted">Retriever mode</span>
            <select
              id={`${idPrefix}-mode`}
              value={mode}
              onChange={(e) => setMode(e.target.value as SearchKnowledgeMode)}
              className="rounded border border-border bg-surface px-2 py-1.5 text-ink"
            >
              <option value="hybrid_rrf">hybrid_rrf (BM25 + SVD)</option>
              <option value="bm25_only">bm25_only</option>
              <option value="dense_svd_only">dense_svd_only</option>
            </select>
          </label>
          <button
            type="submit"
            disabled={searchQuery.isFetching}
            className="rounded bg-accent px-3 py-1.5 font-medium text-white disabled:opacity-50"
          >
            {searchQuery.isFetching ? 'Searching…' : 'Search corpus'}
          </button>
          {validationError && (
            <span id={validationId} role="alert" className="w-full text-critical">
              {validationError}
            </span>
          )}
        </form>

        {searchQuery.isPending && <p role="status">Searching verified technical corpus…</p>}
        {searchQuery.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-xs text-critical">
            Search failed: {describeApiError(searchQuery.error)}
          </p>
        )}
        {searchData && (
          <div className="flex flex-col gap-3">
            <p className="text-xs text-muted">
              Showing {searchData.items.length} result(s) across {searchData.accessible_chunks}{' '}
              authorized chunk(s). {searchData.safety_note}
            </p>
            {searchData.items.length === 0 ? (
              <p
                role="status"
                className="rounded-lg border border-border bg-panel p-3 text-xs text-muted"
              >
                No matching chunks found in authorized documentation.
              </p>
            ) : (
              searchData.items.map((hit) => (
                <article
                  key={hit.chunk_id}
                  className="flex flex-col gap-2 rounded-lg border border-border bg-panel p-3 text-xs"
                >
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <h3 className="font-medium text-ink">
                        #{hit.rank} {hit.title} — {hit.section}
                      </h3>
                      <p className="text-muted">
                        Source: <span className="font-mono">{hit.source_path}</span> · License:{' '}
                        {hit.license} · Origin:{' '}
                        <span className="font-mono">{hit.provenance_origin}</span>
                      </p>
                    </div>
                    <div className="flex flex-wrap gap-1.5 font-mono text-[11px] text-muted">
                      <span className="rounded border border-border px-1.5 py-0.5 text-ink">
                        RRF {hit.rrf_score.toFixed(4)}
                      </span>
                      <span className="rounded border border-border px-1.5 py-0.5">
                        BM25 {hit.bm25_score.toFixed(2)}
                      </span>
                      <span className="rounded border border-border px-1.5 py-0.5">
                        SVD {hit.semantic_score.toFixed(3)}
                      </span>
                    </div>
                  </div>
                  <p className="text-ink">{hit.snippet}</p>
                  {hit.linked_node_ids && hit.linked_node_ids.length > 0 && (
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="text-muted">KG links:</span>
                      {hit.linked_node_ids.map((nid) => (
                        <button
                          key={nid}
                          type="button"
                          onClick={() => setCenterNodeId(nid)}
                          className="rounded border border-border px-1.5 py-0.5 font-mono text-[11px] text-accent hover:bg-border"
                        >
                          {nid}
                        </button>
                      ))}
                    </div>
                  )}
                </article>
              ))
            )}
            {searchData.related_kg_edges && searchData.related_kg_edges.length > 0 && (
              <div className="rounded-lg border border-border bg-panel p-3 text-xs">
                <h3 className="mb-1.5 font-medium text-muted">
                  Related Knowledge Graph relationships ({searchData.related_kg_edges.length})
                </h3>
                <ul className="flex flex-col gap-1">
                  {searchData.related_kg_edges.map((edge) => (
                    <li
                      key={`rel-${edge.source_id}-${edge.relation}-${edge.target_id}-${edge.epistemic_type}`}
                      className="flex flex-wrap items-center gap-2"
                    >
                      <span className="font-mono text-ink">{edge.source_id}</span>
                      <span className="font-mono text-muted">→ {edge.relation} →</span>
                      <span className="font-mono text-ink">{edge.target_id}</span>
                      <span
                        className={`rounded border px-1.5 py-0.5 font-mono text-[10px] ${EPISTEMIC_BADGE[edge.epistemic_type]}`}
                      >
                        {edge.epistemic_type}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </section>

      <section aria-labelledby={`${idPrefix}-kg`} className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <h2 id={`${idPrefix}-kg`} className="text-lg font-medium">
            Typed Knowledge Graph explorer
          </h2>
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {centerNodeId && (
              <button
                type="button"
                onClick={() => setCenterNodeId(undefined)}
                className="rounded border border-border px-2 py-0.5 text-muted hover:bg-border"
              >
                Reset center ({centerNodeId})
              </button>
            )}
            <label htmlFor={`${idPrefix}-depth`} className="flex items-center gap-1 text-muted">
              Depth
              <select
                id={`${idPrefix}-depth`}
                value={depth}
                onChange={(e) => setDepth(Number(e.target.value))}
                className="rounded border border-border bg-surface px-1.5 py-0.5 text-ink"
              >
                <option value={1}>1 hop</option>
                <option value={2}>2 hops</option>
                <option value={3}>3 hops</option>
              </select>
            </label>
            <label htmlFor={`${idPrefix}-epistemic`} className="flex items-center gap-1 text-muted">
              Edge type
              <select
                id={`${idPrefix}-epistemic`}
                value={epistemicFilter}
                onChange={(e) => setEpistemicFilter(e.target.value as EpistemicFilter)}
                className="rounded border border-border bg-surface px-1.5 py-0.5 text-ink"
              >
                <option value="all">all</option>
                <option value="physical_mechanism">physical_mechanism</option>
                <option value="causal_hypothesis">causal_hypothesis</option>
                <option value="correlational_indicator">correlational_indicator</option>
                <option value="structural_membership">structural_membership</option>
              </select>
            </label>
          </div>
        </div>

        {kgQuery.isPending && <p role="status">Loading knowledge graph subgraph…</p>}
        {kgQuery.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-xs text-critical">
            Could not load knowledge graph: {describeApiError(kgQuery.error)}
          </p>
        )}
        {kgData && (
          <div className="flex flex-col gap-3 rounded-lg border border-border bg-panel p-4 text-xs">
            <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
              {Object.entries(kgData.epistemic_legend).map(([key, desc]) => (
                <div key={key} className="rounded border border-border p-2">
                  <span className="font-mono font-medium text-ink">{key}</span>
                  <p className="mt-0.5 text-muted">{desc}</p>
                </div>
              ))}
            </div>

            <div>
              <h3 className="mb-1.5 font-medium text-muted">
                Nodes in view ({kgData.nodes.length})
              </h3>
              <div className="flex flex-wrap gap-1.5">
                {kgData.nodes.map((node) => (
                  <button
                    key={node.node_id}
                    type="button"
                    aria-pressed={centerNodeId === node.node_id}
                    onClick={() => setCenterNodeId(node.node_id)}
                    title={node.summary}
                    className={`rounded border px-2 py-1 text-left ${
                      centerNodeId === node.node_id
                        ? 'border-accent bg-accent/15 font-medium text-ink'
                        : 'border-border text-ink hover:bg-border'
                    }`}
                  >
                    <span className="font-mono text-[11px]">{node.node_id}</span>
                    <span className="block text-[10px] text-muted">{node.kind}</span>
                  </button>
                ))}
              </div>
            </div>

            <div className="overflow-x-auto">
              {kgData.edges.length === 0 ? (
                <p role="status" className="py-2 text-muted">
                  No knowledge graph edges match the current filter criteria.
                </p>
              ) : (
                <table className="w-full text-xs">
                  <caption className="mb-1 text-left text-muted">
                    Directed relationships preserving physical vs causal vs correlational semantics
                    ({kgData.edges.length} edges)
                  </caption>
                  <thead>
                    <tr className="text-left text-muted">
                      {['Source', 'Relation', 'Epistemic type', 'Target', 'Rationale'].map((h) => (
                        <th key={h} scope="col" className="pr-3 font-normal">
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {kgData.edges.map((edge) => (
                      <tr
                        key={`${edge.source_id}-${edge.relation}-${edge.target_id}-${edge.epistemic_type}`}
                        className="border-t border-border align-top"
                      >
                        <th
                          scope="row"
                          className="py-1.5 pr-3 text-left font-mono font-normal text-ink"
                        >
                          {edge.source_id}
                        </th>
                        <td className="py-1.5 pr-3 font-mono text-muted">{edge.relation}</td>
                        <td className="py-1.5 pr-3">
                          <span
                            className={`rounded border px-1.5 py-0.5 font-mono text-[11px] ${EPISTEMIC_BADGE[edge.epistemic_type]}`}
                          >
                            {edge.epistemic_type}
                          </span>
                        </td>
                        <td className="py-1.5 pr-3 font-mono text-ink">{edge.target_id}</td>
                        <td className="py-1.5 text-muted">{edge.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}
      </section>

      <section aria-labelledby={`${idPrefix}-eval`} className="flex flex-col gap-3">
        <h2 id={`${idPrefix}-eval`} className="text-lg font-medium">
          Controlled retrieval ablation metrics
        </h2>
        {evalQuery.isPending && <p role="status">Computing retrieval evaluation metrics…</p>}
        {evalQuery.isError && (
          <p role="alert" className="rounded border border-critical p-3 text-xs text-critical">
            Retrieval evaluation failed: {describeApiError(evalQuery.error)}
          </p>
        )}
        {evalData &&
          (evalData.modes.length === 0 ? (
            <p
              role="status"
              className="rounded-lg border border-border bg-panel p-4 text-xs text-muted"
            >
              {evalData.notes}
            </p>
          ) : (
            <div className="overflow-x-auto rounded-lg border border-border bg-panel p-4 text-xs">
              <table className="w-full text-xs">
                <caption className="mb-1 text-left text-muted">
                  Ablation across {evalData.corpus_chunks} accessible chunks ({evalData.notes})
                </caption>
                <thead>
                  <tr className="text-left text-muted">
                    {['Retriever mode', 'Queries', 'Recall@5', 'MRR@10', 'nDCG@5'].map((h) => (
                      <th key={h} scope="col" className="pr-4 font-normal">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {evalData.modes.map((m) => (
                    <tr key={m.mode} className="border-t border-border">
                      <th
                        scope="row"
                        className="py-1.5 pr-4 text-left font-mono font-medium text-ink"
                      >
                        {m.mode}
                      </th>
                      <td className="py-1.5 pr-4 font-mono">{m.query_count}</td>
                      <td className="py-1.5 pr-4 font-mono">{m.recall_at_5.toFixed(4)}</td>
                      <td className="py-1.5 pr-4 font-mono">{m.mrr_at_10.toFixed(4)}</td>
                      <td className="py-1.5 font-mono">{m.ndcg_at_5.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
      </section>
    </section>
  )
}
