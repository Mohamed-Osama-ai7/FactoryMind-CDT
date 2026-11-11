# M3-WP2 (`WP-C3`) — Hybrid RAG, Knowledge Graph & Retrieval Evaluation

M3-WP2 introduces a hardened technical corpus loader (`pypdf` PDF parsing + Asset Passports + verified repo documentation), a tenant- and asset-scoped **Hybrid RAG** engine combining Okapi BM25 and TF-IDF/TruncatedSVD dense cosine similarity via Reciprocal Rank Fusion (RRF), an epistemically typed **Knowledge Graph**, and a reproducible IR evaluation suite.

---

## 1. Hardened Technical Corpus (`factorymind.knowledge.corpus`)

- **Hardened PDF Ingestion (`extract_pdf_chunks` via `pypdf`):**
  - Parses UCI `Data Description_Metro.pdf` under strict resource guards (`MAX_PDF_BYTES = 10,000,000`, `MAX_PDF_PAGES = 50`, `MAX_EXTRACTED_CHARS = 250,000`), path-containment (`is_relative_to`), and degrades corrupt/oversized/blank files cleanly.
- **Code-Fence-Aware & Lossless Markdown Chunking:**
  - Splits repo documentation on level-2 headings outside fenced code blocks (`_split_markdown_sections`) and segments long sections on word boundaries (`<= 1,800` chars) so no technical prose is dropped.
- **License & Provenance Attribution:**
  - Tags every chunk (`KnowledgeDocumentChunk`) with explicit `asset_id`, optional `tenant_scope`, `source_path`, `license` (`CC BY 4.0 (UCI #791)` for MetroPT-3 PDF, `MIT` for repo contracts/docs), and `provenance_origin` (`observed` vs `derived`).

---

## 2. Hybrid RAG & Knowledge Graph (`rag.py`, `kg.py`)

- **Partition-Isolated Hybrid Retrieval (`HybridRagIndex`) & Relevance Gate:**
  - Combines exact Okapi BM25 (`k1=1.5`, `b=0.75`) lexical matching with TF-IDF (`1–2`-grams) + `TruncatedSVD` latent semantic cosine similarity using Reciprocal Rank Fusion (`rrf_k=60`), excluding grammatical stopwords (`extract_content_tokens`) so question boilerplate cannot manufacture spurious BM25 or normalized SVD scores.
  - Enforces a corpus-size-invariant minimum content relevance gate using Okapi IDF weighting across all non-stopword query terms (assigning $df=0$ IDF weight $\ln(1 + (N + 0.5)/0.5)$ to out-of-vocabulary terms): out-of-scope, empty-content, or weakly overlapping queries return empty `items=[]` across all three modes rather than padded citations. Known limit: matching is lexical (no stemming or neural synonyms).
  - Builds isolated BM25/IDF, `TfidfVectorizer`, and `TruncatedSVD` indices strictly per `(tenant_id, frozenset(allowed_asset_ids))` slice so unauthorized tenants/assets cannot influence vocabulary, IDF statistics, latent subspaces, or corpus counts.
- **Typed Knowledge Graph (`IndustrialKnowledgeGraph`):**
  - Captures typed `Asset`, `Component`, `Sensor`, `Regime`, `EventClass`, `FailureMode`, and `Document` nodes.
  - Explicitly classifies directed edges with `epistemic_type`:
    - `physical_mechanism`
    - `causal_hypothesis`
    - `correlational_indicator` (*correlation does NOT imply causation*)
    - `structural_membership`
  - Uses word-boundary entity linking (`infer_linked_kg_nodes`) scoped to `allowed_asset_ids` and raises `KeyError` (`404`) when traversing ungranted asset nodes.

---

## 3. Controlled Retrieval Ablation (`factorymind.knowledge.eval`)

Live evaluation across verified corpus chunks (including all **3** pages extracted from `Data Description_Metro.pdf`, asset passports, and M1–M3 technical docs):

### Full Corpus (`metropt3-apu-1` + `ai4i-milling-bench`, `61` chunks, `8` queries)

| Mode | Query Count | Recall@5 | MRR@10 | nDCG@5 |
| --- | --- | --- | --- | --- |
| `bm25_only` | `8` | `1.0000` | `1.0000` | `0.7683` |
| `dense_svd_only` | `8` | `1.0000` | `1.0000` | `0.7494` |
| `hybrid_rrf` | `8` | `1.0000` | `1.0000` | `0.7683` |

### Default Single-Asset Slice (`metropt3-apu-1`, `57` chunks, `7` queries)

| Mode | Query Count | Recall@5 | MRR@10 | nDCG@5 |
| --- | --- | --- | --- | --- |
| `bm25_only` | `7` | `1.0000` | `1.0000` | `0.7594` |
| `dense_svd_only` | `7` | `1.0000` | `1.0000` | `0.7103` |
| `hybrid_rrf` | `7` | `1.0000` | `1.0000` | `0.7352` |

---

## 4. API & `/knowledge` Frontend Explorer

- **Read-Only API (`factorymind.api.knowledge`, `Permission.ASSET_READ`):**
  - `GET /api/v1/knowledge/search` (`mode`, `asset_id`, `top_k`; rejects whitespace-only `q` with `422`)
  - `GET /api/v1/knowledge/kg` (`node_id`, `depth`, `epistemic_type`)
  - `GET /api/v1/knowledge/eval` (rate-limit `cost=5`; returns `modes=[]` cleanly when no benchmark cases match granted assets)
- **Frontend (`KnowledgeExplorer.tsx`, `routes/knowledge.tsx`, `__root.tsx`):**
  - Gated `Knowledge & KG` navigation link (`asset:read`), hybrid search results with BM25/SVD/RRF scores, related KG edges, clickable KG node exploration (1–3 hops), causal-vs-correlational edge table with epistemic badges, and live ablation comparison table.
