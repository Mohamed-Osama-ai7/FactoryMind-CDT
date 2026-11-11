# M3-WP2 (`WP-C3`) — Hybrid RAG, Knowledge Graph & Retrieval Evaluation

M3-WP2 introduces a hardened technical corpus loader (`pypdf` PDF parsing + Asset Passports + verified repo documentation), a tenant- and asset-scoped **Hybrid RAG** engine combining Okapi BM25 and TF-IDF/TruncatedSVD dense cosine similarity via Reciprocal Rank Fusion (RRF), an epistemically typed **Knowledge Graph**, and a reproducible IR evaluation suite.

---

## 1. Hardened Technical Corpus (`factorymind.knowledge.corpus`)

- **Hardened PDF Ingestion (`extract_pdf_chunks` via `pypdf`):**
  - Parses UCI `Data Description_Metro.pdf` (dealing cleanly with CIDFont encodings) under strict resource guards (`MAX_PDF_BYTES = 10,000,000`, `MAX_PDF_PAGES = 50`, `MAX_EXTRACTED_CHARS = 250,000`) and degrades corrupt/oversized/blank files cleanly.
- **License & Provenance Attribution:**
  - Tags every chunk (`KnowledgeDocumentChunk`) with explicit `asset_id`, optional `tenant_scope`, `source_path`, `license` (`CC BY 4.0 (UCI #791)` for MetroPT-3 PDF, `MIT` for repo contracts/docs), and `provenance_origin` (`observed` vs `derived`).

---

## 2. Hybrid RAG & Knowledge Graph (`rag.py`, `kg.py`)

- **Hybrid Retrieval (`HybridRagIndex`):**
  - Combines exact Okapi BM25 (`k1=1.5`, `b=0.75`) lexical matching with TF-IDF (`1–2`-grams) + `TruncatedSVD` latent semantic cosine similarity using Reciprocal Rank Fusion (`rrf_k=60`).
  - Strictly filters candidates by caller `(tenant_id, allowed_asset_ids)` **before** ranking and snippet generation so inaccessible documents cannot leak into search results or counts.
- **Typed Knowledge Graph (`IndustrialKnowledgeGraph`):**
  - Captures typed `Asset`, `Component`, `Sensor`, `Regime`, `EventClass`, `FailureMode`, and `Document` nodes.
  - Explicitly classifies directed edges with `epistemic_type`:
    - `physical_mechanism`
    - `causal_hypothesis`
    - `correlational_indicator` (*correlation does NOT imply causation*)
    - `structural_membership`
  - Enforces asset authorization during BFS traversal (`traverse`) and raises `KeyError` (`404`) when querying nodes belonging to ungranted assets.

---

## 3. Controlled Retrieval Ablation (`factorymind.knowledge.eval`)

Live evaluation across all **51** verified corpus chunks (including all **3** pages extracted from live `Data Description_Metro.pdf`, asset passports, and M1–M3 technical docs; 8 ground-truth domain queries):

| Mode | Query Count | Recall@5 | MRR@10 | nDCG@5 |
| --- | --- | --- | --- | --- |
| `bm25_only` | `8` | `1.0000` | `1.0000` | `0.7683` |
| `dense_svd_only` | `8` | `1.0000` | `1.0000` | `0.7330` |
| `hybrid_rrf` | `8` | `1.0000` | `1.0000` | `0.7683` |

---

## 4. API & `/knowledge` Frontend Explorer

- **Read-Only API (`factorymind.api.knowledge`, `Permission.ASSET_READ`):**
  - `GET /api/v1/knowledge/search` (`mode`, `asset_id`, `top_k`)
  - `GET /api/v1/knowledge/kg` (`node_id`, `depth`, `epistemic_type`)
  - `GET /api/v1/knowledge/eval`
- **Frontend (`KnowledgeExplorer.tsx`, `routes/knowledge.tsx`, `__root.tsx`):**
  - Gated `Knowledge & KG` navigation link (`asset:read`), hybrid search results with BM25/SVD/RRF scores and clickable KG node exploration, causal-vs-correlational edge table with epistemic badges, and live ablation comparison table.
