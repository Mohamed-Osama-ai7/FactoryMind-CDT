"""Hybrid RAG engine: Okapi BM25 + TF-IDF/TruncatedSVD dense cosine + Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from factorymind.contracts.knowledge import (
    HybridSearchHit,
    HybridSearchResponse,
    KnowledgeDocumentChunk,
)
from factorymind.knowledge.kg import IndustrialKnowledgeGraph, infer_linked_kg_nodes

SearchMode = Literal["bm25_only", "dense_svd_only", "hybrid_rrf"]
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _snippet_around(text: str, query_tokens: Sequence[str], max_len: int = 280) -> str:
    if len(text) <= max_len:
        return text
    lower = text.lower()
    pos = -1
    for tok in query_tokens:
        idx = lower.find(tok)
        if idx >= 0 and (pos < 0 or idx < pos):
            pos = idx
    if pos < 0:
        return text[:max_len].rstrip() + "…"
    start = max(0, pos - 70)
    end = min(len(text), start + max_len)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


class HybridRagIndex:
    """Tenant- and asset-scoped hybrid lexical + latent-semantic retriever."""

    def __init__(
        self,
        chunks: Sequence[KnowledgeDocumentChunk],
        *,
        kg: IndustrialKnowledgeGraph | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
    ) -> None:
        self._chunks = list(chunks)
        self._kg = kg or IndustrialKnowledgeGraph()
        self._k1 = k1
        self._b = b
        self._rrf_k = rrf_k

        # Precompute corpus tokens & IDF
        self._doc_tokens: list[list[str]] = [
            _tokenize(f"{c.title} {c.section} {c.text}") for c in self._chunks
        ]
        self._doc_tf: list[Counter[str]] = [Counter(toks) for toks in self._doc_tokens]
        self._doc_lens: list[int] = [len(toks) for toks in self._doc_tokens]
        self._avg_len: float = (
            sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 1.0
        ) or 1.0

        df: Counter[str] = Counter()
        for tf in self._doc_tf:
            df.update(tf.keys())
        n_docs = len(self._chunks)
        self._idf: dict[str, float] = {
            tok: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()
        }

        # Dense TF-IDF + TruncatedSVD projection
        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._dense_matrix: np.ndarray | None = None
        if n_docs >= 1 and df:
            corpus_texts = [f"{c.title} {c.section} {c.text}" for c in self._chunks]
            vec = TfidfVectorizer(
                tokenizer=_tokenize,
                token_pattern=None,
                ngram_range=(1, 2),
                sublinear_tf=True,
                lowercase=False,
            )
            tfidf = vec.fit_transform(corpus_texts)
            n_features = tfidf.shape[1]
            self._vectorizer = vec
            max_comp = min(32, n_docs - 1, n_features - 1)
            if max_comp >= 2:
                svd = TruncatedSVD(n_components=max_comp, random_state=42)
                reduced = svd.fit_transform(tfidf)
                self._svd = svd
                self._dense_matrix = np.asarray(normalize(reduced, norm="l2"), dtype="float64")
            else:
                self._dense_matrix = np.asarray(
                    normalize(tfidf.toarray(), norm="l2"), dtype="float64"
                )

    @property
    def chunks(self) -> list[KnowledgeDocumentChunk]:
        return self._chunks

    @property
    def kg(self) -> IndustrialKnowledgeGraph:
        return self._kg

    def _bm25_scores(self, q_tokens: Sequence[str], indices: Sequence[int]) -> dict[int, float]:
        scores: dict[int, float] = {}
        unique_q = list(dict.fromkeys(q_tokens))
        for idx in indices:
            tf = self._doc_tf[idx]
            dl = self._doc_lens[idx]
            total = 0.0
            for tok in unique_q:
                freq = tf.get(tok, 0)
                if freq == 0:
                    continue
                idf = self._idf.get(tok, 0.0)
                denom = freq + self._k1 * (1.0 - self._b + self._b * (dl / self._avg_len))
                total += idf * (freq * (self._k1 + 1.0)) / denom
            scores[idx] = total
        return scores

    def _dense_scores(self, query: str, indices: Sequence[int]) -> dict[int, float]:
        if self._vectorizer is None or self._dense_matrix is None or not indices:
            return {idx: 0.0 for idx in indices}
        q_tfidf = self._vectorizer.transform([query])
        q_vec = self._svd.transform(q_tfidf) if self._svd is not None else q_tfidf.toarray()
        q_norm = np.asarray(normalize(q_vec, norm="l2"), dtype="float64")[0]
        if not np.any(q_norm):
            return {idx: 0.0 for idx in indices}
        sub = self._dense_matrix[indices]
        sims = sub @ q_norm
        return {idx: float(sims[pos]) for pos, idx in enumerate(indices)}

    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        allowed_asset_ids: set[str],
        top_k: int = 5,
        mode: SearchMode = "hybrid_rrf",
    ) -> HybridSearchResponse:
        accessible_indices = [
            idx
            for idx, chunk in enumerate(self._chunks)
            if chunk.asset_id in allowed_asset_ids
            and (chunk.tenant_scope is None or chunk.tenant_scope == tenant_id)
        ]
        q_tokens = _tokenize(query)
        if not accessible_indices or not q_tokens:
            return HybridSearchResponse(
                query=query,
                total_indexed_chunks=len(self._chunks),
                accessible_chunks=len(accessible_indices),
                items=[],
                related_kg_edges=[],
                safety_note=(
                    "Hybrid RAG retrieval across verified documentation only; inspect citations "
                    "and causal-vs-correlation tags before acting."
                ),
            )

        bm25 = self._bm25_scores(q_tokens, accessible_indices)
        dense = self._dense_scores(query, accessible_indices)

        bm25_ranked = sorted(
            accessible_indices, key=lambda idx: (-bm25[idx], self._chunks[idx].chunk_id)
        )
        dense_ranked = sorted(
            accessible_indices, key=lambda idx: (-dense[idx], self._chunks[idx].chunk_id)
        )
        bm25_pos = {idx: pos for pos, idx in enumerate(bm25_ranked, start=1)}
        dense_pos = {idx: pos for pos, idx in enumerate(dense_ranked, start=1)}

        rrf_scores: dict[int, float] = {}
        for idx in accessible_indices:
            if mode == "bm25_only":
                rrf_scores[idx] = bm25[idx]
            elif mode == "dense_svd_only":
                rrf_scores[idx] = dense[idx]
            else:
                b_term = 1.0 / (self._rrf_k + bm25_pos[idx]) if bm25[idx] > 0.0 else 0.0
                d_term = 1.0 / (self._rrf_k + dense_pos[idx]) if dense[idx] > 0.0 else 0.0
                rrf_scores[idx] = b_term + d_term

        ordered = sorted(
            accessible_indices,
            key=lambda idx: (-rrf_scores[idx], -bm25[idx], self._chunks[idx].chunk_id),
        )
        non_empty = [
            idx for idx in ordered if (bm25[idx] > 0.0 or dense[idx] > 0.0 or mode == "hybrid_rrf")
        ][: max(1, min(top_k, 20))]

        hits: list[HybridSearchHit] = []
        linked_all: list[str] = []
        for rank, idx in enumerate(non_empty, start=1):
            c = self._chunks[idx]
            linked = infer_linked_kg_nodes(f"{c.section} {c.text}")
            for nid in linked:
                if nid not in linked_all:
                    linked_all.append(nid)
            hits.append(
                HybridSearchHit(
                    rank=rank,
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    title=c.title,
                    section=c.section,
                    asset_id=c.asset_id,
                    source_path=c.source_path,
                    license=c.license,
                    provenance_origin=c.provenance_origin,
                    snippet=_snippet_around(c.text, q_tokens),
                    bm25_score=round(bm25[idx], 4),
                    semantic_score=round(dense[idx], 4),
                    rrf_score=round(rrf_scores[idx], 6),
                    linked_node_ids=linked,
                )
            )

        related_edges = (
            self._kg.edges_for_nodes(linked_all, allowed_asset_ids=allowed_asset_ids)[:12]
            if linked_all
            else []
        )
        return HybridSearchResponse(
            query=query,
            total_indexed_chunks=len(self._chunks),
            accessible_chunks=len(accessible_indices),
            items=hits,
            related_kg_edges=related_edges,
            safety_note=(
                "Retrieved knowledge preserves source license/origin tags and "
                "causal-vs-correlation KG semantics. Advisory reference only."
            ),
        )
