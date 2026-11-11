"""Hybrid RAG engine: Okapi BM25 + TF-IDF/TruncatedSVD dense cosine + Reciprocal Rank Fusion."""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from typing import Final, Literal

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
_CONTRACTION_NOT_RE = re.compile(r"\b([a-z0-9_]+)n['\u2019]t\b")
_CONTRACTION_SUFFIX_RE = re.compile(r"(?<=[a-z0-9_])['\u2019](?:s|re|ve|ll|d|m)\b")

# Grammatical function words, interrogatives, and generic request framing excluded from scoring
# and relevance gating. Domain state/polarity tokens (e.g. ``off``, ``on``, ``no``, ``not``,
# ``below``, ``above``, ``low``, ``high``) are intentionally kept as content tokens.
_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "having",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "of",
        "in",
        "to",
        "for",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "about",
        "between",
        "during",
        "under",
        "over",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "them",
        "their",
        "we",
        "us",
        "our",
        "you",
        "your",
        "i",
        "me",
        "my",
        "and",
        "or",
        "but",
        "if",
        "so",
        "then",
        "than",
        "nor",
        "too",
        "very",
        "just",
        "can",
        "cannot",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "should",
        "will",
        "would",
        "please",
        "tell",
        "explain",
        "summarize",
        "describe",
        "give",
        "show",
        "list",
        "find",
        "get",
        "now",
        "today",
        "tomorrow",
        "yesterday",
        "vs",
    }
)

# Minimum IDF-weighted query coverage thresholds derived from Okapi BM25 / OOV IDF balance:
# - Raw RRF is strictly rank-based in (0, 2/(rrf_k+1)] (~0.0328 for rrf_k=60), and TruncatedSVD
#   cosine drops OOV query terms before L2-normalizing, so neither detects out-of-scope queries.
# - Weighting every query content token (including OOV tokens using BM25 df=0 IDF) measures how
#   much of the user's substantive information need is actually covered by an authorized chunk.
_MIN_MULTI_TERM_IDF_COV: Final[float] = 0.25
_MIN_OOV_MULTI_TERM_IDF_COV: Final[float] = 0.45
_MIN_SINGLE_HIT_TWO_TERM_IDF_COV: Final[float] = 0.45
_MIN_TAIL_HIT_IDF_COV: Final[float] = 0.15


def _has_alpha(token: str) -> bool:
    return any(ch.isalpha() for ch in token)


def _tokenize(text: str) -> list[str]:
    norm = re.sub(r"\bcan['\u2019]t\b", "cannot", text.lower())
    norm = re.sub(r"\bwon['\u2019]t\b", "will not", norm)
    norm = _CONTRACTION_NOT_RE.sub(r"\1 not", norm)
    norm = _CONTRACTION_SUFFIX_RE.sub("", norm)
    return _TOKEN_RE.findall(norm)


def extract_content_tokens(text: str) -> list[str]:
    """Return lowercase tokens excluding grammatical stopwords and contraction fragments."""
    return [tok for tok in _tokenize(text) if tok not in _STOPWORDS]


def _chunk_passes_relevance_gate(
    *,
    n_matched: int,
    has_alpha_match: bool,
    has_alpha_oov: bool,
    q_len: int,
    tok_cov: float,
    idf_cov: float,
    n_docs: int,
) -> bool:
    """Decide whether one chunk provides substantive non-stopword support for a query.

    Limitations:
    - Exact token matching only (no stemming or neural synonyms), so queries phrased entirely in
      unindexed synonyms or inflections will abstain conservatively.
    """
    if q_len <= 0 or n_matched <= 0 or not has_alpha_match:
        return False
    if q_len == 1:
        return True
    if n_docs <= 2:
        if q_len == 2:
            return n_matched >= 2
        return n_matched >= 2 and tok_cov >= 0.30
    min_multi_idf = (
        _MIN_OOV_MULTI_TERM_IDF_COV
        if (has_alpha_oov and n_matched == 2)
        else _MIN_MULTI_TERM_IDF_COV
    )
    if n_matched >= 2 and idf_cov >= min_multi_idf:
        return True
    return tok_cov >= 0.5 and idf_cov >= _MIN_SINGLE_HIT_TWO_TERM_IDF_COV


def _snippet_around(text: str, query_tokens: Sequence[str], max_len: int = 280) -> str:
    if len(text) <= max_len:
        return text
    lower = text.lower()
    candidates = [idx for tok in query_tokens if (idx := lower.find(tok)) >= 0]
    if not candidates:
        return text[:max_len].rstrip() + "…"
    best_pos = candidates[0]
    best_cov = -1
    for cand in candidates:
        start_c = max(0, cand - 70)
        end_c = min(len(text), start_c + max_len)
        window = lower[start_c:end_c]
        cov = sum(1 for tok in query_tokens if tok in window)
        if cov > best_cov or (cov == best_cov and cand < best_pos):
            best_cov = cov
            best_pos = cand
    start = max(0, best_pos - 70)
    end = min(len(text), start + max_len)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end].strip()}{suffix}"


class _ScopedSliceIndex:
    """BM25 + TF-IDF/TruncatedSVD index built strictly over one authorized chunk partition."""

    def __init__(
        self,
        chunks: Sequence[KnowledgeDocumentChunk],
        *,
        k1: float,
        b: float,
    ) -> None:
        self.chunks = list(chunks)
        self._k1 = k1
        self._b = b
        self._doc_tokens: list[list[str]] = [
            extract_content_tokens(f"{c.title} {c.section} {c.text}") for c in self.chunks
        ]
        self._doc_tf: list[Counter[str]] = [Counter(toks) for toks in self._doc_tokens]
        self._doc_lens: list[int] = [len(toks) for toks in self._doc_tokens]
        self._avg_len: float = (
            sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 1.0
        ) or 1.0

        df: Counter[str] = Counter()
        for tf in self._doc_tf:
            df.update(tf.keys())
        n_docs = len(self.chunks)
        self._idf: dict[str, float] = {
            tok: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5)) for tok, freq in df.items()
        }
        self._idf_oov: float = math.log(1.0 + (n_docs + 0.5) / 0.5) if n_docs > 0 else 0.0

        self._vectorizer: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._dense_matrix: np.ndarray | None = None
        if n_docs >= 1 and df:
            corpus_texts = [f"{c.title} {c.section} {c.text}" for c in self.chunks]
            vec = TfidfVectorizer(
                tokenizer=extract_content_tokens,
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

    def has_alpha_oov(self, unique_q: Sequence[str]) -> bool:
        return any(_has_alpha(tok) and tok not in self._idf for tok in unique_q)

    def chunk_coverages(self, unique_q: Sequence[str]) -> list[tuple[int, float, float, bool]]:
        """Return ``(n_matched, tok_cov, idf_cov, has_alpha_match)`` per chunk (incl. OOV)."""
        q_len = len(unique_q)
        if q_len == 0 or not self.chunks:
            return [(0, 0.0, 0.0, False) for _ in self.chunks]
        weights = {tok: self._idf.get(tok, self._idf_oov) for tok in unique_q}
        total_w = sum(weights.values()) or 1.0
        out: list[tuple[int, float, float, bool]] = []
        for tf in self._doc_tf:
            matched_w = 0.0
            n_m = 0
            alpha_match = False
            for tok in unique_q:
                if tf.get(tok, 0) > 0:
                    n_m += 1
                    matched_w += weights[tok]
                    if _has_alpha(tok):
                        alpha_match = True
            out.append((n_m, n_m / q_len, matched_w / total_w, alpha_match))
        return out

    def bm25_scores(self, q_tokens: Sequence[str]) -> list[float]:
        unique_q = list(dict.fromkeys(q_tokens))
        scores: list[float] = []
        for tf, dl in zip(self._doc_tf, self._doc_lens, strict=True):
            total = 0.0
            for tok in unique_q:
                freq = tf.get(tok, 0)
                if freq == 0:
                    continue
                idf = self._idf.get(tok, 0.0)
                denom = freq + self._k1 * (1.0 - self._b + self._b * (dl / self._avg_len))
                total += idf * (freq * (self._k1 + 1.0)) / denom
            scores.append(total)
        return scores

    def dense_scores(self, query: str) -> list[float]:
        n = len(self.chunks)
        if self._vectorizer is None or self._dense_matrix is None or n == 0:
            return [0.0] * n
        q_tfidf = self._vectorizer.transform([query])
        q_vec = self._svd.transform(q_tfidf) if self._svd is not None else q_tfidf.toarray()
        q_norm = np.asarray(normalize(q_vec, norm="l2"), dtype="float64")[0]
        if not np.any(q_norm):
            return [0.0] * n
        sims = self._dense_matrix @ q_norm
        return [float(sims[pos]) for pos in range(n)]


class HybridRagIndex:
    """Tenant- and asset-scoped hybrid lexical + latent-semantic retriever.

    Security invariant: separate BM25/IDF/TF-IDF/SVD indices are built per
    ``(tenant_id, frozenset(allowed_asset_ids))`` slice so ungranted tenants or assets never
    influence vocabulary, IDF weights, latent SVD components, or returned corpus totals.
    """

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
        self._lock = threading.Lock()
        self._partitions: dict[tuple[str, frozenset[str]], _ScopedSliceIndex] = {}

    @property
    def chunks(self) -> list[KnowledgeDocumentChunk]:
        return self._chunks

    @property
    def kg(self) -> IndustrialKnowledgeGraph:
        return self._kg

    def accessible_chunks(
        self, *, tenant_id: str, allowed_asset_ids: set[str]
    ) -> list[KnowledgeDocumentChunk]:
        """Return only chunks visible to ``(tenant_id, allowed_asset_ids)``."""
        return [
            chunk
            for chunk in self._chunks
            if chunk.asset_id in allowed_asset_ids
            and (chunk.tenant_scope is None or chunk.tenant_scope == tenant_id)
        ]

    def _partition(self, *, tenant_id: str, allowed_asset_ids: set[str]) -> _ScopedSliceIndex:
        key = (tenant_id, frozenset(allowed_asset_ids))
        with self._lock:
            cached = self._partitions.get(key)
        if cached is not None:
            return cached
        scoped = self.accessible_chunks(tenant_id=tenant_id, allowed_asset_ids=allowed_asset_ids)
        built = _ScopedSliceIndex(scoped, k1=self._k1, b=self._b)
        with self._lock:
            existing = self._partitions.get(key)
            if existing is not None:
                return existing
            self._partitions[key] = built
        return built

    def search(
        self,
        query: str,
        *,
        tenant_id: str,
        allowed_asset_ids: set[str],
        top_k: int = 5,
        mode: SearchMode = "hybrid_rrf",
    ) -> HybridSearchResponse:
        slice_index = self._partition(tenant_id=tenant_id, allowed_asset_ids=allowed_asset_ids)
        scoped = slice_index.chunks
        q_tokens = list(dict.fromkeys(extract_content_tokens(query)))
        if not scoped or not q_tokens:
            return HybridSearchResponse(
                query=query,
                total_indexed_chunks=len(scoped),
                accessible_chunks=len(scoped),
                items=[],
                related_kg_edges=[],
                safety_note=(
                    "Hybrid RAG retrieval across verified documentation only; inspect citations "
                    "and causal-vs-correlation tags before acting."
                ),
            )

        bm25 = slice_index.bm25_scores(q_tokens)
        dense = slice_index.dense_scores(query)
        coverages = slice_index.chunk_coverages(q_tokens)
        has_oov = slice_index.has_alpha_oov(q_tokens)
        n_docs = len(scoped)
        q_len = len(q_tokens)
        any_passes = any(
            _chunk_passes_relevance_gate(
                n_matched=n_m,
                has_alpha_match=alpha_m,
                has_alpha_oov=has_oov,
                q_len=q_len,
                tok_cov=tok_c,
                idf_cov=idf_c,
                n_docs=n_docs,
            )
            for n_m, tok_c, idf_c, alpha_m in coverages
        )
        if not any_passes:
            return HybridSearchResponse(
                query=query,
                total_indexed_chunks=n_docs,
                accessible_chunks=n_docs,
                items=[],
                related_kg_edges=[],
                safety_note=(
                    "Query did not meet the minimum content relevance gate for verified "
                    "documentation; abstained instead of returning unrelated matches."
                ),
            )

        indices = list(range(n_docs))
        bm25_ranked = sorted(indices, key=lambda idx: (-bm25[idx], scoped[idx].chunk_id))
        dense_ranked = sorted(indices, key=lambda idx: (-dense[idx], scoped[idx].chunk_id))
        bm25_pos = {idx: pos for pos, idx in enumerate(bm25_ranked, start=1)}
        dense_pos = {idx: pos for pos, idx in enumerate(dense_ranked, start=1)}

        rrf_scores: dict[int, float] = {}
        for idx in indices:
            if mode == "bm25_only":
                rrf_scores[idx] = bm25[idx]
            elif mode == "dense_svd_only":
                rrf_scores[idx] = dense[idx]
            else:
                b_term = 1.0 / (self._rrf_k + bm25_pos[idx]) if bm25[idx] > 0.0 else 0.0
                d_term = 1.0 / (self._rrf_k + dense_pos[idx]) if dense[idx] > 0.0 else 0.0
                rrf_scores[idx] = b_term + d_term

        ordered = sorted(
            indices,
            key=lambda idx: (-rrf_scores[idx], -bm25[idx], scoped[idx].chunk_id),
        )
        limit = max(1, min(top_k, 20))
        non_empty: list[int] = []
        for idx in ordered:
            if rrf_scores[idx] <= 0.0:
                continue
            n_m, _tok_c, idf_c, alpha_m = coverages[idx]
            if bm25[idx] <= 0.0 or not alpha_m:
                continue
            if n_docs >= 3 and n_m < 2 and idf_c < _MIN_TAIL_HIT_IDF_COV:
                continue
            non_empty.append(idx)
            if len(non_empty) >= limit:
                break

        hits: list[HybridSearchHit] = []
        linked_all: list[str] = []
        for rank, idx in enumerate(non_empty, start=1):
            c = scoped[idx]
            linked = infer_linked_kg_nodes(
                f"{c.section} {c.text}",
                allowed_asset_ids=allowed_asset_ids,
                kg=self._kg,
            )
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
            total_indexed_chunks=len(scoped),
            accessible_chunks=len(scoped),
            items=hits,
            related_kg_edges=related_edges,
            safety_note=(
                "Retrieved knowledge preserves source license/origin tags and "
                "causal-vs-correlation KG semantics. Advisory reference only."
            ),
        )
