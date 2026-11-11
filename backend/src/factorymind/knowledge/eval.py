"""Reproducible retrieval ablation evaluator comparing BM25, Dense SVD, and Hybrid RRF."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from factorymind.contracts.knowledge import RetrievalEvalResponse, RetrievalModeMetrics
from factorymind.knowledge.rag import HybridRagIndex, SearchMode


@dataclass(frozen=True)
class RetrievalBenchmarkCase:
    query: str
    asset_id: str
    relevant_chunk_prefixes: tuple[str, ...]


DEFAULT_EVAL_CASES: Final[tuple[RetrievalBenchmarkCase, ...]] = (
    RetrievalBenchmarkCase(
        query="unloaded TP3 pressure decay slope air leakage duty_rule",
        asset_id="metropt3-apu-1",
        relevant_chunk_prefixes=("passport:mode-air_leakage", "m2-wp4-explainable-rca", "m2-wp0"),
    ),
    RetrievalBenchmarkCase(
        query="pca_spe non-leak statistical warning tier separation",
        asset_id="metropt3-apu-1",
        relevant_chunk_prefixes=("m2-wp2", "passport:model-pca_spe"),
    ),
    RetrievalBenchmarkCase(
        query="COMP DV_eletric intake outlet solenoid valve contradiction stale telemetry",
        asset_id="metropt3-apu-1",
        relevant_chunk_prefixes=(
            "passport:mode-telemetry_inconsistency",
            "passport:metropt3-sensors",
            "m2-wp2",
            "metropt3-pdf",
        ),
    ),
    RetrievalBenchmarkCase(
        query="downstream air demand flowmeter unobserved quantities competition",
        asset_id="metropt3-apu-1",
        relevant_chunk_prefixes=(
            "passport:metropt3-regimes",
            "passport:mode-abnormal_high_demand",
            "m2-wp4",
            "m3-wp0",
        ),
    ),
    RetrievalBenchmarkCase(
        query="MPG 8.2 bar LPS 7 bar cyclonic filter H1 motor phase current",
        asset_id="metropt3-apu-1",
        relevant_chunk_prefixes=("passport:metropt3-sensors", "metropt3-pdf"),
    ),
    RetrievalBenchmarkCase(
        query="causal vs correlation evidence origin human observation verification",
        asset_id="metropt3-apu-1",
        relevant_chunk_prefixes=("m2-wp3", "m2-wp4", "passport:"),
    ),
    RetrievalBenchmarkCase(
        query="cross_asset_transfer calibration boundary OOD Mahalanobis",
        asset_id="metropt3-apu-1",
        relevant_chunk_prefixes=("passport:model-cross_asset_transfer", "m3-wp0"),
    ),
    RetrievalBenchmarkCase(
        query="AI4I 2020 Brier ECE Mahalanobis abstention insufficient evidence",
        asset_id="ai4i-milling-bench",
        relevant_chunk_prefixes=("m3-wp1",),
    ),
    RetrievalBenchmarkCase(
        query="compressor duty limit",
        asset_id="metropt3-apu-1",
        relevant_chunk_prefixes=(
            "passport:mode-air_leakage",
            "passport:mode-abnormal_high_demand",
            "passport:mode-telemetry_inconsistency",
            "passport:model-duty_rule",
        ),
    ),
    RetrievalBenchmarkCase(
        query="tp3 pressure limits",
        asset_id="metropt3-apu-1",
        relevant_chunk_prefixes=(
            "passport:mode-air_leakage",
            "passport:metropt3-sensors",
            "m2-wp4",
            "m1-anomaly-detection",
        ),
    ),
    RetrievalBenchmarkCase(
        query="Explain pca_spe warning limits",
        asset_id="metropt3-apu-1",
        relevant_chunk_prefixes=("m2-wp2", "passport:model-pca_spe"),
    ),
)

DEFAULT_OOS_EVAL_QUERIES: Final[tuple[str, ...]] = (
    "What is the capital of France?",
    "Chocolate cake recipe",
    "What is the price of oil and air?",
    "Recipe for oil pressure",
    "What is 3 + 7?",
    "Give me 1 chocolate",
    "Description of banana",
    "Capital of compressor",
    "Write a Python binary search algorithm.",
    "What is the turbine blade temperature limit?",
)


def _is_relevant(chunk_id: str, prefixes: tuple[str, ...]) -> bool:
    return any(chunk_id.startswith(p) for p in prefixes)


def evaluate_retrieval_modes(
    index: HybridRagIndex,
    *,
    tenant_id: str,
    allowed_asset_ids: set[str],
) -> RetrievalEvalResponse:
    """Evaluate Recall@5, MRR@10, and nDCG@5 strictly over caller-accessible benchmark cases."""
    scoped_chunks = index.accessible_chunks(
        tenant_id=tenant_id, allowed_asset_ids=allowed_asset_ids
    )
    cases = [c for c in DEFAULT_EVAL_CASES if c.asset_id in allowed_asset_ids]
    if not cases:
        return RetrievalEvalResponse(
            corpus_chunks=len(scoped_chunks),
            modes=[],
            notes=(
                "No ground-truth retrieval benchmark queries match the caller's authorized "
                "asset grants."
            ),
        )

    modes: tuple[SearchMode, ...] = ("bm25_only", "dense_svd_only", "hybrid_rrf")
    results: list[RetrievalModeMetrics] = []

    for mode in modes:
        recall_sum = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0

        for case in cases:
            resp = index.search(
                case.query,
                tenant_id=tenant_id,
                allowed_asset_ids=allowed_asset_ids,
                top_k=10,
                mode=mode,
            )
            rel_flags = [
                1
                if h.asset_id == case.asset_id
                and _is_relevant(h.chunk_id, case.relevant_chunk_prefixes)
                else 0
                for h in resp.items
            ]
            top5 = rel_flags[:5]
            recall_sum += 1.0 if any(top5) else 0.0

            rr = 0.0
            for rank, flag in enumerate(rel_flags[:10], start=1):
                if flag == 1:
                    rr = 1.0 / rank
                    break
            mrr_sum += rr

            dcg = sum(flag / math.log2(rank + 1) for rank, flag in enumerate(top5, start=1))
            total_rel = sum(
                1
                for c in scoped_chunks
                if c.asset_id == case.asset_id
                and _is_relevant(c.chunk_id, case.relevant_chunk_prefixes)
            )
            ideal_k = min(5, max(total_rel, 1))
            idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_k + 1))
            ndcg_sum += dcg / idcg if idcg > 0 else 0.0

        n = len(cases)
        results.append(
            RetrievalModeMetrics(
                mode=mode,
                query_count=n,
                recall_at_5=round(recall_sum / n, 4),
                mrr_at_10=round(mrr_sum / n, 4),
                ndcg_at_5=round(ndcg_sum / n, 4),
            )
        )

    return RetrievalEvalResponse(
        corpus_chunks=len(scoped_chunks),
        modes=results,
        notes=(
            f"Controlled retrieval ablation across {len(cases)} benchmark queries "
            "evaluated strictly across chunks authorized for the caller's asset grants."
        ),
    )


def evaluate_abstention_rate(
    index: HybridRagIndex,
    *,
    tenant_id: str,
    allowed_asset_ids: set[str],
    queries: tuple[str, ...] = DEFAULT_OOS_EVAL_QUERIES,
) -> dict[SearchMode, float]:
    """Measure out-of-scope abstention rate (1.0 == zero spurious hits) across search modes."""
    modes: tuple[SearchMode, ...] = ("bm25_only", "dense_svd_only", "hybrid_rrf")
    rates: dict[SearchMode, float] = {}
    total = len(queries) or 1
    for mode in modes:
        abstained = sum(
            1
            for q in queries
            if not index.search(
                q,
                tenant_id=tenant_id,
                allowed_asset_ids=allowed_asset_ids,
                top_k=5,
                mode=mode,
            ).items
        )
        rates[mode] = round(abstained / total, 4)
    return rates
