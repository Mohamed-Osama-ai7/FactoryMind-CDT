"""Canonical contracts for Hybrid RAG, typed Knowledge Graph traversal, and retrieval eval."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EpistemicType = Literal[
    "physical_mechanism",
    "causal_hypothesis",
    "correlational_indicator",
    "structural_membership",
]

KgNodeKind = Literal[
    "asset",
    "component",
    "sensor",
    "regime",
    "event_class",
    "failure_mode",
    "document",
]


class KnowledgeDocumentChunk(BaseModel):
    """One verified technical corpus chunk with explicit license, asset, and tenant scope."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    title: str
    section: str
    asset_id: str
    tenant_scope: str | None = None
    source_path: str
    license: str
    provenance_origin: Literal["observed", "derived"] = "observed"
    text: str


class KgNode(BaseModel):
    """Typed entity inside the industrial Knowledge Graph."""

    model_config = ConfigDict(extra="forbid")

    node_id: str
    kind: KgNodeKind
    label: str
    asset_id: str
    summary: str
    attributes: dict[str, str] = Field(default_factory=dict)


class KgEdge(BaseModel):
    """Typed directed edge preserving causal vs correlational distinction."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    relation: str
    epistemic_type: EpistemicType
    rationale: str


class HybridSearchHit(BaseModel):
    """Ranked Hybrid RAG chunk combining BM25, TF-IDF/SVD semantic score, and RRF."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1)
    chunk_id: str
    doc_id: str
    title: str
    section: str
    asset_id: str
    source_path: str
    license: str
    provenance_origin: Literal["observed", "derived"]
    snippet: str
    bm25_score: float
    semantic_score: float
    rrf_score: float
    linked_node_ids: list[str] = Field(default_factory=list)


class HybridSearchResponse(BaseModel):
    """Tenant- and asset-scoped hybrid search result set."""

    model_config = ConfigDict(extra="forbid")

    query: str
    total_indexed_chunks: int = Field(ge=0)
    accessible_chunks: int = Field(ge=0)
    items: list[HybridSearchHit]
    related_kg_edges: list[KgEdge] = Field(default_factory=list)
    safety_note: str


class KgSubgraphResponse(BaseModel):
    """Traversed subgraph response with epistemic legend."""

    model_config = ConfigDict(extra="forbid")

    center_node_id: str | None = None
    depth: int = Field(ge=1, le=3)
    nodes: list[KgNode]
    edges: list[KgEdge]
    epistemic_legend: dict[str, str]


class RetrievalModeMetrics(BaseModel):
    """Aggregated IR metrics for a single retriever family."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["bm25_only", "dense_svd_only", "hybrid_rrf"]
    query_count: int = Field(ge=1)
    recall_at_5: float = Field(ge=0.0, le=1.0)
    mrr_at_10: float = Field(ge=0.0, le=1.0)
    ndcg_at_5: float = Field(ge=0.0, le=1.0)


class RetrievalEvalResponse(BaseModel):
    """Controlled ablation comparing sparse BM25, dense TF-IDF/SVD, and Hybrid RRF."""

    model_config = ConfigDict(extra="forbid")

    corpus_chunks: int = Field(ge=0)
    modes: list[RetrievalModeMetrics]
    notes: str
