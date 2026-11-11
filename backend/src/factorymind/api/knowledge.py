"""Knowledge Explorer & Hybrid RAG / Knowledge Graph read-only API.

- ``GET /api/v1/knowledge/search``  Hybrid BM25 + TF-IDF/SVD + RRF retrieval scoped to grants
- ``GET /api/v1/knowledge/kg``      Typed Knowledge Graph traversal (causal vs correlation)
- ``GET /api/v1/knowledge/eval``    Controlled retrieval evaluation across all 3 modes
"""

from __future__ import annotations

import threading
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from factorymind.auth.deps import require_data_source
from factorymind.auth.principal import Permission, Principal
from factorymind.config import IDENTIFIER, Settings
from factorymind.contracts.knowledge import (
    EpistemicType,
    HybridSearchResponse,
    KgSubgraphResponse,
    RetrievalEvalResponse,
)
from factorymind.knowledge.corpus import build_verified_corpus
from factorymind.knowledge.eval import evaluate_retrieval_modes
from factorymind.knowledge.rag import HybridRagIndex, SearchMode

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])
_INDEX_LOCK = threading.Lock()


def get_knowledge_index(request: Request) -> HybridRagIndex:
    """Thread-safely build once and cache the verified knowledge index on ``app.state``."""
    state = request.app.state
    index: HybridRagIndex | None = getattr(state, "knowledge_index", None)
    if index is None:
        with _INDEX_LOCK:
            index = getattr(state, "knowledge_index", None)
            if index is None:
                settings: Settings = state.settings
                chunks = build_verified_corpus(settings.raw_dir)
                index = HybridRagIndex(chunks)
                state.knowledge_index = index
    return index


def _allowed_assets(principal: Principal, settings: Settings) -> set[str]:
    return {aid for aid in settings.asset_ids if principal.can_access_asset(aid)}


@router.get("/search", response_model=HybridSearchResponse)
def search_knowledge(
    request: Request,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ))],
    q: Annotated[str, Query(min_length=1, max_length=256)],
    asset_id: str | None = None,
    top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    mode: SearchMode = "hybrid_rrf",
) -> HybridSearchResponse:
    cleaned_q = q.strip()
    if not cleaned_q:
        raise HTTPException(
            status_code=422,
            detail="Query string 'q' must not be empty or whitespace-only.",
        )
    settings: Settings = request.app.state.settings
    allowed = _allowed_assets(principal, settings)
    if asset_id is not None:
        if not IDENTIFIER.match(asset_id) or asset_id not in allowed:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Asset not found")
        allowed = {asset_id}
    index = get_knowledge_index(request)
    return index.search(
        cleaned_q,
        tenant_id=principal.tenant_id,
        allowed_asset_ids=allowed,
        top_k=top_k,
        mode=mode,
    )


@router.get("/kg", response_model=KgSubgraphResponse)
def query_knowledge_graph(
    request: Request,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ))],
    node_id: Annotated[str | None, Query(max_length=128)] = None,
    depth: Annotated[int, Query(ge=1, le=3)] = 2,
    epistemic_type: EpistemicType | None = None,
) -> KgSubgraphResponse:
    settings: Settings = request.app.state.settings
    allowed = _allowed_assets(principal, settings)
    index = get_knowledge_index(request)
    try:
        return index.kg.traverse(
            allowed_asset_ids=allowed,
            center_node_id=node_id,
            depth=depth,
            epistemic_type=epistemic_type,
        )
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Knowledge node not found") from exc


@router.get("/eval", response_model=RetrievalEvalResponse)
def evaluate_knowledge_retrieval(
    request: Request,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ, cost=5))],
) -> RetrievalEvalResponse:
    settings: Settings = request.app.state.settings
    allowed = _allowed_assets(principal, settings)
    index = get_knowledge_index(request)
    return evaluate_retrieval_modes(
        index,
        tenant_id=principal.tenant_id,
        allowed_asset_ids=allowed,
    )
