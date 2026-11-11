"""Controlled Industrial Copilot read-only API.

- ``GET  /api/v1/copilot/tools``  List permission-scoped read-only Copilot tools
- ``POST /api/v1/copilot/query``  Run red-team-checked, citation-grounded advisory query
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from factorymind.api.knowledge import get_knowledge_index
from factorymind.api.twin import get_digital_twin
from factorymind.auth.deps import require_data_source
from factorymind.auth.principal import Permission, Principal
from factorymind.config import Settings
from factorymind.contracts.copilot import (
    CopilotQueryRequest,
    CopilotQueryResponse,
    CopilotToolsResponse,
)
from factorymind.copilot.service import ControlledCopilot

router = APIRouter(prefix="/api/v1/copilot", tags=["copilot"])


def _get_copilot(request: Request) -> ControlledCopilot:
    index = get_knowledge_index(request)
    twin = get_digital_twin(request)
    return ControlledCopilot(knowledge_index=index, twin=twin)


@router.get("/tools", response_model=CopilotToolsResponse)
def list_copilot_tools(
    request: Request,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ))],
) -> CopilotToolsResponse:
    copilot = _get_copilot(request)
    return CopilotToolsResponse(tools=copilot.tool_specs(principal))


@router.post("/query", response_model=CopilotQueryResponse)
def query_copilot(
    request: Request,
    payload: CopilotQueryRequest,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ, cost=3))],
) -> CopilotQueryResponse:
    settings: Settings = request.app.state.settings
    if payload.asset_id not in settings.asset_ids or not principal.can_access_asset(
        payload.asset_id
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Asset not found")
    copilot = _get_copilot(request)
    try:
        return copilot.run_query(
            payload,
            principal=principal,
            session_factory=request.app.state.db.session,
        )
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Incident not found") from exc
