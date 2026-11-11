"""Controlled Industrial Copilot read-only API.

- ``GET  /api/v1/copilot/tools``  List permission-scoped read-only Copilot tools
- ``POST /api/v1/copilot/query``  Run red-team-checked, citation-grounded advisory query
"""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, Depends, HTTPException, Request, status

from factorymind.anomaly.service import AnomalyService
from factorymind.api.anomaly import get_anomaly_service
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


def _get_copilot(
    request: Request,
    *,
    principal: Principal | None = None,
) -> ControlledCopilot:
    index = get_knowledge_index(request)
    twin = get_digital_twin(request)

    def _anomaly_for_asset(asset_id: str) -> AnomalyService | None:
        settings: Settings = request.app.state.settings
        if (
            principal is not None and principal.data_source_denial(settings) is not None
        ) or asset_id != settings.data_source_asset_id:
            return None
        overrides = getattr(request.app, "dependency_overrides", {})
        override = overrides.get(get_anomaly_service)
        if override is not None:
            return cast(AnomalyService, override())
        try:
            return get_anomaly_service(request)
        except HTTPException:
            return None

    return ControlledCopilot(
        knowledge_index=index,
        twin=twin,
        anomaly_service_factory=_anomaly_for_asset,
    )


@router.get("/tools", response_model=CopilotToolsResponse)
def list_copilot_tools(
    request: Request,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ))],
) -> CopilotToolsResponse:
    copilot = _get_copilot(request, principal=principal)
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
    copilot = _get_copilot(request, principal=principal)
    try:
        return copilot.run_query(
            payload,
            principal=principal,
            session_factory=request.app.state.db.session,
        )
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Incident not found") from exc
