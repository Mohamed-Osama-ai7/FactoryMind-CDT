"""Read-only OT & edge connector health, stateless batch validation, and supply-chain API.

Paths intentionally use ``/api/v1/edge/...`` and expose zero write/actuation operations.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from factorymind.auth.deps import require_data_source
from factorymind.auth.principal import Permission, Principal
from factorymind.config import IDENTIFIER, Settings
from factorymind.connectors.attestation import build_supply_chain_report
from factorymind.connectors.edge import (
    EdgeSecurityError,
    EdgeValidationError,
    list_edge_sources,
    validate_edge_batch,
)
from factorymind.contracts.connectors import (
    EdgeSampleValidationRequest,
    EdgeSampleValidationResponse,
    EdgeSourcesResponse,
    SupplyChainHardeningResponse,
)

router = APIRouter(prefix="/api/v1/edge", tags=["edge"])


@router.get("/sources", response_model=EdgeSourcesResponse)
def get_edge_sources(
    request: Request,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ))],
    asset_id: Annotated[str | None, Query()] = None,
) -> EdgeSourcesResponse:
    """List SSRF-verified read-only OT & edge telemetry sources visible to this caller."""
    settings: Settings = request.app.state.settings
    if asset_id is not None and (
        not IDENTIFIER.fullmatch(asset_id)
        or asset_id not in settings.asset_ids
        or not principal.can_access_asset(asset_id)
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Asset not found")
    allowed_assets = frozenset(aid for aid in settings.asset_ids if principal.can_access_asset(aid))
    return list_edge_sources(allowed_assets=allowed_assets, asset_id=asset_id)


@router.post("/validate-sample", response_model=EdgeSampleValidationResponse)
def validate_edge_sample(
    request: Request,
    payload: EdgeSampleValidationRequest,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ))],
) -> EdgeSampleValidationResponse:
    """Validate sampling cadence, bounded ring-buffer eviction, and tag allowlists statelessly."""
    settings: Settings = request.app.state.settings
    if payload.asset_id not in settings.asset_ids or not principal.can_access_asset(
        payload.asset_id
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Asset not found")
    try:
        return validate_edge_batch(payload)
    except KeyError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Edge source not found") from exc
    except (EdgeSecurityError, EdgeValidationError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc


@router.get("/hardening", response_model=SupplyChainHardeningResponse)
def get_supply_chain_hardening(
    request: Request,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ))],
) -> SupplyChainHardeningResponse:
    """Inspect repository license, SHA-pinned CI workflow, and artifact attestation status."""
    _ = principal
    settings: Settings = request.app.state.settings
    return build_supply_chain_report(
        models_dir=settings.models_dir,
        model_name=settings.anomaly_model_name,
    )
