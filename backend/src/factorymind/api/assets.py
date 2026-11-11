"""Industrial Asset Digital Passport & Data Foundation API (read-only).

- ``GET /api/v1/assets``             list asset digital passports accessible to the caller
- ``GET /api/v1/assets/{asset_id}``  full digital passport + connected datasets for one asset
- ``GET /api/v1/datasets``           provenance, checksums, quality checks & non-destructive lineage
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from factorymind.assets.passport import (
    get_asset_passport,
    list_asset_passports,
    read_active_model_version,
)
from factorymind.auth.deps import require_data_source
from factorymind.auth.principal import Permission, Principal
from factorymind.config import IDENTIFIER, Settings
from factorymind.contracts.assets import (
    AssetDetailResponse,
    AssetListResponse,
    DatasetListResponse,
)
from factorymind.data.foundation import inspect_connected_datasets

router = APIRouter(prefix="/api/v1", tags=["assets"])


def _accessible_asset_ids(principal: Principal, settings: Settings) -> list[str]:
    return [asset_id for asset_id in settings.asset_ids if principal.can_access_asset(asset_id)]


@router.get("/assets", response_model=AssetListResponse)
def list_assets(
    request: Request,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ))],
) -> AssetListResponse:
    settings: Settings = request.app.state.settings
    items = list_asset_passports(
        allowed_asset_ids=_accessible_asset_ids(principal, settings),
        raw_dir=settings.raw_dir,
        processed_dir=settings.processed_dir,
        models_dir=settings.models_dir,
        model_name=settings.anomaly_model_name,
    )
    return AssetListResponse(items=items)


@router.get("/assets/{asset_id}", response_model=AssetDetailResponse)
def get_asset(
    asset_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ))],
) -> AssetDetailResponse:
    settings: Settings = request.app.state.settings
    if (
        not IDENTIFIER.match(asset_id)
        or asset_id not in settings.asset_ids
        or not principal.can_access_asset(asset_id)
    ):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Asset not found")
    passport = get_asset_passport(asset_id)
    if passport is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Asset passport not registered")
    datasets = inspect_connected_datasets(
        settings.raw_dir,
        settings.processed_dir,
        passport.connected_dataset_ids,
        verify_raw_hash=False,
    )
    runtime_version = read_active_model_version(settings.models_dir, settings.anomaly_model_name)
    return AssetDetailResponse(
        passport=passport,
        datasets=datasets,
        runtime_model_version=runtime_version,
    )


@router.get("/datasets", response_model=DatasetListResponse)
def list_datasets(
    request: Request,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ))],
    asset_id: Annotated[str | None, Query(max_length=64)] = None,
) -> DatasetListResponse:
    settings: Settings = request.app.state.settings
    accessible = _accessible_asset_ids(principal, settings)
    if asset_id is not None:
        if not IDENTIFIER.match(asset_id) or asset_id not in accessible:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Asset not found")
        target_assets = [asset_id]
    else:
        target_assets = accessible

    dataset_ids: list[str] = []
    seen: set[str] = set()
    for aid in target_assets:
        passport = get_asset_passport(aid)
        if passport is None:
            continue
        for did in passport.connected_dataset_ids:
            if did not in seen:
                seen.add(did)
                dataset_ids.append(did)

    items = (
        inspect_connected_datasets(
            settings.raw_dir,
            settings.processed_dir,
            dataset_ids,
            verify_raw_hash=False,
        )
        if dataset_ids
        else []
    )
    return DatasetListResponse(items=items)
