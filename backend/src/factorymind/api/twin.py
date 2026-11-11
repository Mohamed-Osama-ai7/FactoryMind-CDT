"""Physics-Informed Digital Twin read-only & counterfactual simulation API.

- ``GET  /api/v1/twin/model``     Calibrated physical ODE parameters, unidentifiable bounds & eval
- ``POST /api/v1/twin/simulate``  Stateless counterfactual scenario simulator (advisory only)
"""

from __future__ import annotations

import logging
import threading
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from factorymind.auth.deps import require_data_source
from factorymind.auth.principal import Permission, Principal
from factorymind.config import IDENTIFIER, Settings
from factorymind.contracts.digital_twin import (
    TwinModelResponse,
    TwinSimulationRequest,
    TwinSimulationResponse,
)
from factorymind.twin.physics import CompressorReservoirTwin

log = logging.getLogger("factorymind.twin")

router = APIRouter(prefix="/api/v1/twin", tags=["digital_twin"])
_TWIN_LOCK = threading.Lock()


def get_digital_twin(request: Request) -> CompressorReservoirTwin:
    """Thread-safely obtain or initialize the calibrated physics twin on ``app.state``."""
    state = request.app.state
    twin: CompressorReservoirTwin | None = getattr(state, "digital_twin", None)
    if twin is None:
        with _TWIN_LOCK:
            twin = getattr(state, "digital_twin", None)
            if twin is None:
                settings: Settings = state.settings
                parquet = settings.processed_dir / "metropt3" / "telemetry.parquet"
                if parquet.is_file():
                    try:
                        frame = pd.read_parquet(parquet)
                        twin = CompressorReservoirTwin.fit_from_telemetry(frame)
                    except Exception:
                        log.warning(
                            "Falling back to verified baseline twin parameters", exc_info=True
                        )
                        twin = CompressorReservoirTwin()
                else:
                    twin = CompressorReservoirTwin()
                state.digital_twin = twin
    return twin


def _ensure_apu_access(asset_id: str, principal: Principal, settings: Settings) -> None:
    if (
        not IDENTIFIER.match(asset_id)
        or asset_id not in settings.asset_ids
        or not principal.can_access_asset(asset_id)
        or asset_id != "metropt3-apu-1"
    ):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="Digital twin not available for requested asset",
        )


@router.get("/model", response_model=TwinModelResponse)
def get_twin_model(
    request: Request,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ))],
    asset_id: Annotated[str, Query(max_length=64)] = "metropt3-apu-1",
) -> TwinModelResponse:
    _ensure_apu_access(asset_id, principal, request.app.state.settings)
    twin = get_digital_twin(request)
    return twin.model_overview(asset_id=asset_id)


@router.post("/simulate", response_model=TwinSimulationResponse)
def simulate_twin_scenario(
    request: Request,
    payload: TwinSimulationRequest,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ, cost=2))],
) -> TwinSimulationResponse:
    _ensure_apu_access(payload.asset_id, principal, request.app.state.settings)
    twin = get_digital_twin(request)
    return twin.simulate(payload)

