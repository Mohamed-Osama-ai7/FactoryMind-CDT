"""Impact estimation and constrained maintenance optimization API (advisory only).

- ``POST /api/v1/optimization/compare``: stateless comparison of maintenance options
- ``GET  /api/v1/incidents/{incident_id}/decision-comparison``: incident-grounded ranking
"""

from __future__ import annotations

import math
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, status

from factorymind.api.twin import get_digital_twin
from factorymind.auth.deps import require, require_data_source
from factorymind.auth.principal import Permission, Principal
from factorymind.config import IDENTIFIER, Settings
from factorymind.contracts.optimization import (
    EconomicAssumptions,
    ImpactQuantity,
    MaintenanceDecisionComparisonResponse,
    MaintenanceDecisionRequest,
)
from factorymind.contracts.rca import RcaHypothesisCategory
from factorymind.db.repository import IncidentRepo
from factorymind.db.session import Database
from factorymind.decision.optimizer import compare_maintenance_options

router = APIRouter(prefix="/api/v1", tags=["optimization"])


def _ensure_asset_access(asset_id: str, principal: Principal, settings: Settings) -> None:
    if (
        not IDENTIFIER.match(asset_id)
        or asset_id not in settings.asset_ids
        or not principal.can_access_asset(asset_id)
        or asset_id != "metropt3-apu-1"
    ):
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            detail="Optimization target asset not found",
        )


def _finite_in_bounds(value: float | None, lo: float, hi: float) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return max(lo, min(hi, value))


@router.post("/optimization/compare", response_model=MaintenanceDecisionComparisonResponse)
def compare_optimization_options(
    request: Request,
    payload: MaintenanceDecisionRequest,
    principal: Annotated[Principal, Depends(require_data_source(Permission.ASSET_READ, cost=2))],
) -> MaintenanceDecisionComparisonResponse:
    _ensure_asset_access(payload.asset_id, principal, request.app.state.settings)
    twin = get_digital_twin(request)
    return compare_maintenance_options(payload, twin=twin)


@router.get(
    "/incidents/{incident_id}/decision-comparison",
    response_model=MaintenanceDecisionComparisonResponse,
)
def get_incident_decision_comparison(
    incident_id: Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")],
    request: Request,
    principal: Annotated[Principal, Depends(require(Permission.INCIDENT_READ, cost=2))],
    electricity_eur_per_kwh: Annotated[float, Query(ge=0.01, le=2.0)] = 0.18,
    planned_depot_maintenance_eur: Annotated[float, Query(ge=50.0, le=50_000.0)] = 650.0,
    unplanned_service_disruption_eur_per_hr: Annotated[
        float, Query(ge=100.0, le=100_000.0)
    ] = 1800.0,
    leak_rate_bar_per_min: Annotated[float, Query(ge=0.0, le=1.5)] = 0.35,
    planning_horizon_hours: Annotated[float, Query(ge=1.0, le=72.0)] = 8.0,
    dominant_hypothesis: RcaHypothesisCategory | None = None,
) -> MaintenanceDecisionComparisonResponse:
    db: Database = request.app.state.db
    with db.session() as session:
        repo = IncidentRepo(session, principal)
        incident = repo.get(incident_id)
        if incident is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found")
        _ensure_asset_access(incident.asset_id, principal, request.app.state.settings)
        evidence = repo.evidence(incident)

    feat_map = {
        row.feature: float(row.value) for row in evidence if math.isfinite(float(row.value))
    }
    measured_duty = _finite_in_bounds(feat_map.get("load_duty"), 0.0, 1.0)
    measured_current = _finite_in_bounds(feat_map.get("Motor_current_mean"), 0.0, 20.0)
    measured_oil = _finite_in_bounds(
        feat_map.get("Oil_temperature_mean", feat_map.get("Oil_temperature_max")),
        -20.0,
        150.0,
    )

    observed_extras: list[ImpactQuantity] = []
    if measured_duty is None and "DV_eletric_mean" in feat_map:
        measured_duty = _finite_in_bounds(feat_map["DV_eletric_mean"], 0.0, 1.0)
        observed_extras.append(
            ImpactQuantity(
                key="measured_outlet_valve_duty",
                label="Measured Outlet Valve Duty (DV_eletric_mean)",
                value=round(feat_map["DV_eletric_mean"], 4),
                unit="fraction",
                tier="measured",
                basis_note="Stored peak incident evidence feature from operating_state group.",
            )
        )
    elif not feat_map:
        pass
    elif measured_duty is None and measured_current is None:
        # Expose first finite peak evidence item truthfully under tier='measured'
        first_row = next((r for r in evidence if math.isfinite(float(r.value))), None)
        if first_row is not None:
            observed_extras.append(
                ImpactQuantity(
                    key=f"measured_{first_row.feature.lower()}",
                    label=f"Measured Peak Feature ({first_row.feature})",
                    value=round(float(first_row.value), 4),
                    unit="raw",
                    tier="measured",
                    basis_note=f"Stored incident peak evidence ({first_row.group_label}).",
                )
            )

    req = MaintenanceDecisionRequest(
        asset_id=incident.asset_id,
        measured_load_duty=measured_duty,
        measured_motor_current_a=measured_current,
        measured_oil_temp_c=measured_oil,
        dominant_hypothesis=dominant_hypothesis,
        assumptions=EconomicAssumptions(
            electricity_eur_per_kwh=electricity_eur_per_kwh,
            planned_depot_maintenance_eur=planned_depot_maintenance_eur,
            unplanned_service_disruption_eur_per_hr=unplanned_service_disruption_eur_per_hr,
            leak_rate_bar_per_min=leak_rate_bar_per_min,
            planning_horizon_hours=planning_horizon_hours,
        ),
    )
    twin = get_digital_twin(request)
    return compare_maintenance_options(
        req,
        twin=twin,
        incident_id=incident.id,
        observed_extras=observed_extras,
    )
