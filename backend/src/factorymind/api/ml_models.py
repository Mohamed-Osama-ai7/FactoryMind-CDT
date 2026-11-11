"""Multi-model ML orchestrator, uncertainty quantification & AI4I 2020 benchmark API.

All predictions and benchmark reports carry explicit provenance and remain strictly advisory.
AI4I 2020 synthetic milling benchmark metrics are never mixed with MetroPT-3 operational metrics.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field

from factorymind.assets.passport import read_active_model_version
from factorymind.auth.deps import require_data_source
from factorymind.auth.principal import Permission, Principal
from factorymind.contracts.ml_engine import (
    ModelRoutingDecision,
    MultiModelBenchmarkReport,
    MultiModelCatalogResponse,
    QualityGateStatus,
    SpecialistModelEntry,
)
from factorymind.ml.orchestrator import (
    BenchmarkUnavailableError,
    MultiModelOrchestrator,
    load_ai4i_benchmark_report,
    load_verified_ai4i_bundle,
)

router = APIRouter(prefix="/api/v1/ml", tags=["ml"])


def get_ai4i_orchestrator(request: Request) -> MultiModelOrchestrator:
    state = request.app.state
    orchestrator: MultiModelOrchestrator | None = getattr(state, "ai4i_orchestrator", None)
    if orchestrator is None:
        try:
            bundle = load_verified_ai4i_bundle(state.settings.artifacts_dir)
        except BenchmarkUnavailableError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        orchestrator = MultiModelOrchestrator(bundle)
        state.ai4i_orchestrator = orchestrator
    return orchestrator


class Ai4iPredictRequest(BaseModel):
    """Input vector for specialized AI4I 2020 milling failure & OOD uncertainty assessment."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(default="ai4i-milling-bench", min_length=1, max_length=64)
    product_type: str = Field(min_length=1, max_length=8)
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    model_id: str | None = Field(default=None, max_length=64)
    quality_status: QualityGateStatus = "healthy"


@router.get("/catalog", response_model=MultiModelCatalogResponse)
def list_ml_catalog(
    request: Request,
    _: Annotated[Principal, Depends(require_data_source(Permission.ANOMALY_READ))],
) -> MultiModelCatalogResponse:
    settings = request.app.state.settings
    metro_ver = read_active_model_version(settings.models_dir, settings.anomaly_model_name)

    report: MultiModelBenchmarkReport | None = None
    try:
        report = load_ai4i_benchmark_report(settings.artifacts_dir)
    except BenchmarkUnavailableError:
        report = None

    specialists = [
        SpecialistModelEntry(
            model_id="duty_rule",
            asset_id="metropt3-apu-1",
            dataset_id="metropt3",
            domain_kind="real_operational_telemetry",
            model_family="PhysicsDutyBaseline + ExceedanceCalibrator",
            governance_status="active" if metro_ver else "candidate",
            uncertainty_mechanism=(
                "Smoothed conformal p-value calibration + WP2 regime/integrity gate"
            ),
            scope_note="Primary calibrated leak detector on MetroPT-3 railway APU telemetry.",
        ),
        SpecialistModelEntry(
            model_id="pca_spe",
            asset_id="metropt3-apu-1",
            dataset_id="metropt3",
            domain_kind="real_operational_telemetry",
            model_family="StandardizedPCA-SPE (42 features)",
            governance_status="active",
            uncertainty_mechanism=(
                "Routed exclusively to non-leak warning tier; no causal leak diagnosis"
            ),
            scope_note="Detects non-leak multivariate excursions without inflating leak alarms.",
        ),
        SpecialistModelEntry(
            model_id="ai4i-hgb",
            asset_id="ai4i-milling-bench",
            dataset_id="ai4i2020",
            domain_kind="controlled_synthetic_benchmark",
            model_family="HistGradientBoostingClassifier + PlattCalibrator",
            governance_status=(
                "active" if report and report.promotion_contract.promoted else "candidate"
            ),
            uncertainty_mechanism=(
                "Ledoit-Wolf Mahalanobis OOD + quality gate + boundary abstention"
            ),
            scope_note=(
                "Controlled AI4I 2020 failure benchmark classifier (separate from MetroPT-3)."
            ),
        ),
        SpecialistModelEntry(
            model_id="cross_asset_transfer",
            asset_id="*",
            dataset_id="uncalibrated",
            domain_kind="controlled_synthetic_benchmark",
            model_family="ApplicabilityBoundaryGuard",
            governance_status="not_calibrated",
            uncertainty_mechanism="Mandatory abstention on asset/domain mismatch",
            scope_note=(
                "Blocks blind model transfer across unrelated machines without asset calibration."
            ),
        ),
    ]
    return MultiModelCatalogResponse(
        separation_disclaimer=(
            "Real MetroPT-3 railway compressor operational telemetry and the UCI AI4I 2020 "
            "synthetic milling benchmark are evaluated under completely separate governance tracks."
        ),
        specialists=specialists,
        ai4i_benchmark_available=report is not None,
        ai4i_selected_model_id=report.selected_model_id if report else None,
        ai4i_promoted=report.promotion_contract.promoted if report else None,
    )


@router.get("/benchmarks/ai4i2020", response_model=MultiModelBenchmarkReport)
def get_ai4i_benchmark(
    request: Request,
    _: Annotated[Principal, Depends(require_data_source(Permission.ANOMALY_READ))],
) -> MultiModelBenchmarkReport:
    try:
        return load_ai4i_benchmark_report(request.app.state.settings.artifacts_dir)
    except BenchmarkUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/predict", response_model=ModelRoutingDecision)
def predict_ai4i_sample(
    body: Ai4iPredictRequest,
    request: Request,
    _: Annotated[Principal, Depends(require_data_source(Permission.ANOMALY_READ))],
) -> ModelRoutingDecision:
    orchestrator = get_ai4i_orchestrator(request)
    return orchestrator.predict(
        asset_id=body.asset_id,
        product_type=body.product_type,
        air_temperature_k=body.air_temperature_k,
        process_temperature_k=body.process_temperature_k,
        rotational_speed_rpm=body.rotational_speed_rpm,
        torque_nm=body.torque_nm,
        tool_wear_min=body.tool_wear_min,
        model_id=body.model_id,
        quality_status=body.quality_status,
    )
