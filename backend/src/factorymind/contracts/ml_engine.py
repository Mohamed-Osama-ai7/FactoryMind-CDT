"""Canonical contracts for multi-model ML routing, uncertainty quantification, and calibration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ABSTENTION_MESSAGE = "Insufficient evidence to make a reliable diagnosis"

UncertaintyBand = Literal["low", "moderate", "high", "abstained"]
QualityGateStatus = Literal["healthy", "degraded", "insufficient"]


class ModelCalibrationMetrics(BaseModel):
    """Held-out discrimination, probability calibration, and selective-abstention metrics."""

    model_config = ConfigDict(extra="forbid")

    roc_auc: float = Field(ge=0.0, le=1.0)
    pr_auc: float = Field(ge=0.0, le=1.0)
    brier_score: float = Field(ge=0.0, le=1.0)
    ece_10_bins: float = Field(ge=0.0, le=1.0)
    f1_at_threshold: float = Field(ge=0.0, le=1.0)
    precision_at_threshold: float = Field(ge=0.0, le=1.0)
    recall_at_threshold: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    selective_coverage_at_margin: float = Field(ge=0.0, le=1.0)
    selective_accuracy_at_margin: float | None = Field(default=None, ge=0.0, le=1.0)


class ModelComparisonRow(BaseModel):
    """One model family entry on the AI4I 2020 controlled benchmark."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    model_family: str
    dataset_id: str
    cal_metrics: ModelCalibrationMetrics
    test_metrics: ModelCalibrationMetrics
    ood_threshold: float = Field(gt=0.0)


class ModelPromotionGate(BaseModel):
    """Single objective promotion readiness check against held-out criteria."""

    model_config = ConfigDict(extra="forbid")

    gate_id: str
    passed: bool
    detail: str


class ModelPromotionContract(BaseModel):
    """Governance decision verifying leakage checks, calibration, discrimination, and OOD gate."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    dataset_id: str
    candidate_model_id: str
    promoted: bool
    gates: list[ModelPromotionGate]
    summary: str


class MultiModelBenchmarkReport(BaseModel):
    """Complete controlled benchmark report (kept strictly separate from MetroPT-3 telemetry)."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = "ai4i2020"
    dataset_scope_note: str
    archive_sha256: str
    rows_total: int = Field(ge=1)
    split_counts: dict[str, int]
    leakage_free_features: list[str]
    excluded_target_and_id_columns: list[str]
    models: list[ModelComparisonRow]
    selected_model_id: str
    promotion_contract: ModelPromotionContract
    generated_at: str


class UncertaintyEstimate(BaseModel):
    """Mahalanobis OOD distance and calibrated uncertainty summary for a single prediction."""

    model_config = ConfigDict(extra="forbid")

    mahalanobis_distance: float | None = None
    ood_threshold: float
    is_ood: bool
    uncertainty_band: UncertaintyBand
    calibrated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    decision_margin: float | None = Field(default=None, ge=0.0, le=0.5)


class ModelRoutingDecision(BaseModel):
    """Structured multi-model prediction or explicit conservative abstention."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    selected_model_id: str
    model_family: str
    model_version: str
    abstained: bool
    abstention_message: str | None = None
    abstention_reasons: list[str] = Field(default_factory=list)
    prediction_label: Literal["nominal", "failure_risk"] | None = None
    calibrated_failure_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    predicted_failure_modes: list[str] = Field(default_factory=list)
    uncertainty: UncertaintyEstimate
    provenance_origin: Literal["model_prediction"] = "model_prediction"
    advisory_only: bool = True


class SpecialistModelEntry(BaseModel):
    """Catalog summary describing one registered ML/anomaly model specialist."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    asset_id: str
    dataset_id: str
    domain_kind: Literal["real_operational_telemetry", "controlled_synthetic_benchmark"]
    model_family: str
    governance_status: Literal["active", "candidate", "not_calibrated"]
    uncertainty_mechanism: str
    scope_note: str


class MultiModelCatalogResponse(BaseModel):
    """Separated catalog of specialized models across operational and benchmark domains."""

    model_config = ConfigDict(extra="forbid")

    separation_disclaimer: str
    specialists: list[SpecialistModelEntry]
    ai4i_benchmark_available: bool
    ai4i_selected_model_id: str | None = None
    ai4i_promoted: bool | None = None
