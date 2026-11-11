"""Canonical contracts for Industrial Asset Digital Passports and Data Foundation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SensorPassport(BaseModel):
    """Specification and observability contract for one physical or logical sensor."""

    model_config = ConfigDict(extra="forbid")

    sensor_id: str
    display_name: str
    kind: Literal["analog", "digital"]
    unit: str
    physical_role: str
    valid_range: tuple[float, float]
    nominal_envelope: str
    unobservability_note: str | None = None


class FailureModeSpec(BaseModel):
    """Domain failure or instrumentation mode documented in an asset passport."""

    model_config = ConfigDict(extra="forbid")

    mode_id: str
    display_name: str
    description: str
    primary_signatures: list[str]
    competing_explanations: list[str]
    recommended_checks: list[str]


class ModelApplicabilityEntry(BaseModel):
    """Governing applicability and boundary status for a model family on this asset."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    status: Literal["active", "experimental", "not_calibrated"]
    scope_note: str
    limitations: list[str]


class DataLineageStep(BaseModel):
    """One non-destructive transformation stage linking raw archive to normalized parquet."""

    model_config = ConfigDict(extra="forbid")

    step: int = Field(ge=1)
    operation: str
    description: str
    preserves_raw_source: bool = True


class DatasetQualitySummary(BaseModel):
    """Observed quality metrics recorded during non-destructive dataset preparation."""

    model_config = ConfigDict(extra="forbid")

    rows_raw: int = Field(ge=0)
    rows_cleaned: int = Field(ge=0)
    start: str
    end: str
    duplicate_timestamps: int = Field(ge=0)
    null_values: int = Field(ge=0)
    out_of_range: dict[str, int] = Field(default_factory=dict)
    non_binary_digital: dict[str, int] = Field(default_factory=dict)
    median_period_s: float | None = None
    gaps_over_60s: int = Field(ge=0)
    gap_hours: float = Field(ge=0.0)
    gap_policy: str


class QualityCheckItem(BaseModel):
    """Result of one automated dataset provenance or structural quality verification."""

    model_config = ConfigDict(extra="forbid")

    check_id: str
    status: Literal["pass", "warn", "unavailable"]
    summary: str


class SourceDatasetRecord(BaseModel):
    """Provenance, integrity, quality, and non-destructive lineage for a dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    title: str
    asset_ids: list[str]
    source_url: str
    source_page: str
    license: str
    archive_sha256: str
    raw_preserved: bool
    manifest_present: bool
    telemetry_present: bool
    artifact_status: Literal["verified", "partial", "catalog_only"]
    provenance_origin: Literal["observed", "derived"] = "observed"
    failure_report_count: int = Field(ge=0)
    quality: DatasetQualitySummary | None = None
    quality_checks: list[QualityCheckItem]
    lineage: list[DataLineageStep]


class AssetPassport(BaseModel):
    """Industrial Asset Digital Passport describing physics, sensors, failure modes, and models."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    name: str
    asset_type: str
    system_context: str
    site_or_domain: str
    deployment_scope_note: str
    nominal_sampling_period_s: float = Field(gt=0.0)
    operating_regimes: list[str]
    operating_envelopes: dict[str, str]
    sensors: list[SensorPassport]
    failure_modes: list[FailureModeSpec]
    connected_dataset_ids: list[str]
    model_applicability: list[ModelApplicabilityEntry]
    unobserved_quantities: list[str]
    safety_boundary: Literal["advisory-read-only"] = "advisory-read-only"


class AssetPassportSummary(BaseModel):
    """Concise summary row for the /assets registry index."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    name: str
    asset_type: str
    sensor_count: int = Field(ge=0)
    failure_mode_count: int = Field(ge=0)
    connected_dataset_ids: list[str]
    active_models: list[str]
    telemetry_available: bool
    runtime_model_version: str | None = None
    safety_boundary: Literal["advisory-read-only"] = "advisory-read-only"


class AssetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[AssetPassportSummary]


class AssetDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passport: AssetPassport
    datasets: list[SourceDatasetRecord]
    runtime_model_version: str | None = None


class DatasetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[SourceDatasetRecord]
