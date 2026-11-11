"""Shared typed contracts for WP-A Event Evidence Packages, Quality, and Timelines.

Every datum carries an explicit ``origin`` so observed sensor readings are never
confused with derived features, statistical exceedances, model predictions, or
human field observations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from factorymind.anomaly.events import EventClass
from factorymind.anomaly.regimes import OperatingRegime

EvidenceOrigin = Literal[
    "observed",
    "derived",
    "statistical_inference",
    "model_prediction",
    "human_observation",
]
EventPhaseName = Literal[
    "pre_event_baseline",
    "onset",
    "evolution",
    "peak",
    "recovery_or_persistence",
]
EvidenceQualityStatus = Literal["sufficient", "degraded", "insufficient"]
HumanObservationCategory = Literal[
    "field_inspection",
    "sensor_check",
    "maintenance_note",
    "process_context",
]
TimelineCategory = Literal[
    "telemetry_window",
    "regime_change",
    "alarm_episode",
    "consistency_warning",
    "model_output",
    "evidence_group",
    "human_observation",
    "operator_report",
]


class ObservedSignalSummary(BaseModel):
    """Aggregate of directly observed telemetry values for one channel within a phase."""

    signal: str
    unit: str
    origin: Literal["observed"] = "observed"
    sample_count: int
    missing_count: int = 0
    out_of_range_count: int = 0
    min: float | None = None
    mean: float | None = None
    max: float | None = None
    std: float | None = None
    last: float | None = None
    active_fraction: float | None = None
    frozen: bool = False
    provenance: str


class DerivedFeatureSummary(BaseModel):
    """Aggregate of computed 10-minute window features within a phase."""

    name: str
    unit: str
    origin: Literal["derived"] = "derived"
    value: float | None = None
    observable: bool
    note: str | None = None
    provenance: str


class PhaseEvidenceSummary(BaseModel):
    """Evidence summary for one chronological event phase."""

    phase: EventPhaseName
    label: str
    start: datetime
    end: datetime
    available: bool
    reason_unavailable: str | None = None
    expected_windows: int
    total_windows: int
    valid_windows: int
    sample_count: int
    dominant_regime: OperatingRegime | None = None
    regime_counts: dict[str, int] = Field(default_factory=dict)
    primary_peak_score: float | None = None
    primary_min_exceedance: float | None = None
    primary_model_origin: Literal["model_prediction"] = "model_prediction"
    exceedance_origin: Literal["statistical_inference"] = "statistical_inference"
    warning_peak_score: float | None = None
    warning_min_exceedance: float | None = None
    warning_active_windows: int = 0
    observed_signals: list[ObservedSignalSummary] = Field(default_factory=list)
    derived_features: list[DerivedFeatureSummary] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)


class ProvenanceRecord(BaseModel):
    """Dataset, preprocessing, and detector version lineage for one evidence package."""

    dataset_id: str
    dataset_source: str
    dataset_license: str
    archive_sha256: str | None = None
    preprocessing_version: str
    sampling_contract: str
    window_length: str
    primary_detector: str
    primary_model_version: str
    warning_detector: str | None = None
    warning_model_version: str | None = None


class ContradictionRecord(BaseModel):
    """Explicit signal inconsistency or integrity conflict observed around the event."""

    kind: Literal[
        "valve_contradiction",
        "motor_valve_mismatch",
        "sensor_freeze",
        "out_of_range",
        "model_divergence",
    ]
    timestamp: datetime
    severity: Literal["warning", "critical"]
    signals: list[str]
    description: str
    origin: EvidenceOrigin = "derived"


class EvidenceQualityAssessment(BaseModel):
    """Explicit evidence coverage, missing-data, freeze, and contradiction evaluation."""

    overall_status: EvidenceQualityStatus
    completeness_score: float = Field(ge=0.0, le=1.0)
    coverage_ratio: float = Field(ge=0.0, le=1.0)
    baseline_available: bool
    recovery_available: bool
    recovery_status: Literal["recovered", "persisting", "stopped", "unverified"]
    expected_event_windows: int
    valid_event_windows: int
    missing_windows_count: int
    missing_samples_count: int
    stale_windows_count: int
    frozen_sensors: list[str] = Field(default_factory=list)
    out_of_range_signals: list[str] = Field(default_factory=list)
    contradictions: list[ContradictionRecord] = Field(default_factory=list)
    alignment_issues: list[str] = Field(default_factory=list)
    confidence_penalty_factor: float = Field(ge=0.0, le=1.0)
    quality_warnings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class EvidenceItemContract(BaseModel):
    """Grouped feature deviation paired with explicit inferential origin and unit."""

    group: str
    group_label: str
    feature: str
    value: float
    unit: str
    training_median: float
    robust_z: float | None = None
    novel: bool
    meaning: str
    origin: Literal["statistical_inference", "derived"] = "statistical_inference"
    phase: EventPhaseName = "peak"
    source_id: str


class NeighboringEventSummary(BaseModel):
    """Alarm, warning, or weak operator report adjacent to or overlapping the target event."""

    kind: Literal["air_leak_alarm", "sensor_consistency_warning", "operator_report"]
    event_id: str | None = None
    title: str
    start: datetime
    end: datetime
    raised_at: datetime | None = None
    detector_or_source: str
    relationship: Literal["concurrent", "preceding", "following"]
    origin: EvidenceOrigin


class HumanObservationOut(BaseModel):
    """Append-only human field observation linked to an incident."""

    id: str
    incident_id: str
    asset_id: str
    actor_id: str
    category: HumanObservationCategory
    observed_at: datetime | None = None
    note: str
    created_at: datetime
    origin: Literal["human_observation"] = "human_observation"


class TimelineEntry(BaseModel):
    """Chronological event-timeline entry preserving original timestamps and origins."""

    timestamp: datetime
    end_timestamp: datetime | None = None
    category: TimelineCategory
    origin: EvidenceOrigin
    phase: EventPhaseName | None = None
    title: str
    summary: str
    source_id: str
    detector: str | None = None
    model_version: str | None = None
    regime: OperatingRegime | None = None
    event_class: EventClass | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class EventEvidencePackage(BaseModel):
    """Complete multi-phase, provenance-tagged event evidence package and timeline."""

    package_id: str
    asset_id: str
    incident_id: str | None = None
    event_start: datetime
    event_end: datetime
    raised_at: datetime | None = None
    lookback_start: datetime
    recovery_end: datetime
    provenance: ProvenanceRecord
    phases: list[PhaseEvidenceSummary]
    evidence_items: list[EvidenceItemContract]
    quality: EvidenceQualityAssessment
    missing_measurements: list[str]
    conflicting_measurements: list[str]
    neighboring_events: list[NeighboringEventSummary]
    human_observations: list[HumanObservationOut]
    timeline: list[TimelineEntry]
    advisory: str = (
        "Advisory evidence package for human review. Observed sensor values, derived features, "
        "statistical exceedances, model predictions, and human notes remain separated. "
        "FactoryMind never controls equipment."
    )
