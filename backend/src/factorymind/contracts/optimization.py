"""Canonical contracts for impact estimation and bounded maintenance option comparison."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from factorymind.config import IDENTIFIER
from factorymind.contracts.rca import RcaHypothesisCategory

ImpactDataTier = Literal[
    "measured",
    "engineering_estimate",
    "configurable_economic_assumption",
    "simulated_projection",
]
MaintenanceOptionKind = Literal["repair_now", "schedule", "monitor"]


class EconomicAssumptions(BaseModel):
    """Explicitly configurable economic & power-rating assumptions (never hidden)."""

    model_config = ConfigDict(extra="forbid")

    rated_apu_power_kw: float = Field(default=4.5, ge=1.0, le=15.0)
    unloaded_power_fraction: float = Field(default=0.45, ge=0.1, le=0.8)
    nominal_baseline_duty: float = Field(default=0.15, ge=0.05, le=0.5)
    electricity_eur_per_kwh: float = Field(default=0.18, ge=0.01, le=2.0)
    planned_depot_maintenance_eur: float = Field(default=650.0, ge=50.0, le=50_000.0)
    monitoring_check_eur: float = Field(default=120.0, ge=10.0, le=10_000.0)
    unplanned_service_disruption_eur_per_hr: float = Field(default=1800.0, ge=100.0, le=100_000.0)
    planning_horizon_hours: float = Field(default=8.0, ge=1.0, le=72.0)
    leak_rate_bar_per_min: float = Field(default=0.35, ge=0.0, le=1.5)


class OptimizationConstraints(BaseModel):
    """Hard operational safety & duty boundaries used to gate option feasibility."""

    model_config = ConfigDict(extra="forbid")

    require_zero_lps_violation: bool = True
    max_allowable_duty: float = Field(default=0.85, ge=0.5, le=1.0)
    max_allowable_oil_temp_c: float = Field(default=95.0, ge=60.0, le=130.0)


class OptimizationWeights(BaseModel):
    """Normalized relative importance weights across cost, safety margin & downtime."""

    model_config = ConfigDict(extra="forbid")

    cost_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    reliability_weight: float = Field(default=0.40, ge=0.0, le=1.0)
    downtime_weight: float = Field(default=0.25, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _positive_total(self) -> OptimizationWeights:
        if self.cost_weight + self.reliability_weight + self.downtime_weight <= 0.0:
            raise ValueError("Sum of optimization weights must be strictly positive")
        return self


class ImpactQuantity(BaseModel):
    """Single quantified metric tagged cleanly by its epistemic/provenance tier."""

    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    value: float
    unit: str
    tier: ImpactDataTier
    basis_note: str


class MaintenanceOptionEvaluation(BaseModel):
    """Constrained multi-criteria trade-off assessment for one maintenance candidate."""

    model_config = ConfigDict(extra="forbid")

    option: MaintenanceOptionKind
    title: str
    feasible: bool
    constraint_violations: list[str]
    pareto_dominated: bool
    projected_energy_kwh: float = Field(ge=0.0)
    projected_excess_energy_kwh: float
    projected_lps_risk_minutes: float = Field(ge=0.0)
    projected_unplanned_disruption_hours: float = Field(ge=0.0)
    expected_total_cost_eur: float = Field(ge=0.0)
    bounded_utility_score: float = Field(ge=0.0, le=1.0)
    rationale: str


class MaintenanceDecisionRequest(BaseModel):
    """Input bundle combining measured incident features with bounded assumptions."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(
        default="metropt3-apu-1",
        min_length=1,
        max_length=64,
        pattern=IDENTIFIER.pattern,
    )
    measured_load_duty: float = Field(default=0.72, ge=0.0, le=1.0)
    measured_motor_current_a: float = Field(default=6.4, ge=0.0, le=20.0)
    measured_oil_temp_c: float = Field(default=72.0, ge=-20.0, le=150.0)
    dominant_hypothesis: RcaHypothesisCategory | None = None
    assumptions: EconomicAssumptions = Field(default_factory=EconomicAssumptions)
    constraints: OptimizationConstraints = Field(default_factory=OptimizationConstraints)
    weights: OptimizationWeights = Field(default_factory=OptimizationWeights)


class MaintenanceDecisionComparisonResponse(BaseModel):
    """Stratified impact report + constrained repair_now vs schedule vs monitor comparison."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    incident_id: str | None = None
    provenance_origin: Literal["advisory_decision_comparison"] = "advisory_decision_comparison"
    impact_quantities: list[ImpactQuantity]
    options: list[MaintenanceOptionEvaluation]
    recommended_option: MaintenanceOptionKind
    recommendation_note: str
    uncertainty_warnings: list[str]
