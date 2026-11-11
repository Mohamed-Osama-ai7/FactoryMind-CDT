"""Canonical contracts for Physics-Informed Digital Twin calibration, validation & simulation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from factorymind.config import IDENTIFIER

TwinScenarioKind = Literal[
    "baseline",
    "leak",
    "continued_operation",
    "maintenance_repair",
]


class TwinParameterEstimate(BaseModel):
    """Calibrated normalized ODE parameter fitted strictly on normal train/calibration data."""

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    unit: str
    identifiable: bool = True
    fit_source: str
    uncertainty_note: str


class UnidentifiableParameter(BaseModel):
    """Explicit physical quantity that cannot be uniquely identified from MetroPT-3 telemetry."""

    model_config = ConfigDict(extra="forbid")

    name: str
    reason: str
    assumption_or_proxy: str


class TwinHorizonEvaluation(BaseModel):
    """Per-horizon out-of-sample metrics separating teacher-forced vs closed-loop accuracy."""

    model_config = ConfigDict(extra="forbid")

    horizon_minutes: float = Field(gt=0.0)
    segment_count: int = Field(ge=0)
    teacher_forced_tp3_mae_bar: float = Field(ge=0.0)
    teacher_forced_tp3_median_mae_bar: float = Field(ge=0.0)
    teacher_forced_tp3_rmse_bar: float = Field(ge=0.0)
    closed_loop_tp3_mae_bar: float = Field(ge=0.0)
    closed_loop_tp3_median_mae_bar: float = Field(ge=0.0)
    closed_loop_tp3_rmse_bar: float = Field(ge=0.0)
    persistence_tp3_mae_bar: float = Field(ge=0.0)
    persistence_tp3_median_mae_bar: float = Field(ge=0.0)
    train_mean_tp3_mae_bar: float = Field(ge=0.0)
    train_mean_tp3_median_mae_bar: float = Field(ge=0.0)
    teacher_forced_skill_vs_persistence: float
    teacher_forced_skill_vs_train_mean: float
    closed_loop_skill_vs_persistence: float
    closed_loop_skill_vs_train_mean: float
    teacher_forced_supported: bool
    closed_loop_supported: bool
    duty_cycle_mae: float = Field(ge=0.0, le=1.0)
    duty_cycle_median_mae: float = Field(ge=0.0, le=1.0)
    persistence_duty_mae: float = Field(ge=0.0, le=1.0)
    train_mean_duty_mae: float = Field(ge=0.0, le=1.0)
    train_mean_duty_median_mae: float = Field(ge=0.0, le=1.0)
    duty_skill_vs_train_mean: float
    duty_cycle_supported: bool
    observed_cycles_per_segment: float = Field(ge=0.0)
    simulated_cycles_per_segment: float = Field(ge=0.0)
    cycle_count_mae: float = Field(ge=0.0)
    closed_loop_oil_mae_c: float = Field(ge=0.0)
    persistence_oil_mae_c: float = Field(ge=0.0)
    train_mean_oil_mae_c: float = Field(ge=0.0)
    support_note: str


class TwinSensitivityCheck(BaseModel):
    """Pre-evaluation parameter perturbation sensitivity (never touches held-out data)."""

    model_config = ConfigDict(extra="forbid")

    perturbation: str
    evaluation_scope: Literal["pre_evaluation_only"] = "pre_evaluation_only"
    segment_count: int = Field(ge=0)
    baseline_tp3_mae_10min_bar: float = Field(ge=0.0)
    perturbed_tp3_mae_10min_bar: float = Field(ge=0.0)
    delta_tp3_mae_bar: float
    baseline_duty_mae_10min: float = Field(ge=0.0)
    perturbed_duty_mae_10min: float = Field(ge=0.0)
    delta_duty_mae: float


class TwinValidationSummary(BaseModel):
    """Out-of-sample held-out trajectory accuracy, baseline comparisons, stability & limits."""

    model_config = ConfigDict(extra="forbid")

    split_policy: str
    train_cal_samples: int = Field(ge=0)
    heldout_normal_segments: int = Field(ge=0)
    mae_10min_bar: float = Field(ge=0.0)
    mae_60min_bar: float = Field(ge=0.0)
    rmse_60min_bar: float = Field(ge=0.0)
    closed_loop_mae_10min_bar: float = Field(ge=0.0)
    closed_loop_mae_60min_bar: float = Field(ge=0.0)
    closed_loop_rmse_60min_bar: float = Field(ge=0.0)
    max_supported_closed_loop_horizon_minutes: float = Field(ge=0.0)
    max_supported_teacher_forced_horizon_minutes: float = Field(ge=0.0)
    max_supported_duty_horizon_minutes: float = Field(ge=0.0)
    unsupported_horizons: list[str]
    horizon_evaluations: list[TwinHorizonEvaluation]
    sensitivity_checks: list[TwinSensitivityCheck]
    calibration_notes: list[str]
    constraint_violations: int = Field(ge=0)
    step_halving_max_diff_bar: float = Field(ge=0.0)
    leak_vs_normal_residual_ratio: float = Field(ge=0.0)
    provenance_origin: Literal["derived", "empirical_validation"] = "empirical_validation"


class TwinModelResponse(BaseModel):
    """Calibrated Physics-Informed Digital Twin specification & held-out validation."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    model_version: str
    provenance_origin: Literal["advisory_physics_twin"] = "advisory_physics_twin"
    governing_equations: list[str]
    parameters: list[TwinParameterEstimate]
    unidentifiable_parameters: list[UnidentifiableParameter]
    validation: TwinValidationSummary
    advisory_note: str


class TwinSimulationRequest(BaseModel):
    """Counterfactual digital-twin simulation scenario request (advisory only)."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(
        default="metropt3-apu-1",
        min_length=1,
        max_length=64,
        pattern=IDENTIFIER.pattern,
    )
    scenario: TwinScenarioKind = "baseline"
    horizon_minutes: float = Field(default=60.0, ge=10.0, le=240.0)
    initial_tp3_bar: float = Field(default=9.1, ge=1.0, le=12.0)
    initial_oil_temp_c: float = Field(default=62.0, ge=10.0, le=120.0)
    leak_rate_bar_per_min: float = Field(default=0.35, ge=0.0, le=1.5)
    demand_factor: float = Field(default=1.0, ge=0.5, le=3.0)
    repair_at_minute: float | None = Field(default=30.0, ge=1.0, le=239.0)

    @model_validator(mode="after")
    def _validate_repair_horizon(self) -> TwinSimulationRequest:
        if self.scenario == "maintenance_repair":
            repair_min = self.repair_at_minute if self.repair_at_minute is not None else 30.0
            if repair_min >= self.horizon_minutes:
                raise ValueError(
                    "repair_at_minute must be strictly less than horizon_minutes for "
                    "maintenance_repair scenarios"
                )
        return self


class TwinTrajectoryPoint(BaseModel):
    """Single physics-simulated state sample explicitly marked as simulated projection."""

    model_config = ConfigDict(extra="forbid")

    minute: float = Field(ge=0.0)
    tp3_bar: float
    reservoirs_bar: float
    oil_temp_c: float
    motor_loaded: bool
    rolling_duty: float = Field(ge=0.0, le=1.0)
    lps_active: bool
    regime: str
    provenance_origin: Literal["simulated_projection"] = "simulated_projection"


class TwinScenarioSummary(BaseModel):
    """Aggregate KPIs and constraint audit for one counterfactual twin trajectory."""

    model_config = ConfigDict(extra="forbid")

    scenario: TwinScenarioKind
    mean_load_duty: float = Field(ge=0.0, le=1.0)
    peak_load_duty: float = Field(ge=0.0, le=1.0)
    min_tp3_bar: float
    max_tp3_bar: float
    lps_excursion_minutes: float = Field(ge=0.0)
    time_above_duty_threshold_minutes: float = Field(ge=0.0)
    final_oil_temp_c: float
    constraint_violations: int = Field(ge=0)
    advisory_interpretation: str


class TwinSimulationResponse(BaseModel):
    """Controlled physics-informed scenario rollout clearly separated from observed telemetry."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    scenario: TwinScenarioKind
    provenance_origin: Literal["simulated_projection"] = "simulated_projection"
    horizon_supported: bool
    horizon_support_status: Literal["supported_closed_loop", "unsupported_closed_loop_horizon"]
    horizon_warning: str | None = None
    params_used: list[TwinParameterEstimate]
    unidentifiable_parameters: list[UnidentifiableParameter]
    trajectory: list[TwinTrajectoryPoint]
    summary: TwinScenarioSummary
    advisory_note: str
