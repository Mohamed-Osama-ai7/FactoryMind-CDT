"""Anomaly-detection API (advisory, read-only).

- ``GET  /api/v1/anomaly/model``   current model card summary
- ``POST /api/v1/anomaly/score``   score one submitted block of APU samples (stateless)
- ``GET  /api/v1/anomaly/replay``  score recorded MetroPT-3 telemetry over a time range
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime
from itertools import pairwise
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from factorymind.anomaly.events import EventClass
from factorymind.anomaly.events import WarningEpisode as ServiceWarningEpisode
from factorymind.anomaly.evidence import EvidenceItem
from factorymind.anomaly.evidence_package import build_event_evidence_package
from factorymind.anomaly.features import WindowSpec
from factorymind.anomaly.regimes import OperatingRegime, RegimeConfidence
from factorymind.anomaly.service import (
    AnomalyService,
    DataUnavailableError,
    ModelUnavailableError,
    ReplayEpisode,
    WindowScore,
    load_bundle,
    load_warning_bundle,
)
from factorymind.auth.deps import require_data_source
from factorymind.auth.principal import Permission
from factorymind.contracts.events import EventEvidencePackage
from factorymind.data.metropt3 import (
    ANALOG_SENSORS,
    DIGITAL_SENSORS,
    MAX_PERIOD_S,
    MIN_PERIOD_S,
    VALID_RANGES,
    FailureReport,
)

_SCORE_SPEC = WindowSpec()
MAX_SAMPLES = 2000
MAX_REPLAY = pd.Timedelta(days=7)
SCORE_WINDOW = pd.Timedelta(_SCORE_SPEC.length)
SCORE_SPAN_TOLERANCE = pd.Timedelta(seconds=_SCORE_SPEC.span_tolerance_s)
MIN_SCORE_SAMPLES = _SCORE_SPEC.min_samples
MIN_SCORE_SPAN = pd.Timedelta(seconds=_SCORE_SPEC.min_block_span_s)
MAX_SCORE_SPAN = pd.Timedelta(seconds=_SCORE_SPEC.max_block_span_s)

router = APIRouter(prefix="/api/v1/anomaly", tags=["anomaly"])


# ---------------------------------------------------------------- service dependency


def get_anomaly_service(request: Request) -> AnomalyService:
    """Load the current model once per application and keep it on ``app.state``."""
    state = request.app.state
    service: AnomalyService | None = getattr(state, "anomaly_service", None)
    if service is None:
        settings = state.settings
        try:
            bundle, card = load_bundle(settings.models_dir, settings.anomaly_model_name)
            warning_bundle = load_warning_bundle(
                settings.models_dir, settings.anomaly_model_name, bundle
            )
        except ModelUnavailableError as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        service = AnomalyService(
            bundle, card, settings.processed_dir, warning_bundle=warning_bundle
        )
        state.anomaly_service = service
    return service


ServiceDep = Annotated[AnomalyService, Depends(get_anomaly_service)]


# ---------------------------------------------------------------- schemas


class Sample(BaseModel):
    """One APU telemetry sample (MetroPT-3 schema)."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    TP2: float
    TP3: float
    H1: float
    DV_pressure: float
    Reservoirs: float
    Oil_temperature: float
    Motor_current: float
    COMP: Literal[0, 1]
    DV_eletric: Literal[0, 1]
    Towers: Literal[0, 1]
    MPG: Literal[0, 1]
    LPS: Literal[0, 1]
    Pressure_switch: Literal[0, 1]
    Oil_level: Literal[0, 1]
    Caudal_impulses: Literal[0, 1]

    @model_validator(mode="after")
    def _physical_ranges(self) -> Sample:
        for name, (lo, hi) in VALID_RANGES.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name}={value} outside plausible range [{lo}, {hi}]")
        return self


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: list[Sample] = Field(
        min_length=1,
        max_length=MAX_SAMPLES,
        description="Chronological samples forming one 10-minute window at ~0.1 Hz "
        "(>= 48 samples required to score; total span 7-11 min with 60 s tolerance, "
        "median sampling period 5-15 s, internal gaps <= 60 s, no silent resampling).",
    )

    @model_validator(mode="after")
    def _chronological(self) -> ScoreRequest:
        stamps = [s.timestamp for s in self.samples]
        if any(s.tzinfo is not None for s in stamps):
            raise ValueError("timestamps must be naive local train time, like the dataset")
        if any(b <= a for a, b in pairwise(stamps)):
            raise ValueError("sample timestamps must be strictly increasing")
        span = pd.Timestamp(stamps[-1]) - pd.Timestamp(stamps[0])
        if span > MAX_SCORE_SPAN:
            raise ValueError(
                "sample time span exceeds the 10-minute scoring window (60s tolerance)"
            )
        if len(stamps) >= MIN_SCORE_SAMPLES:
            if span < MIN_SCORE_SPAN:
                raise ValueError(
                    "sample time span is below minimum 10-minute temporal coverage "
                    f"({int(_SCORE_SPEC.min_block_span_s)}s required)"
                )
            deltas = [(b - a).total_seconds() for a, b in pairwise(stamps)]
            median_period = float(statistics.median(deltas))
            if not MIN_PERIOD_S <= median_period <= MAX_PERIOD_S:
                raise ValueError(
                    f"median sampling period {median_period:.2f}s violates the "
                    f"{MIN_PERIOD_S:g}-{MAX_PERIOD_S:g}s (~0.1 Hz) contract; "
                    "submit native 0.1 Hz telemetry without silent resampling"
                )
        return self

    def to_frame(self) -> pd.DataFrame:
        rows = [s.model_dump() for s in self.samples]
        frame = pd.DataFrame(rows).set_index("timestamp")
        frame.index = pd.DatetimeIndex(frame.index)
        return frame[[*ANALOG_SENSORS, *DIGITAL_SENSORS]].astype("float64")


class Evidence(BaseModel):
    group: str = Field(description="Physical signal group; at most one item per group.")
    group_label: str
    feature: str
    value: float
    training_median: float
    robust_z: float | None = Field(
        description="Deviation from the training distribution; null when the feature never "
        "varied in training (see `novel`). A deviation is not a cause."
    )
    novel: bool = Field(description="The feature never varied in training and now differs.")
    meaning: str


class RegimeInfo(BaseModel):
    regime: OperatingRegime
    confidence: RegimeConfidence
    reason: str
    summary: str
    quality_flags: list[str]
    missing_signals: list[str]
    off_fraction: float | None = None
    unloaded_fraction: float | None = None
    load_duty: float | None = None
    load_starts: float | None = None
    inrush_samples: int = 0
    valve_contradiction_fraction: float | None = None
    motor_valve_mismatch_fraction: float | None = None
    minutes_since_resume: float | None = None
    stale_telemetry: bool = False


class WindowResult(BaseModel):
    window_start: datetime
    window_end: datetime
    n_samples: int
    status: Literal["normal", "anomalous", "stopped", "insufficient_data"] = Field(
        description="`stopped` is an explained state (machine off), not an anomaly."
    )
    score: float | None
    exceedance: float | None = Field(
        description="Share of reference-normal windows scoring at least this high (smoothed). "
        "Not a failure probability."
    )
    evidence: list[Evidence]
    regime: RegimeInfo
    event_class: EventClass
    event_meaning: str
    warning_active: bool = False
    warning_score: float | None = None
    warning_exceedance: float | None = None
    warning_detector: str | None = None
    warning_reasons: list[str] = Field(default_factory=list)
    warning_evidence: list[Evidence] = Field(default_factory=list)


class ScoreResponse(BaseModel):
    model_version: str
    threshold: float
    exceedance_at_threshold: float
    warning_detector: str | None = None
    warning_threshold: float | None = None
    result: WindowResult
    advisory: str = "Advisory output for human review. FactoryMind never controls equipment."


class AlarmEpisode(BaseModel):
    start: datetime
    end: datetime
    raised_at: datetime
    peak_score: float
    truncated: bool = Field(description="The run may have started before the scanned range.")
    ongoing: bool = Field(description="Still anomalous at the end of the requested range.")
    event_class: EventClass = "air_leak_alarm"
    hypothesis: str = ""
    uncertainty_note: str = ""
    dominant_regime: OperatingRegime = "insufficient_evidence"
    regime_breakdown: dict[str, int] = Field(default_factory=dict)


class WarningEpisode(BaseModel):
    start: datetime
    end: datetime
    raised_at: datetime
    peak_score: float | None = None
    peak_exceedance: float | None = None
    detector_name: str
    warning_kind: str
    sources: list[str]
    truncated: bool
    ongoing: bool
    event_class: EventClass = "sensor_consistency_warning"
    hypothesis: str
    uncertainty_note: str
    dominant_regime: OperatingRegime
    regime_breakdown: dict[str, int] = Field(default_factory=dict)


class OperatorReport(BaseModel):
    report_id: str
    start: datetime
    end: datetime
    failure: str


class ReplayResponse(BaseModel):
    model_version: str
    threshold: float
    persistence: int
    warning_detector: str | None = None
    warning_threshold: float | None = None
    warning_persistence: int | None = None
    start: datetime
    end: datetime
    windows: list[WindowResult]
    episodes: list[AlarmEpisode]
    warnings: list[WarningEpisode] = Field(default_factory=list)
    operator_reports: list[OperatorReport]


class LoeoFold(BaseModel):
    held_out: str
    detected: bool
    delay_minutes: float | None


class LoeoSummary(BaseModel):
    events_detected: int
    events_total: int
    mean_false_alarms_per_day: float = Field(
        description="Mean false alarms per all valid evaluation days (includes stopped days)."
    )
    mean_operating_false_alarms_per_day: float = Field(
        description="Mean false alarms per operating evaluation day (excludes stopped days)."
    )
    median_abs_delay_minutes: float | None
    folds: list[LoeoFold]


class FaultSummary(BaseModel):
    detected: int
    cases: int
    gross_rate: float | None = Field(
        description="Share of faulted intervals that raised an alarm (includes control alarms)."
    )
    control_detected: int
    control_alarm_rate: float | None = Field(
        description="Share of unmodified control intervals that already raised an alarm."
    )
    control_alarms: str
    clean_control_cases: int
    clean_control_detected: int
    clean_control_sensitivity: float | None = Field(
        description="Share of clean (non-alarming) control intervals where the injected fault "
        "raised an alarm."
    )
    incremental_lift: float | None = Field(description="gross_rate minus control_alarm_rate.")
    per_type: dict[str, str]
    simulated: bool


class ReferenceHoldoutCheck(BaseModel):
    alpha: float = Field(description="Nominal exceedance alpha tested on split-half holdout.")
    share_at_or_below_alpha: float | None = Field(
        description="Observed share of held-out reference windows at or below alpha. "
        "Values well above alpha show reference drift; exceedance is not a failure probability."
    )
    windows: int


class ExceedanceCheck(BaseModel):
    alpha: float
    share_at_or_below_alpha: float = Field(
        description="Observed share of far-from-report operating windows at or below alpha."
    )
    by_month: dict[str, float]
    windows: int


class RegimeSplitSummary(BaseModel):
    valid_windows: int
    invalid_gapped_windows: int
    counts: dict[str, int]
    insufficient_reasons: dict[str, int]


class Wp2EventSeparation(BaseModel):
    baseline_leak_episodes_wp0: int
    baseline_false_alarm_episodes_wp0: int
    gated_leak_episodes_wp2: int
    gated_false_alarm_episodes_wp2: int
    integrity_suppressed_leak_episodes: int
    events_detected_wp2: int
    events_total: int
    false_alarms_per_day_wp2: float
    operating_false_alarms_per_day_wp2: float
    sensor_consistency_warning_episodes: int
    warnings_per_day: float
    operating_warnings_per_day: float
    warning_only_episodes: int
    warning_overlapping_leak_episodes: int
    warning_kinds: dict[str, int]
    false_alarms_by_dominant_regime: dict[str, int]
    starting_overlapping_false_alarms: int
    mixed_cycle_overlapping_false_alarms: int


class Wp2RegimePrimaryMetrics(BaseModel):
    scored_windows: int
    exceedance_windows: int
    exceedance_rate: float | None = None
    false_alarm_episodes: int
    overlapping_false_alarm_episodes: int


class Wp2RegimeWarningMetrics(BaseModel):
    scored_windows: int
    exceedance_windows: int
    exceedance_rate: float | None = None
    warning_episodes: int


class Wp2PerRegimeEntry(BaseModel):
    windows: int
    share: float
    primary: Wp2RegimePrimaryMetrics
    warning: Wp2RegimeWarningMetrics | None = None


class Wp2LoeoAblation(LoeoSummary):
    suppressed_states: list[str]
    note: str


class Wp2SimulatedFaultDetector(BaseModel):
    detected: int
    cases: int
    control_detected: int
    clean_control_sensitivity: float | None = None
    incremental_lift: float | None = None
    per_type: dict[str, str]


class Wp2SimulatedFaultSeparation(BaseModel):
    simulated: bool
    real_operational_data: bool
    note: str
    detectors: dict[str, Wp2SimulatedFaultDetector]


class Wp2EvaluationSummary(BaseModel):
    primary_detector: str
    warning_detector: str | None = None
    regime_distribution: dict[str, RegimeSplitSummary]
    event_separation: Wp2EventSeparation
    per_regime_metrics: dict[str, Wp2PerRegimeEntry]
    regime_gating_ablations: dict[str, Wp2LoeoAblation]
    simulated_fault_separation: Wp2SimulatedFaultSeparation


class ModelInfo(BaseModel):
    model_version: str
    detector: str
    selected_as_default: bool
    selection_rationale: str
    dataset_id: str
    dataset_license: str
    dataset_source: str
    data_kind: str
    training_period: list[str]
    threshold: float
    persistence: int
    exceedance_at_threshold: float
    window_length: str
    loeo: LoeoSummary
    fault_injection: FaultSummary | None
    reference_holdout_check: ReferenceHoldoutCheck | None = Field(
        default=None,
        description="Split-half calibration reference stability check. Exceedance is a share of "
        "reference-normal windows, never a failure probability.",
    )
    exceedance_check: ExceedanceCheck | None = Field(
        default=None,
        description="Observed exceedance rate on far-from-report operating evaluation windows.",
    )
    wp2: Wp2EvaluationSummary | None = Field(
        default=None,
        description="WP2 operating-regime separation, consistency warnings, and gating ablations.",
    )
    intended_use: str
    limitations: list[str]
    replay_available_from: datetime | None
    replay_available_to: datetime | None
    operator_reports: list[OperatorReport] = Field(
        description="Operator failure reports used as weak evaluation labels."
    )


# ---------------------------------------------------------------- helpers


def _window(w: WindowScore) -> WindowResult:
    reg = w.regime
    return WindowResult(
        window_start=w.window_start.to_pydatetime(),
        window_end=w.window_end.to_pydatetime(),
        n_samples=w.n_samples,
        status=w.status,
        score=w.score,
        exceedance=w.exceedance,
        evidence=[_evidence(e) for e in w.evidence],
        regime=RegimeInfo(
            regime=reg.regime,
            confidence=reg.confidence,
            reason=reg.reason,
            summary=reg.summary,
            quality_flags=list(reg.quality_flags),
            missing_signals=list(reg.missing_signals),
            off_fraction=reg.off_fraction,
            unloaded_fraction=reg.unloaded_fraction,
            load_duty=reg.load_duty,
            load_starts=reg.load_starts,
            inrush_samples=reg.inrush_samples,
            valve_contradiction_fraction=reg.valve_contradiction_fraction,
            motor_valve_mismatch_fraction=reg.motor_valve_mismatch_fraction,
            minutes_since_resume=reg.minutes_since_resume,
            stale_telemetry=reg.stale_telemetry,
        ),
        event_class=w.event_class,
        event_meaning=w.event_meaning,
        warning_active=w.warning_active,
        warning_score=w.warning_score,
        warning_exceedance=w.warning_exceedance,
        warning_detector=w.warning_detector,
        warning_reasons=list(w.warning_reasons),
        warning_evidence=[_evidence(e) for e in w.warning_evidence],
    )


def _evidence(e: EvidenceItem) -> Evidence:
    return Evidence(
        group=e.group,
        group_label=e.group_label,
        feature=e.feature,
        value=e.value,
        training_median=e.training_median,
        robust_z=e.robust_z,
        novel=e.novel,
        meaning=e.meaning,
    )


def _report(r: FailureReport) -> OperatorReport:
    return OperatorReport(report_id=r.report_id, start=r.start, end=r.end, failure=r.failure)


def _episode(e: ReplayEpisode) -> AlarmEpisode:
    return AlarmEpisode(
        start=e.episode.start.to_pydatetime(),
        end=e.episode.end.to_pydatetime(),
        raised_at=e.episode.raised_at.to_pydatetime(),
        peak_score=e.episode.peak_score,
        truncated=e.truncated,
        ongoing=e.ongoing,
        event_class=e.event_class,
        hypothesis=e.hypothesis,
        uncertainty_note=e.uncertainty_note,
        dominant_regime=e.dominant_regime,
        regime_breakdown=dict(e.regime_breakdown),
    )


def _warning_episode(w: ServiceWarningEpisode) -> WarningEpisode:
    return WarningEpisode(
        start=w.start.to_pydatetime(),
        end=w.end.to_pydatetime(),
        raised_at=w.raised_at.to_pydatetime(),
        peak_score=w.peak_score,
        peak_exceedance=w.peak_exceedance,
        detector_name=w.detector_name,
        warning_kind=w.warning_kind,
        sources=list(w.sources),
        truncated=w.truncated,
        ongoing=w.ongoing,
        event_class=w.event_class,
        hypothesis=w.hypothesis,
        uncertainty_note=w.uncertainty_note,
        dominant_regime=w.dominant_regime,
        regime_breakdown=dict(w.regime_breakdown),
    )


def _loeo(summary: dict[str, Any]) -> LoeoSummary:
    return LoeoSummary(
        events_detected=summary["events_detected"],
        events_total=summary["events_total"],
        mean_false_alarms_per_day=summary["mean_false_alarms_per_day"],
        mean_operating_false_alarms_per_day=summary["mean_operating_false_alarms_per_day"],
        median_abs_delay_minutes=summary["median_abs_delay_minutes"],
        folds=[
            LoeoFold(
                held_out=f["held_out"], detected=f["detected"], delay_minutes=f["delay_minutes"]
            )
            for f in summary["folds"]
        ],
    )


# ---------------------------------------------------------------- routes


@router.get(
    "/model",
    response_model=ModelInfo,
    dependencies=[Depends(require_data_source(Permission.ANOMALY_READ))],
)
def anomaly_model(service: ServiceDep) -> ModelInfo:
    card, bundle = service.card, service.bundle
    data_range = service.data_range()
    evaluation = card["evaluation"]
    faults = evaluation["fault_injection"]
    holdout = evaluation.get("reference_holdout_check")
    exceedance_eval = evaluation.get("exceedance_check")
    wp2_raw = evaluation.get("wp2")
    return ModelInfo(
        model_version=bundle.version,
        detector=bundle.detector_name,
        selected_as_default=evaluation["selected_as_default"],
        selection_rationale=evaluation["selection_rationale"],
        dataset_id=card["dataset"]["id"],
        dataset_license=card["dataset"]["license"],
        dataset_source=card["dataset"]["source"],
        data_kind=card["dataset"]["data_kind"],
        training_period=card["training_period"],
        threshold=bundle.policy.threshold,
        persistence=bundle.policy.persistence,
        exceedance_at_threshold=card["exceedance_at_threshold"],
        window_length=bundle.spec.length,
        loeo=_loeo(evaluation["loeo"]),
        fault_injection=FaultSummary(**{k: faults[k] for k in FaultSummary.model_fields})
        if faults
        else None,
        reference_holdout_check=ReferenceHoldoutCheck(
            **{k: holdout[k] for k in ReferenceHoldoutCheck.model_fields}
        )
        if holdout
        else None,
        exceedance_check=ExceedanceCheck(
            **{k: exceedance_eval[k] for k in ExceedanceCheck.model_fields}
        )
        if exceedance_eval
        else None,
        wp2=Wp2EvaluationSummary.model_validate(wp2_raw) if wp2_raw else None,
        intended_use=card["intended_use"],
        limitations=card["limitations"],
        replay_available_from=data_range[0].to_pydatetime() if data_range else None,
        replay_available_to=data_range[1].to_pydatetime() if data_range else None,
        operator_reports=[_report(r) for r in service.reports],
    )


@router.post(
    "/score",
    response_model=ScoreResponse,
    dependencies=[Depends(require_data_source(Permission.ANOMALY_READ, cost=2))],
)
def anomaly_score(request: ScoreRequest, service: ServiceDep) -> ScoreResponse:
    result = service.score_block(request.to_frame())
    warn_bundle = service.warning_bundle
    return ScoreResponse(
        model_version=service.bundle.version,
        threshold=service.bundle.policy.threshold,
        exceedance_at_threshold=float(
            service.bundle.calibrator.p_value(service.bundle.policy.threshold)[0]
        ),
        warning_detector=warn_bundle.detector_name if warn_bundle else None,
        warning_threshold=warn_bundle.policy.threshold if warn_bundle else None,
        result=_window(result),
    )


@router.get(
    "/replay",
    response_model=ReplayResponse,
    dependencies=[Depends(require_data_source(Permission.ANOMALY_READ, cost=5))],
)
def anomaly_replay(
    service: ServiceDep,
    start: Annotated[datetime, Query(description="Range start (naive local train time)")],
    end: Annotated[datetime, Query(description="Range end, exclusive; at most 7 days after start")],
) -> ReplayResponse:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    if lo.tzinfo is not None or hi.tzinfo is not None:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "use naive timestamps")
    if hi <= lo or hi - lo > MAX_REPLAY:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "end must be after start and within 7 days"
        )
    try:
        out = service.replay_details(lo, hi)
    except DataUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    warn_bundle = service.warning_bundle
    return ReplayResponse(
        model_version=service.bundle.version,
        threshold=service.bundle.policy.threshold,
        persistence=service.bundle.policy.persistence,
        warning_detector=warn_bundle.detector_name if warn_bundle else None,
        warning_threshold=warn_bundle.policy.threshold if warn_bundle else None,
        warning_persistence=warn_bundle.policy.persistence if warn_bundle else None,
        start=start,
        end=end,
        windows=[_window(w) for w in out.windows],
        episodes=[_episode(e) for e in out.episodes],
        warnings=[_warning_episode(w) for w in out.warnings],
        operator_reports=[_report(r) for r in out.reports],
    )


@router.get(
    "/event-evidence",
    response_model=EventEvidencePackage,
    dependencies=[Depends(require_data_source(Permission.ANOMALY_READ, cost=5))],
)
def anomaly_event_evidence(
    request: Request,
    service: ServiceDep,
    start: Annotated[datetime, Query(description="Event start (naive local train time)")],
    end: Annotated[datetime, Query(description="Event end, exclusive; at most 7 days after start")],
    raised_at: Annotated[
        datetime | None, Query(description="Optional alarm raised_at timestamp")
    ] = None,
    baseline_hours: Annotated[float, Query(ge=0.0, le=24.0)] = 4.0,
    recovery_hours: Annotated[float, Query(ge=0.0, le=24.0)] = 2.0,
) -> EventEvidencePackage:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    if (
        lo.tzinfo is not None
        or hi.tzinfo is not None
        or (raised_at is not None and raised_at.tzinfo is not None)
    ):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "use naive timestamps")
    if hi <= lo or hi - lo > MAX_REPLAY:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "end must be after start and within 7 days"
        )
    settings = request.app.state.settings
    try:
        return build_event_evidence_package(
            service,
            asset_id=settings.data_source_asset_id,
            event_start=lo,
            event_end=hi,
            raised_at=pd.Timestamp(raised_at) if raised_at is not None else None,
            baseline_hours=baseline_hours,
            recovery_hours=recovery_hours,
        )
    except DataUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
