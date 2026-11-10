"""Anomaly-detection API (advisory, read-only).

- ``GET  /api/v1/anomaly/model``   current model card summary
- ``POST /api/v1/anomaly/score``   score one submitted block of APU samples (stateless)
- ``GET  /api/v1/anomaly/replay``  score recorded MetroPT-3 telemetry over a time range
"""

from __future__ import annotations

import math
from datetime import datetime
from functools import lru_cache
from itertools import pairwise
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, model_validator

from factorymind.anomaly.service import (
    AnomalyService,
    DataUnavailableError,
    FeatureEvidence,
    ModelUnavailableError,
    WindowScore,
    load_bundle,
)
from factorymind.config import get_settings
from factorymind.data.metropt3 import (
    ANALOG_SENSORS,
    DIGITAL_SENSORS,
    VALID_RANGES,
    FailureReport,
)

MAX_SAMPLES = 2000
MAX_REPLAY = pd.Timedelta(days=7)

router = APIRouter(prefix="/api/v1/anomaly", tags=["anomaly"])


# ---------------------------------------------------------------- service dependency


@lru_cache(maxsize=1)
def _cached_service() -> AnomalyService:
    settings = get_settings()
    bundle, card = load_bundle(settings.models_dir, settings.anomaly_model_name)
    return AnomalyService(bundle, card, settings.processed_dir)


def get_anomaly_service() -> AnomalyService:
    try:
        return _cached_service()
    except ModelUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


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
        description="Chronological samples forming one window (nominally 10 min at ~0.1 Hz).",
    )

    @model_validator(mode="after")
    def _chronological(self) -> ScoreRequest:
        stamps = [s.timestamp for s in self.samples]
        if any(s.tzinfo is not None for s in stamps):
            raise ValueError("timestamps must be naive local train time, like the dataset")
        if any(b <= a for a, b in pairwise(stamps)):
            raise ValueError("sample timestamps must be strictly increasing")
        return self

    def to_frame(self) -> pd.DataFrame:
        rows = [s.model_dump() for s in self.samples]
        frame = pd.DataFrame(rows).set_index("timestamp")
        frame.index = pd.DatetimeIndex(frame.index)
        return frame[[*ANALOG_SENSORS, *DIGITAL_SENSORS]].astype("float64")


class Evidence(BaseModel):
    feature: str
    value: float | None
    training_median: float
    robust_z: float | None = Field(description="Deviation from training distribution, not a cause.")


class WindowResult(BaseModel):
    window_start: datetime
    window_end: datetime
    n_samples: int
    status: Literal["normal", "anomalous", "insufficient_data"]
    score: float | None
    evidence: list[Evidence]


class ScoreResponse(BaseModel):
    model_version: str
    threshold: float
    result: WindowResult
    advisory: str = "Advisory output for human review. FactoryMind never controls equipment."


class AlarmEpisode(BaseModel):
    start: datetime
    end: datetime
    raised_at: datetime
    peak_score: float


class OperatorReport(BaseModel):
    report_id: str
    start: datetime
    end: datetime
    failure: str


class ReplayResponse(BaseModel):
    model_version: str
    threshold: float
    persistence: int
    start: datetime
    end: datetime
    windows: list[WindowResult]
    episodes: list[AlarmEpisode]
    operator_reports: list[OperatorReport]


class SplitMetrics(BaseModel):
    events_detected: int
    events_total: int
    false_alarms_per_day: float
    average_precision: float | None
    valid_days: float


class ModelInfo(BaseModel):
    model_version: str
    detector: str
    dataset_id: str
    dataset_license: str
    dataset_source: str
    data_kind: str
    training_period: list[str]
    threshold: float
    persistence: int
    window_length: str
    validation: SplitMetrics
    test: SplitMetrics
    intended_use: str
    limitations: list[str]
    replay_available_from: datetime | None
    replay_available_to: datetime | None
    operator_reports: list[OperatorReport] = Field(
        description="Operator failure reports used as weak evaluation labels."
    )


# ---------------------------------------------------------------- helpers


def _window(w: WindowScore) -> WindowResult:
    return WindowResult(
        window_start=w.window_start.to_pydatetime(),
        window_end=w.window_end.to_pydatetime(),
        n_samples=w.n_samples,
        status=w.status,
        score=w.score,
        evidence=[_evidence(e) for e in w.evidence],
    )


def _evidence(e: FeatureEvidence) -> Evidence:
    return Evidence(
        feature=e.feature, value=e.value, training_median=e.training_median, robust_z=e.robust_z
    )


def _report(r: FailureReport) -> OperatorReport:
    return OperatorReport(report_id=r.report_id, start=r.start, end=r.end, failure=r.failure)


def _split(metrics: dict[str, Any]) -> SplitMetrics:
    return SplitMetrics(
        events_detected=metrics["events_detected"],
        events_total=metrics["events_total"],
        false_alarms_per_day=metrics["false_alarms_per_day"],
        average_precision=metrics["average_precision"],
        valid_days=metrics["valid_days"],
    )


# ---------------------------------------------------------------- routes


@router.get("/model", response_model=ModelInfo)
def anomaly_model(service: ServiceDep) -> ModelInfo:
    card, bundle = service.card, service.bundle
    data_range = service.data_range()
    return ModelInfo(
        model_version=bundle.version,
        detector=bundle.detector_name,
        dataset_id=card["dataset"]["id"],
        dataset_license=card["dataset"]["license"],
        dataset_source=card["dataset"]["source"],
        data_kind=card["dataset"]["data_kind"],
        training_period=card["training_period"],
        threshold=bundle.policy.threshold,
        persistence=bundle.policy.persistence,
        window_length=bundle.spec.length,
        validation=_split(card["evaluation"]["validation"]),
        test=_split(card["evaluation"]["test"]),
        intended_use=card["intended_use"],
        limitations=card["limitations"],
        replay_available_from=data_range[0].to_pydatetime() if data_range else None,
        replay_available_to=data_range[1].to_pydatetime() if data_range else None,
        operator_reports=[_report(r) for r in service.reports],
    )


@router.post("/score", response_model=ScoreResponse)
def anomaly_score(request: ScoreRequest, service: ServiceDep) -> ScoreResponse:
    result = service.score_block(request.to_frame())
    return ScoreResponse(
        model_version=service.bundle.version,
        threshold=service.bundle.policy.threshold,
        result=_window(result),
    )


@router.get("/replay", response_model=ReplayResponse)
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
        windows, eps, reports = service.replay(lo, hi)
    except DataUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return ReplayResponse(
        model_version=service.bundle.version,
        threshold=service.bundle.policy.threshold,
        persistence=service.bundle.policy.persistence,
        start=start,
        end=end,
        windows=[_window(w) for w in windows],
        episodes=[
            AlarmEpisode(
                start=e.start.to_pydatetime(),
                end=e.end.to_pydatetime(),
                raised_at=e.raised_at.to_pydatetime(),
                peak_score=e.peak_score,
            )
            for e in eps
        ],
        operator_reports=[_report(r) for r in reports],
    )
