"""Incident, review-decision, audit and identity routes (all authenticated, deny by default).

An incident is created from an alarm episode that the *server* re-detects (the client only points at
it), so stored evidence cannot be forged by a caller. A decision is a human review record about an
alarm. It never triggers anything outside this database and never reaches industrial systems.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, cast

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from factorymind.anomaly.evidence_package import build_event_evidence_package
from factorymind.anomaly.rca import analyze_event_evidence
from factorymind.anomaly.service import DataUnavailableError, WindowScore
from factorymind.api.anomaly import MAX_REPLAY, ServiceDep
from factorymind.api.security import request_id_of
from factorymind.auth.deps import get_principal, record_audit, require
from factorymind.auth.principal import Permission, Principal
from factorymind.config import IDENTIFIER, Settings
from factorymind.contracts.events import (
    EventEvidencePackage,
    HumanObservationCategory,
    HumanObservationOut,
)
from factorymind.contracts.rca import RcaAssessment
from factorymind.db.audit import append_audit, verify_chain
from factorymind.db.models import AuditEvent, Decision, EvidenceRow, HumanObservation, Incident
from factorymind.db.repository import (
    DuplicateIncidentError,
    IncidentRepo,
    NewEvidence,
    VersionConflictError,
)
from factorymind.db.session import Database

router = APIRouter(prefix="/api/v1", tags=["incidents"])
MAX_OFFSET = 10_000


def _db(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


def _settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


# ---------------------------------------------------------------- schemas


class MeResponse(BaseModel):
    actor_id: str
    tenant_id: str
    roles: list[str]
    permissions: list[str]
    asset_ids: list[str]


class IncidentCreate(BaseModel):
    """Point at an alarm episode; the server re-detects it and stores the evidence."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(pattern=IDENTIFIER.pattern)
    range_start: datetime
    range_end: datetime
    episode_start: datetime

    @model_validator(mode="after")
    def _naive_and_bounded(self) -> IncidentCreate:
        stamps = (self.range_start, self.range_end, self.episode_start)
        if any(s.tzinfo is not None for s in stamps):
            raise ValueError("timestamps must be naive local train time")
        span = pd.Timestamp(self.range_end) - pd.Timestamp(self.range_start)
        if span <= pd.Timedelta(0) or span > MAX_REPLAY:
            raise ValueError("range must be positive and at most 7 days")
        return self


class DecisionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["confirm", "dismiss", "defer"]
    rationale: str = Field(min_length=1, max_length=2000)
    expected_version: int = Field(ge=1, le=1_000_000)

    @field_validator("rationale")
    @classmethod
    def _stripped(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("rationale must not be blank")
        return value


class EvidenceOut(BaseModel):
    group: str
    group_label: str
    feature: str
    value: float
    training_median: float
    robust_z: float | None
    novel: bool
    meaning: str


class DecisionOut(BaseModel):
    id: str
    actor_id: str
    decision: Literal["confirm", "dismiss", "defer"]
    rationale: str
    incident_version: int
    created_at: datetime


class IncidentOut(BaseModel):
    id: str
    tenant_id: str
    asset_id: str
    status: Literal["open", "confirmed", "dismissed"]
    version: int
    detector: str
    model_version: str
    episode_start: datetime
    episode_end: datetime
    raised_at: datetime
    peak_score: float
    created_by: str
    created_at: datetime


class HumanObservationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: HumanObservationCategory
    note: str = Field(min_length=1, max_length=2000)
    observed_at: datetime | None = None

    @field_validator("note")
    @classmethod
    def _clean_note(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("observation note must not be blank")
        return cleaned

    @field_validator("observed_at")
    @classmethod
    def _naive_observed_at(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is not None:
            raise ValueError("observed_at must be naive local train time")
        return value


class IncidentDetail(IncidentOut):
    evidence: list[EvidenceOut]
    decisions: list[DecisionOut]
    observations: list[HumanObservationOut] = Field(default_factory=list)


class IncidentList(BaseModel):
    items: list[IncidentOut]
    total: int
    limit: int
    offset: int


class AuditEventOut(BaseModel):
    seq: int
    ts: datetime
    actor_id: str
    roles: str
    action: str
    object_type: str
    object_id: str
    outcome: str
    request_id: str
    details: dict[str, object]
    hash: str


class AuditList(BaseModel):
    items: list[AuditEventOut]
    limit: int
    offset: int


class AuditVerification(BaseModel):
    valid: bool
    events: int
    first_invalid_seq: int | None


# ---------------------------------------------------------------- helpers


def _incident_out(i: Incident) -> IncidentOut:
    return IncidentOut(
        id=i.id,
        tenant_id=i.tenant_id,
        asset_id=i.asset_id,
        status=i.status,
        version=i.version,
        detector=i.detector,
        model_version=i.model_version,
        episode_start=i.episode_start,
        episode_end=i.episode_end,
        raised_at=i.raised_at,
        peak_score=i.peak_score,
        created_by=i.created_by,
        created_at=i.created_at,
    )


def _evidence_out(e: EvidenceRow) -> EvidenceOut:
    return EvidenceOut(
        group=e.group_name,
        group_label=e.group_label,
        feature=e.feature,
        value=e.value,
        training_median=e.training_median,
        robust_z=e.robust_z,
        novel=e.novel,
        meaning=e.meaning,
    )


def _decision_out(d: Decision) -> DecisionOut:
    return DecisionOut(
        id=d.id,
        actor_id=d.actor_id,
        decision=d.decision,
        rationale=d.rationale,
        incident_version=d.incident_version,
        created_at=d.created_at,
    )


def _observation_out(obs: HumanObservation) -> HumanObservationOut:
    return HumanObservationOut(
        id=obs.id,
        incident_id=obs.incident_id,
        asset_id=obs.asset_id,
        actor_id=obs.actor_id,
        category=cast(HumanObservationCategory, obs.category),
        observed_at=obs.observed_at,
        note=obs.note,
        created_at=obs.created_at,
    )


def _detail(repo: IncidentRepo, incident: Incident) -> IncidentDetail:
    return IncidentDetail(
        **_incident_out(incident).model_dump(),
        evidence=[_evidence_out(e) for e in repo.evidence(incident)],
        decisions=[_decision_out(d) for d in repo.decisions(incident)],
        observations=[_observation_out(o) for o in repo.observations(incident)],
    )


def _audit_denied(
    request: Request, principal: Principal, reason: str, object_id: str = "-"
) -> None:
    record_audit(
        request,
        action="authz.denied",
        outcome="denied",
        tenant_id=principal.tenant_id,
        actor_id=principal.actor_id,
        roles=principal.roles,
        object_type="incident",
        object_id=object_id,
        reason=reason,
    )


def _not_found() -> HTTPException:
    return HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found")


# ---------------------------------------------------------------- routes


@router.get("/me", response_model=MeResponse)
def me(principal: Annotated[Principal, Depends(get_principal)]) -> MeResponse:
    return MeResponse(
        actor_id=principal.actor_id,
        tenant_id=principal.tenant_id,
        roles=sorted(r.value for r in principal.roles),
        permissions=sorted(p.value for p in principal.permissions),
        asset_ids=sorted(principal.asset_ids),
    )


@router.post("/incidents", response_model=IncidentDetail, status_code=status.HTTP_201_CREATED)
def create_incident(
    body: IncidentCreate,
    request: Request,
    response: Response,
    # authentication/authorization must be resolved before any other dependency (model loading)
    principal: Annotated[Principal, Depends(require(Permission.INCIDENT_CREATE, write=True))],
    service: ServiceDep,
) -> IncidentDetail:
    settings = _settings(request)
    # the replayed telemetry belongs to exactly one asset and tenant; anything else is a mismatch
    if body.asset_id != settings.data_source_asset_id:
        _audit_denied(request, principal, "asset_mismatch", body.asset_id)
        raise _not_found()
    denial = principal.data_source_denial(settings)
    if denial is not None:
        _audit_denied(request, principal, denial, body.asset_id)
        raise _not_found()
    try:
        windows, episodes, _ = service.replay(
            pd.Timestamp(body.range_start), pd.Timestamp(body.range_end)
        )
    except DataUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    match = next((e for e in episodes if e.episode.start == pd.Timestamp(body.episode_start)), None)
    if match is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="No such alarm episode in range")
    evidence = _peak_evidence(windows, match.episode.start, match.episode.end)

    def work(session: Session) -> IncidentDetail:
        repo = IncidentRepo(session, principal)
        existing = repo.find_by_episode(
            body.asset_id, service.bundle.version, match.episode.start.to_pydatetime()
        )
        if existing is not None:
            response.status_code = status.HTTP_200_OK
            return _detail(repo, existing)
        try:
            incident = repo.create(
                asset_id=body.asset_id,
                detector=service.bundle.detector_name,
                model_version=service.bundle.version,
                episode_start=match.episode.start.to_pydatetime(),
                episode_end=match.episode.end.to_pydatetime(),
                raised_at=match.episode.raised_at.to_pydatetime(),
                peak_score=match.episode.peak_score,
                evidence=evidence,
            )
        except DuplicateIncidentError:
            raise HTTPException(
                status.HTTP_409_CONFLICT, detail="Incident already exists"
            ) from None
        append_audit(
            session,
            tenant_id=principal.tenant_id,
            actor_id=principal.actor_id,
            roles=principal.roles,
            action="incident.create",
            object_type="incident",
            object_id=incident.id,
            outcome="allowed",
            request_id=request_id_of(request),
            details={"asset_id": body.asset_id, "evidence_items": len(evidence)},
        )
        return _detail(repo, incident)

    return _db(request).run_with_retry(work)


def _peak_evidence(
    windows: list[WindowScore], start: pd.Timestamp, end: pd.Timestamp
) -> list[NewEvidence]:
    inside = [
        w
        for w in windows
        if w.status == "anomalous" and w.score is not None and start <= w.window_start < end
    ]
    if not inside:
        return []
    peak = max(inside, key=lambda w: w.score if w.score is not None else float("-inf"))
    return [
        NewEvidence(
            group_name=e.group,
            group_label=e.group_label,
            feature=e.feature,
            value=e.value,
            training_median=e.training_median,
            robust_z=e.robust_z,
            novel=e.novel,
            meaning=e.meaning,
        )
        for e in peak.evidence
    ]


@router.get("/incidents", response_model=IncidentList)
def list_incidents(
    request: Request,
    principal: Annotated[Principal, Depends(require(Permission.INCIDENT_READ))],
    status_filter: Annotated[
        Literal["open", "confirmed", "dismissed"] | None, Query(alias="status")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 25,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
) -> IncidentList:
    with _db(request).session() as session:
        items, total = IncidentRepo(session, principal).list_page(
            status=status_filter, limit=limit, offset=offset
        )
        return IncidentList(
            items=[_incident_out(i) for i in items], total=total, limit=limit, offset=offset
        )


@router.get("/incidents/{incident_id}", response_model=IncidentDetail)
def get_incident(
    incident_id: Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")],
    request: Request,
    principal: Annotated[Principal, Depends(require(Permission.INCIDENT_READ))],
) -> IncidentDetail:
    with _db(request).session() as session:
        repo = IncidentRepo(session, principal)
        incident = repo.get(incident_id)
        if incident is None:
            raise _not_found()
        return _detail(repo, incident)


def _build_package_for_incident(
    incident_id: str,
    request: Request,
    principal: Principal,
    service: ServiceDep,
    baseline_hours: float,
    recovery_hours: float,
) -> EventEvidencePackage:
    settings = _settings(request)
    denial = principal.data_source_denial(settings)
    with _db(request).session() as session:
        repo = IncidentRepo(session, principal)
        incident = repo.get(incident_id)
        if incident is None:
            raise _not_found()
        if denial is not None:
            _audit_denied(request, principal, denial, incident.id)
            raise _not_found()
        if incident.asset_id != settings.data_source_asset_id:
            _audit_denied(request, principal, "asset_mismatch", incident.id)
            raise _not_found()
        observations = [_observation_out(o) for o in repo.observations(incident)]
    try:
        return build_event_evidence_package(
            service,
            asset_id=incident.asset_id,
            event_start=pd.Timestamp(incident.episode_start),
            event_end=pd.Timestamp(incident.episode_end),
            raised_at=pd.Timestamp(incident.raised_at),
            incident_id=incident.id,
            human_observations=observations,
            baseline_hours=baseline_hours,
            recovery_hours=recovery_hours,
        )
    except DataUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.get(
    "/incidents/{incident_id}/evidence-package",
    response_model=EventEvidencePackage,
)
def get_incident_evidence_package(
    incident_id: Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")],
    request: Request,
    principal: Annotated[Principal, Depends(require(Permission.INCIDENT_READ, cost=5))],
    service: ServiceDep,
    baseline_hours: Annotated[float, Query(ge=0.0, le=24.0)] = 4.0,
    recovery_hours: Annotated[float, Query(ge=0.0, le=24.0)] = 2.0,
) -> EventEvidencePackage:
    return _build_package_for_incident(
        incident_id, request, principal, service, baseline_hours, recovery_hours
    )


@router.get(
    "/incidents/{incident_id}/rca",
    response_model=RcaAssessment,
)
def get_incident_rca(
    incident_id: Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")],
    request: Request,
    principal: Annotated[Principal, Depends(require(Permission.INCIDENT_READ, cost=5))],
    service: ServiceDep,
    baseline_hours: Annotated[float, Query(ge=0.0, le=24.0)] = 4.0,
    recovery_hours: Annotated[float, Query(ge=0.0, le=24.0)] = 2.0,
) -> RcaAssessment:
    pkg = _build_package_for_incident(
        incident_id, request, principal, service, baseline_hours, recovery_hours
    )
    return analyze_event_evidence(pkg)


@router.post(
    "/incidents/{incident_id}/observations",
    response_model=HumanObservationOut,
    status_code=status.HTTP_201_CREATED,
)
def create_observation(
    incident_id: Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")],
    body: HumanObservationCreate,
    request: Request,
    principal: Annotated[Principal, Depends(require(Permission.OBSERVATION_CREATE, write=True))],
) -> HumanObservationOut:
    def work(session: Session) -> HumanObservationOut:
        repo = IncidentRepo(session, principal)
        incident = repo.get(incident_id)
        if incident is None:
            raise _not_found()
        obs = repo.add_observation(
            incident,
            category=body.category,
            note=body.note,
            observed_at=body.observed_at,
        )
        append_audit(
            session,
            tenant_id=principal.tenant_id,
            actor_id=principal.actor_id,
            roles=principal.roles,
            action="observation.create",
            object_type="incident",
            object_id=incident.id,
            outcome="allowed",
            request_id=request_id_of(request),
            details={"category": body.category, "observation_id": obs.id},
        )
        return _observation_out(obs)

    return _db(request).run_with_retry(work)


@router.post(
    "/incidents/{incident_id}/decisions",
    response_model=IncidentDetail,
    status_code=status.HTTP_201_CREATED,
)
def create_decision(
    incident_id: Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")],
    body: DecisionCreate,
    request: Request,
    principal: Annotated[Principal, Depends(require(Permission.DECISION_CREATE, write=True))],
) -> IncidentDetail:
    settings = _settings(request)
    with _db(request).session() as session:
        incident = IncidentRepo(session, principal).get(incident_id)
        creator = incident.created_by if incident is not None else None
    if incident is None:
        raise _not_found()
    if settings.require_separation_of_duties and creator == principal.actor_id:
        _audit_denied(request, principal, "separation_of_duties", incident_id)
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Forbidden")

    def work(session: Session) -> IncidentDetail:
        repo = IncidentRepo(session, principal)
        current = repo.get(incident_id)
        if current is None:
            raise _not_found()
        try:
            decision = repo.add_decision(
                current,
                decision=body.decision,
                rationale=body.rationale,
                expected_version=body.expected_version,
            )
        except VersionConflictError:
            raise HTTPException(
                status.HTTP_409_CONFLICT, detail="Incident changed or already decided"
            ) from None
        append_audit(
            session,
            tenant_id=principal.tenant_id,
            actor_id=principal.actor_id,
            roles=principal.roles,
            action="decision.create",
            object_type="incident",
            object_id=incident_id,
            outcome="allowed",
            request_id=request_id_of(request),
            details={"decision": decision.decision, "version": decision.incident_version},
        )
        return _detail(repo, current)

    return _db(request).run_with_retry(work)


@router.get("/audit/events", response_model=AuditList)
def audit_events(
    request: Request,
    principal: Annotated[Principal, Depends(require(Permission.AUDIT_READ))],
    action: Annotated[str | None, Query(max_length=64, pattern=r"^[a-z_.]+$")] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0, le=MAX_OFFSET)] = 0,
) -> AuditList:
    with _db(request).session() as session:
        stmt = select(AuditEvent).where(AuditEvent.tenant_id == principal.tenant_id)
        if action:
            stmt = stmt.where(AuditEvent.action == action)
        rows = session.execute(stmt.order_by(AuditEvent.seq.desc()).limit(limit).offset(offset))
        return AuditList(
            items=[
                AuditEventOut(
                    seq=e.seq,
                    ts=e.ts,
                    actor_id=e.actor_id,
                    roles=e.roles,
                    action=e.action,
                    object_type=e.object_type,
                    object_id=e.object_id,
                    outcome=e.outcome,
                    request_id=e.request_id,
                    details=dict(e.details),
                    hash=e.hash,
                )
                for e in rows.scalars().all()
            ],
            limit=limit,
            offset=offset,
        )


@router.get("/audit/verify", response_model=AuditVerification)
def audit_verify(
    request: Request,
    principal: Annotated[Principal, Depends(require(Permission.AUDIT_READ))],
) -> AuditVerification:
    with _db(request).session() as session:
        status_ = verify_chain(session, principal.tenant_id)
        return AuditVerification(
            valid=status_.valid, events=status_.events, first_invalid_seq=status_.first_invalid_seq
        )
