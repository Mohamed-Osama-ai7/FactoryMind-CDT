"""Tenant- and asset-scoped repositories.

Every query is built from the authenticated principal; no method accepts a tenant or an unchecked
asset from the caller. An object outside the principal's scope is indistinguishable from a missing
one (callers return 404), so existence is not leaked.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from sqlalchemy import Select, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from factorymind.auth.principal import Principal
from factorymind.db.audit import utcnow
from factorymind.db.models import Decision, EvidenceRow, HumanObservation, Incident

DecisionKind = Literal["confirm", "dismiss", "defer"]
_NEW_STATUS: dict[str, str] = {"confirm": "confirmed", "dismiss": "dismissed", "defer": "open"}
MAX_PAGE = 100


class VersionConflictError(Exception):
    """The incident changed since the reviewer loaded it, or is already decided."""


class DuplicateIncidentError(Exception):
    """A concurrent request created the same incident first."""


@dataclass(frozen=True)
class NewEvidence:
    group_name: str
    group_label: str
    feature: str
    value: float
    training_median: float
    robust_z: float | None
    novel: bool
    meaning: str


class IncidentRepo:
    def __init__(self, session: Session, principal: Principal) -> None:
        self._s = session
        self._p = principal

    def _scoped(self, stmt: Select[Any]) -> Select[Any]:
        return stmt.where(
            Incident.tenant_id == self._p.tenant_id,
            Incident.asset_id.in_(sorted(self._p.asset_ids)),
        )

    def get(self, incident_id: str) -> Incident | None:
        return self._s.execute(
            self._scoped(select(Incident).where(Incident.id == incident_id))
        ).scalar_one_or_none()

    def list_page(
        self, *, status: str | None, limit: int, offset: int
    ) -> tuple[list[Incident], int]:
        limit = max(1, min(limit, MAX_PAGE))
        base = self._scoped(select(Incident))
        if status:
            base = base.where(Incident.status == status)
        total = self._s.execute(
            select(func.count()).select_from(base.order_by(None).subquery())
        ).scalar_one()
        rows = (
            self._s.execute(
                base.order_by(Incident.created_at.desc(), Incident.id).limit(limit).offset(offset)
            )
            .scalars()
            .all()
        )
        return list(rows), int(total)

    def find_by_episode(
        self, asset_id: str, model_version: str, episode_start: datetime
    ) -> Incident | None:
        return self._s.execute(
            self._scoped(
                select(Incident).where(
                    Incident.asset_id == asset_id,
                    Incident.model_version == model_version,
                    Incident.episode_start == episode_start,
                )
            )
        ).scalar_one_or_none()

    def create(
        self,
        *,
        asset_id: str,
        detector: str,
        model_version: str,
        episode_start: datetime,
        episode_end: datetime,
        raised_at: datetime,
        peak_score: float,
        evidence: list[NewEvidence],
    ) -> Incident:
        if not self._p.can_access_asset(asset_id):
            raise PermissionError("asset outside the principal's grants")
        now = utcnow()
        incident = Incident(
            id=uuid.uuid4().hex,
            tenant_id=self._p.tenant_id,
            asset_id=asset_id,
            status="open",
            version=1,
            detector=detector,
            model_version=model_version,
            episode_start=episode_start,
            episode_end=episode_end,
            raised_at=raised_at,
            peak_score=peak_score,
            created_by=self._p.actor_id,
            created_at=now,
        )
        self._s.add(incident)
        try:
            self._s.flush()
        except IntegrityError as exc:
            self._s.rollback()
            raise DuplicateIncidentError from exc
        for e in evidence:
            self._s.add(
                EvidenceRow(
                    tenant_id=self._p.tenant_id,
                    asset_id=asset_id,
                    incident_id=incident.id,
                    group_name=e.group_name,
                    group_label=e.group_label,
                    feature=e.feature,
                    value=e.value,
                    training_median=e.training_median,
                    robust_z=e.robust_z,
                    novel=e.novel,
                    meaning=e.meaning,
                    created_at=now,
                )
            )
        self._s.flush()
        return incident

    def evidence(self, incident: Incident) -> list[EvidenceRow]:
        return list(
            self._s.execute(
                select(EvidenceRow)
                .where(
                    EvidenceRow.incident_id == incident.id,
                    EvidenceRow.tenant_id == self._p.tenant_id,
                )
                .order_by(EvidenceRow.id)
            )
            .scalars()
            .all()
        )

    def decisions(self, incident: Incident) -> list[Decision]:
        return list(
            self._s.execute(
                select(Decision)
                .where(
                    Decision.incident_id == incident.id,
                    Decision.tenant_id == self._p.tenant_id,
                )
                .order_by(Decision.created_at, Decision.id)
            )
            .scalars()
            .all()
        )

    def add_decision(
        self, incident: Incident, *, decision: DecisionKind, rationale: str, expected_version: int
    ) -> Decision:
        """Append a review decision with optimistic concurrency; never edits earlier decisions."""
        result = self._s.execute(
            update(Incident)
            .where(
                Incident.id == incident.id,
                Incident.tenant_id == self._p.tenant_id,
                Incident.version == expected_version,
                Incident.status == "open",
            )
            .values(status=_NEW_STATUS[decision], version=Incident.version + 1)
        )
        if result.rowcount != 1:  # type: ignore[attr-defined]
            raise VersionConflictError
        row = Decision(
            id=uuid.uuid4().hex,
            tenant_id=self._p.tenant_id,
            asset_id=incident.asset_id,
            incident_id=incident.id,
            actor_id=self._p.actor_id,
            decision=decision,
            rationale=rationale,
            incident_version=expected_version,
            created_at=utcnow(),
        )
        self._s.add(row)
        self._s.flush()
        self._s.refresh(incident)
        return row

    def observations(self, incident: Incident) -> list[HumanObservation]:
        return list(
            self._s.execute(
                select(HumanObservation)
                .where(
                    HumanObservation.incident_id == incident.id,
                    HumanObservation.tenant_id == self._p.tenant_id,
                )
                .order_by(HumanObservation.created_at, HumanObservation.id)
            )
            .scalars()
            .all()
        )

    def add_observation(
        self,
        incident: Incident,
        *,
        category: str,
        note: str,
        observed_at: datetime | None = None,
    ) -> HumanObservation:
        if not self._p.can_access_asset(incident.asset_id):
            raise PermissionError("asset outside the principal's grants")
        row = HumanObservation(
            id=uuid.uuid4().hex,
            tenant_id=self._p.tenant_id,
            asset_id=incident.asset_id,
            incident_id=incident.id,
            actor_id=self._p.actor_id,
            category=category,
            observed_at=observed_at,
            note=note,
            created_at=utcnow(),
        )
        self._s.add(row)
        self._s.flush()
        return row
