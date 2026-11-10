"""Append-only, hash-chained audit log (one chain per tenant).

Each event stores the SHA-256 of its canonical content plus the previous event's hash, so an edited,
removed or reordered row breaks verification from that point on. This detects tampering; it does
not prevent a database administrator from rewriting the whole chain (an external anchor is future
work). ``details`` must stay small and free of tokens, payloads and personal data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from factorymind.db.models import AuditEvent

GENESIS = "0" * 64
PRE_AUTH_TENANT = "-"
MAX_DETAIL_VALUE = 200


class AuditContentionError(RuntimeError):
    """Two writers raced for the same position in a tenant's chain; the caller may retry."""


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _clean(details: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (details or {}).items():
        if isinstance(value, bool | int | float) or value is None:
            out[str(key)[:40]] = value
        else:
            out[str(key)[:40]] = str(value)[:MAX_DETAIL_VALUE]
    return out


def compute_hash(content: dict[str, Any], prev_hash: str) -> str:
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev_hash + canonical).encode("utf-8")).hexdigest()


def _content(ev: AuditEvent) -> dict[str, Any]:
    return {
        "tenant_id": ev.tenant_id,
        "seq": ev.seq,
        "ts": ev.ts.isoformat(timespec="microseconds"),
        "actor_id": ev.actor_id,
        "roles": ev.roles,
        "action": ev.action,
        "object_type": ev.object_type,
        "object_id": ev.object_id,
        "outcome": ev.outcome,
        "request_id": ev.request_id,
        "details": ev.details,
    }


def append_audit(
    session: Session,
    *,
    tenant_id: str,
    actor_id: str,
    roles: frozenset[Any] | set[Any] | list[Any],
    action: str,
    object_type: str,
    object_id: str,
    outcome: str,
    request_id: str,
    details: dict[str, Any] | None = None,
) -> AuditEvent:
    last = session.execute(
        select(AuditEvent.seq, AuditEvent.hash)
        .where(AuditEvent.tenant_id == tenant_id)
        .order_by(AuditEvent.seq.desc())
        .limit(1)
    ).first()
    event = AuditEvent(
        tenant_id=tenant_id,
        seq=(last.seq + 1) if last else 1,
        ts=utcnow(),
        actor_id=actor_id[:128],
        roles=",".join(sorted(str(r) for r in roles))[:128],
        action=action[:64],
        object_type=object_type[:32],
        object_id=object_id[:64],
        outcome=outcome,
        request_id=request_id[:64],
        details=_clean(details),
        prev_hash=last.hash if last else GENESIS,
        hash="",
    )
    event.hash = compute_hash(_content(event), event.prev_hash)
    session.add(event)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise AuditContentionError("audit chain contention") from exc
    return event


@dataclass(frozen=True)
class ChainStatus:
    valid: bool
    events: int
    first_invalid_seq: int | None


def verify_chain(session: Session, tenant_id: str) -> ChainStatus:
    rows = (
        session.execute(
            select(AuditEvent).where(AuditEvent.tenant_id == tenant_id).order_by(AuditEvent.seq)
        )
        .scalars()
        .all()
    )
    prev, expected_seq = GENESIS, 1
    for ev in rows:
        if (
            ev.seq != expected_seq
            or ev.prev_hash != prev
            or ev.hash != compute_hash(_content(ev), prev)
        ):
            return ChainStatus(False, len(rows), ev.seq if ev.seq == expected_seq else expected_seq)
        prev, expected_seq = ev.hash, expected_seq + 1
    return ChainStatus(True, len(rows), None)
