"""Persistence models. Every table carries ``tenant_id``; incident-related tables also ``asset_id``.

Timestamps: ``created_at`` and audit ``ts`` are naive UTC. ``episode_*`` and ``raised_at`` keep the
dataset's naive local train time (the source does not state a timezone).

Evidence, decisions and audit events are append-only: the application never updates or deletes them
and, on SQLite, triggers abort any UPDATE or DELETE (defence in depth; the PostgreSQL equivalent is
part of ADR-010).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA_VERSION = "1"
APPEND_ONLY_TABLES = ("evidence", "decision", "audit_event")


class Base(DeclarativeBase):
    pass


class SchemaMeta(Base):
    __tablename__ = "schema_meta"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(64))


class Incident(Base):
    __tablename__ = "incident"
    __table_args__ = (
        UniqueConstraint("tenant_id", "asset_id", "model_version", "episode_start"),
        Index("ix_incident_tenant_status", "tenant_id", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    asset_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16))  # open | confirmed | dismissed
    version: Mapped[int] = mapped_column(Integer, default=1)  # optimistic concurrency
    detector: Mapped[str] = mapped_column(String(64))
    model_version: Mapped[str] = mapped_column(String(128))
    episode_start: Mapped[datetime] = mapped_column(DateTime)
    episode_end: Mapped[datetime] = mapped_column(DateTime)
    raised_at: Mapped[datetime] = mapped_column(DateTime)
    peak_score: Mapped[float] = mapped_column(Float)
    created_by: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime)


class EvidenceRow(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    asset_id: Mapped[str] = mapped_column(String(64))
    incident_id: Mapped[str] = mapped_column(ForeignKey("incident.id"), index=True)
    group_name: Mapped[str] = mapped_column(String(32))
    group_label: Mapped[str] = mapped_column(String(64))
    feature: Mapped[str] = mapped_column(String(64))
    value: Mapped[float] = mapped_column(Float)
    training_median: Mapped[float] = mapped_column(Float)
    robust_z: Mapped[float | None] = mapped_column(Float, nullable=True)
    novel: Mapped[bool] = mapped_column(Boolean)
    meaning: Mapped[str] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime)


class Decision(Base):
    __tablename__ = "decision"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)
    asset_id: Mapped[str] = mapped_column(String(64))
    incident_id: Mapped[str] = mapped_column(ForeignKey("incident.id"), index=True)
    actor_id: Mapped[str] = mapped_column(String(128))
    decision: Mapped[str] = mapped_column(String(16))  # confirm | dismiss | defer
    rationale: Mapped[str] = mapped_column(String(2000))
    incident_version: Mapped[int] = mapped_column(Integer)  # version the reviewer saw
    created_at: Mapped[datetime] = mapped_column(DateTime)


class AuditEvent(Base):
    __tablename__ = "audit_event"
    __table_args__ = (UniqueConstraint("tenant_id", "seq"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(String(64), index=True)  # "-" before identification
    seq: Mapped[int] = mapped_column(Integer)  # per-tenant, gap-free
    ts: Mapped[datetime] = mapped_column(DateTime)
    actor_id: Mapped[str] = mapped_column(String(128))
    roles: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(64), index=True)
    object_type: Mapped[str] = mapped_column(String(32))
    object_id: Mapped[str] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16))  # allowed | denied | failure
    request_id: Mapped[str] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(JSON)
    prev_hash: Mapped[str] = mapped_column(String(64))
    hash: Mapped[str] = mapped_column(String(64))
