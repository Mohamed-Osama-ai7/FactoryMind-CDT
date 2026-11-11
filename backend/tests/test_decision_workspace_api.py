"""Tests for WP-F1: Decision Workspace, append-only Outcome Verification, SoD, and Audit."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from factorymind.anomaly.service import AnomalyService
from factorymind.db.session import Database, SchemaMismatchError
from tests.auth_helpers import ASSET, AuthKit, make_client


def test_decision_outcome_verification_sod_triggers_and_audit(
    kit: AuthKit,
    anomaly_service: AnomalyService,
) -> None:
    client: TestClient = make_client(kit, anomaly_service)
    r_start, r_end = "2020-01-09T00:00:00", "2020-01-09T12:00:00"
    replay = client.get(
        f"/api/v1/anomaly/replay?start={r_start}&end={r_end}", headers=kit.headers()
    ).json()
    ep_start = replay["episodes"][0]["start"]

    # Engineer creates incident (version 1, status open)
    created = client.post(
        "/api/v1/incidents",
        json={
            "asset_id": ASSET,
            "range_start": r_start,
            "range_end": r_end,
            "episode_start": ep_start,
        },
        headers=kit.headers(roles=["engineer"], sub="eng-1"),
    )
    assert created.status_code == 201
    inc = created.json()
    inc_id = inc["id"]
    assert inc["outcomes"] == []

    # Separation of duties: same actor holding approver role cannot decide their own incident
    sod = client.post(
        f"/api/v1/incidents/{inc_id}/decisions",
        json={"decision": "confirm", "rationale": "Self approval blocked", "expected_version": 1},
        headers=kit.headers(roles=["approver"], sub="eng-1"),
    )
    assert sod.status_code == 403

    # Separate approver defers incident (version bumps 1 -> 2, remains open)
    deferred = client.post(
        f"/api/v1/incidents/{inc_id}/decisions",
        json={
            "decision": "defer",
            "rationale": "Hold pending ultrasonic purge inspection.",
            "expected_version": 1,
        },
        headers=kit.headers(roles=["approver"], sub="app-1"),
    )
    assert deferred.status_code == 201
    assert deferred.json()["status"] == "open"
    assert deferred.json()["version"] == 2

    # Stale expected_version=1 is rejected with 409
    stale = client.post(
        f"/api/v1/incidents/{inc_id}/decisions",
        json={"decision": "confirm", "rationale": "Stale attempt", "expected_version": 1},
        headers=kit.headers(roles=["approver"], sub="app-1"),
    )
    assert stale.status_code == 409

    # Approver confirms incident at expected_version=2
    confirmed = client.post(
        f"/api/v1/incidents/{inc_id}/decisions",
        json={
            "decision": "confirm",
            "rationale": "Corroborated leak duty rise and TP3 decay.",
            "expected_version": 2,
        },
        headers=kit.headers(roles=["approver"], sub="app-1"),
    )
    assert confirmed.status_code == 201
    assert confirmed.json()["status"] == "confirmed"
    assert confirmed.json()["version"] == 3
    assert len(confirmed.json()["decisions"]) == 2

    # Engineer records append-only physical outcome verification
    assert (
        client.post(
            f"/api/v1/incidents/{inc_id}/outcomes",
            json={
                "outcome": "confirmed_leak",
                "maintenance_action": "Inspected hose fitting",
                "verified_at": "2020-01-09T11:00:00+00:00",
                "notes": "Reject tz-aware stamp.",
            },
            headers=kit.headers(roles=["engineer"], sub="eng-1"),
        ).status_code
        == 422
    )

    out_resp = client.post(
        f"/api/v1/incidents/{inc_id}/outcomes",
        json={
            "outcome": "confirmed_leak",
            "maintenance_action": "Tightened air-dryer purge line coupling",
            "verified_at": "2020-01-09T11:00:00",
            "notes": "Soap bubble check showed leak at purge coupling before tightening.",
        },
        headers=kit.headers(roles=["engineer"], sub="eng-1"),
    )
    assert out_resp.status_code == 201
    out_body = out_resp.json()
    assert out_body["origin"] == "human_observation"
    assert out_body["outcome"] == "confirmed_leak"

    # Cross-tenant caller receives 404 on outcomes
    assert (
        client.post(
            f"/api/v1/incidents/{inc_id}/outcomes",
            json={
                "outcome": "false_positive",
                "maintenance_action": "Cross tenant probe",
                "notes": "Should return 404.",
            },
            headers=kit.headers(tenant="beta", roles=["engineer"], sub="eve-beta"),
        ).status_code
        == 404
    )

    # Same-tenant caller without asset authorization receives 404 on outcomes
    for denied_assets in (["pump-7"], []):
        assert (
            client.post(
                f"/api/v1/incidents/{inc_id}/outcomes",
                json={
                    "outcome": "false_positive",
                    "maintenance_action": "Ungrounded asset probe",
                    "notes": "Should return 404.",
                },
                headers=kit.headers(
                    roles=["engineer"], sub="eng-other-asset", assets=denied_assets
                ),
            ).status_code
            == 404
        )

    # IncidentDetail now carries both recorded decisions and append-only outcomes
    detail = client.get(
        f"/api/v1/incidents/{inc_id}",
        headers=kit.headers(roles=["auditor"], sub="aud-1"),
    ).json()
    assert len(detail["decisions"]) == 2
    assert len(detail["outcomes"]) == 1
    assert detail["outcomes"][0]["id"] == out_body["id"]

    # SQLite triggers block UPDATE and DELETE on outcome_verification
    db: Database = client.app.state.db  # type: ignore[attr-defined]
    with (
        db.engine.begin() as conn,
        pytest.raises(DatabaseError, match="append-only"),
    ):
        conn.execute(text("UPDATE outcome_verification SET notes = 'tampered'"))
    with (
        db.engine.begin() as conn,
        pytest.raises(DatabaseError, match="append-only"),
    ):
        conn.execute(text("DELETE FROM outcome_verification"))

    # Auditor sees outcome.create and decision.create in audit events and verifies intact chain
    verify = client.get(
        "/api/v1/audit/verify", headers=kit.headers(roles=["auditor"], sub="aud-1")
    ).json()
    assert verify["valid"] is True
    assert verify["events"] >= 4

    outcome_events = client.get(
        "/api/v1/audit/events",
        params={"action": "outcome.create"},
        headers=kit.headers(roles=["auditor"], sub="aud-1"),
    ).json()["items"]
    assert len(outcome_events) == 1
    assert outcome_events[0]["object_id"] == inc_id


def test_schema_v3_refuses_v2_database(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'v2.sqlite3').as_posix()}"
    db = Database(url)
    db.ensure_ready()
    with db.engine.begin() as conn:
        conn.execute(text("UPDATE schema_meta SET value = '2'"))
    with pytest.raises(SchemaMismatchError):
        Database(url).ensure_ready()
