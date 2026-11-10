from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from factorymind.anomaly.service import AnomalyService
from factorymind.db.audit import compute_hash, verify_chain
from factorymind.db.session import Database, SchemaMismatchError
from tests.auth_helpers import ASSET, AuthKit, make_client

RANGE_START, RANGE_END = "2020-01-09T00:00:00", "2020-01-09T12:00:00"


@dataclass
class World:
    kit: AuthKit
    client: TestClient

    def episode_start(self) -> str:
        replay = self.client.get(
            f"/api/v1/anomaly/replay?start={RANGE_START}&end={RANGE_END}",
            headers=self.kit.headers(),
        ).json()
        assert replay["episodes"], "the simulated leak must raise an alarm"
        return str(replay["episodes"][0]["start"])

    def create(self, **auth: Any) -> Any:
        body = {
            "asset_id": ASSET,
            "range_start": RANGE_START,
            "range_end": RANGE_END,
            "episode_start": self.episode_start(),
        }
        headers = self.kit.headers(**{"roles": ["engineer"], "sub": "eve", **auth})
        return self.client.post("/api/v1/incidents", json=body, headers=headers)

    def decide(self, incident: dict[str, Any], decision: str = "confirm", **auth: Any) -> Any:
        body = {
            "decision": decision,
            "rationale": "Reviewed the evidence; leak signature is consistent.",
            "expected_version": incident["version"],
        }
        headers = self.kit.headers(**{"roles": ["approver"], "sub": "ada", **auth})
        return self.client.post(
            f"/api/v1/incidents/{incident['id']}/decisions", json=body, headers=headers
        )

    def audit(self, tenant: str = "acme", **params: Any) -> dict[str, Any]:
        headers = self.kit.headers(roles=["auditor"], tenant=tenant, sub="audrey")
        return self.client.get("/api/v1/audit/events", params=params, headers=headers).json()


@pytest.fixture
def world(kit: AuthKit, anomaly_service: AnomalyService) -> World:
    return World(kit, make_client(kit, anomaly_service))


# ---------------------------------------------------------------- creation


def test_incident_is_created_with_server_side_evidence(world: World) -> None:
    response = world.create()
    assert response.status_code == 201
    body = response.json()
    assert body["tenant_id"] == "acme" and body["asset_id"] == ASSET
    assert body["status"] == "open" and body["version"] == 1 and body["created_by"] == "eve"
    groups = [e["group"] for e in body["evidence"]]
    assert (
        groups and len(groups) == len(set(groups)) and all(e["meaning"] for e in body["evidence"])
    )
    assert body["decisions"] == []


def test_creation_is_idempotent_per_detected_episode(world: World) -> None:
    first, second = world.create(), world.create(sub="someone-else")
    assert (first.status_code, second.status_code) == (201, 200)
    assert first.json()["id"] == second.json()["id"]
    listing = world.client.get("/api/v1/incidents", headers=world.kit.headers()).json()
    assert listing["total"] == 1


def test_unknown_episode_unknown_asset_and_ungranted_asset_are_404(world: World) -> None:
    body = {
        "asset_id": ASSET,
        "range_start": RANGE_START,
        "range_end": RANGE_END,
        "episode_start": "2020-01-09T00:10:00",
    }
    headers = world.kit.headers(roles=["engineer"])
    assert world.client.post("/api/v1/incidents", json=body, headers=headers).status_code == 404
    good = {**body, "episode_start": world.episode_start()}
    other = world.client.post(
        "/api/v1/incidents", json={**good, "asset_id": "pump-9"}, headers=headers
    )
    assert other.status_code == 404
    scoped = world.kit.headers(roles=["engineer"], assets=[])
    assert world.client.post("/api/v1/incidents", json=good, headers=scoped).status_code == 404
    reasons = [e["details"].get("reason") for e in world.audit()["items"]]
    assert reasons.count("asset_not_granted") == 2


# ---------------------------------------------------------------- tenant and asset isolation


def test_other_tenants_cannot_see_or_decide_an_incident_and_get_404(world: World) -> None:
    incident = world.create().json()
    path = f"/api/v1/incidents/{incident['id']}"
    stranger = world.kit.headers(tenant="globex", roles=["viewer"])
    assert world.client.get(path, headers=stranger).status_code == 404
    assert world.client.get("/api/v1/incidents", headers=stranger).json()["total"] == 0
    assert world.decide(incident, tenant="globex").status_code == 404
    # the same episode in another tenant is a separate incident, not a duplicate
    other = world.create(tenant="globex")
    assert other.status_code == 201 and other.json()["id"] != incident["id"]
    # and tenant A still sees exactly its own
    assert world.client.get("/api/v1/incidents", headers=world.kit.headers()).json()["total"] == 1


def test_asset_grants_restrict_reads(world: World) -> None:
    incident = world.create().json()
    path = f"/api/v1/incidents/{incident['id']}"
    no_assets = world.kit.headers(roles=["viewer"], assets=[])
    assert world.client.get(path, headers=no_assets).status_code == 404
    assert world.client.get("/api/v1/incidents", headers=no_assets).json()["total"] == 0
    assert world.client.get(path, headers=world.kit.headers()).status_code == 200


def test_audit_is_tenant_scoped(world: World) -> None:
    world.create()
    assert [e["action"] for e in world.audit()["items"]] == ["incident.create"]
    assert world.audit(tenant="globex")["items"] == []


# ---------------------------------------------------------------- decisions


def test_decision_flow_closes_the_incident_and_is_append_only(world: World) -> None:
    incident = world.create().json()
    decided = world.decide(incident, "confirm")
    assert decided.status_code == 201
    body = decided.json()
    assert body["status"] == "confirmed" and body["version"] == 2
    assert [d["decision"] for d in body["decisions"]] == ["confirm"]
    assert (
        body["decisions"][0]["actor_id"] == "ada" and body["decisions"][0]["incident_version"] == 1
    )
    # stale version, and a decided incident refuses anything further
    assert world.decide(incident).status_code == 409
    assert world.decide(body).status_code == 409


def test_defer_keeps_the_incident_open_and_bumps_the_version(world: World) -> None:
    incident = world.create().json()
    deferred = world.decide(incident, "defer").json()
    assert deferred["status"] == "open" and deferred["version"] == 2
    final = world.decide(deferred, "dismiss").json()
    assert final["status"] == "dismissed" and [d["decision"] for d in final["decisions"]] == [
        "defer",
        "dismiss",
    ]
    filtered = world.client.get("/api/v1/incidents?status=dismissed", headers=world.kit.headers())
    assert filtered.json()["total"] == 1


@pytest.mark.parametrize(
    "override",
    [
        {"rationale": "   "},
        {"rationale": ""},
        {"rationale": "x" * 2001},
        {"decision": "approve"},
        {"expected_version": 0},
        {"extra": 1},
    ],
)
def test_invalid_decisions_are_422(world: World, override: dict[str, Any]) -> None:
    incident = world.create().json()
    body = {"decision": "confirm", "rationale": "ok", "expected_version": 1, **override}
    headers = world.kit.headers(roles=["approver"], sub="ada")
    response = world.client.post(
        f"/api/v1/incidents/{incident['id']}/decisions", json=body, headers=headers
    )
    assert response.status_code == 422


def test_creator_cannot_decide_their_own_incident(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    world = World(kit, make_client(kit, anomaly_service))
    both = {"roles": ["engineer", "approver"], "sub": "carol"}
    incident = world.create(**both).json()
    denied = world.decide(incident, **both)
    assert denied.status_code == 403
    assert "separation_of_duties" in [e["details"].get("reason") for e in world.audit()["items"]]
    relaxed = World(kit, make_client(kit, anomaly_service, require_separation_of_duties=False))
    assert relaxed.decide(incident, **both).status_code == 201


# ---------------------------------------------------------------- audit trail


def test_every_state_change_is_audited_in_order_with_request_ids(world: World) -> None:
    incident = world.create().json()
    response = world.decide(incident)
    events = world.audit()["items"]
    assert [e["action"] for e in reversed(events)] == ["incident.create", "decision.create"]
    newest = events[0]
    assert newest["actor_id"] == "ada" and newest["outcome"] == "allowed"
    assert newest["request_id"] == response.headers["x-request-id"]
    assert newest["object_id"] == incident["id"]
    only = world.audit(action="decision.create")["items"]
    assert [e["action"] for e in only] == ["decision.create"]
    assert len(world.audit(limit=1)["items"]) == 1


def test_audit_chain_verifies_and_detects_tampering(world: World) -> None:
    incident = world.create().json()
    world.decide(incident)
    headers = world.kit.headers(roles=["auditor"], sub="audrey")
    assert world.client.get("/api/v1/audit/verify", headers=headers).json() == {
        "valid": True,
        "events": 2,
        "first_invalid_seq": None,
    }
    db: Database = world.client.app.state.db  # type: ignore[attr-defined]
    # the database itself refuses edits ...
    with db.engine.begin() as conn, pytest.raises(DatabaseError, match="append-only"):
        conn.execute(text("UPDATE audit_event SET actor_id = 'mallory' WHERE seq = 1"))
    with db.engine.begin() as conn:
        # ... and an attacker who removes the guard still breaks the hash chain
        conn.execute(text("DROP TRIGGER audit_event_no_update"))
        conn.execute(text("UPDATE audit_event SET actor_id = 'mallory' WHERE seq = 1"))
    result = world.client.get("/api/v1/audit/verify", headers=headers).json()
    assert result["valid"] is False and result["first_invalid_seq"] == 1


def test_deleting_an_audit_event_breaks_the_chain(world: World) -> None:
    incident = world.create().json()
    world.decide(incident)
    db: Database = world.client.app.state.db  # type: ignore[attr-defined]
    with db.engine.begin() as conn:
        conn.execute(text("DROP TRIGGER audit_event_no_delete"))
        conn.execute(text("DELETE FROM audit_event WHERE seq = 1"))
    with db.session() as session:
        status = verify_chain(session, "acme")
    assert not status.valid


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE evidence SET tenant_id = 'x'",
        "DELETE FROM evidence",
        "UPDATE decision SET tenant_id = 'x'",
        "DELETE FROM decision",
    ],
)
def test_evidence_and_decisions_are_append_only_in_the_database(
    world: World, statement: str
) -> None:
    world.decide(world.create().json())
    db: Database = world.client.app.state.db  # type: ignore[attr-defined]
    with db.engine.begin() as conn, pytest.raises(DatabaseError, match="append-only"):
        conn.execute(text(statement))


# ---------------------------------------------------------------- database foundations


def test_schema_version_mismatch_refuses_to_start(tmp_path: Any) -> None:
    url = f"sqlite:///{(tmp_path / 'old.sqlite3').as_posix()}"
    db = Database(url)
    db.ensure_ready()
    with db.engine.begin() as conn:
        conn.execute(text("UPDATE schema_meta SET value = '0'"))
    with pytest.raises(SchemaMismatchError):
        Database(url).ensure_ready()


def test_audit_hash_is_deterministic_and_chained() -> None:
    content = {"seq": 1, "action": "x"}
    assert compute_hash(content, "0" * 64) == compute_hash(dict(content), "0" * 64)
    assert compute_hash(content, "0" * 64) != compute_hash(content, "1" * 64)
    assert compute_hash(content, "0" * 64) != compute_hash({**content, "action": "y"}, "0" * 64)


def test_audit_details_are_bounded(world: World) -> None:
    from factorymind.db.audit import MAX_DETAIL_VALUE, append_audit

    db: Database = world.client.app.state.db  # type: ignore[attr-defined]
    with db.session() as session:
        event = append_audit(
            session,
            tenant_id="acme",
            actor_id="a",
            roles=set(),
            action="x",
            object_type="t",
            object_id="1",
            outcome="allowed",
            request_id="r",
            details={"long": "z" * 5000, "ok": 3},
        )
    assert len(event.details["long"]) == MAX_DETAIL_VALUE and event.details["ok"] == 3
