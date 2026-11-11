import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from factorymind.anomaly.service import AnomalyService
from factorymind.db.models import AuditEvent
from tests.auth_helpers import ASSET, AuthKit, make_client
from tests.route_helpers import all_operations

ZERO_ID = "0" * 32
HEALTH = "/api/v1/health"


def fill(path: str) -> str:
    return path.replace("{incident_id}", ZERO_ID).replace("{asset_id}", ASSET)


def audit_rows(client: TestClient, tenant: str) -> list[AuditEvent]:
    with client.app.state.db.session() as session:  # type: ignore[attr-defined]
        return list(
            session.execute(
                select(AuditEvent).where(AuditEvent.tenant_id == tenant).order_by(AuditEvent.seq)
            ).scalars()
        )


# ---------------------------------------------------------------- AC1: deny by default


def test_every_route_except_health_requires_authentication(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    client = make_client(kit, anomaly_service)
    checked = 0
    for method, path in all_operations(client.app):  # type: ignore[arg-type]
        if path == HEALTH:
            continue
        response = client.request(method, fill(path))
        assert response.status_code == 401, (method, path)
        assert response.headers["www-authenticate"].startswith("Bearer")
        checked += 1
    assert checked >= 10  # the check must actually cover the application's routes


def test_health_is_public_and_reports_auth_state(kit: AuthKit) -> None:
    body = make_client(kit).get(HEALTH).json()
    assert body["status"] == "ok" and body["auth"] == "configured"
    unconfigured = make_client(kit, auth_issuer=None)
    assert unconfigured.get(HEALTH).json()["auth"] == "not_configured"


def test_unconfigured_authentication_fails_closed(kit: AuthKit) -> None:
    client = make_client(kit, auth_issuer=None)
    assert client.get("/api/v1/incidents", headers=kit.headers()).status_code == 401


def test_unreadable_jwks_fails_closed(kit: AuthKit) -> None:
    kit.jwks_path.write_text("not json", encoding="utf-8")
    client = make_client(kit)
    assert client.get("/api/v1/me", headers=kit.headers()).status_code == 401
    assert client.get(HEALTH).json()["auth"] == "not_configured"


def test_bad_tokens_get_a_generic_401_and_are_audited_without_the_token(kit: AuthKit) -> None:
    client = make_client(kit)
    token = kit.sign(kit.claims(exp=1))
    response = client.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.json() == {"detail": "Authentication required"}
    assert token not in response.text
    rows = audit_rows(client, "-")
    assert [r.action for r in rows] == ["authn.failure"]
    assert rows[0].details["reason"] == "expired" and token not in json.dumps(rows[0].details)
    assert rows[0].request_id == response.headers["x-request-id"]


@pytest.mark.parametrize(
    "header", ["", "Basic abc", "Bearer", "Bearer   ", "bearer-only", "Token abc.def.ghi"]
)
def test_malformed_authorization_headers_are_rejected(kit: AuthKit, header: str) -> None:
    response = make_client(kit).get(
        "/api/v1/me", headers={"Authorization": header} if header else {}
    )
    assert response.status_code == 401


# ---------------------------------------------------------------- AC3: role x route matrix

ALL_ROLES = ["viewer", "engineer", "approver", "auditor", "admin"]
VIEW_ROLES = {"viewer", "engineer", "approver", "admin"}  # anomaly:read
INCIDENT_READERS = set(ALL_ROLES)
RANGE = {
    "asset_id": ASSET,
    "range_start": "2020-01-09T00:00:00",
    "range_end": "2020-01-09T01:00:00",
    "episode_start": "2020-01-09T00:00:00",
}
DECISION = {"decision": "dismiss", "rationale": "ok", "expected_version": 1}
OBSERVATION = {"category": "field_inspection", "note": "Verified intake valve visually."}
OUTCOME = {
    "outcome": "confirmed_leak",
    "maintenance_action": "Tightened purge hose coupling during inspection",
    "notes": "Verified leak cleared post-tightening.",
}

MATRIX: list[tuple[str, str, dict[str, Any] | None, set[str]]] = [
    ("GET", "/api/v1/anomaly/model", None, VIEW_ROLES),
    ("GET", "/api/v1/assets", None, VIEW_ROLES),
    ("GET", f"/api/v1/assets/{ASSET}", None, VIEW_ROLES),
    ("GET", "/api/v1/datasets", None, VIEW_ROLES),
    (
        "GET",
        "/api/v1/anomaly/replay?start=2020-01-09T00:00:00&end=2020-01-09T01:00:00",
        None,
        VIEW_ROLES,
    ),
    (
        "GET",
        "/api/v1/anomaly/event-evidence?start=2020-01-09T00:00:00&end=2020-01-09T01:00:00",
        None,
        VIEW_ROLES,
    ),
    (
        "GET",
        "/api/v1/anomaly/rca?start=2020-01-09T00:00:00&end=2020-01-09T01:00:00",
        None,
        VIEW_ROLES,
    ),
    ("POST", "/api/v1/anomaly/score", {}, VIEW_ROLES),
    ("GET", "/api/v1/incidents", None, INCIDENT_READERS),
    ("GET", f"/api/v1/incidents/{ZERO_ID}", None, INCIDENT_READERS),
    ("GET", f"/api/v1/incidents/{ZERO_ID}/evidence-package", None, INCIDENT_READERS),
    ("GET", f"/api/v1/incidents/{ZERO_ID}/rca", None, INCIDENT_READERS),
    ("POST", "/api/v1/incidents", RANGE, {"engineer", "admin"}),
    (
        "POST",
        f"/api/v1/incidents/{ZERO_ID}/observations",
        OBSERVATION,
        {"engineer", "approver", "admin"},
    ),
    (
        "POST",
        f"/api/v1/incidents/{ZERO_ID}/outcomes",
        OUTCOME,
        {"engineer", "approver", "admin"},
    ),
    ("POST", f"/api/v1/incidents/{ZERO_ID}/decisions", DECISION, {"approver"}),
    ("GET", "/api/v1/audit/events", None, {"auditor", "admin"}),
    ("GET", "/api/v1/audit/verify", None, {"auditor", "admin"}),
]


@pytest.mark.parametrize("role", ALL_ROLES)
@pytest.mark.parametrize(("method", "path", "body", "allowed"), MATRIX)
def test_role_route_matrix(
    kit: AuthKit,
    anomaly_service: AnomalyService,
    role: str,
    method: str,
    path: str,
    body: dict | None,
    allowed: set[str],
) -> None:
    client = make_client(kit, anomaly_service)
    response = client.request(method, path, json=body, headers=kit.headers(roles=[role]))
    if role in allowed:
        assert response.status_code not in (401, 403), (role, method, path, response.text)
    else:
        assert response.status_code == 403, (role, method, path)
        assert response.json() == {"detail": "Forbidden"}


def test_a_token_without_recognised_roles_can_only_ask_who_it_is(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    client = make_client(kit, anomaly_service)
    headers = kit.headers(roles=["superuser"])
    assert client.get("/api/v1/me", headers=headers).json()["permissions"] == []
    for method, path, body, _ in MATRIX:
        assert client.request(method, path, json=body, headers=headers).status_code == 403


def test_denials_are_audited_in_the_callers_tenant(kit: AuthKit) -> None:
    client = make_client(kit)
    client.get("/api/v1/audit/events", headers=kit.headers(sub="vera", roles=["viewer"]))
    rows = audit_rows(client, "acme")
    assert [(r.action, r.outcome, r.actor_id) for r in rows] == [("authz.denied", "denied", "vera")]
    assert rows[0].details["reason"] == "missing:audit:read"


def test_missing_tenant_is_forbidden_not_defaulted(kit: AuthKit) -> None:
    client = make_client(kit)
    assert client.get("/api/v1/me", headers=kit.headers(tenant=None)).status_code == 403
    # a default tenant alone does not enable the fallback: single-tenant mode is an explicit opt-in
    assert (
        make_client(kit, default_tenant_id="solo")
        .get("/api/v1/me", headers=kit.headers(tenant=None))
        .status_code
        == 403
    )
    solo = make_client(kit, single_tenant_mode=True, default_tenant_id="solo")
    assert solo.get("/api/v1/me", headers=kit.headers(tenant=None)).json()["tenant_id"] == "solo"


def test_me_reports_the_principal(kit: AuthKit) -> None:
    body = (
        make_client(kit)
        .get("/api/v1/me", headers=kit.headers(sub="eve", roles=["engineer"]))
        .json()
    )
    assert (
        body["actor_id"] == "eve" and body["tenant_id"] == "acme" and body["asset_ids"] == [ASSET]
    )
    assert "incident:create" in body["permissions"] and "decision:create" not in body["permissions"]


# ---------------------------------------------------------------- AC7: rate limits


def test_per_principal_rate_limit_returns_429_and_is_independent_per_principal(
    kit: AuthKit,
) -> None:
    client = make_client(kit, rate_limit_read_per_min=3)
    alice, bob = kit.headers(sub="alice"), kit.headers(sub="bob")
    assert [client.get("/api/v1/incidents", headers=alice).status_code for _ in range(3)] == [
        200
    ] * 3
    limited = client.get("/api/v1/incidents", headers=alice)
    assert limited.status_code == 429 and int(limited.headers["retry-after"]) >= 1
    assert client.get("/api/v1/incidents", headers=bob).status_code == 200


def test_expensive_routes_cost_more(kit: AuthKit, anomaly_service: AnomalyService) -> None:
    client = make_client(kit, anomaly_service, rate_limit_read_per_min=5)
    url = "/api/v1/anomaly/replay?start=2020-01-09T00:00:00&end=2020-01-09T01:00:00"
    assert client.get(url, headers=kit.headers()).status_code == 200  # costs 5
    assert client.get(url, headers=kit.headers()).status_code == 429


def test_write_limit_is_separate_from_the_read_limit(kit: AuthKit) -> None:
    client = make_client(kit, rate_limit_write_per_min=1)
    headers = kit.headers(roles=["engineer"])
    assert client.post("/api/v1/incidents", json=RANGE, headers=headers).status_code != 429
    assert client.post("/api/v1/incidents", json=RANGE, headers=headers).status_code == 429
    assert client.get("/api/v1/incidents", headers=headers).status_code == 200


def test_repeated_authentication_failures_are_throttled(kit: AuthKit) -> None:
    client = make_client(kit, rate_limit_auth_failures_per_min=3)
    codes = [
        client.get("/api/v1/me", headers={"Authorization": "Bearer junk"}).status_code
        for _ in range(5)
    ]
    assert codes == [401, 401, 401, 429, 429]
    # still-failing callers carry Retry-After, but a valid token is never locked out
    assert "retry-after" in client.get("/api/v1/me", headers={"Authorization": "Bearer x"}).headers
    assert client.get("/api/v1/me", headers=kit.headers()).status_code == 200
    # throttled attempts are not written to the audit log (no audit flooding)
    assert len(audit_rows(client, "-")) == 3


# ---------------------------------------------------------------- AC8 / AC9: limits and hygiene


def test_oversized_bodies_are_rejected_before_parsing(kit: AuthKit) -> None:
    client = make_client(kit, max_body_bytes=2048)
    headers = {**kit.headers(roles=["engineer"]), "Content-Type": "application/json"}
    big = b'{"asset_id":"' + b"a" * 5000 + b'"}'
    assert client.post("/api/v1/incidents", content=big, headers=headers).status_code == 413

    def chunks():  # no Content-Length: the streaming guard must still stop it
        for _ in range(10):
            yield b"x" * 1000

    streamed = client.post("/api/v1/incidents", content=chunks(), headers=headers)
    assert streamed.status_code == 413 and streamed.json()["detail"] == "Request body too large"


def test_unknown_fields_and_bad_values_are_422(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    client = make_client(kit, anomaly_service)
    headers = kit.headers(roles=["engineer"])
    assert (
        client.post(
            "/api/v1/incidents", json={**RANGE, "tenant_id": "evil"}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/incidents", json={**RANGE, "range_end": "2020-02-09T00:00:00"}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/incidents", json={**RANGE, "asset_id": "../etc"}, headers=headers
        ).status_code
        == 422
    )
    assert client.get("/api/v1/incidents?limit=1000", headers=headers).status_code == 422
    assert client.get("/api/v1/incidents/not-a-uuid", headers=headers).status_code == 422


@pytest.mark.parametrize("path", [HEALTH, "/api/v1/me", "/nope"])
def test_security_headers_and_request_id_on_every_response(kit: AuthKit, path: str) -> None:
    response = make_client(kit).get(path)
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"
    assert len(response.headers["x-request-id"]) >= 8


def test_request_id_is_propagated_only_when_safe(kit: AuthKit) -> None:
    client = make_client(kit)
    assert (
        client.get(HEALTH, headers={"X-Request-ID": "trace-12345678"}).headers["x-request-id"]
        == "trace-12345678"
    )
    replaced = client.get(HEALTH, headers={"X-Request-ID": "bad id\r\nX-Injected: 1"})
    assert replaced.headers["x-request-id"] != "bad id"
    assert "x-injected" not in replaced.headers


def test_unhandled_errors_return_a_generic_500_without_internals(kit: AuthKit) -> None:
    client = make_client(kit)
    app = client.app

    @app.get("/boom")  # type: ignore[attr-defined]
    def boom() -> None:
        raise RuntimeError("secret detail at D:\\FactoryMind\\data\\private.db")

    response = TestClient(app, raise_server_exceptions=False).get("/boom")
    assert response.status_code == 500
    assert "secret" not in response.text and "FactoryMind" not in response.text
    assert response.json()["request_id"] == response.headers["x-request-id"]
    assert response.headers["x-content-type-options"] == "nosniff"


def test_interactive_docs_are_not_served_in_production(kit: AuthKit) -> None:
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert make_client(kit, environment="production").get(path).status_code == 404
        assert make_client(kit).get(path).status_code == 200
