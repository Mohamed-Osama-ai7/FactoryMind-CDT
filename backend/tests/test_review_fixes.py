"""Regression tests for the independent review of PR #5.

Covers asset/tenant grants, token errors, throttling, data-source binding, tenancy mode and
contention."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from factorymind.anomaly.service import AnomalyService
from factorymind.auth.tokens import RemoteJwks, TokenError
from factorymind.config import Settings
from factorymind.db.audit import verify_chain
from tests.auth_helpers import ASSET, KID, AuthKit, b64, make_client

RS, RE = "2020-01-09T00:00:00", "2020-01-09T12:00:00"
ANOMALY_ROUTES = [
    ("GET", "/api/v1/anomaly/model", None),
    ("GET", f"/api/v1/anomaly/replay?start={RS}&end={RE}", None),
    ("POST", "/api/v1/anomaly/score", {"samples": []}),
]


def call(client: TestClient, method: str, url: str, body: Any, headers: dict[str, str]) -> int:
    return client.request(method, url, json=body, headers=headers).status_code


def first_episode(client: TestClient, kit: AuthKit) -> str:
    replay = client.get(f"/api/v1/anomaly/replay?start={RS}&end={RE}", headers=kit.headers())
    return str(replay.json()["episodes"][0]["start"])


# ------------------------------------------------ anomaly routes honour tenant and asset grants


@pytest.mark.parametrize(("method", "url", "body"), ANOMALY_ROUTES)
def test_anomaly_routes_deny_other_tenants_and_ungranted_assets(
    kit: AuthKit, anomaly_service: AnomalyService, method: str, url: str, body: Any
) -> None:
    client = make_client(kit, anomaly_service)
    assert call(client, method, url, body, kit.headers(roles=["viewer"])) != 403  # control
    for headers in (
        kit.headers(roles=["viewer"], tenant="globex"),  # another tenant
        kit.headers(roles=["viewer"], assets=[]),  # no grants
        kit.headers(roles=["viewer"], assets=["pump-7"]),  # grant for some other asset
    ):
        assert call(client, method, url, body, headers) == 403


def test_anomaly_denied_without_a_data_source_owner(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    client = make_client(kit, anomaly_service, data_source_tenant_id=None)
    url = f"/api/v1/anomaly/replay?start={RS}&end={RE}"
    assert call(client, "GET", url, None, kit.headers()) == 403  # deny by default
    solo = make_client(
        kit,
        anomaly_service,
        data_source_tenant_id=None,
        single_tenant_mode=True,
        default_tenant_id="acme",
    )
    assert call(solo, "GET", url, None, kit.headers()) == 200


def test_anomaly_denials_are_audited_with_a_reason(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    client = make_client(kit, anomaly_service)
    client.get("/api/v1/anomaly/model", headers=kit.headers(roles=["viewer"], assets=[]))
    events = client.get(
        "/api/v1/audit/events", headers=kit.headers(roles=["auditor"], sub="audrey")
    ).json()["items"]
    reasons = [e["details"]["reason"] for e in events if e["action"] == "authz.denied"]
    assert reasons == ["asset_not_granted"]


# ------------------------------------------------ incident creation is bound to the data source


def test_incident_for_another_configured_asset_is_rejected(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    client = make_client(kit, anomaly_service, asset_ids=[ASSET, "pump-7"])
    headers = kit.headers(roles=["engineer"], assets=[ASSET, "pump-7"])
    body = {
        "asset_id": "pump-7",
        "range_start": RS,
        "range_end": RE,
        "episode_start": first_episode(client, kit),
    }
    assert client.post("/api/v1/incidents", json=body, headers=headers).status_code == 404
    listing = client.get("/api/v1/incidents", headers=kit.headers()).json()
    assert listing["total"] == 0  # nothing was stored


def test_data_source_asset_must_be_a_configured_asset(kit: AuthKit) -> None:
    with pytest.raises(ValidationError):
        kit.settings(data_source_asset_id="elsewhere")


# ------------------------------------------------ token failures are controlled, never 500


def _token_with_header(kit: AuthKit, header: dict[str, Any]) -> str:
    h = b64(json.dumps(header).encode())
    return f"{h}.{b64(json.dumps(kit.claims()).encode())}.{b64(b'x' * 64)}"


@pytest.mark.parametrize("alg", ["ES256", "ES384", "PS256", "RS512"])
def test_key_type_or_algorithm_mismatch_is_a_401_not_a_500(kit: AuthKit, alg: str) -> None:
    client = make_client(kit, auth_algorithms=["RS256", "RS512", "ES256", "ES384", "PS256"])
    token = _token_with_header(kit, {"alg": alg, "kid": KID})
    response = client.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert "Traceback" not in response.text


@pytest.mark.parametrize("roles", [[["viewer"]], [{"a": 1}], [None, 5, "viewer"], {"x": "viewer"}])
def test_malformed_role_claims_never_crash_and_grant_only_valid_roles(
    kit: AuthKit, roles: Any
) -> None:
    client = make_client(kit)
    token = kit.sign(kit.claims(roles=roles))
    response = client.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    expected = ["viewer"] if isinstance(roles, list) and "viewer" in roles else []
    assert response.json()["roles"] == expected


# ------------------------------------------------ throttling cannot lock out valid users


def test_a_flood_of_bad_tokens_does_not_lock_out_a_valid_user(kit: AuthKit) -> None:
    client = make_client(kit, rate_limit_auth_failures_per_min=3)
    bad = [
        client.get("/api/v1/me", headers={"Authorization": "Bearer junk"}).status_code
        for _ in range(8)
    ]
    assert bad[:3] == [401] * 3 and set(bad[3:]) == {429}
    assert client.get("/api/v1/me", headers=kit.headers()).status_code == 200


# ------------------------------------------------ JWKS refresh is bounded


class FakeClient:
    def __init__(self) -> None:
        self.fetches = 0

    def get_signing_keys(self, refresh: bool = False) -> list[Any]:
        self.fetches += int(refresh)
        return []


def test_unknown_kids_trigger_at_most_one_refetch_per_cooldown(kit: AuthKit) -> None:
    remote = RemoteJwks("https://idp.test/jwks", refresh_cooldown_s=3600)
    fake = FakeClient()
    remote._client = fake  # type: ignore[assignment]
    token = kit.sign(kit.claims(), kid="rotated-away")
    for _ in range(25):
        with pytest.raises(TokenError) as info:
            remote.key_for(token)
        assert info.value.reason == "unknown_key"
    assert fake.fetches == 1


# ------------------------------------------------ tenancy mode


def test_single_tenant_mode_requires_a_default_tenant() -> None:
    with pytest.raises(ValidationError):
        Settings(single_tenant_mode=True)


# ------------------------------------------------ concurrency and contention


def _run_threads(count: int, target: Any) -> list[Any]:
    results: list[Any] = [None] * count
    barrier = threading.Barrier(count)

    def worker(i: int) -> None:
        barrier.wait()
        results[i] = target(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_concurrent_decisions_on_one_incident_yield_exactly_one_winner(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    client = make_client(kit, anomaly_service)
    created = client.post(
        "/api/v1/incidents",
        json={
            "asset_id": ASSET,
            "range_start": RS,
            "range_end": RE,
            "episode_start": first_episode(client, kit),
        },
        headers=kit.headers(roles=["engineer"], sub="eve"),
    ).json()

    def decide(i: int) -> int:
        body = {"decision": "confirm", "rationale": f"review {i}", "expected_version": 1}
        return client.post(
            f"/api/v1/incidents/{created['id']}/decisions",
            json=body,
            headers=kit.headers(roles=["approver"], sub=f"approver-{i}"),
        ).status_code

    codes = _run_threads(4, decide)
    assert codes.count(201) == 1, codes
    assert set(codes) <= {201, 409}, codes  # losers get a clean conflict, never a server error
    detail = client.get(f"/api/v1/incidents/{created['id']}", headers=kit.headers()).json()
    assert len(detail["decisions"]) == 1 and detail["status"] == "confirmed"
    with client.app.state.db.session() as session:  # type: ignore[attr-defined]
        assert verify_chain(session, "acme").valid


def test_concurrent_denials_keep_the_audit_chain_valid(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    client = make_client(kit, anomaly_service)

    def denied(_: int) -> int:
        return client.get(
            "/api/v1/anomaly/model", headers=kit.headers(roles=["viewer"], assets=[])
        ).status_code

    codes = _run_threads(6, denied)
    assert set(codes) == {403}  # audit contention must not change the outcome of a request
    with client.app.state.db.session() as session:  # type: ignore[attr-defined]
        assert verify_chain(session, "acme").valid
