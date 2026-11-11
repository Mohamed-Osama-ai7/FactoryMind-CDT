"""Audit-chain contention: nothing is lost or swallowed, and state changes fail closed."""

from __future__ import annotations

import logging
import threading
from typing import Any

import pytest
from fastapi.testclient import TestClient

from factorymind.anomaly.service import AnomalyService
from factorymind.db import audit as audit_module
from factorymind.db.audit import PRE_AUTH_TENANT, AuditContentionError, verify_chain
from tests.auth_helpers import ASSET, AuthKit, make_client

RS, RE = "2020-01-09T00:00:00", "2020-01-09T12:00:00"
CONCURRENCY = 12


def run_threads(count: int, target: Any) -> list[Any]:
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


def chain(client: TestClient, tenant: str) -> Any:
    with client.app.state.db.session() as session:  # type: ignore[attr-defined]
        return verify_chain(session, tenant)


def audit_actions(client: TestClient, kit: AuthKit) -> list[str]:
    headers = kit.headers(roles=["auditor"], sub="audrey")
    response = client.get("/api/v1/audit/events", params={"limit": 100}, headers=headers)
    return [e["action"] for e in response.json()["items"]]


def denied(client: TestClient, kit: AuthKit) -> int:
    headers = kit.headers(roles=["viewer"], assets=[])
    return client.get("/api/v1/anomaly/model", headers=headers).status_code


def episode(client: TestClient, kit: AuthKit) -> str:
    replay = client.get(f"/api/v1/anomaly/replay?start={RS}&end={RE}", headers=kit.headers())
    return str(replay.json()["episodes"][0]["start"])


# ------------------------------------------------ concurrency: nothing is lost


def test_twelve_concurrent_denials_record_twelve_events_in_a_complete_valid_chain(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    client = make_client(kit, anomaly_service)
    codes = run_threads(CONCURRENCY, lambda _: denied(client, kit))
    assert codes == [403] * CONCURRENCY
    status = chain(client, "acme")
    assert status.valid and status.events == CONCURRENCY
    assert audit_actions(client, kit).count("authz.denied") == CONCURRENCY
    assert client.app.state.db.audit_failures == 0  # type: ignore[attr-defined]


def test_twelve_concurrent_authentication_failures_are_all_recorded(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    client = make_client(kit, anomaly_service, rate_limit_auth_failures_per_min=1000)

    def bad(_: int) -> int:
        return client.get("/api/v1/me", headers={"Authorization": "Bearer junk"}).status_code

    assert run_threads(CONCURRENCY, bad) == [401] * CONCURRENCY
    status = chain(client, PRE_AUTH_TENANT)
    assert status.valid and status.events == CONCURRENCY


def test_a_state_change_racing_with_denials_is_applied_once_and_everything_is_recorded(
    kit: AuthKit, anomaly_service: AnomalyService
) -> None:
    client = make_client(kit, anomaly_service)
    body = {"asset_id": ASSET, "range_start": RS, "range_end": RE}
    body["episode_start"] = episode(client, kit)

    def work(i: int) -> int:
        if i == 0:
            headers = kit.headers(roles=["engineer"], sub="eve")
            return client.post("/api/v1/incidents", json=body, headers=headers).status_code
        return denied(client, kit)

    codes = run_threads(CONCURRENCY, work)
    assert codes[0] == 201 and set(codes[1:]) == {403}, codes  # no 500/503 under contention
    status = chain(client, "acme")
    assert status.valid and status.events == CONCURRENCY
    actions = audit_actions(client, kit)
    assert (
        actions.count("authz.denied") == CONCURRENCY - 1 and actions.count("incident.create") == 1
    )
    assert client.get("/api/v1/incidents", headers=kit.headers()).json()["total"] == 1


# ------------------------------------------------ retries: bounded, using fresh state


def _fail_first(real: Any, times: int) -> Any:
    calls = {"n": 0}

    def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] <= times:
            raise AuditContentionError("simulated collision")
        return real(*args, **kwargs)

    flaky.calls = calls  # type: ignore[attr-defined]
    return flaky


def test_a_collision_is_retried_and_the_event_is_recorded_exactly_once(
    kit: AuthKit, anomaly_service: AnomalyService, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = make_client(kit, anomaly_service)
    flaky = _fail_first(audit_module.append_audit, times=3)
    monkeypatch.setattr("factorymind.auth.deps.append_audit", flaky)
    assert denied(client, kit) == 403
    assert flaky.calls["n"] == 4  # three collisions, then success
    status = chain(client, "acme")
    assert status.valid and status.events == 1
    assert client.app.state.db.audit_failures == 0  # type: ignore[attr-defined]


# ------------------------------------------------ failure is loud and fails closed


def test_an_unrecordable_denial_is_still_denied_but_loud(
    kit: AuthKit,
    anomaly_service: AnomalyService,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = make_client(kit, anomaly_service)
    flaky = _fail_first(audit_module.append_audit, times=10_000)
    monkeypatch.setattr("factorymind.auth.deps.append_audit", flaky)
    with caplog.at_level(logging.CRITICAL, logger="factorymind.security"):
        assert denied(client, kit) == 403  # a denial never depends on the audit log
    assert flaky.calls["n"] == 8  # bounded: exactly the retry budget
    assert client.app.state.db.audit_failures == 1  # type: ignore[attr-defined]
    assert any("AUDIT FAILURE" in r.getMessage() for r in caplog.records)
    assert chain(client, "acme").events == 0


def test_a_state_change_whose_audit_cannot_be_recorded_is_rolled_back_with_503(
    kit: AuthKit,
    anomaly_service: AnomalyService,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = make_client(kit, anomaly_service)
    flaky = _fail_first(audit_module.append_audit, times=10_000)
    monkeypatch.setattr("factorymind.api.incidents.append_audit", flaky)
    body = {"asset_id": ASSET, "range_start": RS, "range_end": RE}
    body["episode_start"] = episode(client, kit)
    with caplog.at_level(logging.CRITICAL):
        response = client.post(
            "/api/v1/incidents", json=body, headers=kit.headers(roles=["engineer"], sub="eve")
        )
    assert response.status_code == 503 and "retry-after" in response.headers
    assert "Traceback" not in response.text
    assert client.get("/api/v1/incidents", headers=kit.headers()).json()["total"] == 0
    assert client.app.state.db.audit_failures == 1  # type: ignore[attr-defined]
    assert any("AUDIT FAILURE" in r.getMessage() for r in caplog.records)
