from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from factorymind.anomaly import experiment
from factorymind.anomaly.service import AnomalyService, load_bundle
from factorymind.api.anomaly import get_anomaly_service
from factorymind.api.app import create_app
from factorymind.data import metropt3
from tests.conftest import simulate_apu
from tests.test_anomaly_experiment import REPORTS, config, telemetry


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> AnomalyService:
    results, bundles, scored = experiment.run(config(), telemetry(), REPORTS)
    root = tmp_path_factory.mktemp("api")
    experiment.save(results, bundles, scored, root / "artifacts", "unit-model")
    processed = root / "processed"
    (processed / metropt3.DATASET_ID).mkdir(parents=True)
    telemetry().to_parquet(processed / metropt3.DATASET_ID / "telemetry.parquet")
    bundle, card = load_bundle(root / "artifacts" / "models", "unit-model")
    return AnomalyService(bundle, card, processed, REPORTS)


@pytest.fixture
def client(service: AnomalyService) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_anomaly_service] = lambda: service
    yield TestClient(app)


def payload(frame: pd.DataFrame) -> dict[str, Any]:
    rows = frame.reset_index()
    rows["timestamp"] = rows["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    for col in metropt3.DIGITAL_SENSORS:
        rows[col] = rows[col].astype(int)
    return {"samples": rows.to_dict("records")}


def test_model_info(client: TestClient) -> None:
    body = client.get("/api/v1/anomaly/model").json()
    assert body["dataset_license"] == "CC BY 4.0"
    assert body["persistence"] >= 1
    assert body["replay_available_from"].startswith("2020-01-01")


def test_score_normal_window(client: TestClient) -> None:
    frame = simulate_apu("2020-01-02 12:00", 60, seed=11)
    response = client.post("/api/v1/anomaly/score", json=payload(frame))
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "normal"
    assert result["n_samples"] == 60
    assert len(result["evidence"]) == 5


def test_score_anomalous_window(client: TestClient) -> None:
    frame = simulate_apu(
        "2020-01-02 11:00",
        6 * 70,
        leak_from="2020-01-02 11:00",
        leak_to="2020-01-02 13:00",
        seed=12,
    ).loc["2020-01-02 12:00":"2020-01-02 12:09:59"]
    result = client.post("/api/v1/anomaly/score", json=payload(frame)).json()["result"]
    assert result["status"] == "anomalous"
    assert result["score"] > client.get("/api/v1/anomaly/model").json()["threshold"]
    assert {e["feature"] for e in result["evidence"]}


def test_score_too_few_samples_is_insufficient_data(client: TestClient) -> None:
    frame = simulate_apu("2020-01-02 12:00", 10)
    result = client.post("/api/v1/anomaly/score", json=payload(frame)).json()["result"]
    assert result["status"] == "insufficient_data"
    assert result["score"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["samples"][3].update(TP3=99.0),  # outside physical range
        lambda p: p["samples"][3].update(LPS=2),  # non-binary digital
        lambda p: p["samples"][3].pop("H1"),  # missing sensor
        lambda p: p["samples"][3].update(extra=1),  # unknown field
        lambda p: p["samples"].reverse(),  # not chronological
        lambda p: p.update(samples=[]),  # empty
        lambda p: p["samples"][0].update(timestamp="2020-01-02T12:00:00+00:00"),  # tz-aware
    ],
)
def test_score_rejects_invalid_input(client: TestClient, mutate: Any) -> None:
    body = payload(simulate_apu("2020-01-02 12:00", 60))
    mutate(body)
    assert client.post("/api/v1/anomaly/score", json=body).status_code == 422


def test_replay_returns_windows_episodes_and_reports(client: TestClient) -> None:
    response = client.get(
        "/api/v1/anomaly/replay",
        params={"start": "2020-01-05T00:00:00", "end": "2020-01-05T12:00:00"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["windows"]) == 72
    assert [r["report_id"] for r in body["operator_reports"]] == ["T1"]
    assert body["episodes"], "the simulated leak should raise an alarm"
    anomalous = [w for w in body["windows"] if w["status"] == "anomalous"]
    assert anomalous and all(w["evidence"] for w in anomalous)


@pytest.mark.parametrize(
    "params",
    [
        {"start": "2020-01-05T00:00:00", "end": "2020-01-04T00:00:00"},
        {"start": "2020-01-01T00:00:00", "end": "2020-01-09T00:00:00"},
        {"start": "2020-01-05T00:00:00+00:00", "end": "2020-01-05T01:00:00+00:00"},
    ],
)
def test_replay_rejects_bad_ranges(client: TestClient, params: dict[str, str]) -> None:
    assert client.get("/api/v1/anomaly/replay", params=params).status_code == 422


def test_replay_outside_data_returns_empty(client: TestClient) -> None:
    body = client.get(
        "/api/v1/anomaly/replay",
        params={"start": "2021-01-01T00:00:00", "end": "2021-01-02T00:00:00"},
    ).json()
    assert body["windows"] == [] and body["episodes"] == []


def test_replay_without_processed_data_is_503(service: AnomalyService, tmp_path: Path) -> None:
    app = create_app()
    app.dependency_overrides[get_anomaly_service] = lambda: AnomalyService(
        service.bundle, service.card, tmp_path
    )
    response = TestClient(app).get(
        "/api/v1/anomaly/replay",
        params={"start": "2020-01-05T00:00:00", "end": "2020-01-05T01:00:00"},
    )
    assert response.status_code == 503


def test_missing_model_returns_503(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clear_service_cache: None
) -> None:
    from factorymind.config import get_settings

    monkeypatch.setenv("FACTORYMIND_ARTIFACTS_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        client = TestClient(create_app())
        response = client.get("/api/v1/anomaly/model")
        assert response.status_code == 503
        assert "No current model" in response.json()["detail"]
    finally:
        get_settings.cache_clear()
