from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
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
    # the simulated duty rule is the default: pin it so the API tests are deterministic
    output = experiment.run(config(), telemetry(), REPORTS)
    root = tmp_path_factory.mktemp("api")
    experiment.save(output, root / "artifacts", "unit-model")
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


def test_model_info_reports_protocol_results_and_provenance(client: TestClient) -> None:
    body = client.get("/api/v1/anomaly/model").json()
    assert body["dataset_license"] == "CC BY 4.0"
    assert body["persistence"] >= 1 and 0 < body["exceedance_at_threshold"] <= 1
    assert body["loeo"]["events_total"] == 2
    assert [f["held_out"] for f in body["loeo"]["folds"]] == ["V1", "T1"]
    assert body["replay_available_from"].startswith("2020-01-01")
    assert [r["report_id"] for r in body["operator_reports"]] == ["V1", "T1"]
    assert isinstance(body["selection_rationale"], str) and body["selected_as_default"] is True


def leak_block(start: str = "2020-01-02 11:00") -> pd.DataFrame:
    frame = simulate_apu(start, 6 * 70, leak_from=start, leak_to="2020-01-02 13:00", seed=12)
    return frame.loc["2020-01-02 12:00":"2020-01-02 12:09:59"]


def test_score_normal_window(client: TestClient) -> None:
    frame = simulate_apu("2020-01-05 12:00", 60, seed=11)
    response = client.post("/api/v1/anomaly/score", json=payload(frame))
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["status"] == "normal" and result["n_samples"] == 60
    assert result["evidence"] == [] and 0 < result["exceedance"] <= 1


def test_score_anomalous_window_returns_grouped_evidence(client: TestClient) -> None:
    response = client.post("/api/v1/anomaly/score", json=payload(leak_block())).json()
    result = response["result"]
    assert result["status"] == "anomalous" and result["score"] > response["threshold"]
    groups = [e["group"] for e in result["evidence"]]
    assert groups and len(groups) == len(set(groups))  # at most one item per physical group
    assert all(e["meaning"] for e in result["evidence"])


def test_score_stopped_machine_is_explained_not_anomalous(client: TestClient) -> None:
    frame = simulate_apu(
        "2020-01-05 12:00", 60, off_from="2020-01-05 12:00", off_to="2020-01-05 12:10", seed=3
    )
    result = client.post("/api/v1/anomaly/score", json=payload(frame)).json()["result"]
    assert result["status"] == "stopped" and result["score"] is None and result["evidence"] == []


def test_score_too_few_samples_is_insufficient_data(client: TestClient) -> None:
    frame = simulate_apu("2020-01-05 12:00", 10)
    result = client.post("/api/v1/anomaly/score", json=payload(frame)).json()["result"]
    assert result["status"] == "insufficient_data" and result["score"] is None


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
    body = payload(simulate_apu("2020-01-05 12:00", 60))
    mutate(body)
    assert client.post("/api/v1/anomaly/score", json=body).status_code == 422


def test_replay_returns_windows_episodes_and_reports(client: TestClient) -> None:
    response = client.get(
        "/api/v1/anomaly/replay",
        params={"start": "2020-01-09T00:00:00", "end": "2020-01-09T12:00:00"},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["windows"]) == 72
    assert [r["report_id"] for r in body["operator_reports"]] == ["T1"]
    assert body["episodes"], "the simulated leak should raise an alarm"
    assert {e["truncated"] for e in body["episodes"]} == {False}
    anomalous = [w for w in body["windows"] if w["status"] == "anomalous"]
    assert anomalous and all(w["evidence"] for w in anomalous)


def test_replay_marks_stopped_windows(client: TestClient) -> None:
    body = client.get(
        "/api/v1/anomaly/replay",
        params={"start": "2020-01-09T19:00:00", "end": "2020-01-09T23:30:00"},
    ).json()
    statuses = {w["status"] for w in body["windows"]}
    assert "stopped" in statuses
    assert all(w["score"] is None for w in body["windows"] if w["status"] == "stopped")


def _episodes(service: AnomalyService, start: str, end: str) -> list[Any]:
    _, eps, _ = service.replay(pd.Timestamp(start), pd.Timestamp(end))
    return eps


@pytest.mark.parametrize("seed", range(12))
def test_incident_identity_does_not_depend_on_how_the_range_is_chunked(
    service: AnomalyService, seed: int
) -> None:
    """A sub-range reports the same start and raised-at as a pass over the whole leak day."""
    rng = np.random.default_rng(seed)
    day = pd.Timestamp("2020-01-09")
    whole = {
        (e.episode.start, e.episode.raised_at)
        for e in _episodes(service, str(day), str(day + pd.Timedelta(days=1)))
    }
    assert whole, "reference day must contain incidents"
    lo = day + pd.Timedelta(minutes=int(rng.integers(0, 12 * 60)))
    hi = lo + pd.Timedelta(minutes=int(rng.integers(60, 12 * 60)))
    for e in _episodes(service, str(lo), str(hi)):
        assert not e.truncated  # the 24 h look-back is longer than any simulated run
        assert (e.episode.start, e.episode.raised_at) in whole


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


def test_replay_without_processed_data_is_503_without_paths(
    service: AnomalyService, tmp_path: Path
) -> None:
    app = create_app()
    app.dependency_overrides[get_anomaly_service] = lambda: AnomalyService(
        service.bundle, service.card, tmp_path
    )
    response = TestClient(app).get(
        "/api/v1/anomaly/replay",
        params={"start": "2020-01-05T00:00:00", "end": "2020-01-05T01:00:00"},
    )
    assert response.status_code == 503 and str(tmp_path) not in response.text


def test_missing_model_returns_503_without_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, clear_service_cache: None
) -> None:
    from factorymind.config import get_settings

    monkeypatch.setenv("FACTORYMIND_ARTIFACTS_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        response = TestClient(create_app()).get("/api/v1/anomaly/model")
        assert response.status_code == 503
        assert "No current model" in response.json()["detail"]
        assert str(tmp_path) not in response.text
    finally:
        get_settings.cache_clear()
