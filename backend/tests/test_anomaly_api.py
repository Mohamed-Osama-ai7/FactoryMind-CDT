from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from factorymind.anomaly.service import AnomalyService
from factorymind.data import metropt3
from tests.auth_helpers import AuthKit, make_client
from tests.conftest import simulate_apu


@pytest.fixture
def service(anomaly_service: AnomalyService) -> AnomalyService:
    return anomaly_service


@pytest.fixture
def client(kit: AuthKit, anomaly_service: AnomalyService) -> TestClient:
    test_client = make_client(kit, anomaly_service)
    test_client.headers.update(kit.headers(roles=["viewer"]))
    return test_client


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
    assert (
        body["loeo"]["mean_operating_false_alarms_per_day"]
        >= body["loeo"]["mean_false_alarms_per_day"]
    )
    assert [f["held_out"] for f in body["loeo"]["folds"]] == ["V1", "T1"]
    assert body["fault_injection"]["simulated"] is True
    assert "gross_rate" in body["fault_injection"]
    assert "control_alarm_rate" in body["fault_injection"]
    assert "clean_control_sensitivity" in body["fault_injection"]
    assert "incremental_lift" in body["fault_injection"]
    assert body["reference_holdout_check"]["alpha"] > 0
    assert body["exceedance_check"]["windows"] > 0
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


def test_score_accepts_slight_timestamp_irregularities_within_tolerance(client: TestClient) -> None:
    frame = simulate_apu("2020-01-05 12:00", 60, seed=11)
    # Introduce slight uneven sampling jitter pushing total span to 10 min 30 s (<= 11 min)
    rng = np.random.default_rng(4)
    deltas = rng.integers(8, 14, size=len(frame) - 1).astype(float)
    deltas = deltas * (630.0 / float(deltas.sum()))
    offsets = np.concatenate([[0.0], np.cumsum(deltas)])
    frame.index = pd.Timestamp("2020-01-05 12:00:00") + pd.to_timedelta(offsets, unit="s")
    frame.index.name = "timestamp"
    response = client.post("/api/v1/anomaly/score", json=payload(frame))
    assert response.status_code == 200
    assert response.json()["result"]["status"] in {"normal", "anomalous"}


def test_score_rejects_20_minute_oversized_window(client: TestClient) -> None:
    frame = simulate_apu("2020-01-05 12:00", 120, seed=11)  # 19m 50s span
    response = client.post("/api/v1/anomaly/score", json=payload(frame))
    assert response.status_code == 422


def test_score_rejects_max_sample_count_with_oversized_duration(client: TestClient) -> None:
    frame = simulate_apu("2020-01-05 12:00", 2000, seed=11)  # ~5.5 h span at 10 s
    response = client.post("/api/v1/anomaly/score", json=payload(frame))
    assert response.status_code == 422


def test_score_rejects_short_duration_48_sample_burst(client: TestClient) -> None:
    # 48 samples ~1 s apart (47 s span) must never be scored as a valid 10-minute window.
    frame = simulate_apu("2020-01-05 12:00", 48, seed=11)
    frame.index = pd.date_range("2020-01-05 12:00:00", periods=48, freq="1s", name="timestamp")
    response = client.post("/api/v1/anomaly/score", json=payload(frame))
    assert response.status_code == 422


def test_score_rejects_dense_high_frequency_10_minute_block(client: TestClient) -> None:
    # 600 samples at 1 Hz over ~10 min violates the 5-15 s (~0.1 Hz) contract; no silent resampling.
    frame = simulate_apu("2020-01-05 12:00", 600, seed=11)
    frame.index = pd.date_range("2020-01-05 12:00:00", periods=600, freq="1s", name="timestamp")
    response = client.post("/api/v1/anomaly/score", json=payload(frame))
    assert response.status_code == 422


def test_score_handles_irregular_timestamps(client: TestClient) -> None:
    # Irregular timestamps within 10 min containing an internal gap > 60 s => insufficient_data
    frame = simulate_apu("2020-01-05 12:00", 60, seed=11)
    gapped = frame.drop(frame.index[25:34])  # 90 s gap inside the 10-min block
    ok_resp = client.post("/api/v1/anomaly/score", json=payload(gapped))
    assert ok_resp.status_code == 200
    assert ok_resp.json()["result"]["status"] == "insufficient_data"

    # Irregular sparse timestamps exceeding 11 min tolerance (48 samples at 15 s = 11m 45s) => 422
    sparse = simulate_apu("2020-01-05 12:00", 48, seed=11)
    sparse.index = pd.date_range("2020-01-05 12:00:00", periods=48, freq="15s")
    sparse.index.name = "timestamp"
    bad_resp = client.post("/api/v1/anomaly/score", json=payload(sparse))
    assert bad_resp.status_code == 422


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
    kit: AuthKit, service: AnomalyService, tmp_path: Path
) -> None:
    broken = AnomalyService(service.bundle, service.card, tmp_path)
    client = make_client(kit, broken)
    response = client.get(
        "/api/v1/anomaly/replay",
        params={"start": "2020-01-05T00:00:00", "end": "2020-01-05T01:00:00"},
        headers=kit.headers(),
    )
    assert response.status_code == 503 and str(tmp_path) not in response.text


def test_missing_model_returns_503_without_paths(kit: AuthKit, tmp_path: Path) -> None:
    client = make_client(kit, artifacts_dir=tmp_path)  # no model has been trained there
    response = client.get("/api/v1/anomaly/model", headers=kit.headers())
    assert response.status_code == 503
    assert "No current model" in response.json()["detail"]
    assert str(tmp_path) not in response.text


def test_model_endpoint_exposes_wp2_summary(client: TestClient) -> None:
    body = client.get("/api/v1/anomaly/model").json()
    wp2 = body["wp2"]
    assert wp2 is not None
    assert wp2["primary_detector"] == body["detector"]
    assert set(wp2["per_regime_metrics"]) == {
        "loaded",
        "unloaded",
        "starting",
        "stopped",
        "insufficient_evidence",
    }
    assert wp2["simulated_fault_separation"]["simulated"] is True
    assert wp2["simulated_fault_separation"]["real_operational_data"] is False


def test_score_exposes_regime_and_separates_frozen_or_inconsistent_telemetry(
    client: TestClient,
) -> None:
    frozen = simulate_apu("2020-01-05 12:00", 60, seed=5).copy()
    frozen["TP3"] = 8.9
    frozen["TP2"] = 8.9
    frozen["Motor_current"] = 7.0
    frozen["COMP"] = 0.0
    frozen["DV_eletric"] = 1.0
    resp_frozen = client.post("/api/v1/anomaly/score", json=payload(frozen))
    assert resp_frozen.status_code == 200
    rf = resp_frozen.json()["result"]
    assert rf["status"] == "insufficient_data"
    assert rf["score"] is None
    assert rf["event_class"] == "sensor_consistency_warning"
    assert rf["warning_active"] is True
    assert rf["regime"]["regime"] == "insufficient_evidence"
    assert rf["regime"]["reason"] == "stale_telemetry"

    contradictory = simulate_apu("2020-01-05 12:10", 60, seed=6).copy()
    contradictory["COMP"] = 1.0
    contradictory["DV_eletric"] = 1.0
    resp_contra = client.post("/api/v1/anomaly/score", json=payload(contradictory))
    assert resp_contra.status_code == 200
    rc = resp_contra.json()["result"]
    assert rc["status"] == "insufficient_data"
    assert rc["event_class"] == "sensor_consistency_warning"
    assert rc["regime"]["reason"] == "signal_inconsistency"
    assert "valve_contradiction" in rc["regime"]["quality_flags"]


def test_replay_exposes_episode_hypotheses_and_warnings_field(client: TestClient) -> None:
    body = client.get(
        "/api/v1/anomaly/replay",
        params={"start": "2020-01-09T00:00:00", "end": "2020-01-09T12:00:00"},
    ).json()
    assert "warnings" in body
    assert body["episodes"]
    ep = body["episodes"][0]
    assert ep["event_class"] == "air_leak_alarm"
    assert ep["hypothesis"]
    assert ep["uncertainty_note"]
    assert ep["dominant_regime"] in {
        "loaded",
        "unloaded",
        "starting",
        "stopped",
        "insufficient_evidence",
    }
    assert isinstance(ep["regime_breakdown"], dict) and ep["regime_breakdown"]
    assert all("regime" in w and "event_class" in w for w in body["windows"])
