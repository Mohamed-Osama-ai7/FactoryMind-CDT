"""Tests for WP-A: Event Evidence Foundation, Quality, Timelines, Observations, and API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from factorymind.anomaly import experiment
from factorymind.anomaly.evidence_eval import evaluate_package, evaluate_service_report_events
from factorymind.anomaly.evidence_package import build_event_evidence_package
from factorymind.anomaly.service import AnomalyService, load_bundle, load_warning_bundle
from factorymind.data import metropt3
from factorymind.db.session import Database
from tests.auth_helpers import ASSET, AuthKit, make_client
from tests.conftest import simulate_apu
from tests.test_anomaly_events import _wp2_config


def _make_wp_a_service(
    tmp_path: Path,
    *,
    frame_override: pd.DataFrame | None = None,
) -> tuple[AnomalyService, pd.DataFrame, tuple[metropt3.FailureReport, ...]]:
    base = simulate_apu(
        "2020-01-01",
        6 * 60 * 24 * 9,
        leak_from="2020-01-06 06:00",
        leak_to="2020-01-06 09:00",
        off_from="2020-01-05 12:00",
        off_to="2020-01-05 14:00",
        seed=17,
    ).copy()
    reports = (
        metropt3.FailureReport("F1", "2020-01-06 06:00", "2020-01-06 09:00", "Air leak"),
        metropt3.FailureReport("F2", "2020-01-09 06:00", "2020-01-09 08:00", "Air leak"),
    )
    f2_mask = (base.index >= pd.Timestamp("2020-01-09 06:00")) & (
        base.index < pd.Timestamp("2020-01-09 08:00")
    )
    base.loc[f2_mask, "Motor_current"] = 7.1
    base.loc[f2_mask, "COMP"] = 0.0
    base.loc[f2_mask, "DV_eletric"] = 1.0

    train_frame = base
    replay_frame = frame_override if frame_override is not None else base.copy()

    out = experiment.run(_wp2_config(), train_frame, reports)
    experiment.save(out, tmp_path / "artifacts", "wp-a-model")
    processed = tmp_path / "processed"
    (processed / metropt3.DATASET_ID).mkdir(parents=True)
    replay_frame.to_parquet(processed / metropt3.DATASET_ID / "telemetry.parquet")

    models_dir = tmp_path / "artifacts" / "models"
    bundle, card = load_bundle(models_dir, "wp-a-model")
    warning_bundle = load_warning_bundle(models_dir, "wp-a-model", bundle)
    return (
        AnomalyService(bundle, card, processed, reports, warning_bundle=warning_bundle),
        replay_frame,
        reports,
    )


def test_event_evidence_package_phases_provenance_and_eval(tmp_path: Path) -> None:
    service, _, _ = _make_wp_a_service(tmp_path)
    pkg = build_event_evidence_package(
        service,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-06 06:00"),
        event_end=pd.Timestamp("2020-01-06 09:00"),
        raised_at=pd.Timestamp("2020-01-06 06:20"),
    )
    assert [p.phase for p in pkg.phases] == [
        "pre_event_baseline",
        "onset",
        "evolution",
        "peak",
        "recovery_or_persistence",
    ]
    assert pkg.provenance.primary_detector == service.bundle.detector_name
    assert pkg.provenance.warning_detector == (
        service.warning_bundle.detector_name if service.warning_bundle else None
    )
    assert pkg.quality.overall_status == "sufficient"
    assert pkg.quality.completeness_score == 1.0
    assert pkg.quality.baseline_available and pkg.quality.recovery_available
    assert any(n.event_id == "F1" for n in pkg.neighboring_events)
    assert any(t.category == "operator_report" for t in pkg.timeline)
    assert any(t.category == "alarm_episode" for t in pkg.timeline)

    peak = next(p for p in pkg.phases if p.phase == "peak")
    assert all(s.origin == "observed" and s.unit for s in peak.observed_signals)
    assert all(d.origin == "derived" for d in peak.derived_features)
    assert all(item.origin in ("statistical_inference", "derived") for item in pkg.evidence_items)

    ev_res = evaluate_package("F1", pkg, kind="synthetic_clean")
    assert ev_res.boundary_aligned
    assert ev_res.origins_pure
    assert ev_res.missing_data_honest

    suite = evaluate_service_report_events(service, asset_id=ASSET, kind="synthetic_suite")
    assert suite["cases"] == 2
    assert suite["boundary_alignment_rate"] == 1.0
    assert suite["origin_purity_rate"] == 1.0
    assert suite["missing_data_honesty_rate"] == 1.0


def test_quality_handles_missing_gaps_freezes_and_contradictions(tmp_path: Path) -> None:
    service_clean, base, _ = _make_wp_a_service(tmp_path / "clean")

    # 1. Missing baseline & gapped event + missing recovery -> honest degraded/insufficient quality
    gapped = base.copy()
    gapped = gapped[gapped.index >= pd.Timestamp("2020-01-06 06:00")]
    gapped = gapped[
        ~(
            (gapped.index >= pd.Timestamp("2020-01-06 06:20"))
            & (gapped.index < pd.Timestamp("2020-01-06 07:30"))
        )
    ]
    gapped = gapped[gapped.index < pd.Timestamp("2020-01-06 09:00")]
    proc_gap = tmp_path / "gap_proc"
    (proc_gap / metropt3.DATASET_ID).mkdir(parents=True)
    gapped.to_parquet(proc_gap / metropt3.DATASET_ID / "telemetry.parquet")
    svc_gap = AnomalyService(
        service_clean.bundle,
        service_clean.card,
        proc_gap,
        service_clean.reports,
        warning_bundle=service_clean.warning_bundle,
    )
    pkg_gap = build_event_evidence_package(
        svc_gap,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-06 06:00"),
        event_end=pd.Timestamp("2020-01-06 09:00"),
    )
    assert not pkg_gap.quality.baseline_available
    assert not pkg_gap.quality.recovery_available
    assert pkg_gap.quality.recovery_status == "unverified"
    assert pkg_gap.quality.missing_windows_count > 0
    assert pkg_gap.quality.missing_samples_count > 0
    assert pkg_gap.quality.completeness_score < 0.6
    assert pkg_gap.quality.overall_status == "insufficient"
    baseline_phase = next(p for p in pkg_gap.phases if p.phase == "pre_event_baseline")
    assert all(s.mean is None for s in baseline_phase.observed_signals)
    gap_eval = evaluate_package("gapped", pkg_gap, kind="synthetic_stress")
    assert gap_eval.missing_data_honest and gap_eval.origins_pure

    # 2. Single-sensor TP3 freeze + valve contradiction inside an event
    faulted = base.copy()
    tp3_freeze_mask = (faulted.index >= pd.Timestamp("2020-01-08 06:00")) & (
        faulted.index < pd.Timestamp("2020-01-08 06:30")
    )
    faulted.loc[tp3_freeze_mask, "TP3"] = 8.9
    faulted.loc[tp3_freeze_mask, "COMP"] = 1.0
    faulted.loc[tp3_freeze_mask, "DV_eletric"] = 1.0
    proc_fault = tmp_path / "fault_proc"
    (proc_fault / metropt3.DATASET_ID).mkdir(parents=True)
    faulted.to_parquet(proc_fault / metropt3.DATASET_ID / "telemetry.parquet")
    svc_fault = AnomalyService(
        service_clean.bundle,
        service_clean.card,
        proc_fault,
        service_clean.reports,
        warning_bundle=service_clean.warning_bundle,
    )
    pkg_fault = build_event_evidence_package(
        svc_fault,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-08 06:00"),
        event_end=pd.Timestamp("2020-01-08 06:30"),
    )
    assert "TP3" in pkg_fault.quality.frozen_sensors
    assert pkg_fault.quality.overall_status == "degraded"
    assert pkg_fault.quality.confidence_penalty_factor < 1.0
    kinds = {c.kind for c in pkg_fault.quality.contradictions}
    assert "valve_contradiction" in kinds
    assert "sensor_freeze" in kinds
    fault_eval = evaluate_package("freeze_contra", pkg_fault, kind="synthetic_stress")
    assert fault_eval.contradictions_detected >= 2 and fault_eval.origins_pure


def test_evidence_api_observations_audit_and_tenant_isolation(
    kit: AuthKit,
    anomaly_service: AnomalyService,
) -> None:
    client: TestClient = make_client(kit, anomaly_service)
    r_start, r_end = "2020-01-09T00:00:00", "2020-01-09T12:00:00"

    replay = client.get(
        f"/api/v1/anomaly/replay?start={r_start}&end={r_end}", headers=kit.headers()
    ).json()
    ep_start = replay["episodes"][0]["start"]
    ep_end = replay["episodes"][0]["end"]

    # Pre-incident event-evidence route works for anomaly readers
    ev_resp = client.get(
        "/api/v1/anomaly/event-evidence",
        params={"start": ep_start, "end": ep_end},
        headers=kit.headers(roles=["viewer"]),
    )
    assert ev_resp.status_code == 200
    ev_pkg = ev_resp.json()
    assert ev_pkg["asset_id"] == ASSET and ev_pkg["incident_id"] is None
    assert len(ev_pkg["phases"]) == 5

    # Reject tz-aware & invalid ranges with 422
    assert (
        client.get(
            "/api/v1/anomaly/event-evidence",
            params={"start": "2020-01-09T00:00:00+00:00", "end": ep_end},
            headers=kit.headers(),
        ).status_code
        == 422
    )

    # Create incident, append human observation, verify package & append-only DB triggers
    inc = client.post(
        "/api/v1/incidents",
        json={
            "asset_id": ASSET,
            "range_start": r_start,
            "range_end": r_end,
            "episode_start": ep_start,
        },
        headers=kit.headers(roles=["engineer"], sub="eng-1"),
    ).json()
    inc_id = inc["id"]

    # Viewer cannot post observations
    assert (
        client.post(
            f"/api/v1/incidents/{inc_id}/observations",
            json={"category": "field_inspection", "note": "Audible hiss near manifold."},
            headers=kit.headers(roles=["viewer"]),
        ).status_code
        == 403
    )

    obs_resp = client.post(
        f"/api/v1/incidents/{inc_id}/observations",
        json={
            "category": "field_inspection",
            "note": "Audible hiss near dryer purge line.",
            "observed_at": ep_start,
        },
        headers=kit.headers(roles=["engineer"], sub="eng-1"),
    )
    assert obs_resp.status_code == 201
    obs = obs_resp.json()
    assert obs["origin"] == "human_observation"

    # Cross-tenant caller receives 404 on both observations and evidence-package
    assert (
        client.post(
            f"/api/v1/incidents/{inc_id}/observations",
            json={"category": "field_inspection", "note": "Other tenant probe"},
            headers=kit.headers(tenant="beta", roles=["engineer"], sub="eve-beta"),
        ).status_code
        == 404
    )
    assert (
        client.get(
            f"/api/v1/incidents/{inc_id}/evidence-package",
            headers=kit.headers(tenant="beta", roles=["viewer"], sub="vera-beta"),
        ).status_code
        == 404
    )

    # Owner tenant retrieves full incident evidence package including the human observation
    pkg_resp = client.get(
        f"/api/v1/incidents/{inc_id}/evidence-package",
        headers=kit.headers(roles=["auditor"], sub="aud-1"),
    )
    assert pkg_resp.status_code == 200
    pkg_body = pkg_resp.json()
    assert pkg_body["incident_id"] == inc_id
    assert len(pkg_body["human_observations"]) == 1
    assert any(
        t["category"] == "human_observation" and "Audible hiss" in t["summary"]
        for t in pkg_body["timeline"]
    )

    # HumanObservation table is strictly append-only in SQLite (both UPDATE and DELETE refused)
    db: Database = client.app.state.db  # type: ignore[attr-defined]
    with (
        db.engine.begin() as conn,
        pytest.raises(DatabaseError, match="append-only"),
    ):
        conn.execute(text("UPDATE human_observation SET note = 'tampered'"))
    with (
        db.engine.begin() as conn,
        pytest.raises(DatabaseError, match="append-only"),
    ):
        conn.execute(text("DELETE FROM human_observation"))

    # Observation creation was recorded in the audit log
    audit_events = client.get(
        "/api/v1/audit/events",
        params={"action": "observation.create"},
        headers=kit.headers(roles=["auditor"], sub="aud-1"),
    ).json()["items"]
    assert len(audit_events) == 1
    assert audit_events[0]["object_id"] == inc_id


def test_schema_v2_refuses_v1_database(tmp_path: Any) -> None:
    url = f"sqlite:///{(tmp_path / 'v1.sqlite3').as_posix()}"
    db = Database(url)
    db.ensure_ready()
    with db.engine.begin() as conn:
        conn.execute(text("UPDATE schema_meta SET value = '1'"))
    from factorymind.db.session import SchemaMismatchError

    with pytest.raises(SchemaMismatchError):
        Database(url).ensure_ready()


def test_unaligned_boundaries_origin_purity_and_dedup(tmp_path: Path) -> None:
    service_clean, base, _ = _make_wp_a_service(tmp_path / "clean")

    # 1. Inject whole-feed stale window in baseline BEFORE unaligned 06:05..06:35 event,
    #    plus Oil_temperature single-sensor freeze spanning onset & evolution.
    mod = base.copy()
    stale_before = (mod.index >= pd.Timestamp("2020-01-04 03:00")) & (
        mod.index < pd.Timestamp("2020-01-04 03:20")
    )
    mod.loc[stale_before, ["TP3", "TP2", "Motor_current"]] = [8.2, 8.1, 4.0]
    ev_span = (mod.index >= pd.Timestamp("2020-01-04 06:05")) & (
        mod.index < pd.Timestamp("2020-01-04 06:35")
    )
    mod.loc[ev_span, "Oil_temperature"] = 61.5
    proc = tmp_path / "proc"
    (proc / metropt3.DATASET_ID).mkdir(parents=True)
    mod.to_parquet(proc / metropt3.DATASET_ID / "telemetry.parquet")
    svc = AnomalyService(
        service_clean.bundle,
        service_clean.card,
        proc,
        service_clean.reports,
        warning_bundle=service_clean.warning_bundle,
    )

    pkg = build_event_evidence_package(
        svc,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-04 06:05"),
        event_end=pd.Timestamp("2020-01-04 06:35"),
        baseline_hours=4.0,
        recovery_hours=2.0,
    )
    res = evaluate_package("unaligned", pkg, kind="synthetic_stress")
    assert res.boundary_aligned
    assert res.origins_pure
    assert res.missing_data_honest

    # Unaligned onset [06:05..06:15) should still observe anchored derived features cleanly
    onset = next(p for p in pkg.phases if p.phase == "onset")
    assert onset.available
    assert onset.valid_windows == 1
    assert any(d.name == "load_duty" and d.observable for d in onset.derived_features)

    # Whole-feed stale neighbor window in baseline uses derived origin, and single-sensor freeze
    # across onset/evolution emits exactly ONE deduplicated ContradictionRecord for Oil_temperature.
    oil_freezes = [
        c
        for c in pkg.quality.contradictions
        if c.kind == "sensor_freeze" and c.signals == ["Oil_temperature"]
    ]
    assert len(oil_freezes) == 1

    # 2. Zero baseline/recovery durations yield accurate reason messages (not "single-window")
    pkg_zero = build_event_evidence_package(
        svc,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-06 06:00"),
        event_end=pd.Timestamp("2020-01-06 09:00"),
        baseline_hours=0.0,
        recovery_hours=0.0,
    )
    base_zero = next(p for p in pkg_zero.phases if p.phase == "pre_event_baseline")
    assert not base_zero.available
    assert "Single-window" not in (base_zero.reason_unavailable or "")

    # 3. Alarm episodes outside [event_start, onset_end) receive their true chronological phase
    ep_pkg = build_event_evidence_package(
        svc,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-06 07:00"),
        event_end=pd.Timestamp("2020-01-06 09:00"),
        baseline_hours=4.0,
        recovery_hours=2.0,
    )
    ep_entries = [t for t in ep_pkg.timeline if t.category == "alarm_episode"]
    if ep_entries:
        assert all(t.phase != "onset" or t.timestamp >= ep_pkg.event_start for t in ep_entries)
