"""Tests for WP-B: Explainable RCA engine, Casebook evaluation, abstention, and RCA API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from factorymind.anomaly.evidence_package import build_event_evidence_package
from factorymind.anomaly.rca import analyze_event_evidence
from factorymind.anomaly.rca_eval import (
    evaluate_rca_case,
    evaluate_service_report_rca,
    evaluate_unlabeled_warning_slice,
)
from factorymind.anomaly.service import AnomalyService
from factorymind.data import metropt3
from tests.auth_helpers import ASSET, AuthKit, make_client
from tests.test_evidence_foundation import _make_wp_a_service


def test_rca_clean_leak_vs_concurrent_and_grounding(tmp_path: Path) -> None:
    svc, _, _ = _make_wp_a_service(tmp_path)

    # F1 synthetic event has clean leak around Jan 6 06:00-09:00
    pkg_clean = build_event_evidence_package(
        svc,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-06 06:00"),
        event_end=pd.Timestamp("2020-01-06 09:00"),
        raised_at=pd.Timestamp("2020-01-06 06:20"),
    )
    res_clean, rca_clean = evaluate_rca_case(
        "clean_leak",
        pkg_clean,
        kind="synthetic_clean",
        expected_categories=["air_leakage"],
        expect_abstain=False,
    )
    assert res_clean.citations_grounded
    assert res_clean.correlation_disclosed
    assert res_clean.matches_expectation
    assert rca_clean.primary_category == "air_leakage"
    assert not rca_clean.abstained
    assert any(h.category == "abnormal_high_demand" for h in rca_clean.ranked_hypotheses)
    assert all(
        "never actuate" in step.safety_note.lower()
        for hyp in rca_clean.ranked_hypotheses
        for step in hyp.verification_steps
    )

    # F2 in _make_wp_a_service has high motor current + COMP=0 + DV_eletric=1
    suite = evaluate_service_report_rca(svc, asset_id=ASSET)
    assert suite["cases"] == 2
    assert suite["evidence_citation_grounding_rate"] == 1.0
    assert suite["correlation_disclaimer_rate"] == 1.0
    assert suite["expectation_agreement_rate"] == 1.0


def test_rca_separates_sensor_freeze_valve_inconsistency_and_abstains_on_gaps(
    tmp_path: Path,
) -> None:
    svc_clean, base, _ = _make_wp_a_service(tmp_path / "clean")

    # 1. Single-sensor TP3 freeze + valve contradiction without air leak duty elevation
    mod = base.copy()
    mask = (mod.index >= pd.Timestamp("2020-01-04 06:00")) & (
        mod.index < pd.Timestamp("2020-01-04 07:00")
    )
    mod.loc[mask, "TP3"] = 8.9
    mod.loc[mask, "COMP"] = 1.0
    mod.loc[mask, "DV_eletric"] = 1.0
    proc = tmp_path / "fault"
    (proc / metropt3.DATASET_ID).mkdir(parents=True)
    mod.to_parquet(proc / metropt3.DATASET_ID / "telemetry.parquet")
    svc_fault = AnomalyService(
        svc_clean.bundle,
        svc_clean.card,
        proc,
        svc_clean.reports,
        warning_bundle=svc_clean.warning_bundle,
    )
    pkg_fault = build_event_evidence_package(
        svc_fault,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-04 06:00"),
        event_end=pd.Timestamp("2020-01-04 07:00"),
    )
    res_fault, rca_fault = evaluate_rca_case(
        "fault_valve_tp3",
        pkg_fault,
        kind="synthetic_stress",
        expected_categories=["telemetry_inconsistency", "sensor_malfunction"],
        expect_abstain=False,
    )
    assert res_fault.citations_grounded
    assert res_fault.correlation_disclosed
    assert res_fault.matches_expectation
    assert rca_fault.primary_category in ("telemetry_inconsistency", "sensor_malfunction")
    leak_rank = next(h for h in rca_fault.ranked_hypotheses if h.category == "air_leakage")
    assert leak_rank.rank > 1
    assert leak_rank.relative_support in ("weak", "ruled_out")

    # 2. Missing baseline & gapped event (<50% coverage) -> mandatory abstention
    gapped = base.copy()
    gapped = gapped[gapped.index >= pd.Timestamp("2020-01-06 06:00")]
    gapped = gapped[
        ~(
            (gapped.index >= pd.Timestamp("2020-01-06 06:20"))
            & (gapped.index < pd.Timestamp("2020-01-06 08:20"))
        )
    ]
    gapped = gapped[gapped.index < pd.Timestamp("2020-01-06 09:00")]
    proc_gap = tmp_path / "gap"
    (proc_gap / metropt3.DATASET_ID).mkdir(parents=True)
    gapped.to_parquet(proc_gap / metropt3.DATASET_ID / "telemetry.parquet")
    svc_gap = AnomalyService(
        svc_clean.bundle,
        svc_clean.card,
        proc_gap,
        svc_clean.reports,
        warning_bundle=svc_clean.warning_bundle,
    )
    pkg_gap = build_event_evidence_package(
        svc_gap,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-06 06:00"),
        event_end=pd.Timestamp("2020-01-06 09:00"),
    )
    res_gap, rca_gap = evaluate_rca_case(
        "gapped_abstain",
        pkg_gap,
        kind="synthetic_stress",
        expected_categories=["insufficient_evidence"],
        expect_abstain=True,
    )
    assert res_gap.matches_expectation
    assert res_gap.citations_grounded
    assert rca_gap.abstained
    assert rca_gap.primary_category == "insufficient_evidence"
    assert all(h.confidence_band == "abstained" for h in rca_gap.ranked_hypotheses)

    # 3. Unlabeled warning slice helper checks grounding and non-leak separation
    w_eval = evaluate_unlabeled_warning_slice(
        svc_fault,
        pd.Timestamp("2020-01-04 00:00"),
        pd.Timestamp("2020-01-04 12:00"),
        asset_id=ASSET,
    )
    assert w_eval["cases"] >= 1
    assert w_eval["evidence_citation_grounding_rate"] == 1.0
    assert w_eval["non_leak_primary_rate"] == 1.0


def test_rca_api_rbac_tenant_isolation_and_human_observation_incorporation(
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

    # Viewer can query pre-incident RCA endpoint
    pre = client.get(
        "/api/v1/anomaly/rca",
        params={"start": ep_start, "end": ep_end},
        headers=kit.headers(roles=["viewer"]),
    )
    assert pre.status_code == 200
    pre_body = pre.json()
    assert pre_body["incident_id"] is None
    assert pre_body["primary_category"] == "air_leakage"
    assert len(pre_body["correlation_vs_causation_boundary"]) >= 1

    # Invalid tz-aware timestamps rejected with 422
    assert (
        client.get(
            "/api/v1/anomaly/rca",
            params={"start": "2020-01-09T00:00:00Z", "end": ep_end},
            headers=kit.headers(),
        ).status_code
        == 422
    )

    # Create incident, log a corroborating human field observation, verify incident RCA cites it
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

    assert (
        client.post(
            f"/api/v1/incidents/{inc_id}/observations",
            json={
                "category": "field_inspection",
                "note": "Audible air leak hiss found near dryer purge coupling.",
                "observed_at": ep_start,
            },
            headers=kit.headers(roles=["engineer"], sub="eng-1"),
        ).status_code
        == 201
    )

    # Cross-tenant caller receives 404
    assert (
        client.get(
            f"/api/v1/incidents/{inc_id}/rca",
            headers=kit.headers(tenant="beta", roles=["viewer"], sub="vera-beta"),
        ).status_code
        == 404
    )

    inc_rca = client.get(
        f"/api/v1/incidents/{inc_id}/rca",
        headers=kit.headers(roles=["auditor"], sub="aud-1"),
    )
    assert inc_rca.status_code == 200
    body = inc_rca.json()
    assert body["incident_id"] == inc_id
    leak_hyp = next(h for h in body["ranked_hypotheses"] if h["category"] == "air_leakage")
    assert any(ref["origin"] == "human_observation" for ref in leak_hyp["supporting_evidence"])

    # Verify citations remain grounded even after adding human observations
    pkg = client.get(
        f"/api/v1/incidents/{inc_id}/evidence-package",
        headers=kit.headers(roles=["viewer"]),
    ).json()
    from factorymind.anomaly.rca_eval import citations_grounded
    from factorymind.contracts.events import EventEvidencePackage
    from factorymind.contracts.rca import RcaAssessment

    assert citations_grounded(
        EventEvidencePackage.model_validate(pkg),
        RcaAssessment.model_validate(body),
    )

    # Standalone analyzer remains deterministic on direct validated package
    assert analyze_event_evidence(EventEvidencePackage.model_validate(pkg)).rca_id == body["rca_id"]


def test_rca_regression_for_frozen_motor_zero_nonstale_fake_pca_negated_obs_and_rank_order(
    tmp_path: Path,
) -> None:
    svc_clean, base, _ = _make_wp_a_service(tmp_path / "clean")

    # 1. Single-sensor Motor_current freeze at 7.0 A -> sensor_malfunction (not concurrent leak)
    frozen_motor = base.copy()
    m_mask = (frozen_motor.index >= pd.Timestamp("2020-01-04 06:00")) & (
        frozen_motor.index < pd.Timestamp("2020-01-04 07:00")
    )
    frozen_motor.loc[m_mask, "Motor_current"] = 7.0
    p1 = tmp_path / "frozen_motor"
    (p1 / metropt3.DATASET_ID).mkdir(parents=True)
    frozen_motor.to_parquet(p1 / metropt3.DATASET_ID / "telemetry.parquet")
    svc_m = AnomalyService(
        svc_clean.bundle, svc_clean.card, p1, svc_clean.reports, warning_bundle=None
    )
    pkg_m = build_event_evidence_package(
        svc_m,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-04 06:00"),
        event_end=pd.Timestamp("2020-01-04 07:00"),
    )
    rca_m = analyze_event_evidence(pkg_m)
    assert rca_m.primary_category == "sensor_malfunction"
    scores_m = [h.support_score for h in rca_m.ranked_hypotheses]
    assert scores_m == sorted(scores_m, reverse=True)

    # 2. Mixed 2 stale + 2 gapped windows (0 non-stale valid windows) -> abstained + monotonic rank
    mixed = base.copy()
    stale_span = (mixed.index >= pd.Timestamp("2020-01-04 06:00")) & (
        mixed.index < pd.Timestamp("2020-01-04 06:20")
    )
    mixed.loc[stale_span, ["TP3", "TP2", "Motor_current"]] = [8.5, 8.3, 7.0]
    mixed = mixed[
        ~(
            (mixed.index >= pd.Timestamp("2020-01-04 06:20"))
            & (mixed.index < pd.Timestamp("2020-01-04 06:40"))
        )
    ]
    p2 = tmp_path / "mixed_stale_gap"
    (p2 / metropt3.DATASET_ID).mkdir(parents=True)
    mixed.to_parquet(p2 / metropt3.DATASET_ID / "telemetry.parquet")
    svc_mix = AnomalyService(
        svc_clean.bundle, svc_clean.card, p2, svc_clean.reports, warning_bundle=None
    )
    pkg_mix = build_event_evidence_package(
        svc_mix,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-04 06:00"),
        event_end=pd.Timestamp("2020-01-04 06:40"),
    )
    rca_mix = analyze_event_evidence(pkg_mix)
    assert rca_mix.abstained
    assert rca_mix.primary_category == "insufficient_evidence"
    scores_mix = [h.support_score for h in rca_mix.ranked_hypotheses]
    assert scores_mix == sorted(scores_mix, reverse=True)

    # 3. warning_bundle=None + valve_contradiction must never fabricate pca_spe citation
    valve_only = base.copy()
    v_mask = (valve_only.index >= pd.Timestamp("2020-01-04 08:00")) & (
        valve_only.index < pd.Timestamp("2020-01-04 08:30")
    )
    valve_only.loc[v_mask, ["COMP", "DV_eletric"]] = 1.0
    p3 = tmp_path / "valve_no_warn"
    (p3 / metropt3.DATASET_ID).mkdir(parents=True)
    valve_only.to_parquet(p3 / metropt3.DATASET_ID / "telemetry.parquet")
    svc_v = AnomalyService(
        svc_clean.bundle, svc_clean.card, p3, svc_clean.reports, warning_bundle=None
    )
    pkg_v = build_event_evidence_package(
        svc_v,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-04 08:00"),
        event_end=pd.Timestamp("2020-01-04 08:30"),
    )
    from factorymind.anomaly.rca_eval import citations_grounded

    rca_v = analyze_event_evidence(pkg_v)
    assert citations_grounded(pkg_v, rca_v)
    assert not any(
        ref.ref_id == "sensor:pca_spe"
        for hyp in rca_v.ranked_hypotheses
        for ref in hyp.supporting_evidence
    )

    # 4. Normal nominal telemetry -> abstained cleanly (no 0.0 primary fault promotion)
    pkg_norm = build_event_evidence_package(
        svc_clean,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-03 06:00"),
        event_end=pd.Timestamp("2020-01-03 09:00"),
    )
    rca_norm = analyze_event_evidence(pkg_norm)
    assert rca_norm.abstained
    assert rca_norm.primary_category == "insufficient_evidence"

    # 5. Negated human observation must not boost leak or sensor fault
    from factorymind.contracts.events import HumanObservationOut

    obs_neg = HumanObservationOut(
        id="obsneg",
        incident_id="a" * 32,
        asset_id=ASSET,
        actor_id="eng-1",
        category="field_inspection",
        note="Checked gauge: no leak found, hose intact.",
        created_at=pd.Timestamp("2020-01-06 07:00").to_pydatetime(),
    )
    pkg_leak = build_event_evidence_package(
        svc_clean,
        asset_id=ASSET,
        event_start=pd.Timestamp("2020-01-06 06:00"),
        event_end=pd.Timestamp("2020-01-06 09:00"),
        human_observations=[obs_neg],
    )
    rca_neg = analyze_event_evidence(pkg_leak)
    assert rca_neg.primary_category == "air_leakage"
    assert not any(
        ref.origin == "human_observation"
        for hyp in rca_neg.ranked_hypotheses
        for ref in (*hyp.supporting_evidence, *hyp.contradicting_evidence)
    )
