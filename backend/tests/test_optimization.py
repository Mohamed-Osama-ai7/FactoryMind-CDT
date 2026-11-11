"""Tests for WP-E (M4-WP1): Impact estimation and constrained maintenance optimization."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from factorymind.contracts.optimization import (
    EconomicAssumptions,
    MaintenanceDecisionRequest,
    OptimizationConstraints,
    OptimizationWeights,
)
from factorymind.db.repository import IncidentRepo, NewEvidence
from factorymind.decision.optimizer import compare_maintenance_options
from factorymind.twin import CompressorReservoirTwin, TwinCalibrationError
from tests.auth_helpers import ASSET, AuthKit, make_client


@lru_cache(maxsize=1)
def _calibrated_twin() -> CompressorReservoirTwin:
    idx = pd.date_range("2020-02-15", periods=500, freq="10s").append(
        pd.date_range("2020-04-10", periods=420, freq="10s")
    )
    n = len(idx)
    tp3 = np.empty(n, dtype="float64")
    oil = np.empty(n, dtype="float64")
    loaded = np.zeros(n, dtype="float64")
    p, t_oil, on = 9.0, 60.0, False
    dt = 1.0 / 6.0
    for i in range(n):
        if p <= 8.1:
            on = True
        elif p >= 10.1:
            on = False
        loaded[i] = 1.0 if on else 0.0
        tp3[i], oil[i] = p, t_oil
        p += (0.86 if on else -0.085) * dt
        t_oil += (2.1 * loaded[i] - 0.05 * (t_oil - 55.0)) * dt
    df = pd.DataFrame(
        {
            "TP3": tp3,
            "Reservoirs": tp3 - 0.05,
            "Oil_temperature": oil,
            "Motor_current": np.where(loaded > 0.5, 7.0, 4.0),
            "DV_eletric": loaded,
        },
        index=idx,
    )
    return CompressorReservoirTwin.fit_from_telemetry(df)


def test_impact_stratifies_tiers_and_respects_safety_and_diagnostic_constraints() -> None:
    twin = _calibrated_twin()

    with pytest.raises(TwinCalibrationError, match="Calibrated CompressorReservoirTwin"):
        compare_maintenance_options(MaintenanceDecisionRequest(), twin=None)

    # Case 1: Active high leak rate under air_leakage hypothesis -> monitor violates LPS safety
    leak_res = compare_maintenance_options(
        MaintenanceDecisionRequest(
            measured_load_duty=0.82,
            measured_motor_current_a=6.8,
            measured_oil_temp_c=74.0,
            dominant_hypothesis="air_leakage",
            assumptions=EconomicAssumptions(leak_rate_bar_per_min=0.95),
        ),
        twin=twin,
    )
    tiers = {q.tier for q in leak_res.impact_quantities}
    assert tiers == {
        "measured",
        "engineering_estimate",
        "configurable_economic_assumption",
        "simulated_projection",
    }
    by_opt = {o.option: o for o in leak_res.options}
    assert set(by_opt) == {"repair_now", "schedule", "monitor"}
    assert by_opt["monitor"].feasible is False
    assert by_opt["monitor"].pareto_dominated is True  # dominated by feasible repair_now
    assert len(by_opt["monitor"].constraint_violations) >= 1
    assert by_opt["repair_now"].feasible is True
    assert by_opt["repair_now"].pareto_dominated is False
    assert leak_res.recommended_option == "repair_now"

    # Case 2: max_allowable_duty and peak oil temp apply to ALL options (not just monitor)
    strict_res = compare_maintenance_options(
        MaintenanceDecisionRequest(
            assumptions=EconomicAssumptions(leak_rate_bar_per_min=0.70),
            constraints=OptimizationConstraints(
                max_allowable_duty=0.60,
                max_allowable_oil_temp_c=85.0,
            ),
        ),
        twin=twin,
    )
    s_by_opt = {o.option: o for o in strict_res.options}
    assert s_by_opt["schedule"].feasible is False
    assert any(
        "max_allowable_duty" in v or "Peak oil temp" in v
        for v in s_by_opt["schedule"].constraint_violations
    )

    # Case 3: Pareto tie-breaking at reliability-only weights picks non-dominated cheaper option
    tie_res = compare_maintenance_options(
        MaintenanceDecisionRequest(
            assumptions=EconomicAssumptions(leak_rate_bar_per_min=0.10),
            weights=OptimizationWeights(
                cost_weight=0.0, reliability_weight=1.0, downtime_weight=0.0
            ),
        ),
        twin=twin,
    )
    assert tie_res.recommended_option == "monitor"
    t_by_opt = {o.option: o for o in tie_res.options}
    assert t_by_opt["repair_now"].pareto_dominated is True
    assert t_by_opt["monitor"].pareto_dominated is False

    # Case 4: 72h horizon continues repaired options at nominal baseline (no LPS scaling)
    long_res = compare_maintenance_options(
        MaintenanceDecisionRequest(
            assumptions=EconomicAssumptions(leak_rate_bar_per_min=0.95, planning_horizon_hours=72.0)
        ),
        twin=twin,
    )
    long_repair = next(o for o in long_res.options if o.option == "repair_now")
    assert long_repair.projected_lps_risk_minutes == 0.0
    assert any("exceeds" in w or "validated only up to" in w for w in long_res.uncertainty_warnings)

    # Case 5: sensor_malfunction vs abnormal_high_demand
    sensor_res = compare_maintenance_options(
        MaintenanceDecisionRequest(
            measured_load_duty=0.20,
            measured_motor_current_a=4.2,
            measured_oil_temp_c=61.0,
            dominant_hypothesis="sensor_malfunction",
            assumptions=EconomicAssumptions(leak_rate_bar_per_min=0.35),
        ),
        twin=twin,
    )
    assert sensor_res.recommended_option == "monitor"
    assert any("verify instrumentation/demand" in w for w in sensor_res.uncertainty_warnings)

    demand_res = compare_maintenance_options(
        MaintenanceDecisionRequest(
            measured_load_duty=0.80,
            dominant_hypothesis="abnormal_high_demand",
            assumptions=EconomicAssumptions(leak_rate_bar_per_min=0.20),
        ),
        twin=twin,
    )
    assert any("verify instrumentation/demand" in w for w in demand_res.uncertainty_warnings)

    # Case 6: Configurable tariff assumptions directly change expected energy cost
    low_tariff = compare_maintenance_options(
        MaintenanceDecisionRequest(assumptions=EconomicAssumptions(electricity_eur_per_kwh=0.05)),
        twin=twin,
    )
    high_tariff = compare_maintenance_options(
        MaintenanceDecisionRequest(assumptions=EconomicAssumptions(electricity_eur_per_kwh=1.50)),
        twin=twin,
    )
    low_c = next(o.expected_total_cost_eur for o in low_tariff.options if o.option == "schedule")
    high_c = next(o.expected_total_cost_eur for o in high_tariff.options if o.option == "schedule")
    assert high_c > low_c


def test_optimization_api_nan_safety_and_incident_provenance(kit: AuthKit) -> None:
    client = make_client(kit)
    viewer = kit.headers(roles=["viewer"], assets=[ASSET])

    # Unconfigured twin telemetry surfaces 503 explicitly instead of silent fallback
    assert (
        client.post(
            "/api/v1/optimization/compare",
            json={"asset_id": ASSET, "dominant_hypothesis": "air_leakage"},
            headers=viewer,
        ).status_code
        == 503
    )

    client.app.state.digital_twin = _calibrated_twin()  # type: ignore[union-attr]

    # Raw JSON NaN / Infinity returns clean 422 instead of 500
    for bad_json in ['{"measured_load_duty": NaN}', '{"measured_load_duty": Infinity}']:
        bad_resp = client.post(
            "/api/v1/optimization/compare",
            content=bad_json,
            headers={**viewer, "Content-Type": "application/json"},
        )
        assert bad_resp.status_code == 422

    # When measured fields are omitted in POST, they MUST NOT be labeled tier='measured'
    omitted_resp = client.post(
        "/api/v1/optimization/compare",
        json={"asset_id": ASSET, "dominant_hypothesis": "air_leakage"},
        headers=viewer,
    )
    assert omitted_resp.status_code == 200
    assert all(q["tier"] != "measured" for q in omitted_resp.json()["impact_quantities"])

    # Non-APU asset cleanly 404
    assert (
        client.post(
            "/api/v1/optimization/compare",
            json={"asset_id": "ai4i-milling-bench"},
            headers=viewer,
        ).status_code
        == 404
    )

    # Incident with group-representative peak evidence (e.g. DV_eletric_mean instead of load_duty)
    # must NEVER fabricate measured_load_duty=0.72 under tier='measured'
    with client.app.state.db.session() as session:  # type: ignore[union-attr]
        from factorymind.auth.principal import Principal, Role

        creator = Principal(
            actor_id="eng-seed",
            tenant_id="acme",
            roles=frozenset({Role.ENGINEER}),
            asset_ids=frozenset({ASSET}),
        )
        inc = IncidentRepo(session, creator).create(
            asset_id=ASSET,
            detector="duty_rule",
            model_version="m2-wp2",
            episode_start=datetime(2020, 4, 18, 10, 0),
            episode_end=datetime(2020, 4, 18, 12, 0),
            raised_at=datetime(2020, 4, 18, 10, 30),
            peak_score=0.85,
            evidence=[
                NewEvidence(
                    group_name="operating_state",
                    group_label="Operating state",
                    feature="DV_eletric_mean",
                    value=0.3833,
                    training_median=0.14,
                    robust_z=5.2,
                    novel=True,
                    meaning="High outlet valve active duty",
                ),
            ],
        )
        inc_id = inc.id

    inc_comp = client.get(
        f"/api/v1/incidents/{inc_id}/decision-comparison",
        params={"leak_rate_bar_per_min": 0.9, "dominant_hypothesis": "air_leakage"},
        headers=viewer,
    )
    assert inc_comp.status_code == 200
    ic_body = inc_comp.json()
    assert ic_body["incident_id"] == inc_id
    measured_keys = {q["key"] for q in ic_body["impact_quantities"] if q["tier"] == "measured"}
    assert "measured_motor_current_a" not in measured_keys
    assert "measured_load_duty" in measured_keys
    val_q = next(q for q in ic_body["impact_quantities"] if q["key"] == "measured_load_duty")
    assert val_q["value"] == 0.3833

    # Cross-tenant isolation -> 404
    assert (
        client.get(
            f"/api/v1/incidents/{inc_id}/decision-comparison",
            headers=kit.headers(tenant="beta", roles=["auditor"]),
        ).status_code
        == 404
    )
