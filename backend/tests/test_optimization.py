"""Tests for WP-E (M4-WP1): Impact estimation and constrained maintenance optimization."""

from __future__ import annotations

from datetime import datetime

from factorymind.contracts.optimization import EconomicAssumptions, MaintenanceDecisionRequest
from factorymind.db.repository import IncidentRepo, NewEvidence
from factorymind.decision.optimizer import compare_maintenance_options
from tests.auth_helpers import ASSET, AuthKit, make_client


def test_impact_stratifies_tiers_and_respects_safety_and_diagnostic_constraints() -> None:
    # Case 1: Active high leak rate under air_leakage hypothesis -> monitor violates LPS safety
    leak_res = compare_maintenance_options(
        MaintenanceDecisionRequest(
            measured_load_duty=0.82,
            measured_motor_current_a=6.8,
            measured_oil_temp_c=74.0,
            dominant_hypothesis="air_leakage",
            assumptions=EconomicAssumptions(leak_rate_bar_per_min=0.95),
        )
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
    assert len(by_opt["monitor"].constraint_violations) >= 1
    assert by_opt["repair_now"].feasible is True
    assert leak_res.recommended_option in {"repair_now", "schedule"}

    # Case 2: Dominant hypothesis is sensor_malfunction -> monitor avoids unnecessary depot pull-in
    sensor_res = compare_maintenance_options(
        MaintenanceDecisionRequest(
            measured_load_duty=0.20,
            measured_motor_current_a=4.2,
            measured_oil_temp_c=61.0,
            dominant_hypothesis="sensor_malfunction",
            assumptions=EconomicAssumptions(leak_rate_bar_per_min=0.35),
        )
    )
    assert sensor_res.recommended_option == "monitor"
    assert any("verify instrumentation/demand" in w for w in sensor_res.uncertainty_warnings)

    # Case 3: Configurable tariff assumptions directly change expected energy cost
    low_tariff = compare_maintenance_options(
        MaintenanceDecisionRequest(assumptions=EconomicAssumptions(electricity_eur_per_kwh=0.05))
    )
    high_tariff = compare_maintenance_options(
        MaintenanceDecisionRequest(assumptions=EconomicAssumptions(electricity_eur_per_kwh=1.50))
    )
    low_c = next(o.expected_total_cost_eur for o in low_tariff.options if o.option == "schedule")
    high_c = next(o.expected_total_cost_eur for o in high_tariff.options if o.option == "schedule")
    assert high_c > low_c


def test_optimization_api_and_incident_integration(kit: AuthKit) -> None:
    client = make_client(kit)
    viewer = kit.headers(roles=["viewer"], assets=[ASSET])

    resp = client.post(
        "/api/v1/optimization/compare",
        json={
            "asset_id": ASSET,
            "dominant_hypothesis": "air_leakage",
            "assumptions": {"leak_rate_bar_per_min": 0.9, "planning_horizon_hours": 12.0},
        },
        headers=viewer,
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["options"]) == 3
    assert body["provenance_origin"] == "advisory_decision_comparison"

    # Non-APU asset cleanly 404
    assert (
        client.post(
            "/api/v1/optimization/compare",
            json={"asset_id": "ai4i-milling-bench"},
            headers=viewer,
        ).status_code
        == 404
    )

    # Create a seeded incident in SQLite and fetch GET /incidents/{id}/decision-comparison
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
                    group_name="duty",
                    group_label="Duty",
                    feature="load_duty",
                    value=0.88,
                    training_median=0.14,
                    robust_z=5.2,
                    novel=True,
                    meaning="High loaded duty",
                ),
                NewEvidence(
                    group_name="motor",
                    group_label="Motor",
                    feature="Motor_current_mean",
                    value=6.7,
                    training_median=2.5,
                    robust_z=3.1,
                    novel=True,
                    meaning="High current",
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
    duty_q = next(q for q in ic_body["impact_quantities"] if q["key"] == "measured_load_duty")
    assert duty_q["value"] == 0.88
    assert duty_q["tier"] == "measured"

    # Cross-tenant isolation -> 404
    assert (
        client.get(
            f"/api/v1/incidents/{inc_id}/decision-comparison",
            headers=kit.headers(tenant="beta", roles=["auditor"]),
        ).status_code
        == 404
    )
