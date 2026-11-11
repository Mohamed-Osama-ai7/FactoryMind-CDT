"""Tests for WP-D (M4-WP0): Physics-Informed Compressor & Reservoir Digital Twin."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorymind.contracts.digital_twin import TwinSimulationRequest
from factorymind.twin.physics import (
    DEFAULT_TWIN_PARAMS,
    CompressorReservoirTwin,
)
from tests.auth_helpers import ASSET, AuthKit, make_client


def _synthetic_apu_telemetry() -> pd.DataFrame:
    """Create synthetic pre-eval and eval APU telemetry honoring physical charging/decay laws."""
    idx_pre = pd.date_range("2020-02-15 00:00:00", periods=400, freq="10s")
    idx_eval = pd.date_range("2020-04-10 00:00:00", periods=400, freq="10s")
    idx = idx_pre.append(idx_eval)

    n = len(idx)
    tp3 = np.empty(n, dtype="float64")
    loaded = np.zeros(n, dtype="float64")
    p = 9.0
    is_load = False
    for i in range(n):
        if p <= 8.1:
            is_load = True
        elif p >= 10.1:
            is_load = False
        loaded[i] = 1.0 if is_load else 0.0
        tp3[i] = p
        # Charge net +0.86 bar/min when loaded; drain -0.085 bar/min when unloaded (dt = 1/6 min)
        p += (0.86 if is_load else -0.085) * (1.0 / 6.0)

    return pd.DataFrame(
        {
            "TP2": np.where(loaded > 0.5, tp3 - 0.1, 0.0),
            "TP3": tp3,
            "H1": np.where(loaded > 0.5, 0.0, tp3),
            "DV_pressure": np.zeros(n),
            "Reservoirs": tp3,
            "Oil_temperature": 60.0 + 0.01 * np.arange(n),
            "Motor_current": np.where(loaded > 0.5, 7.0, 4.0),
            "COMP": np.where(loaded > 0.5, 0.0, 1.0),
            "DV_eletric": loaded,
            "Towers": np.zeros(n),
            "MPG": (tp3 < 8.2).astype(float),
            "LPS": (tp3 < 7.0).astype(float),
            "Pressure_switch": np.zeros(n),
            "Oil_level": np.zeros(n),
            "Caudal_impulses": np.ones(n),
        },
        index=idx,
    )


def test_fit_strictly_pre_evaluation_zero_leakage() -> None:
    df = _synthetic_apu_telemetry()
    twin_a = CompressorReservoirTwin.fit_from_telemetry(df)

    # Corrupt all evaluation telemetry heavily — calibration parameters MUST remain identical
    df_mutated = df.copy()
    ev_mask = df_mutated.index >= pd.Timestamp("2020-04-01")
    df_mutated.loc[ev_mask, "TP3"] = 2.0
    df_mutated.loc[ev_mask, "Oil_temperature"] = 110.0
    twin_b = CompressorReservoirTwin.fit_from_telemetry(df_mutated)

    assert twin_a.params == twin_b.params
    assert twin_a.validation.heldout_normal_segments >= 1
    assert twin_a.validation.mae_60min_bar < 0.25
    assert twin_a.validation.constraint_violations == 0


def test_four_counterfactual_scenarios_and_physical_invariants() -> None:
    twin = CompressorReservoirTwin(params=DEFAULT_TWIN_PARAMS)
    overview = twin.model_overview()
    assert len(overview.unidentifiable_parameters) >= 4
    assert overview.validation.constraint_violations == 0
    assert overview.validation.step_halving_max_diff_bar < 0.5

    base = twin.simulate(TwinSimulationRequest(scenario="baseline", horizon_minutes=120.0))
    leak = twin.simulate(
        TwinSimulationRequest(
            scenario="leak",
            horizon_minutes=120.0,
            leak_rate_bar_per_min=0.95,
        )
    )
    cont = twin.simulate(
        TwinSimulationRequest(
            scenario="continued_operation",
            horizon_minutes=120.0,
            leak_rate_bar_per_min=0.95,
        )
    )
    repair = twin.simulate(
        TwinSimulationRequest(
            scenario="maintenance_repair",
            horizon_minutes=120.0,
            leak_rate_bar_per_min=0.95,
            repair_at_minute=30.0,
        )
    )

    assert base.provenance_origin == "simulated_projection"
    assert all(pt.provenance_origin == "simulated_projection" for pt in base.trajectory)
    assert base.summary.lps_excursion_minutes == 0.0
    assert base.summary.mean_load_duty < 0.25
    assert leak.summary.mean_load_duty > base.summary.mean_load_duty + 0.5
    assert cont.summary.min_tp3_bar <= leak.summary.min_tp3_bar
    assert repair.summary.mean_load_duty < cont.summary.mean_load_duty
    assert repair.trajectory[-1].tp3_bar >= 8.0


def test_twin_api_auth_isolation_and_validation(kit: AuthKit) -> None:
    client = make_client(kit)
    viewer = kit.headers(roles=["viewer"], assets=[ASSET])

    model_resp = client.get("/api/v1/twin/model", headers=viewer)
    assert model_resp.status_code == 200
    m_body = model_resp.json()
    assert m_body["provenance_origin"] == "advisory_physics_twin"
    assert len(m_body["unidentifiable_parameters"]) >= 4

    # Non-APU asset cleanly rejected with 404
    assert (
        client.get(
            "/api/v1/twin/model",
            params={"asset_id": "ai4i-milling-bench"},
            headers=viewer,
        ).status_code
        == 404
    )

    sim_resp = client.post(
        "/api/v1/twin/simulate",
        json={"scenario": "leak", "horizon_minutes": 60.0, "leak_rate_bar_per_min": 0.45},
        headers=viewer,
    )
    assert sim_resp.status_code == 200
    s_body = sim_resp.json()
    assert s_body["provenance_origin"] == "simulated_projection"
    assert len(s_body["trajectory"]) >= 60
    assert s_body["summary"]["mean_load_duty"] > 0.3

    # Bounds checked strictly
    assert (
        client.post(
            "/api/v1/twin/simulate",
            json={"scenario": "baseline", "horizon_minutes": 1000.0},
            headers=viewer,
        ).status_code
        == 422
    )

    # Auditor lacks asset:read -> 403
    assert (
        client.get("/api/v1/twin/model", headers=kit.headers(roles=["auditor"])).status_code == 403
    )
