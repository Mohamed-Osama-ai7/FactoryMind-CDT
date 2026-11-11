"""Tests for WP-D / WP-P0C: Physics-Informed Digital Twin calibration, baselines & validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from factorymind.contracts.digital_twin import TwinSimulationRequest
from factorymind.data import metropt3
from factorymind.twin import CompressorReservoirTwin, TwinCalibrationError
from tests.auth_helpers import ASSET, AuthKit, make_client


def _synthetic_apu_telemetry(
    *,
    eval_periods: int = 420,
    eval_demand_drift: bool = False,
) -> pd.DataFrame:
    """Create physical pre-eval and eval APU telemetry honoring pressure & thermal ODEs."""
    idx_pre = pd.date_range("2020-02-15 00:00:00", periods=500, freq="10s")
    idx_eval = pd.date_range("2020-04-10 00:00:00", periods=eval_periods, freq="10s")
    idx = idx_pre.append(idx_eval)

    n = len(idx)
    tp3 = np.empty(n, dtype="float64")
    oil = np.empty(n, dtype="float64")
    loaded = np.zeros(n, dtype="float64")
    p = 9.0
    t_oil = 60.0
    is_load = False
    dt_min = 1.0 / 6.0

    for i in range(n):
        if p <= 8.1:
            is_load = True
        elif p >= 10.1:
            is_load = False
        loaded[i] = 1.0 if is_load else 0.0
        tp3[i] = p
        oil[i] = t_oil

        if eval_demand_drift and i >= len(idx_pre) + 70:
            # Large unmodeled downstream demand bursts cause closed-loop phase drift at 30m/60m
            phase = ((i - len(idx_pre)) // 18) % 3
            drain = 0.04 if phase == 0 else (0.34 if phase == 1 else 0.19)
            p += (0.94 * (1.0 if is_load else 0.0) - drain) * dt_min
        else:
            p += (0.86 if is_load else -0.085) * dt_min
        p = max(1.5, min(11.5, p))
        t_oil += (2.1 * loaded[i] - 0.05 * (t_oil - 55.0)) * dt_min

    return pd.DataFrame(
        {
            "TP2": np.where(loaded > 0.5, tp3 - 0.1, 0.0),
            "TP3": tp3,
            "H1": np.where(loaded > 0.5, 0.0, tp3),
            "DV_pressure": np.zeros(n),
            "Reservoirs": tp3 - 0.05,
            "Oil_temperature": oil,
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


def test_fit_strictly_pre_evaluation_zero_leakage_and_sensitivity() -> None:
    df = _synthetic_apu_telemetry()
    twin_a = CompressorReservoirTwin.fit_from_telemetry(df)

    # Corrupt all evaluation telemetry heavily — fit params, train baselines, and pre-eval
    # sensitivity checks MUST remain strictly invariant (zero evaluation-to-train leakage).
    df_mutated = df.copy()
    ev_mask = df_mutated.index >= pd.Timestamp("2020-04-01")
    # Mutate held-out eval TP3 and oil temperature heavily
    df_mutated.loc[ev_mask, "TP3"] = np.clip(df_mutated.loc[ev_mask, "TP3"] * 0.8 + 1.1, 1.0, 12.0)
    df_mutated.loc[ev_mask, "Oil_temperature"] = df_mutated.loc[ev_mask, "Oil_temperature"] + 15.0
    twin_b = CompressorReservoirTwin.fit_from_telemetry(df_mutated)

    assert twin_a.params == twin_b.params
    assert twin_a.validation.sensitivity_checks == twin_b.validation.sensitivity_checks
    assert twin_a.validation.heldout_normal_segments >= 1
    assert twin_a.validation.constraint_violations == 0
    assert twin_a.validation.max_supported_closed_loop_horizon_minutes == 60.0
    assert twin_a.validation.max_supported_teacher_forced_horizon_minutes == 60.0
    assert [h.horizon_minutes for h in twin_a.validation.horizon_evaluations] == [10.0, 30.0, 60.0]
    for h in twin_a.validation.horizon_evaluations:
        assert h.teacher_forced_supported is True
        assert h.closed_loop_supported is True
        assert h.teacher_forced_skill_vs_persistence > 0.0
        assert h.teacher_forced_skill_vs_train_mean > 0.0
        assert h.closed_loop_skill_vs_persistence > 0.0
        assert h.closed_loop_skill_vs_train_mean > 0.0
        assert h.observed_cycles_per_segment > 0.0
        if h.horizon_minutes == 10.0:
            assert h.duty_cycle_supported is True
            assert h.duty_cycle_mae < h.train_mean_duty_mae
    assert len(twin_a.validation.sensitivity_checks) >= 4
    assert all(
        s.evaluation_scope == "pre_evaluation_only" for s in twin_a.validation.sensitivity_checks
    )


def test_twin_fit_and_calibration_fail_loudly_without_silent_fallbacks() -> None:
    base = _synthetic_apu_telemetry()

    # 1. Non-DatetimeIndex
    with pytest.raises(TwinCalibrationError, match="DatetimeIndex"):
        CompressorReservoirTwin.fit_from_telemetry(base.reset_index(drop=True))

    # 2. Missing required column
    with pytest.raises(TwinCalibrationError, match="missing required columns"):
        CompressorReservoirTwin.fit_from_telemetry(base.drop(columns=["Oil_temperature"]))

    # 3. Insufficient pre-eval rows
    too_short = base[base.index >= pd.Timestamp("2020-02-15 01:00:00")].iloc[:100]
    with pytest.raises(TwinCalibrationError, match="Insufficient finite"):
        CompressorReservoirTwin.fit_from_telemetry(too_short)

    # 4. Unloaded telemetry fails decay check (no silent 1e-4 clamp)
    no_decay = base.copy()
    pre_mask = no_decay.index < pd.Timestamp("2020-04-01")
    unloaded_pre = pre_mask & (no_decay["DV_eletric"] < 0.5)
    no_decay.loc[unloaded_pre, "TP3"] = 9.0
    with pytest.raises(TwinCalibrationError, match="physical decay check"):
        CompressorReservoirTwin.fit_from_telemetry(no_decay)

    # 5. Missing governor transitions (no silent 8.09 / 10.078 fallback)
    no_trans = base.copy()
    no_trans.loc[pre_mask, "DV_eletric"] = 1.0
    no_trans.loc[pre_mask, "Motor_current"] = 7.0
    with pytest.raises(TwinCalibrationError, match=r"steady loaded/unloaded|transitions"):
        CompressorReservoirTwin.fit_from_telemetry(no_trans)

    # 6. Degenerate thermal profile: flat Oil_temperature (no silent 2.30353 / 0.01734 fallback)
    flat_oil = base.copy()
    flat_oil.loc[pre_mask, "Oil_temperature"] = 62.0
    with pytest.raises(TwinCalibrationError, match="Thermal OLS"):
        CompressorReservoirTwin.fit_from_telemetry(flat_oil)

    # 7. Zero held-out contiguous >=60m segments (no silent 0.0 bar MAE)
    short_eval = _synthetic_apu_telemetry(eval_periods=100)
    with pytest.raises(TwinCalibrationError, match="Zero contiguous >=60m held-out"):
        CompressorReservoirTwin.fit_from_telemetry(short_eval)

    # 8. Non-physical degenerate hysteresis (cut_out <= cut_in + 0.2)
    inverted = base.copy()
    # Set all transitions to identical TP3=9.0 -> cut_in == cut_out == 9.0
    ld = inverted["DV_eletric"].to_numpy() >= 0.5
    trans = np.zeros(len(inverted), dtype=bool)
    trans[1:] = ld[1:] != ld[:-1]
    inverted.loc[pre_mask & trans, "TP3"] = 9.0
    # Shift adjacent point to transition too so np.where(trans)[0] reads 9.0
    trans_prev = np.zeros(len(inverted), dtype=bool)
    trans_prev[:-1] = ld[1:] != ld[:-1]
    inverted.loc[pre_mask & trans_prev, "TP3"] = 9.0
    with pytest.raises(TwinCalibrationError, match=r"hysteresis invalid|physical"):
        CompressorReservoirTwin.fit_from_telemetry(inverted)


def test_unsupported_horizon_explicitly_reported_against_baselines() -> None:
    drifting = _synthetic_apu_telemetry(eval_periods=450, eval_demand_drift=True)
    twin = CompressorReservoirTwin.fit_from_telemetry(drifting)

    by_h = {h.horizon_minutes: h for h in twin.validation.horizon_evaluations}
    assert by_h[10.0].closed_loop_supported is True
    # Demand-driven phase drift causes 30m/60m closed-loop TP3 to underperform Train-Mean
    assert by_h[60.0].closed_loop_supported is False
    assert "UNSUPPORTED" in by_h[60.0].support_note
    assert any("60m" in label for label in twin.validation.unsupported_horizons)
    assert twin.validation.max_supported_closed_loop_horizon_minutes < 60.0

    ok_sim = twin.simulate(TwinSimulationRequest(scenario="baseline", horizon_minutes=10.0))
    assert ok_sim.horizon_supported is True
    assert ok_sim.horizon_support_status == "supported_closed_loop"
    assert ok_sim.horizon_warning is None

    unsup_sim = twin.simulate(TwinSimulationRequest(scenario="leak", horizon_minutes=60.0))
    assert unsup_sim.horizon_supported is False
    assert unsup_sim.horizon_support_status == "unsupported_closed_loop_horizon"
    assert unsup_sim.horizon_warning is not None
    assert "UNSUPPORTED FORECAST HORIZON" in unsup_sim.summary.advisory_interpretation


def test_four_counterfactual_scenarios_and_physical_invariants() -> None:
    df = _synthetic_apu_telemetry()
    twin = CompressorReservoirTwin.fit_from_telemetry(df)
    overview = twin.model_overview()
    assert len(overview.unidentifiable_parameters) >= 4
    assert overview.validation.constraint_violations == 0
    assert overview.validation.step_halving_max_diff_bar < 1.0

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
    assert leak.summary.lps_excursion_minutes <= 120.0
    assert cont.summary.min_tp3_bar <= leak.summary.min_tp3_bar
    assert repair.summary.mean_load_duty < cont.summary.mean_load_duty
    assert repair.trajectory[-1].tp3_bar >= 8.0
    assert base.horizon_supported is False  # 120m > 60m validated horizon on this dataset

    # Non-finite rows in calibration telemetry are excluded cleanly without NaN propagation
    df_nan = _synthetic_apu_telemetry()
    df_nan.iloc[5:10, 0:3] = float("nan")
    nan_fit = CompressorReservoirTwin.fit_from_telemetry(df_nan)
    assert all(np.isfinite(p.value) for p in nan_fit.params.to_contracts())
    reval = nan_fit.validate_held_out(df_nan)
    assert reval.calibration_notes == nan_fit.validation.calibration_notes


def test_twin_api_fail_closed_auth_isolation_and_validation(kit: AuthKit) -> None:
    viewer = kit.headers(roles=["viewer"], assets=[ASSET])

    # 1. Missing telemetry.parquet MUST surface 503 explicitly (zero hardcoded fallback!)
    unconfigured = make_client(kit)
    assert unconfigured.get("/api/v1/twin/model", headers=viewer).status_code == 503
    assert (
        unconfigured.post(
            "/api/v1/twin/simulate", json={"scenario": "baseline"}, headers=viewer
        ).status_code
        == 503
    )

    # 2. Invalid/uncalibrated telemetry.parquet MUST ALSO surface 503 explicitly!
    bad_dir = kit.root / "bad_data" / "processed" / metropt3.DATASET_ID
    bad_dir.mkdir(parents=True)
    bad_df = _synthetic_apu_telemetry()
    bad_df.loc[bad_df.index < pd.Timestamp("2020-04-01"), "Oil_temperature"] = 60.0
    bad_df.to_parquet(bad_dir / "telemetry.parquet")
    bad_client = make_client(kit, data_dir=kit.root / "bad_data")
    bad_model = bad_client.get("/api/v1/twin/model", headers=viewer)
    assert bad_model.status_code == 503
    assert "calibration failed" in bad_model.json()["detail"].lower()

    # 2b. Truncated/corrupted parquet file MUST ALSO return 503 instead of HTTP 500
    corrupt_dir = kit.root / "corrupt_data" / "processed" / metropt3.DATASET_ID
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "telemetry.parquet").write_bytes(b"not-a-valid-parquet-stream")
    corrupt_client = make_client(kit, data_dir=kit.root / "corrupt_data")
    corrupt_model = corrupt_client.get("/api/v1/twin/model", headers=viewer)
    assert corrupt_model.status_code == 503
    assert "calibration failed" in corrupt_model.json()["detail"].lower()

    # 3. Valid calibrated telemetry.parquet succeeds with live fit_source & horizon validation
    good_dir = kit.root / "data" / "processed" / metropt3.DATASET_ID
    good_dir.mkdir(parents=True)
    _synthetic_apu_telemetry().to_parquet(good_dir / "telemetry.parquet")
    client = make_client(kit)

    model_resp = client.get("/api/v1/twin/model", headers=viewer)
    assert model_resp.status_code == 200
    m_body = model_resp.json()
    assert m_body["provenance_origin"] == "advisory_physics_twin"
    assert all(p["fit_source"] == "live_pre_eval_fit" for p in m_body["parameters"])
    assert len(m_body["unidentifiable_parameters"]) >= 4
    assert len(m_body["validation"]["horizon_evaluations"]) == 3
    assert len(m_body["validation"]["sensitivity_checks"]) >= 4

    # Non-APU asset cleanly rejected with 404 on both GET /model and POST /simulate
    assert (
        client.get(
            "/api/v1/twin/model",
            params={"asset_id": "ai4i-milling-bench"},
            headers=viewer,
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/v1/twin/simulate",
            json={"asset_id": "ai4i-milling-bench", "scenario": "baseline"},
            headers=viewer,
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/v1/twin/simulate",
            json={"asset_id": "../etc", "scenario": "baseline"},
            headers=viewer,
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/twin/simulate",
            json={
                "scenario": "maintenance_repair",
                "horizon_minutes": 20.0,
                "repair_at_minute": 30.0,
            },
            headers=viewer,
        ).status_code
        == 422
    )

    sim_resp = client.post(
        "/api/v1/twin/simulate",
        json={"scenario": "leak", "horizon_minutes": 60.0, "leak_rate_bar_per_min": 0.45},
        headers=viewer,
    )
    assert sim_resp.status_code == 200
    s_body = sim_resp.json()
    assert s_body["provenance_origin"] == "simulated_projection"
    assert s_body["horizon_supported"] is True
    assert s_body["horizon_support_status"] == "supported_closed_loop"
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
