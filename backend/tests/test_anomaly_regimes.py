"""Tests for causal operating-regime classification and uncertainty handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from factorymind.anomaly.features import WindowSpec, block_features, window_features
from factorymind.anomaly.regimes import RegimeTracker, classify_sequence, classify_window
from tests.conftest import simulate_apu

SPEC = WindowSpec()


def _steady_frame(
    start: str,
    minutes: int,
    *,
    motor_current: float,
    comp: float,
    dv_eletric: float,
    seed: int = 1,
) -> pd.DataFrame:
    df = simulate_apu(start, 6 * minutes, seed=seed).copy()
    rng = np.random.default_rng(seed + 100)
    n = len(df)
    df["Motor_current"] = motor_current + rng.normal(0.0, 0.05, n)
    df["COMP"] = comp
    df["DV_eletric"] = dv_eletric
    return df


def test_classifies_all_four_operating_regimes() -> None:
    stopped_df = simulate_apu(
        "2020-01-01 00:00", 60, off_from="2020-01-01 00:00", off_to="2020-01-01 00:10", seed=1
    )
    unloaded_df = _steady_frame("2020-01-01 00:10", 10, motor_current=4.0, comp=1.0, dv_eletric=0.0)
    loaded_df = _steady_frame("2020-01-01 00:20", 10, motor_current=7.0, comp=0.0, dv_eletric=1.0)
    inrush_df = unloaded_df.copy()
    inrush_df.index = pd.date_range("2020-01-01 00:30", periods=len(inrush_df), freq="10s")
    inrush_df.iloc[0, inrush_df.columns.get_loc("Motor_current")] = 8.9

    r_stop = classify_window(
        pd.Timestamp("2020-01-01 00:00"),
        pd.Timestamp("2020-01-01 00:10"),
        block_features(stopped_df, SPEC),
        SPEC,
    )
    r_unl = classify_window(
        pd.Timestamp("2020-01-01 00:10"),
        pd.Timestamp("2020-01-01 00:20"),
        block_features(unloaded_df, SPEC),
        SPEC,
    )
    r_load = classify_window(
        pd.Timestamp("2020-01-01 00:20"),
        pd.Timestamp("2020-01-01 00:30"),
        block_features(loaded_df, SPEC),
        SPEC,
    )
    r_start = classify_window(
        pd.Timestamp("2020-01-01 00:30"),
        pd.Timestamp("2020-01-01 00:40"),
        block_features(inrush_df, SPEC),
        SPEC,
    )

    assert r_stop.regime == "stopped" and r_stop.confidence == "confirmed"
    assert r_unl.regime == "unloaded" and r_unl.confidence == "confirmed"
    assert r_load.regime == "loaded" and r_load.confidence == "confirmed"
    assert r_start.regime == "starting" and r_start.confidence == "transitional"
    assert r_start.reason == "startup_inrush" and r_start.inrush_samples == 1
    assert r_unl.window_start == pd.Timestamp("2020-01-01 00:10")
    assert r_unl.window_end == pd.Timestamp("2020-01-01 00:20")


def test_transitions_and_resume_gaps_detect_starting_without_rewriting_history() -> None:
    stopped = simulate_apu(
        "2020-01-01 00:00", 60, off_from="2020-01-01 00:00", off_to="2020-01-01 00:10", seed=2
    )
    # 50 % off -> 50 % unloaded cycling right after stopped triggers starting transition
    resume = _steady_frame("2020-01-01 00:10", 10, motor_current=4.0, comp=1.0, dv_eletric=0.0)
    resume.iloc[:24, resume.columns.get_loc("Motor_current")] = 0.04
    steady_loaded = _steady_frame(
        "2020-01-01 00:20", 10, motor_current=7.0, comp=0.0, dv_eletric=1.0
    )
    # After >10 min gap at 00:30..00:50, resuming into active mixed load/unloaded is starting
    post_gap = _steady_frame("2020-01-01 00:50", 10, motor_current=4.0, comp=1.0, dv_eletric=0.0)
    post_gap.iloc[-18:, post_gap.columns.get_loc("Motor_current")] = 7.0
    post_gap.iloc[-18:, post_gap.columns.get_loc("COMP")] = 0.0
    post_gap.iloc[-18:, post_gap.columns.get_loc("DV_eletric")] = 1.0

    full = pd.concat([stopped, resume, steady_loaded, post_gap])
    wf = window_features(full, SPEC)
    seq_full = classify_sequence(wf, SPEC)
    seq_prefix = classify_sequence(wf.iloc[:2], SPEC)

    assert [a.regime for a in seq_full] == ["stopped", "starting", "loaded", "starting"]
    assert seq_full[1].confidence == "transitional"
    assert seq_full[1].missing_signals  # documents 0.1 Hz inrush limitation
    assert seq_full[:2] == seq_prefix  # causal: future windows never rewrite history

    # Isolated single block with minutes_since_resume == 0 and no inrush is not marked starting
    isolated = classify_window(
        pd.Timestamp("2020-01-01 00:50"),
        pd.Timestamp("2020-01-01 01:00"),
        block_features(post_gap, SPEC),
        SPEC,
    )
    assert isolated.regime == "insufficient_evidence" and isolated.reason == "mixed_duty_cycle"


def test_hysteresis_prevents_oscillation_and_starting_chatter() -> None:
    # Loaded >=0.9 -> 0.85 dip -> >=0.9 should stay loaded via hysteresis_hold (no flicker).
    w0 = _steady_frame("2020-01-01 00:00", 10, motor_current=7.0, comp=0.0, dv_eletric=1.0, seed=1)
    w1 = _steady_frame("2020-01-01 00:10", 10, motor_current=7.0, comp=0.0, dv_eletric=1.0, seed=2)
    w1.iloc[:9, w1.columns.get_loc("Motor_current")] = 4.0
    w1.iloc[:9, w1.columns.get_loc("COMP")] = 1.0
    w1.iloc[:9, w1.columns.get_loc("DV_eletric")] = 0.0
    w2 = _steady_frame("2020-01-01 00:20", 10, motor_current=7.0, comp=0.0, dv_eletric=1.0, seed=3)

    seq = classify_sequence(window_features(pd.concat([w0, w1, w2]), SPEC), SPEC)
    assert [a.regime for a in seq] == ["loaded", "loaded", "loaded"]
    assert seq[1].reason == "hysteresis_hold"
    assert "transition_hysteresis_hold" in seq[1].quality_flags

    # Noise dip near stopped boundary (0.88 off, zero starts/load) avoids stopped/starting chatter.
    s0 = simulate_apu(
        "2020-01-02 00:00", 60, off_from="2020-01-02 00:00", off_to="2020-01-02 00:10", seed=4
    )
    s1 = s0.copy()
    s1.index = pd.date_range("2020-01-02 00:10", periods=60, freq="10s")
    s1.iloc[:7, s1.columns.get_loc("Motor_current")] = 3.8  # 88.3 % off, zero load_starts
    s2 = simulate_apu(
        "2020-01-02 00:20", 60, off_from="2020-01-02 00:20", off_to="2020-01-02 00:30", seed=5
    )
    seq_stop = classify_sequence(window_features(pd.concat([s0, s1, s2]), SPEC), SPEC)
    assert seq_stop[0].regime == "stopped" and seq_stop[2].regime == "stopped"
    assert seq_stop[1].regime == "insufficient_evidence"
    assert seq_stop[1].reason == "hysteresis_hold"


def test_unloaded_hysteresis_hold() -> None:
    u0 = _steady_frame("2020-01-01 00:00", 10, motor_current=4.0, comp=1.0, dv_eletric=0.0, seed=1)
    u1 = _steady_frame("2020-01-01 00:10", 10, motor_current=4.0, comp=1.0, dv_eletric=0.0, seed=2)
    u1.iloc[:9, u1.columns.get_loc("Motor_current")] = 0.04
    seq = classify_sequence(window_features(pd.concat([u0, u1]), SPEC), SPEC)
    assert [a.regime for a in seq] == ["unloaded", "unloaded"]
    assert seq[1].reason == "hysteresis_hold"


def test_stale_frozen_and_gapped_telemetry_yield_insufficient_evidence() -> None:
    # Frozen telemetry even at loaded or stopped levels must never be trusted as a confirmed regime.
    frozen = simulate_apu("2020-01-01 00:00", 60, seed=1).copy()
    for col in ("TP3", "TP2", "Motor_current"):
        frozen[col] = 6.2
    frozen["COMP"] = 0.0
    frozen["DV_eletric"] = 1.0
    a_frozen = classify_window(
        pd.Timestamp("2020-01-01 00:00"),
        pd.Timestamp("2020-01-01 00:10"),
        block_features(frozen, SPEC),
        SPEC,
    )
    assert a_frozen.regime == "insufficient_evidence"
    assert a_frozen.reason == "stale_telemetry"
    assert a_frozen.stale_telemetry is True
    assert "stale_telemetry" in a_frozen.quality_flags

    # Gapped / invalid window
    gapped = simulate_apu("2020-01-01 00:00", 60, seed=1).drop(
        pd.date_range("2020-01-01 00:03", periods=10, freq="10s")
    )
    a_gap = classify_window(
        pd.Timestamp("2020-01-01 00:00"),
        pd.Timestamp("2020-01-01 00:10"),
        block_features(gapped, SPEC),
        SPEC,
    )
    assert a_gap.regime == "insufficient_evidence"
    assert a_gap.reason == "incomplete_or_gapped"

    # Non-finite required field
    bad_row = block_features(
        _steady_frame("2020-01-01 00:00", 10, motor_current=4.0, comp=1.0, dv_eletric=0.0), SPEC
    ).copy()
    bad_row["load_duty"] = np.nan
    a_nan = classify_window(
        pd.Timestamp("2020-01-01 00:00"), pd.Timestamp("2020-01-01 00:10"), bad_row, SPEC
    )
    assert a_nan.regime == "insufficient_evidence"
    assert "missing_signals" in a_nan.quality_flags


def test_inconsistent_valves_or_motor_signals_yield_insufficient_evidence() -> None:
    # Contradictory digital valves: COMP=1 and DV_eletric=1 simultaneously
    contra = _steady_frame("2020-01-01 00:00", 10, motor_current=4.0, comp=1.0, dv_eletric=1.0)
    a_contra = classify_window(
        pd.Timestamp("2020-01-01 00:00"),
        pd.Timestamp("2020-01-01 00:10"),
        block_features(contra, SPEC),
        SPEC,
    )
    assert a_contra.regime == "insufficient_evidence"
    assert a_contra.reason == "signal_inconsistency"
    assert "valve_contradiction" in a_contra.quality_flags

    # Motor off (<1 A) while outlet load valve DV_eletric=1
    mismatch = _steady_frame("2020-01-01 00:00", 10, motor_current=0.04, comp=0.0, dv_eletric=1.0)
    a_mismatch = classify_window(
        pd.Timestamp("2020-01-01 00:00"),
        pd.Timestamp("2020-01-01 00:10"),
        block_features(mismatch, SPEC),
        SPEC,
    )
    assert a_mismatch.regime == "insufficient_evidence"
    assert "motor_valve_mismatch" in a_mismatch.quality_flags


def test_ambiguous_mixed_cycle_is_not_forced_into_steady_state() -> None:
    mixed = _steady_frame("2020-01-01 00:00", 10, motor_current=4.0, comp=1.0, dv_eletric=0.0)
    mixed.iloc[:24, mixed.columns.get_loc("Motor_current")] = 0.04
    mixed.iloc[-18:, mixed.columns.get_loc("Motor_current")] = 7.0
    mixed.iloc[-18:, mixed.columns.get_loc("COMP")] = 0.0
    mixed.iloc[-18:, mixed.columns.get_loc("DV_eletric")] = 1.0

    res = RegimeTracker(SPEC).push(pd.Timestamp("2020-01-01 00:00"), block_features(mixed, SPEC))
    assert res.regime == "insufficient_evidence"
    assert res.reason == "mixed_duty_cycle"
    assert "mixed_duty_cycle" in res.quality_flags
    assert res.missing_signals
    assert (
        res.off_fraction is not None
        and res.unloaded_fraction is not None
        and res.load_duty is not None
    )
