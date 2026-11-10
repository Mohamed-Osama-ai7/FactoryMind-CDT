import numpy as np
import pandas as pd
import pytest

from factorymind.anomaly.features import (
    FEATURE_NAMES,
    WindowSpec,
    block_features,
    window_features,
)
from tests.conftest import simulate_apu

SPEC = WindowSpec()


def test_windows_are_clock_aligned_and_complete() -> None:
    frame = simulate_apu("2020-01-01 00:00", 6 * 60)  # one hour
    out = window_features(frame, SPEC)
    assert len(out) == 6
    assert out.index[0] == pd.Timestamp("2020-01-01 00:00")
    assert out["valid"].all()
    assert (out["n_samples"] == 60).all()
    assert list(out.columns[-len(FEATURE_NAMES) :]) == list(FEATURE_NAMES)


def test_sparse_or_gapped_windows_are_invalid_not_interpolated() -> None:
    frame = simulate_apu("2020-01-01 00:00", 6 * 60)
    gapped = frame.drop(frame.index[70:80])  # 100 s gap inside the second window
    sparse = frame.iloc[: 6 * 10 + 20]  # third window only partly covered
    out_gap = window_features(gapped, SPEC)
    assert not out_gap.loc["2020-01-01 00:10", "valid"]
    assert out_gap.loc["2020-01-01 00:20", "valid"]
    out_sparse = window_features(sparse, SPEC)
    assert not out_sparse.loc["2020-01-01 00:10", "valid"]


def test_leak_raises_duty_and_steepens_unloaded_decay() -> None:
    frame = simulate_apu(
        "2020-01-01", 6 * 60 * 4, leak_from="2020-01-01 02:00", leak_to="2020-01-01 04:00"
    )
    out = window_features(frame, SPEC)
    normal = out.loc[:"2020-01-01 01:50"]
    leak = out.loc["2020-01-01 02:30":]
    assert leak["load_duty"].mean() > 2 * normal["load_duty"].mean()
    assert leak["unloaded_tp3_slope"].mean() < 3 * normal["unloaded_tp3_slope"].mean()
    assert leak["load_starts"].sum() > normal["load_starts"].sum()


def test_minutes_since_resume_tracks_restarts() -> None:
    a = simulate_apu("2020-01-01 00:00", 6 * 30)
    b = simulate_apu("2020-01-01 02:00", 6 * 30)
    out = window_features(pd.concat([a, b]), SPEC)
    assert out.loc["2020-01-01 00:20", "minutes_since_resume"] == pytest.approx(20)
    assert out.loc["2020-01-01 02:10", "minutes_since_resume"] == pytest.approx(10)


def test_block_features_match_window_features_for_aligned_block() -> None:
    frame = simulate_apu("2020-01-01 00:00", 60)
    block = block_features(frame, SPEC)
    window = window_features(frame, SPEC).iloc[0]
    np.testing.assert_allclose(
        block[list(FEATURE_NAMES)].astype(float).to_numpy(),
        window[list(FEATURE_NAMES)].astype(float).to_numpy(),
        equal_nan=True,
    )


def test_empty_input_rejected() -> None:
    empty = simulate_apu("2020-01-01", 1).iloc[:0]
    with pytest.raises(ValueError, match="empty"):
        window_features(empty, SPEC)
    with pytest.raises(ValueError, match="empty"):
        block_features(empty, SPEC)


def test_stopped_windows_are_explained_and_features_describe_the_state() -> None:
    frame = simulate_apu(
        "2020-01-01 00:00", 6 * 60 * 2, off_from="2020-01-01 00:20", off_to="2020-01-01 01:00"
    )
    out = window_features(frame, SPEC)
    assert out.loc["2020-01-01 00:20":"2020-01-01 00:50", "stopped"].all()
    assert (out.loc["2020-01-01 00:20":"2020-01-01 00:50", "off_fraction"] > 0.95).all()
    assert not out.loc["2020-01-01 01:10":, "stopped"].any()
    assert (out["off_fraction"] + out["load_duty"] <= 1.0 + 1e-9).all()


def test_loaded_rise_is_observable_during_loading_and_slower_when_leaking() -> None:
    normal = simulate_apu("2020-01-01", 6 * 60 * 4, seed=1)
    leak = simulate_apu(
        "2020-01-01", 6 * 60 * 4, leak_from="2020-01-01", leak_to="2020-01-02", seed=1
    )
    n = window_features(normal, SPEC)["loaded_tp3_rise"].dropna()
    assert (n > 0).all() and len(n) > 3  # TP3 rises while the compressor loads
    # under a sustained leak the unloaded phase is short, so the unloaded slope is rarely defined
    assert (
        window_features(leak, SPEC)["load_duty"].mean()
        > 2 * window_features(normal, SPEC)["load_duty"].mean()
    )


def test_stopped_threshold_is_configurable() -> None:
    frame = simulate_apu("2020-01-01", 60, off_from="2020-01-01", off_to="2020-01-01 00:05")
    assert not window_features(frame, WindowSpec(stopped_off_fraction=0.9))["stopped"].iloc[0]
    assert window_features(frame, WindowSpec(stopped_off_fraction=0.4))["stopped"].iloc[0]


def test_block_features_enforces_window_span_tolerance() -> None:
    # 10 min 30 s span at ~0.1 Hz cadence is within 60 s tolerance
    jittered = simulate_apu("2020-01-01 00:00", 60)
    idx = pd.DatetimeIndex(jittered.index).to_numpy().copy()
    idx[-1] = pd.Timestamp("2020-01-01 00:10:30")
    jittered.index = pd.DatetimeIndex(idx, name="timestamp")
    ok = block_features(jittered, SPEC)
    assert bool(ok["valid"])

    # 48 samples packed ~1 s apart (47 s span) fails temporal coverage and cadence checks
    burst = simulate_apu("2020-01-01 00:00", 48)
    burst.index = pd.date_range("2020-01-01 00:00:00", periods=48, freq="1s", name="timestamp")
    assert not bool(block_features(burst, SPEC)["valid"])

    # Dense 1 Hz block spanning 10 min (600 samples) fails the 5-15 s cadence contract
    dense_10m = simulate_apu("2020-01-01 00:00", 600)
    dense_10m.index = pd.date_range("2020-01-01 00:00:00", periods=600, freq="1s", name="timestamp")
    assert not bool(block_features(dense_10m, SPEC)["valid"])

    # 20-minute span exceeds 10 min + 60 s tolerance
    oversized = simulate_apu("2020-01-01 00:00", 120)
    with pytest.raises(ValueError, match="exceeds single-window limit"):
        block_features(oversized, SPEC)
