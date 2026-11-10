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
