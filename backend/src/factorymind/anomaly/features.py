"""Gap-aware window features for the APU compressor.

The same functions are used for offline training/evaluation and for online inference, so the
API and the experiments cannot diverge. Missing data is never interpolated: windows with too
few samples or an internal gap are marked invalid instead.

Operating states follow the motor current, which is clearly trimodal in MetroPT-3 (~0.04 A off,
~4 A running unloaded, ~7 A under load): *off* < 1 A, *unloaded* 1-5.5 A, *loaded* > 5.5 A.

Physically motivated derived features (an air leak makes the compressor run loaded more often,
slows the pressure rise while loading and speeds up the decay while unloaded):
- ``load_duty``: fraction of samples with the motor under load.
- ``off_fraction``: fraction of samples with the motor off.
- ``load_starts``: rising edges of the outlet-valve signal (load cycles started).
- ``unloaded_tp3_slope``: median TP3 slope (bar/min) while unloaded. Observable only when the
  compressor actually unloads; during a sustained leak it never does.
- ``loaded_tp3_rise``: median TP3 slope (bar/min) while loading; a leak slows it.
- ``tp3_reservoir_gap``: mean TP3 minus reservoir pressure.

A window is ``stopped`` (an *explained* state, not an anomaly) when the machine was off for at
least ``stopped_off_fraction`` of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from factorymind.data.metropt3 import ANALOG_SENSORS, DIGITAL_SENSORS, NOMINAL_PERIOD_S

LOADED_CURRENT_A: Final = 5.5
OFF_CURRENT_A: Final = 1.0
_ANALOG_STATS: Final = ("mean", "std", "min", "max")
DERIVED_FEATURES: Final = (
    "load_duty",
    "off_fraction",
    "load_starts",
    "unloaded_tp3_slope",
    "loaded_tp3_rise",
    "tp3_reservoir_gap",
)
FEATURE_NAMES: Final[tuple[str, ...]] = (
    *(f"{s}_{stat}" for s in ANALOG_SENSORS for stat in _ANALOG_STATS),
    *(f"{s}_mean" for s in DIGITAL_SENSORS),
    *DERIVED_FEATURES,
)


@dataclass(frozen=True)
class WindowSpec:
    """How telemetry is cut into windows."""

    length: str = "10min"
    min_coverage: float = 0.8  # fraction of nominal samples required
    max_gap_s: float = 60.0  # an internal gap longer than this invalidates the window
    resume_gap_s: float = 600.0  # a gap longer than this starts a new operating segment
    stopped_off_fraction: float = 0.9  # off this share of a window => explained "stopped"

    @property
    def length_s(self) -> float:
        return float(pd.Timedelta(self.length).total_seconds())

    @property
    def min_samples(self) -> int:
        return int(np.ceil(self.min_coverage * self.length_s / NOMINAL_PERIOD_S))


def _sample_helpers(frame: pd.DataFrame, spec: WindowSpec) -> pd.DataFrame:
    """Per-sample columns needed for derived features, with gap-aware differencing."""
    index = pd.DatetimeIndex(frame.index)
    dt = index.to_series().diff().dt.total_seconds().to_numpy()
    contiguous = (dt > 0) & (dt <= spec.max_gap_s)
    loading = frame["DV_eletric"].to_numpy() >= 0.5
    prev_loading = np.roll(loading, 1)
    tp3 = frame["TP3"].to_numpy()
    slope = np.full(len(frame), np.nan)
    ok = contiguous & ~loading & ~prev_loading
    slope[ok] = (tp3[ok] - np.roll(tp3, 1)[ok]) / dt[ok] * 60.0
    rise = np.full(len(frame), np.nan)
    ok_rise = contiguous & loading & prev_loading
    rise[ok_rise] = (tp3[ok_rise] - np.roll(tp3, 1)[ok_rise]) / dt[ok_rise] * 60.0
    starts = (contiguous & loading & ~prev_loading).astype("float64")
    current = frame["Motor_current"].to_numpy()
    resume = ~((dt > 0) & (dt <= spec.resume_gap_s))
    segment_start = index.to_series().where(resume).ffill().to_numpy()
    return pd.DataFrame(
        {
            "loaded": (current > LOADED_CURRENT_A).astype("float64"),
            "off": (current < OFF_CURRENT_A).astype("float64"),
            "load_start": starts,
            "unloaded_slope": slope,
            "loaded_rise": rise,
            "tp3_res_gap": tp3 - frame["Reservoirs"].to_numpy(),
            "gap_s": np.where(np.isnan(dt), 0.0, dt),
            "segment_start": segment_start,
        },
        index=index,
    )


def _aggregate(frame: pd.DataFrame, keys: np.ndarray, spec: WindowSpec) -> pd.DataFrame:
    helpers = _sample_helpers(frame, spec)
    grouped = frame.groupby(keys, sort=True)
    parts = [grouped[list(ANALOG_SENSORS)].agg(list(_ANALOG_STATS))]
    # agg() yields (sensor, stat) MultiIndex columns; flatten to "sensor_stat".
    parts[0].columns = ["_".join(map(str, col)) for col in parts[0].columns.to_flat_index()]
    digital = grouped[list(DIGITAL_SENSORS)].mean()
    digital.columns = [f"{c}_mean" for c in digital.columns]
    parts.append(digital)
    hg = helpers.groupby(keys, sort=True)
    derived = pd.DataFrame(
        {
            "load_duty": hg["loaded"].mean(),
            "off_fraction": hg["off"].mean(),
            "load_starts": hg["load_start"].sum(),
            "unloaded_tp3_slope": hg["unloaded_slope"].median(),
            "loaded_tp3_rise": hg["loaded_rise"].median(),
            "tp3_reservoir_gap": hg["tp3_res_gap"].mean(),
        }
    )
    parts.append(derived)
    features = pd.concat(parts, axis=1)[list(FEATURE_NAMES)]

    ts = pd.Series(pd.DatetimeIndex(frame.index), index=frame.index)
    meta = pd.DataFrame(
        {
            "first_sample": ts.groupby(keys).min().to_numpy(),
            "last_sample": ts.groupby(keys).max().to_numpy(),
            "n_samples": grouped.size().to_numpy(),
            # the first sample's gap belongs to the previous window, so exclude it
            "max_internal_gap_s": helpers["gap_s"]
            .groupby(keys)
            .apply(lambda g: float(g.iloc[1:].max()) if len(g) > 1 else 0.0)
            .to_numpy(),
            "segment_start": helpers["segment_start"].groupby(keys).min().to_numpy(),
        },
        index=features.index,
    )
    meta["minutes_since_resume"] = (
        meta["first_sample"] - meta["segment_start"]
    ).dt.total_seconds() / 60.0
    meta["valid"] = (meta["n_samples"] >= spec.min_samples) & (
        meta["max_internal_gap_s"] <= spec.max_gap_s
    )
    meta["stopped"] = features["off_fraction"] >= spec.stopped_off_fraction
    return pd.concat([meta.drop(columns="segment_start"), features], axis=1)


def window_features(frame: pd.DataFrame, spec: WindowSpec) -> pd.DataFrame:
    """Cut a time-indexed telemetry frame into clock-aligned windows and compute features.

    Returns one row per non-empty window, indexed by window start, with metadata columns
    (``n_samples``, ``valid``, ``minutes_since_resume``...) followed by ``FEATURE_NAMES``.
    """
    if frame.empty:
        raise ValueError("Cannot compute features for an empty frame")
    keys = pd.DatetimeIndex(frame.index).floor(spec.length).to_numpy()
    out = _aggregate(frame, keys, spec)
    out.index = pd.DatetimeIndex(out.index, name="window_start")
    return out


def block_features(frame: pd.DataFrame, spec: WindowSpec) -> pd.Series:
    """Features for one submitted block treated as a single window (used by the API)."""
    if frame.empty:
        raise ValueError("Cannot compute features for an empty frame")
    out = _aggregate(frame, np.zeros(len(frame), dtype=np.int64), spec)
    return out.iloc[0]
