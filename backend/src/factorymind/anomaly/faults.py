"""Labelled *simulated* sensor faults injected into real normal telemetry.

MetroPT-3 has a single real failure type, so it cannot show whether a detector sees problems a
simple rule cannot. This module creates controlled test cases: intervals of real, normal,
operating telemetry in which one sensor is corrupted in a documented way. They are always reported
as **simulated** and never mixed with the real-leak results.

Fault catalogue (magnitudes are fixed here so results are reproducible and reviewable):

- ``stuck``: TP3 frozen at its value at the start of the interval (frozen pressure reading).
- ``offset``: TP3 shifted by +0.5 bar (about 5 % of range; calibration bias).
- ``flatline``: Oil_temperature frozen at its starting value.
- ``drift``: Reservoirs reading ramps linearly by +1.0 bar over the interval.
- ``noise``: Gaussian noise with sigma 0.3 bar added to TP2.

The compressor current, valve signals and duty cycle are untouched by all of them, so a rule that
only watches the duty cycle cannot see these faults by construction; the question is whether the
learned detectors can, without raising false alarms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from factorymind.anomaly.features import WindowSpec, window_features
from factorymind.data.metropt3 import FailureReport

FaultType = Literal["stuck", "offset", "flatline", "drift", "noise"]
FAULT_TYPES: Final[tuple[FaultType, ...]] = ("stuck", "offset", "flatline", "drift", "noise")

OFFSET_BAR: Final = 0.5
DRIFT_BAR: Final = 1.0
NOISE_SIGMA_BAR: Final = 0.3


def inject(frame: pd.DataFrame, fault: FaultType, rng: np.random.Generator) -> pd.DataFrame:
    """Return a copy of ``frame`` with the fault applied to the whole interval."""
    out = frame.copy()
    n = len(out)
    if fault == "stuck":
        out["TP3"] = float(out["TP3"].iloc[0])
    elif fault == "offset":
        out["TP3"] = out["TP3"] + OFFSET_BAR
    elif fault == "flatline":
        out["Oil_temperature"] = float(out["Oil_temperature"].iloc[0])
    elif fault == "drift":
        out["Reservoirs"] = out["Reservoirs"] + np.linspace(0.0, DRIFT_BAR, n)
    elif fault == "noise":
        out["TP2"] = out["TP2"] + rng.normal(0.0, NOISE_SIGMA_BAR, n)
    else:  # pragma: no cover - exhaustive literal
        raise ValueError(f"unknown fault {fault!r}")
    return out


@dataclass(frozen=True)
class FaultCase:
    start: pd.Timestamp
    end: pd.Timestamp
    fault: FaultType
    control: pd.DataFrame  # window features of the untouched interval
    faulted: pd.DataFrame  # window features with the fault injected


def build_cases(
    telemetry: pd.DataFrame,
    candidates: pd.DataFrame,
    spec: WindowSpec,
    reports: tuple[FailureReport, ...],
    *,
    n_per_type: int,
    duration_hours: float,
    min_gap_from_reports_hours: float,
    seed: int,
    period: tuple[pd.Timestamp, pd.Timestamp],
) -> list[FaultCase]:
    """Choose non-overlapping operating intervals far from real failures and corrupt them.

    ``candidates`` holds the window features of the evaluation period; an interval qualifies when
    all its windows are valid and the machine was operating (not stopped) in most of them.
    """
    window = pd.Timedelta(spec.length)
    n_windows = round(pd.Timedelta(hours=duration_hours) / window)
    margin = pd.Timedelta(hours=min_gap_from_reports_hours)
    index = pd.DatetimeIndex(candidates.index)
    valid = candidates["valid"].to_numpy(dtype=bool)
    operating = ~candidates["stopped"].to_numpy(dtype=bool)

    def near_report(ts: pd.Timestamp) -> bool:
        return any(
            pd.Timestamp(r.start) - margin <= ts <= pd.Timestamp(r.end) + margin for r in reports
        )

    lo, hi = period
    eligible: list[pd.Timestamp] = []
    for i in range(len(index) - n_windows):
        start = index[i]
        if start < lo or start + n_windows * window > hi:
            continue
        if index[i + n_windows - 1] - start != (n_windows - 1) * window:
            continue  # a gap inside the interval
        if not valid[i : i + n_windows].all() or operating[i : i + n_windows].mean() < 0.9:
            continue
        if near_report(start) or near_report(start + n_windows * window):
            continue
        eligible.append(start)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(eligible))
    wanted = n_per_type * len(FAULT_TYPES)
    chosen: list[pd.Timestamp] = []
    for k in order:
        start = eligible[int(k)]
        end = start + n_windows * window
        if all(end + window <= c or c + (n_windows + 1) * window <= start for c in chosen):
            chosen.append(start)
        if len(chosen) == wanted:
            break
    if len(chosen) < wanted:
        raise ValueError(f"only {len(chosen)} of {wanted} fault intervals could be placed")
    chosen.sort()

    cases: list[FaultCase] = []
    for i, start in enumerate(chosen):
        end = start + n_windows * window
        fault = FAULT_TYPES[i % len(FAULT_TYPES)]
        segment = telemetry[(telemetry.index >= start) & (telemetry.index < end)]
        control = window_features(segment, spec)
        faulted = window_features(inject(segment, fault, rng), spec)
        cases.append(FaultCase(start, end, fault, control, faulted))
    return cases
