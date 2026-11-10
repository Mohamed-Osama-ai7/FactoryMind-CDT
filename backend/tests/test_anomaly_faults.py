import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from factorymind.anomaly.faults import FAULT_TYPES, build_cases, inject
from factorymind.anomaly.features import WindowSpec, window_features
from factorymind.data.metropt3 import FailureReport
from tests.conftest import simulate_apu

SPEC = WindowSpec()
REPORT = FailureReport("R", datetime(2020, 1, 3, 6), datetime(2020, 1, 3, 9), "Air leak")


def segment() -> pd.DataFrame:
    return simulate_apu("2020-01-01", 6 * 60 * 3, seed=4)  # three hours


def test_faults_change_only_their_documented_signal() -> None:
    base = segment()
    rng = np.random.default_rng(0)
    expected = {
        "stuck": "TP3",
        "offset": "TP3",
        "flatline": "Oil_temperature",
        "drift": "Reservoirs",
        "noise": "TP2",
    }
    for fault, column in expected.items():
        out = inject(base, fault, rng)  # type: ignore[arg-type]
        changed = [c for c in base.columns if not np.allclose(base[c], out[c])]
        assert changed == [column], fault
    assert base["TP3"].std() > 0  # input untouched


def test_stuck_and_flatline_remove_variation() -> None:
    base = segment()
    rng = np.random.default_rng(0)
    assert inject(base, "stuck", rng)["TP3"].nunique() == 1
    assert inject(base, "flatline", rng)["Oil_temperature"].nunique() == 1


def test_offset_and_drift_have_documented_magnitude() -> None:
    base = segment()
    rng = np.random.default_rng(0)
    assert (inject(base, "offset", rng)["TP3"] - base["TP3"]).mean() == pytest.approx(0.5)
    drift = inject(base, "drift", rng)["Reservoirs"] - base["Reservoirs"]
    assert drift.iloc[0] == pytest.approx(0.0) and drift.iloc[-1] == pytest.approx(1.0)


def telemetry_and_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    tele = simulate_apu("2020-01-01", 6 * 60 * 24 * 6, seed=2)
    feats = window_features(tele, SPEC)
    return tele, feats


def test_cases_are_balanced_non_overlapping_far_from_reports_and_reproducible() -> None:
    tele, feats = telemetry_and_features()
    kwargs = {
        "n_per_type": 2,
        "duration_hours": 3.0,
        "min_gap_from_reports_hours": 6.0,
        "seed": 5,
        "period": (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-07")),
    }
    cases = build_cases(tele, feats, SPEC, (REPORT,), **kwargs)
    assert len(cases) == 2 * len(FAULT_TYPES)
    for fault in FAULT_TYPES:
        assert sum(c.fault == fault for c in cases) == 2
    for a, b in itertools.pairwise(cases):
        assert b.start >= a.end  # chronological, non-overlapping
    for c in cases:
        assert not (REPORT.start - pd.Timedelta(hours=6) <= c.start <= REPORT.end)
        assert len(c.control) == len(c.faulted) == 18
    again = build_cases(tele, feats, SPEC, (REPORT,), **kwargs)
    assert [c.start for c in again] == [c.start for c in cases]


def test_faulted_windows_differ_from_control_in_the_affected_feature() -> None:
    tele, feats = telemetry_and_features()
    cases = build_cases(
        tele,
        feats,
        SPEC,
        (),
        n_per_type=1,
        duration_hours=3.0,
        min_gap_from_reports_hours=0.0,
        seed=1,
        period=(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-07")),
    )
    stuck = next(c for c in cases if c.fault == "stuck")
    assert (stuck.faulted["TP3_std"] == 0).all()
    assert (stuck.control["TP3_std"] > 0).any()


def test_too_few_eligible_intervals_is_an_error() -> None:
    tele, feats = telemetry_and_features()
    with pytest.raises(ValueError, match="could be placed"):
        build_cases(
            tele,
            feats,
            SPEC,
            (),
            n_per_type=50,
            duration_hours=3.0,
            min_gap_from_reports_hours=0.0,
            seed=1,
            period=(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")),
        )
