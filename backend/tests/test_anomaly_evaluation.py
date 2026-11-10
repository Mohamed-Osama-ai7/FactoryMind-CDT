from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from factorymind.anomaly.evaluation import AlarmPolicy, episodes, evaluate, window_labels
from factorymind.data.metropt3 import FailureReport

W = pd.Timedelta("10min")
REPORT = FailureReport("R1", datetime(2020, 1, 2, 0, 0), datetime(2020, 1, 2, 2, 0), "Air leak")


def _starts(n: int, start: str = "2020-01-01 00:00") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="10min")


def test_window_labels_mark_overlap_only() -> None:
    starts = _starts(6 * 24 * 2)
    labels = window_labels(starts, W, [REPORT])
    positives = starts[labels]
    assert positives.min() == pd.Timestamp("2020-01-01 23:55") + pd.Timedelta("5min")
    assert positives.max() == pd.Timestamp("2020-01-02 02:00")


def test_persistence_requires_consecutive_windows_and_gaps_break_runs() -> None:
    starts = _starts(10)
    scores = np.array([0, 1, 1, 0, 1, 1, 1, 0, 0, 0], dtype=float)
    eps = episodes(starts, scores, W, AlarmPolicy(threshold=0.5, persistence=3))
    assert len(eps) == 1
    assert eps[0].start == starts[4]
    assert eps[0].raised_at == starts[6] + W

    gapped = starts.delete(5).insert(5, starts[5] + pd.Timedelta("1h"))
    eps_gap = episodes(gapped.sort_values(), scores, W, AlarmPolicy(threshold=0.5, persistence=3))
    assert eps_gap == []


def test_alarm_policy_validates_persistence() -> None:
    with pytest.raises(ValueError):
        AlarmPolicy(threshold=1.0, persistence=0)


def test_event_detection_delay_and_false_alarms() -> None:
    starts = _starts(6 * 24 * 3)
    scores = np.zeros(len(starts))
    # early alarm 1 h before the report start
    scores[(starts >= "2020-01-01 22:50") & (starts < "2020-01-01 23:20")] = 1.0
    # an isolated false alarm far from the report
    scores[(starts >= "2020-01-03 12:00") & (starts < "2020-01-03 12:30")] = 1.0
    result = evaluate(
        starts,
        scores,
        W,
        AlarmPolicy(threshold=0.5, persistence=3),
        [REPORT],
        early_horizon=pd.Timedelta(hours=6),
        post_tolerance=pd.Timedelta(hours=24),
    )
    assert result.events_detected == 1
    # raised at the end of the third consecutive window (23:20), 40 min before the start
    assert result.events[0].delay_minutes == pytest.approx(-40.0)
    assert result.alarm_episodes == 2
    assert result.false_alarm_episodes == 1
    assert result.false_alarms_per_day == pytest.approx(1 / 3, abs=1e-3)
    assert result.confusion["tp"] == 0  # the early alarm precedes the labelled window


def test_missed_event_and_metrics_without_positives() -> None:
    starts = _starts(30)
    result = evaluate(
        starts,
        np.zeros(30),
        W,
        AlarmPolicy(threshold=0.5),
        [],
        early_horizon=pd.Timedelta(hours=6),
        post_tolerance=pd.Timedelta(hours=24),
    )
    assert result.events_total == 0
    assert result.average_precision is None
    assert result.precision is None


def test_length_mismatch_rejected() -> None:
    with pytest.raises(ValueError):
        evaluate(
            _starts(3),
            np.zeros(2),
            W,
            AlarmPolicy(threshold=0.5),
            [],
            early_horizon=pd.Timedelta(0),
            post_tolerance=pd.Timedelta(0),
        )


def test_operating_false_alarms_per_day_excludes_suppressed_stopped_windows() -> None:
    starts = _starts(6 * 24 * 2)  # 2 days = 288 windows
    scores = np.zeros(len(starts))
    # One isolated false alarm
    scores[10:14] = 1.0
    # Half of the period (144 windows = 1 day) is stopped/suppressed
    suppress = np.zeros(len(starts), dtype=bool)
    suppress[144:] = True
    result = evaluate(
        starts,
        scores,
        W,
        AlarmPolicy(threshold=0.5, persistence=3),
        [],
        early_horizon=pd.Timedelta(0),
        post_tolerance=pd.Timedelta(0),
        suppress=suppress,
    )
    assert result.valid_days == pytest.approx(2.0)
    assert result.operating_days == pytest.approx(1.0)
    assert result.false_alarms_per_day == pytest.approx(0.5)
    assert result.operating_false_alarms_per_day == pytest.approx(1.0)
