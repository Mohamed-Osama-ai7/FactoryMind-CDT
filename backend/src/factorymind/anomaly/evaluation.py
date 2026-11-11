"""Leakage-safe, label-aware evaluation for window anomaly scores.

Labels are the operator's air-leak report windows, which are coarse (whole days in one case) and
incomplete (unreported anomalies may exist). Metrics therefore focus on what such labels can
support:

- Event level: was each reported failure detected (alarm within ``early_horizon`` before its
  start up to its end), and with what delay relative to the reported start?
- False alarms: raised alarm episodes that do not touch any report window (plus tolerance),
  normalised per day of valid telemetry.
- Window level: average precision (PR-AUC) of raw scores and precision/recall/F1 of the alarm
  decision, reported as secondary because window labels inherit the reports' coarseness.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from factorymind.anomaly.alarm import AlarmPolicy, Episode, episodes
from factorymind.data.metropt3 import FailureReport


@dataclass(frozen=True)
class EventResult:
    report_id: str
    detected: bool
    delay_minutes: float | None  # negative = before the reported start


@dataclass(frozen=True)
class EvaluationResult:
    valid_windows: int
    valid_days: float
    operating_windows: int
    operating_days: float
    positive_windows: int
    average_precision: float | None
    precision: float | None
    recall: float | None
    f1: float | None
    confusion: dict[str, int]
    events: list[EventResult]
    events_detected: int
    events_total: int
    alarm_episodes: int
    false_alarm_episodes: int
    false_alarms_per_day: float
    operating_false_alarms_per_day: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def window_labels(
    starts: pd.DatetimeIndex, window: pd.Timedelta, reports: Sequence[FailureReport]
) -> np.ndarray:
    """1 where [start, start+window) overlaps any report interval."""
    labels = np.zeros(len(starts), dtype=bool)
    ends = starts + window
    for r in reports:
        labels |= (starts <= pd.Timestamp(r.end)) & (ends > pd.Timestamp(r.start))
    return labels


def is_outside_report_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    reports: Sequence[FailureReport],
    *,
    early_horizon: pd.Timedelta,
    post_tolerance: pd.Timedelta,
) -> bool:
    """Return True when ``[start, end)`` falls outside all padded failure report windows."""
    for r in reports:
        lo = pd.Timestamp(r.start) - early_horizon
        hi = pd.Timestamp(r.end) + post_tolerance
        if start <= hi and end >= lo:
            return False
    return True


def evaluate(
    starts: pd.DatetimeIndex,
    scores: np.ndarray,
    window: pd.Timedelta,
    policy: AlarmPolicy,
    reports: Sequence[FailureReport],
    *,
    early_horizon: pd.Timedelta,
    post_tolerance: pd.Timedelta,
    suppress: np.ndarray | None = None,
    precomputed_episodes: Sequence[Episode] | None = None,
) -> EvaluationResult:
    if len(starts) != len(scores):
        raise ValueError("starts and scores must have the same length")
    labels = window_labels(starts, window, reports)
    eps = (
        list(precomputed_episodes)
        if precomputed_episodes is not None
        else episodes(starts, scores, window, policy, suppress)
    )

    decision = np.zeros(len(starts), dtype=bool)
    for ep in eps:
        decision |= (starts >= ep.start) & (starts < ep.end)

    tp = int((decision & labels).sum())
    fp = int((decision & ~labels).sum())
    fn = int((~decision & labels).sum())
    tn = int((~decision & ~labels).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else (0.0 if precision is not None and recall is not None else None)
    )
    ap = (
        float(average_precision_score(labels, scores)) if labels.any() and (~labels).any() else None
    )

    events: list[EventResult] = []
    for r in reports:
        lo, hi = pd.Timestamp(r.start) - early_horizon, pd.Timestamp(r.end)
        hits = [ep.raised_at for ep in eps if lo <= ep.raised_at <= hi + window]
        if hits:
            first = min(hits)
            delay = (first - pd.Timestamp(r.start)).total_seconds() / 60.0
            events.append(EventResult(r.report_id, True, round(delay, 1)))
        else:
            events.append(EventResult(r.report_id, False, None))

    false_eps = sum(
        1
        for ep in eps
        if is_outside_report_windows(
            ep.start,
            ep.end,
            reports,
            early_horizon=early_horizon,
            post_tolerance=post_tolerance,
        )
    )
    valid_days = len(starts) * window.total_seconds() / 86400.0
    operating_windows = (
        int((~np.asarray(suppress, dtype=bool)).sum()) if suppress is not None else len(starts)
    )
    operating_days = operating_windows * window.total_seconds() / 86400.0
    return EvaluationResult(
        valid_windows=len(starts),
        valid_days=round(valid_days, 2),
        operating_windows=operating_windows,
        operating_days=round(operating_days, 2),
        positive_windows=int(labels.sum()),
        average_precision=ap,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        events=events,
        events_detected=sum(e.detected for e in events),
        events_total=len(events),
        alarm_episodes=len(eps),
        false_alarm_episodes=false_eps,
        false_alarms_per_day=round(false_eps / valid_days, 3) if valid_days else 0.0,
        operating_false_alarms_per_day=round(false_eps / operating_days, 3)
        if operating_days
        else 0.0,
    )


__all__ = [
    "AlarmPolicy",
    "Episode",
    "EvaluationResult",
    "EventResult",
    "episodes",
    "evaluate",
    "window_labels",
]
