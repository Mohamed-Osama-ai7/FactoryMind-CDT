"""Stateful alarm tracking: windows in time order in, incident events out.

This is the single implementation of the alarm rule "``persistence`` consecutive anomalous
windows". Offline evaluation, API replay and any future live ingestion all use it, so an
incident's identity (start, raised-at) never depends on how a time range was chunked.

A window counts as anomalous only if it was scored, is not *explained* (for example a stopped
machine) and exceeds the threshold. A window that is missing, unscored, explained or normal ends
the current run; so does a gap between windows, because runs must be exactly one window apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlarmPolicy:
    threshold: float
    persistence: int = 1  # consecutive anomalous windows required to raise an alarm

    def __post_init__(self) -> None:
        if self.persistence < 1:
            raise ValueError("persistence must be >= 1")


@dataclass(frozen=True)
class Episode:
    start: pd.Timestamp  # first anomalous window start
    end: pd.Timestamp  # last anomalous window end
    raised_at: pd.Timestamp  # end of the window that satisfied persistence
    peak_score: float


EventKind = Literal["opened", "updated", "closed"]


@dataclass(frozen=True)
class IncidentEvent:
    kind: EventKind
    episode: Episode


class AlarmTracker:
    """Consumes windows in chronological order and emits incident events."""

    def __init__(self, window: pd.Timedelta, policy: AlarmPolicy) -> None:
        self._window = window
        self._policy = policy
        self._count = 0
        self._run_start: pd.Timestamp | None = None
        self._last_start: pd.Timestamp | None = None
        self._raised_at: pd.Timestamp | None = None
        self._peak = float("-inf")
        self._latest: pd.Timestamp | None = None

    @property
    def open_episode(self) -> Episode | None:
        """The incident currently open (alarm raised, run still continuing), if any."""
        return self._episode() if self._count >= self._policy.persistence else None

    def push(
        self, start: pd.Timestamp, score: float | None, *, explained: bool = False
    ) -> list[IncidentEvent]:
        if self._latest is not None and start <= self._latest:
            raise ValueError("windows must be pushed in strictly increasing start order")
        self._latest = start
        anomalous = score is not None and not explained and score > self._policy.threshold
        events: list[IncidentEvent] = []
        contiguous = self._last_start is not None and start - self._last_start == self._window
        if self._count and not (anomalous and contiguous):
            events.extend(self._close())
        if not anomalous or score is None:
            return events
        if self._count == 0:
            self._run_start = start
            self._peak = score
        else:
            self._peak = max(self._peak, score)
        self._count += 1
        self._last_start = start
        if self._count == self._policy.persistence:
            self._raised_at = start + self._window
            events.append(IncidentEvent("opened", self._episode()))
        elif self._count > self._policy.persistence:
            events.append(IncidentEvent("updated", self._episode()))
        return events

    def finish(self) -> list[IncidentEvent]:
        """Close any open run at the end of the input."""
        return self._close() if self._count else []

    def _episode(self) -> Episode:
        if self._run_start is None or self._last_start is None or self._raised_at is None:
            raise RuntimeError("no alarm run is open")
        return Episode(
            start=self._run_start,
            end=self._last_start + self._window,
            raised_at=self._raised_at,
            peak_score=self._peak,
        )

    def _close(self) -> list[IncidentEvent]:
        events: list[IncidentEvent] = []
        if self._count >= self._policy.persistence:
            events.append(IncidentEvent("closed", self._episode()))
        self._count = 0
        self._run_start = self._last_start = self._raised_at = None
        self._peak = float("-inf")
        return events


def episodes(
    starts: pd.DatetimeIndex,
    scores: np.ndarray | Sequence[float],
    window: pd.Timedelta,
    policy: AlarmPolicy,
    suppress: np.ndarray | None = None,
) -> list[Episode]:
    """Closed alarm episodes for a complete series of windows (single pass of the tracker)."""
    values = np.asarray(scores, dtype="float64")
    tracker = AlarmTracker(window, policy)
    result: list[Episode] = []
    for i, start in enumerate(starts):
        score = None if np.isnan(values[i]) else float(values[i])
        explained = bool(suppress[i]) if suppress is not None else False
        events = tracker.push(pd.Timestamp(start), score, explained=explained)
        result.extend(e.episode for e in events if e.kind == "closed")
    result.extend(e.episode for e in tracker.finish() if e.kind == "closed")
    return result
