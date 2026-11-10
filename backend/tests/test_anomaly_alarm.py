import numpy as np
import pandas as pd
import pytest

from factorymind.anomaly.alarm import AlarmPolicy, AlarmTracker, episodes

W = pd.Timedelta("10min")


def starts(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2020-01-01", periods=n, freq="10min")


def run(scores: list[float | None], k: int, explained: list[bool] | None = None):
    tracker = AlarmTracker(W, AlarmPolicy(0.5, k))
    events = []
    for i, (ts, s) in enumerate(zip(starts(len(scores)), scores, strict=True)):
        events.extend(tracker.push(ts, s, explained=bool(explained and explained[i])))
    events.extend(tracker.finish())
    return events


def test_lifecycle_opens_at_persistence_updates_and_closes() -> None:
    events = run([0, 1, 1, 1, 1, 0], k=3)
    assert [e.kind for e in events] == ["opened", "updated", "closed"]
    opened, _, closed = (e.episode for e in events)
    assert opened.start == starts(6)[1]
    assert opened.raised_at == starts(6)[3] + W  # end of the third anomalous window
    assert closed.end == starts(6)[4] + W
    assert closed.peak_score == 1


def test_runs_shorter_than_persistence_never_alarm() -> None:
    assert run([1, 1, 0, 1, 1, 0], k=3) == []


def test_unscored_and_explained_windows_break_runs() -> None:
    assert run([1, 1, None, 1, 1], k=3) == []
    assert run([1, 1, 1, 1, 1], k=3, explained=[False, False, True, False, False]) == []
    # explained windows are never anomalous even with a huge score
    assert run([9, 9, 9], k=1, explained=[True, True, True]) == []


def test_time_gaps_break_runs() -> None:
    tracker = AlarmTracker(W, AlarmPolicy(0.5, 2))
    t0 = pd.Timestamp("2020-01-01")
    events = []
    for offset in (0, 1, 3, 4):  # window 2 is missing
        events += tracker.push(t0 + offset * W, 1.0)
    events += tracker.finish()
    closed = [e.episode for e in events if e.kind == "closed"]
    assert len(closed) == 2 and closed[0].end == t0 + 2 * W and closed[1].start == t0 + 3 * W


def test_windows_must_arrive_in_order() -> None:
    tracker = AlarmTracker(W, AlarmPolicy(0.5, 1))
    tracker.push(starts(2)[1], 0.0)
    with pytest.raises(ValueError, match="increasing"):
        tracker.push(starts(2)[0], 0.0)


def test_open_episode_is_visible_before_it_closes() -> None:
    tracker = AlarmTracker(W, AlarmPolicy(0.5, 2))
    assert tracker.open_episode is None
    tracker.push(starts(3)[0], 1.0)
    assert tracker.open_episode is None
    tracker.push(starts(3)[1], 1.0)
    assert tracker.open_episode is not None
    tracker.push(starts(3)[2], 0.0)
    assert tracker.open_episode is None


def test_policy_validates_persistence() -> None:
    with pytest.raises(ValueError):
        AlarmPolicy(threshold=1.0, persistence=0)


@pytest.mark.parametrize("seed", range(100))
def test_chunked_processing_equals_single_pass(seed: int) -> None:
    """One persistent tracker fed in arbitrary chunks gives exactly the single-pass episodes."""
    rng = np.random.default_rng(seed)
    n = 400
    idx = starts(n)
    # runs of anomalous windows of random length, random missing windows (gaps), random stops
    scores = np.zeros(n)
    i = 0
    while i < n:
        length = int(rng.integers(1, 20))
        scores[i : i + length] = float(rng.integers(0, 2))
        i += length
    keep = rng.random(n) > 0.03
    idx, scores = idx[keep], scores[keep]
    suppress = rng.random(len(idx)) < 0.05
    policy = AlarmPolicy(0.5, int(rng.integers(1, 7)))

    expected = episodes(idx, scores, W, policy, suppress)

    tracker = AlarmTracker(W, policy)
    cuts = sorted(rng.choice(np.arange(1, len(idx)), size=5, replace=False).tolist())
    got = []
    for chunk in np.split(np.arange(len(idx)), cuts):
        for j in chunk:
            got += [
                e.episode
                for e in tracker.push(idx[j], float(scores[j]), explained=bool(suppress[j]))
                if e.kind == "closed"
            ]
    got += [e.episode for e in tracker.finish() if e.kind == "closed"]
    assert got == expected
