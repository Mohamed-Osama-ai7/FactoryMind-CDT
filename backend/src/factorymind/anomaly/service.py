"""Inference service: loads the current model bundle and scores telemetry.

Advisory only: results are anomaly scores and evidence for human review. Nothing here writes to
industrial systems or triggers actions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from factorymind.anomaly.alarm import AlarmTracker, Episode
from factorymind.anomaly.evidence import EvidenceItem, build_evidence
from factorymind.anomaly.experiment import ModelBundle
from factorymind.anomaly.features import block_features, window_features
from factorymind.data import metropt3

WindowStatus = Literal["normal", "anomalous", "stopped", "insufficient_data"]
# How far back a replay scans so that an alarm run that began before the requested range keeps its
# true start and raised-at time. Runs longer than this are reported as truncated.
REPLAY_LOOKBACK = pd.Timedelta(hours=24)


class ModelUnavailableError(RuntimeError):
    """The model artifact is missing, unreadable or fails its integrity check."""


class DataUnavailableError(RuntimeError):
    """Processed telemetry needed for replay is not available."""


@dataclass(frozen=True)
class WindowScore:
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_samples: int
    status: WindowStatus
    score: float | None
    exceedance: float | None
    evidence: list[EvidenceItem]


@dataclass(frozen=True)
class ReplayEpisode:
    episode: Episode
    truncated: bool  # the run may have started before the scanned range
    ongoing: bool  # still anomalous at the end of the requested range


def load_bundle(models_dir: Path, model_name: str) -> tuple[ModelBundle, dict[str, Any]]:
    root = models_dir / model_name
    pointer = root / "CURRENT"
    if not pointer.exists():
        raise ModelUnavailableError("No current model is available; run the training pipeline")
    version = pointer.read_text(encoding="utf-8").strip()
    model_dir = root / version
    artifact, card_path = model_dir / "model.joblib", model_dir / "model_card.json"
    if not artifact.exists() or not card_path.exists():
        raise ModelUnavailableError(f"Model {version} is incomplete")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != card.get("artifact_sha256"):
        raise ModelUnavailableError(f"Model {version} failed its integrity check")
    try:
        bundle = joblib.load(artifact)
    except Exception as exc:
        raise ModelUnavailableError(f"Model {version} could not be loaded") from exc
    if not isinstance(bundle, ModelBundle) or card.get("schema_version") != 2:
        raise ModelUnavailableError(f"Model {version} has an unsupported format")
    return bundle, card


class AnomalyService:
    def __init__(
        self,
        bundle: ModelBundle,
        card: dict[str, Any],
        processed_dir: Path,
        reports: tuple[metropt3.FailureReport, ...] = metropt3.FAILURE_REPORTS,
    ) -> None:
        self.bundle = bundle
        self.card = card
        self._processed_dir = processed_dir
        self._reports = reports

    @property
    def reports(self) -> tuple[metropt3.FailureReport, ...]:
        return self._reports

    @cached_property
    def telemetry(self) -> pd.DataFrame:
        try:
            return metropt3.load_processed(self._processed_dir)
        except FileNotFoundError as exc:
            raise DataUnavailableError("Processed telemetry is not available") from exc

    def _evidence(self, row: pd.Series) -> list[EvidenceItem]:
        return build_evidence(row, self.bundle.profile, self.bundle.feature_names)

    def _score_rows(self, rows: pd.DataFrame) -> np.ndarray:
        return self.bundle.detector.score(rows[list(self.bundle.feature_names)])

    def _window(
        self, start: pd.Timestamp, end: pd.Timestamp, row: pd.Series, score: float | None
    ) -> WindowScore:
        n = int(row["n_samples"])
        if not bool(row["valid"]):
            return WindowScore(start, end, n, "insufficient_data", None, None, [])
        if bool(row["stopped"]):
            return WindowScore(start, end, n, "stopped", None, None, [])
        if score is None:
            raise ValueError("a valid operating window needs a score")
        p = float(self.bundle.calibrator.p_value(score)[0])
        if score > self.bundle.policy.threshold:
            return WindowScore(start, end, n, "anomalous", score, p, self._evidence(row))
        return WindowScore(start, end, n, "normal", score, p, [])

    def score_block(self, frame: pd.DataFrame) -> WindowScore:
        """Score one submitted block of samples as a single window."""
        row = block_features(frame, self.bundle.spec)
        start, end = pd.Timestamp(row["first_sample"]), pd.Timestamp(row["last_sample"])
        score: float | None = None
        if bool(row["valid"]) and not bool(row["stopped"]):
            score = float(self._score_rows(row.to_frame().T)[0])
        return self._window(start, end, row, score)

    def replay(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[list[WindowScore], list[ReplayEpisode], list[metropt3.FailureReport]]:
        """Score recorded telemetry in [start, end) window by window.

        Alarm runs are tracked from ``start - REPLAY_LOOKBACK`` so an incident's start and
        raised-at time do not depend on where the requested range begins.
        """
        data = self.telemetry
        window = pd.Timedelta(self.bundle.spec.length)
        scan_start = start - REPLAY_LOOKBACK
        frame = data[(data.index >= scan_start) & (data.index < end)]
        if frame.empty:
            return [], [], []
        features = window_features(frame, self.bundle.spec)
        usable = features["valid"] & ~features["stopped"]
        scores = np.full(len(features), np.nan)
        if usable.any():
            scores[usable.to_numpy()] = self._score_rows(features[usable])

        tracker = AlarmTracker(window, self.bundle.policy)
        closed: list[Episode] = []
        windows: list[WindowScore] = []
        for i, (ts, row) in enumerate(features.iterrows()):
            stamp = pd.Timestamp(str(ts))
            score = None if np.isnan(scores[i]) else float(scores[i])
            events = tracker.push(stamp, score, explained=bool(row["stopped"]))
            closed.extend(e.episode for e in events if e.kind == "closed")
            if stamp >= start:
                windows.append(self._window(stamp, stamp + window, row, score))
        ongoing = tracker.open_episode
        closed.extend(e.episode for e in tracker.finish() if e.kind == "closed")

        first_window = pd.Timestamp(str(features.index[0]))
        episodes = [
            ReplayEpisode(
                ep,
                truncated=ep.start <= first_window,
                ongoing=ongoing is not None and ep.start == ongoing.start,
            )
            for ep in closed
            if ep.end > start and ep.start < end
        ]
        reports = [
            r for r in self._reports if pd.Timestamp(r.start) < end and pd.Timestamp(r.end) >= start
        ]
        return windows, episodes, reports

    def data_range(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        try:
            index = pd.DatetimeIndex(self.telemetry.index)
        except DataUnavailableError:
            return None
        return pd.Timestamp(index.min()), pd.Timestamp(index.max())
