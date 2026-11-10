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

from factorymind.anomaly.evaluation import Episode, episodes
from factorymind.anomaly.experiment import ModelBundle
from factorymind.anomaly.features import block_features, window_features
from factorymind.data import metropt3

WindowStatus = Literal["normal", "anomalous", "insufficient_data"]
EVIDENCE_TOP_K = 5


class ModelUnavailableError(RuntimeError):
    """The model artifact is missing, unreadable or fails its integrity check."""


class DataUnavailableError(RuntimeError):
    """Processed telemetry needed for replay is not available."""


@dataclass(frozen=True)
class FeatureEvidence:
    feature: str
    value: float | None
    training_median: float
    robust_z: float | None


@dataclass(frozen=True)
class WindowScore:
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_samples: int
    status: WindowStatus
    score: float | None
    evidence: list[FeatureEvidence]


def load_bundle(models_dir: Path, model_name: str) -> tuple[ModelBundle, dict[str, Any]]:
    root = models_dir / model_name
    pointer = root / "CURRENT"
    if not pointer.exists():
        raise ModelUnavailableError(f"No current model under {root}; run the training pipeline")
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
        raise ModelUnavailableError(f"Model {version} could not be loaded: {exc}") from exc
    if not isinstance(bundle, ModelBundle):
        raise ModelUnavailableError(f"Model {version} has an unexpected type")
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
            raise DataUnavailableError(str(exc)) from exc

    def _evidence(self, row: pd.Series) -> list[FeatureEvidence]:
        values = row[list(self.bundle.feature_names)].astype("float64")
        z = self.bundle.profile.z(values.to_frame().T).iloc[0]
        ranked = z.abs().sort_values(ascending=False).dropna().index[:EVIDENCE_TOP_K]
        return [
            FeatureEvidence(
                feature=name,
                value=None if np.isnan(values[name]) else float(values[name]),
                training_median=float(self.bundle.profile.median[name]),
                robust_z=float(z[name]),
            )
            for name in ranked
        ]

    def _score_rows(self, rows: pd.DataFrame) -> np.ndarray:
        return self.bundle.detector.score(rows[list(self.bundle.feature_names)])

    def score_block(self, frame: pd.DataFrame) -> WindowScore:
        """Score one submitted block of samples as a single window."""
        row = block_features(frame, self.bundle.spec)
        start = pd.Timestamp(row["first_sample"])
        end = pd.Timestamp(row["last_sample"])
        if not bool(row["valid"]):
            return WindowScore(start, end, int(row["n_samples"]), "insufficient_data", None, [])
        score = float(self._score_rows(row.to_frame().T)[0])
        status: WindowStatus = "anomalous" if score > self.bundle.policy.threshold else "normal"
        return WindowScore(start, end, int(row["n_samples"]), status, score, self._evidence(row))

    def replay(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[list[WindowScore], list[Episode], list[metropt3.FailureReport]]:
        """Score recorded telemetry in [start, end) window by window, applying alarm persistence."""
        data = self.telemetry
        lo = start - pd.Timedelta(self.bundle.spec.length)
        frame = data[(data.index >= lo) & (data.index < end)]
        if frame.empty:
            return [], [], []
        features = window_features(frame, self.bundle.spec)
        features = features[(features.index >= start) & (features.index < end)]
        window = pd.Timedelta(self.bundle.spec.length)
        valid = features[features["valid"]]
        scores = self._score_rows(valid) if len(valid) else np.array([])
        score_by_start = dict(zip(valid.index, scores, strict=True))
        results: list[WindowScore] = []
        for ts, row in features.iterrows():
            stamp = pd.Timestamp(str(ts))
            if stamp in score_by_start:
                s = float(score_by_start[stamp])
                status: WindowStatus = "anomalous" if s > self.bundle.policy.threshold else "normal"
                evidence = self._evidence(row) if status == "anomalous" else []
                results.append(
                    WindowScore(stamp, stamp + window, int(row["n_samples"]), status, s, evidence)
                )
            else:
                results.append(
                    WindowScore(
                        stamp, stamp + window, int(row["n_samples"]), "insufficient_data", None, []
                    )
                )
        eps = episodes(pd.DatetimeIndex(valid.index), scores, window, self.bundle.policy)
        reports = [
            r for r in self._reports if pd.Timestamp(r.start) < end and pd.Timestamp(r.end) >= start
        ]
        return results, eps, reports

    def data_range(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        try:
            index = pd.DatetimeIndex(self.telemetry.index)
        except DataUnavailableError:
            return None
        return pd.Timestamp(index.min()), pd.Timestamp(index.max())
