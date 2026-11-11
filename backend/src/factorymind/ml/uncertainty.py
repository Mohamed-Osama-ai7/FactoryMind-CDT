"""Uncertainty quantification: Mahalanobis OOD detector and probability calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from factorymind.contracts.ml_engine import ModelCalibrationMetrics


@dataclass(frozen=True)
class MahalanobisOodDetector:
    """Regularized Mahalanobis distance estimator using Ledoit-Wolf shrinkage covariance."""

    mean: np.ndarray
    inv_cov: np.ndarray
    threshold: float

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        *,
        quantile: float = 0.995,
    ) -> MahalanobisOodDetector:
        arr = np.asarray(features, dtype="float64")
        if arr.ndim != 2 or arr.shape[0] < 5:
            raise ValueError("Need at least 5 samples to fit Mahalanobis OOD detector")
        mean = np.mean(arr, axis=0)
        lw = LedoitWolf(assume_centered=False).fit(arr)
        inv_cov = np.asarray(lw.precision_, dtype="float64")
        dists = cls._compute_distances(arr, mean, inv_cov)
        threshold = float(np.quantile(dists, quantile))
        if threshold <= 0.0:
            threshold = 1.0
        return cls(mean=mean, inv_cov=inv_cov, threshold=threshold)

    @staticmethod
    def _compute_distances(arr: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
        diff = arr - mean
        quad = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        return np.asarray(np.sqrt(np.maximum(quad, 0.0)), dtype="float64")

    def distance(self, vector: np.ndarray) -> float | None:
        arr = np.asarray(vector, dtype="float64")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if not np.all(np.isfinite(arr)):
            return None
        return float(self._compute_distances(arr, self.mean, self.inv_cov)[0])

    def is_ood(self, vector: np.ndarray) -> tuple[bool, float | None]:
        dist = self.distance(vector)
        if dist is None:
            return True, None
        return dist > self.threshold, dist


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Compute weighted Expected Calibration Error (ECE) across equally spaced probability bins."""
    yt = np.asarray(y_true, dtype="float64")
    yp = np.clip(np.asarray(y_prob, dtype="float64"), 0.0, 1.0)
    if len(yt) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(yt)
    for idx in range(n_bins):
        lo, hi = edges[idx], edges[idx + 1]
        in_bin = (yp >= lo) & (yp <= hi) if idx == 0 else (yp > lo) & (yp <= hi)
        count = int(np.sum(in_bin))
        if count == 0:
            continue
        acc = float(np.mean(yt[in_bin]))
        conf = float(np.mean(yp[in_bin]))
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    margin_half_width: float = 0.10,
) -> ModelCalibrationMetrics:
    """Compute held-out discrimination, calibration (Brier/ECE), and selective accuracy."""
    yt = np.asarray(y_true, dtype="int64")
    yp = np.clip(np.asarray(y_prob, dtype="float64"), 0.0, 1.0)
    preds = (yp >= threshold).astype("int64")
    roc = float(roc_auc_score(yt, yp)) if len(np.unique(yt)) > 1 else 0.5
    pr = float(average_precision_score(yt, yp)) if len(np.unique(yt)) > 1 else float(np.mean(yt))
    brier = float(brier_score_loss(yt, yp))
    ece = expected_calibration_error(yt, yp, n_bins=10)
    prec, rec, f1, _ = precision_recall_fscore_support(yt, preds, average="binary", zero_division=0)

    confident = np.abs(yp - threshold) >= margin_half_width
    coverage = float(np.mean(confident)) if len(yt) else 0.0
    sel_acc = float(np.mean(preds[confident] == yt[confident])) if np.any(confident) else None

    return ModelCalibrationMetrics(
        roc_auc=round(roc, 6),
        pr_auc=round(pr, 6),
        brier_score=round(brier, 6),
        ece_10_bins=round(ece, 6),
        f1_at_threshold=round(float(f1), 6),
        precision_at_threshold=round(float(prec), 6),
        recall_at_threshold=round(float(rec), 6),
        threshold=round(float(threshold), 4),
        selective_coverage_at_margin=round(coverage, 6),
        selective_accuracy_at_margin=round(sel_acc, 6) if sel_acc is not None else None,
    )
