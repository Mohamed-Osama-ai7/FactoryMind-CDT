"""Anomaly detectors over window features.

All detectors are fitted on training windows only and return a score where *higher means more
anomalous*. Scores are not probabilities; the alarm threshold is calibrated separately.

- ``lps_rule``: the plant's own low-pressure switch (fraction of samples with LPS active). It
  represents the alarm that already exists, and is the baseline the models must beat.
- ``duty_rule``: compressor load duty alone, the physics baseline.
- ``robust_z``: max absolute robust z-score across features (median/MAD from training).
- ``isolation_forest``: scikit-learn IsolationForest on imputed, robust-scaled features.
- ``pca_spe``: PCA squared prediction error (SPE/Q statistic), a classic process-monitoring
  method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

DetectorName = Literal["lps_rule", "duty_rule", "robust_z", "isolation_forest", "pca_spe"]

_MAD_TO_SIGMA = 1.4826
_MIN_SCALE = 1e-6
_Z_99 = 2.3263  # standard-normal 99th percentile


class Detector(Protocol):
    name: str

    def fit(self, features: pd.DataFrame) -> Detector: ...

    def score(self, features: pd.DataFrame) -> np.ndarray: ...


@dataclass
class FeatureProfile:
    """Robust training-distribution statistics used for scoring and for evidence."""

    median: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    scale: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))

    def fit(self, features: pd.DataFrame) -> FeatureProfile:
        self.median = features.median(skipna=True)
        mad = (features - self.median).abs().median(skipna=True) * _MAD_TO_SIGMA
        # Several APU signals are bimodal (e.g. TP2 ~0 bar unloaded, ~10 bar loaded). Their MAD
        # only describes the dominant mode, which made every loaded window look extreme. Floor
        # the scale with a sigma estimate from the central 98% span of the training data.
        span = features.quantile(0.99) - features.quantile(0.01)
        span_sigma = span / (2 * _Z_99)
        scale = pd.concat([mad.fillna(0.0), span_sigma.fillna(0.0)], axis=1).max(axis=1)
        self.scale = scale.where(scale > _MIN_SCALE, _MIN_SCALE)
        return self

    def z(self, features: pd.DataFrame) -> pd.DataFrame:
        return (features[self.median.index] - self.median) / self.scale


@dataclass
class LpsRule:
    name: str = "lps_rule"

    def fit(self, features: pd.DataFrame) -> LpsRule:
        if "LPS_mean" not in features:
            raise ValueError("LPS_mean feature required")
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        return features["LPS_mean"].to_numpy(dtype="float64")


@dataclass
class DutyRule:
    """Physics baseline: share of the window the compressor spent under load.

    A healthy APU runs loaded roughly 7 % of the time (median over normal windows); an air leak
    forces it to stay loaded for hours. One interpretable feature, no learning.
    """

    name: str = "duty_rule"

    def fit(self, features: pd.DataFrame) -> DutyRule:
        if "load_duty" not in features:
            raise ValueError("load_duty feature required")
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        return features["load_duty"].to_numpy(dtype="float64")


@dataclass
class RobustZ:
    name: str = "robust_z"
    clip: float = 50.0
    profile: FeatureProfile = field(default_factory=FeatureProfile)

    def fit(self, features: pd.DataFrame) -> RobustZ:
        self.profile.fit(features)
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        z = self.profile.z(features).abs().clip(upper=self.clip).fillna(0.0)
        return z.max(axis=1).to_numpy(dtype="float64")


@dataclass
class IsolationForestDetector:
    name: str = "isolation_forest"
    n_estimators: int = 300
    max_samples: int | float | str = "auto"
    seed: int = 42
    pipeline: Pipeline | None = None

    def fit(self, features: pd.DataFrame) -> IsolationForestDetector:
        self.pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
                (
                    "model",
                    IsolationForest(
                        n_estimators=self.n_estimators,
                        max_samples=self.max_samples,
                        random_state=self.seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ).fit(features.to_numpy(dtype="float64"))
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Detector is not fitted")
        return -np.asarray(self.pipeline.score_samples(features.to_numpy(dtype="float64")))


@dataclass
class PcaSpe:
    name: str = "pca_spe"
    variance: float = 0.95
    seed: int = 42
    pre: Pipeline | None = None
    pca: PCA | None = None

    def fit(self, features: pd.DataFrame) -> PcaSpe:
        self.pre = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        x = self.pre.fit_transform(features.to_numpy(dtype="float64"))
        self.pca = PCA(n_components=self.variance, svd_solver="full", random_state=self.seed).fit(x)
        return self

    def score(self, features: pd.DataFrame) -> np.ndarray:
        if self.pre is None or self.pca is None:
            raise RuntimeError("Detector is not fitted")
        x = self.pre.transform(features.to_numpy(dtype="float64"))
        reconstructed = self.pca.inverse_transform(self.pca.transform(x))
        return np.asarray(((x - reconstructed) ** 2).sum(axis=1), dtype="float64")


def build_detector(name: DetectorName, params: dict[str, object], seed: int) -> Detector:
    if name == "lps_rule":
        return LpsRule()
    if name == "duty_rule":
        return DutyRule()
    if name == "robust_z":
        return RobustZ(**params)  # type: ignore[arg-type]
    if name == "isolation_forest":
        return IsolationForestDetector(seed=seed, **params)  # type: ignore[arg-type]
    if name == "pca_spe":
        return PcaSpe(seed=seed, **params)  # type: ignore[arg-type]
    raise ValueError(f"Unknown detector {name!r}")
