"""Score calibration: turn a detector score into an honestly defined exceedance probability.

Raw scores (IsolationForest, PCA SPE, load duty) are not comparable and are not probabilities.
``ExceedanceCalibrator`` stores scores of *reference-normal* windows and reports, for any new score,
the (smoothed) share of reference windows scoring at least as high: a conformal-style p-value.

``p = 0.01`` therefore means "a reference-normal window scores this high about 1 % of the time".
It is *not* the probability of a failure. A score beyond every reference window gets the minimum
p-value ``1 / (n + 1)``, so p cannot express severity beyond the reference range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ExceedanceCalibrator:
    reference: np.ndarray  # sorted ascending, finite

    @classmethod
    def fit(cls, scores: np.ndarray) -> ExceedanceCalibrator:
        values = np.asarray(scores, dtype="float64")
        if values.size == 0 or not np.isfinite(values).all():
            raise ValueError("reference scores must be a non-empty, finite array")
        return cls(reference=np.sort(values))

    @property
    def n(self) -> int:
        return int(self.reference.size)

    @property
    def min_p(self) -> float:
        return 1.0 / (self.n + 1)

    def p_value(self, scores: np.ndarray | float) -> np.ndarray:
        values = np.atleast_1d(np.asarray(scores, dtype="float64"))
        at_least = self.n - np.searchsorted(self.reference, values, side="left")
        return np.asarray((1.0 + at_least) / (self.n + 1), dtype="float64")

    def threshold_for(self, p: float) -> float:
        """Score such that windows above it have exceedance probability below ``p``."""
        if not 0 < p < 1:
            raise ValueError("p must be in (0, 1)")
        return float(np.quantile(self.reference, 1.0 - p))
