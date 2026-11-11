"""Specialized multi-model orchestrator enforcing applicability, quality, and OOD abstention."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from factorymind.contracts.ml_engine import (
    ABSTENTION_MESSAGE,
    ModelRoutingDecision,
    MultiModelBenchmarkReport,
    QualityGateStatus,
    UncertaintyBand,
    UncertaintyEstimate,
)
from factorymind.data import ai4i2020
from factorymind.ml.ai4i_benchmark import Ai4iModelBundle


class BenchmarkUnavailableError(RuntimeError):
    """Raised when benchmark artifacts or attested bundles are absent or fail integrity."""


def load_ai4i_benchmark_report(artifacts_dir: Path) -> MultiModelBenchmarkReport:
    path = artifacts_dir / "benchmarks" / ai4i2020.DATASET_ID / "benchmark_report.json"
    if not path.is_file():
        raise BenchmarkUnavailableError("AI4I 2020 benchmark report not generated yet")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return MultiModelBenchmarkReport.model_validate(raw)
    except Exception as exc:
        raise BenchmarkUnavailableError("AI4I 2020 benchmark report is corrupt") from exc


def load_verified_ai4i_bundle(artifacts_dir: Path) -> Ai4iModelBundle:
    bench_dir = artifacts_dir / "benchmarks" / ai4i2020.DATASET_ID
    card_path = bench_dir / "model_card.json"
    bundle_path = bench_dir / "model.joblib"
    if not card_path.is_file() or not bundle_path.is_file():
        raise BenchmarkUnavailableError("AI4I 2020 model artifacts not prepared yet")
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise BenchmarkUnavailableError("AI4I 2020 model card unreadable") from exc
    schema_version = card.get("schema_version") if isinstance(card, dict) else None
    if not isinstance(card, dict) or type(schema_version) is not int or schema_version != 1:
        raise BenchmarkUnavailableError("AI4I 2020 model card unsupported")
    try:
        raw_bytes = bundle_path.read_bytes()
    except OSError as exc:
        raise BenchmarkUnavailableError("AI4I 2020 model bundle unreadable") from exc
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if digest != card.get("artifact_sha256"):
        raise BenchmarkUnavailableError("AI4I 2020 model bundle failed SHA-256 verification")
    try:
        bundle = joblib.load(bundle_path)
    except Exception as exc:
        raise BenchmarkUnavailableError("AI4I 2020 model bundle failed to load") from exc
    if not isinstance(bundle, Ai4iModelBundle):
        raise BenchmarkUnavailableError("AI4I 2020 bundle has invalid type")
    return bundle


def _check_input_quality(
    product_type: str,
    features: dict[str, float],
    quality_status: QualityGateStatus,
) -> list[str]:
    reasons: list[str] = []
    if quality_status != "healthy":
        reasons.append(f"data_quality_{quality_status}")
    if product_type not in ai4i2020.PRODUCT_TYPES:
        reasons.append("invalid_product_type")
    for col, (lo, hi) in ai4i2020.VALID_RANGES.items():
        val = features.get(col)
        if val is None or not math.isfinite(val):
            reasons.append("corrupt_or_non_finite")
            break
        if val < lo or val > hi:
            reasons.append(f"out_of_range:{col}")
    return reasons


class MultiModelOrchestrator:
    """Routes single-sample requests through applicability, quality, OOD, and calibration gates."""

    MARGIN_HALF_WIDTH = 0.10

    def __init__(self, bundle: Ai4iModelBundle) -> None:
        self.bundle = bundle

    def predict(
        self,
        *,
        asset_id: str,
        product_type: str,
        air_temperature_k: float,
        process_temperature_k: float,
        rotational_speed_rpm: float,
        torque_nm: float,
        tool_wear_min: float,
        model_id: str | None = None,
        quality_status: QualityGateStatus = "healthy",
    ) -> ModelRoutingDecision:
        if model_id is not None and model_id not in self.bundle.candidates:
            ood_threshold = round(self.bundle.ood_detector.threshold, 4)
            return ModelRoutingDecision(
                asset_id=asset_id,
                selected_model_id=model_id,
                model_family="unknown",
                model_version=self.bundle.model_version,
                abstained=True,
                abstention_message=ABSTENTION_MESSAGE,
                abstention_reasons=["unknown_model_id"],
                uncertainty=UncertaintyEstimate(
                    mahalanobis_distance=None,
                    ood_threshold=ood_threshold,
                    is_ood=True,
                    uncertainty_band="abstained",
                    calibrated_probability=None,
                    decision_margin=None,
                ),
            )
        candidate = (
            self.bundle.candidates[model_id]
            if model_id is not None
            else self.bundle.selected_candidate
        )

        ood_threshold = round(self.bundle.ood_detector.threshold, 4)
        base_meta: dict[str, Any] = {
            "asset_id": asset_id,
            "selected_model_id": candidate.model_id,
            "model_family": candidate.model_family,
            "model_version": self.bundle.model_version,
        }

        if asset_id != self.bundle.asset_id:
            return ModelRoutingDecision(
                **base_meta,
                abstained=True,
                abstention_message=ABSTENTION_MESSAGE,
                abstention_reasons=["asset_mismatch"],
                uncertainty=UncertaintyEstimate(
                    mahalanobis_distance=None,
                    ood_threshold=ood_threshold,
                    is_ood=True,
                    uncertainty_band="abstained",
                    calibrated_probability=None,
                    decision_margin=None,
                ),
            )

        feat_map = {
            "Air temperature [K]": air_temperature_k,
            "Process temperature [K]": process_temperature_k,
            "Rotational speed [rpm]": rotational_speed_rpm,
            "Torque [Nm]": torque_nm,
            "Tool wear [min]": tool_wear_min,
        }
        quality_reasons = _check_input_quality(product_type, feat_map, quality_status)
        if quality_reasons:
            return ModelRoutingDecision(
                **base_meta,
                abstained=True,
                abstention_message=ABSTENTION_MESSAGE,
                abstention_reasons=quality_reasons,
                uncertainty=UncertaintyEstimate(
                    mahalanobis_distance=None,
                    ood_threshold=ood_threshold,
                    is_ood=True,
                    uncertainty_band="abstained",
                    calibrated_probability=None,
                    decision_margin=None,
                ),
            )

        num_arr = np.array([[feat_map[c] for c in ai4i2020.NUMERIC_FEATURES]], dtype="float64")
        scaled_num = self.bundle.numeric_scaler.transform(num_arr)
        is_ood, m_dist = self.bundle.ood_detector.is_ood(scaled_num)
        if is_ood:
            return ModelRoutingDecision(
                **base_meta,
                abstained=True,
                abstention_message=ABSTENTION_MESSAGE,
                abstention_reasons=["ood_mahalanobis_exceeded"],
                uncertainty=UncertaintyEstimate(
                    mahalanobis_distance=round(m_dist, 4) if m_dist is not None else None,
                    ood_threshold=ood_threshold,
                    is_ood=True,
                    uncertainty_band="abstained",
                    calibrated_probability=None,
                    decision_margin=None,
                ),
            )

        row_df = pd.DataFrame([{"Type": product_type, **feat_map}])
        row_enc = np.asarray(self.bundle.preprocessor.transform(row_df), dtype="float64")
        prob = float(candidate.predict_proba(row_enc)[0])
        prob = min(max(prob, 0.0), 1.0)
        margin = min(abs(prob - candidate.threshold), 0.5)

        if margin < self.MARGIN_HALF_WIDTH:
            return ModelRoutingDecision(
                **base_meta,
                abstained=True,
                abstention_message=ABSTENTION_MESSAGE,
                abstention_reasons=["low_margin_ambiguous"],
                prediction_label=None,
                calibrated_failure_probability=round(prob, 6),
                uncertainty=UncertaintyEstimate(
                    mahalanobis_distance=round(m_dist, 4) if m_dist is not None else None,
                    ood_threshold=ood_threshold,
                    is_ood=False,
                    uncertainty_band="high",
                    calibrated_probability=round(prob, 6),
                    decision_margin=round(margin, 6),
                ),
            )

        band: UncertaintyBand = "low" if margin >= 0.25 else "moderate"
        label = "failure_risk" if prob >= candidate.threshold else "nominal"
        modes: list[str] = []
        if label == "failure_risk":
            for mode_name, clf in self.bundle.mode_heads.items():
                classes = list(getattr(clf, "classes_", []))
                if 1 not in classes:
                    continue
                pos_idx = classes.index(1)
                proba_row = clf.predict_proba(row_enc)[0]
                if float(proba_row[pos_idx]) >= 0.35:
                    modes.append(mode_name)

        return ModelRoutingDecision(
            **base_meta,
            abstained=False,
            abstention_message=None,
            abstention_reasons=[],
            prediction_label=label,
            calibrated_failure_probability=round(prob, 6),
            predicted_failure_modes=modes,
            uncertainty=UncertaintyEstimate(
                mahalanobis_distance=round(m_dist, 4) if m_dist is not None else None,
                ood_threshold=ood_threshold,
                is_ood=False,
                uncertainty_band=band,
                calibrated_probability=round(prob, 6),
                decision_margin=round(margin, 6),
            ),
        )
