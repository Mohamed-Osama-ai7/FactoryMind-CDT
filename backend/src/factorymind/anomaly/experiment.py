"""Reproducible MetroPT-3 anomaly-detection experiment.

Protocol (chronological, leakage-safe):
1. ``train``: fit feature profile and detectors (no labels used; no failure reports here).
2. ``calibration``: held-out period after training, assumed normal; sets each detector's alarm
   threshold as a score quantile (a false-alarm budget), never from failure labels.
3. ``validation``: contains report F1; selects the threshold quantile, persistence and the
   detector. This is the only place labels influence choices.
4. ``test``: contains reports F2-F4; evaluated once with the frozen choices.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from factorymind import __version__
from factorymind.anomaly.detectors import Detector, DetectorName, FeatureProfile, build_detector
from factorymind.anomaly.evaluation import AlarmPolicy, EvaluationResult, evaluate
from factorymind.anomaly.features import FEATURE_NAMES, WindowSpec, window_features
from factorymind.data import metropt3

SPLITS = ("train", "calibration", "validation", "test")


class WindowConfig(BaseModel):
    length: str = "10min"
    min_coverage: float = Field(0.8, gt=0, le=1)
    max_gap_s: float = Field(60.0, gt=0)
    resume_gap_s: float = Field(600.0, gt=0)

    def spec(self) -> WindowSpec:
        return WindowSpec(**self.model_dump())


class ExperimentConfig(BaseModel):
    experiment: str
    seed: int = 42
    window: WindowConfig = WindowConfig()
    splits: dict[str, tuple[str, str]]
    startup_exclusion_minutes: float = Field(0.0, ge=0)
    early_horizon_hours: float = Field(6.0, ge=0)
    post_tolerance_hours: float = Field(24.0, ge=0)
    threshold_quantiles: list[float]
    persistence_grid: list[int]
    # Each detector maps to one or more parameter sets; variants are compared on validation.
    detectors: dict[DetectorName, list[dict[str, Any]]]

    @field_validator("detectors")
    @classmethod
    def _variants(
        cls, value: dict[DetectorName, list[dict[str, Any]]]
    ) -> dict[DetectorName, list[dict[str, Any]]]:
        if not value or any(not variants for variants in value.values()):
            raise ValueError("each detector needs at least one parameter set")
        return value

    @field_validator("splits")
    @classmethod
    def _chronological(cls, value: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
        if list(value) != list(SPLITS):
            raise ValueError(f"splits must be exactly {SPLITS} in order")
        bounds = [pd.Timestamp(b) for name in SPLITS for b in value[name]]
        if any(b2 < b1 for b1, b2 in pairwise(bounds)):
            raise ValueError("split boundaries must be chronological and non-overlapping")
        return value

    @field_validator("threshold_quantiles")
    @classmethod
    def _quantiles(cls, value: list[float]) -> list[float]:
        if not value or any(not 0.5 < q < 1 for q in value):
            raise ValueError("threshold quantiles must be in (0.5, 1)")
        return value

    def fingerprint(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:10]

    @classmethod
    def load(cls, path: Path) -> ExperimentConfig:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass
class ModelBundle:
    """Everything needed for inference, saved as one artifact."""

    version: str
    detector_name: str
    detector: Detector
    profile: FeatureProfile
    spec: WindowSpec
    policy: AlarmPolicy
    feature_names: tuple[str, ...]
    card: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Choice:
    detector: str
    params: dict[str, Any]
    quantile: float
    threshold: float
    persistence: int
    validation: EvaluationResult


def _selection_key(result: EvaluationResult) -> tuple[float, float, float]:
    # Detect the labelled events first, then minimise false alarms, then rank quality.
    return (
        -result.events_detected,
        result.false_alarms_per_day,
        -(result.average_precision or 0.0),
    )


def split_windows(features: pd.DataFrame, config: ExperimentConfig) -> dict[str, pd.DataFrame]:
    usable = features[features["valid"]]
    if config.startup_exclusion_minutes > 0:
        usable = usable[usable["minutes_since_resume"] >= config.startup_exclusion_minutes]
    out: dict[str, pd.DataFrame] = {}
    for name in SPLITS:
        lo, hi = (pd.Timestamp(b) for b in config.splits[name])
        out[name] = usable[(usable.index >= lo) & (usable.index < hi)]
        if out[name].empty:
            raise ValueError(f"split {name!r} has no valid windows")
    return out


def run(
    config: ExperimentConfig,
    telemetry: pd.DataFrame,
    reports: tuple[metropt3.FailureReport, ...] = metropt3.FAILURE_REPORTS,
) -> tuple[dict[str, Any], dict[str, ModelBundle], dict[str, pd.DataFrame]]:
    """Run the full protocol. Returns (results, bundles per detector, scored splits)."""
    spec = config.window.spec()
    window = pd.Timedelta(spec.length)
    features = window_features(telemetry, spec)
    splits = split_windows(features, config)
    x = {name: frame[list(FEATURE_NAMES)] for name, frame in splits.items()}

    reports_in = {
        name: tuple(
            r
            for r in reports
            if pd.Timestamp(config.splits[name][0])
            <= pd.Timestamp(r.start)
            < pd.Timestamp(config.splits[name][1])
        )
        for name in SPLITS
    }
    if reports_in["train"] or reports_in["calibration"]:
        raise ValueError("training/calibration periods must not contain reported failures")

    eval_kwargs = {
        "early_horizon": pd.Timedelta(hours=config.early_horizon_hours),
        "post_tolerance": pd.Timedelta(hours=config.post_tolerance_hours),
    }
    profile = FeatureProfile().fit(x["train"])
    version = f"{datetime.now(UTC):%Y%m%d}-{config.fingerprint()}"

    results: dict[str, Any] = {"detectors": {}}
    bundles: dict[str, ModelBundle] = {}
    scored: dict[str, pd.DataFrame] = {
        name: frame[["n_samples", "minutes_since_resume"]].copy() for name, frame in splits.items()
    }
    choices: list[Choice] = []

    for name, variants in config.detectors.items():
        candidates: list[Choice] = []
        fitted: list[tuple[Detector, dict[str, np.ndarray]]] = []
        for params in variants:
            det = build_detector(name, params, config.seed).fit(x["train"])
            det_scores = {split: det.score(frame) for split, frame in x.items()}
            fitted.append((det, det_scores))
            for q in config.threshold_quantiles:
                threshold = float(np.quantile(det_scores["calibration"], q))
                for k in config.persistence_grid:
                    policy = AlarmPolicy(threshold=threshold, persistence=k)
                    res = evaluate(
                        pd.DatetimeIndex(splits["validation"].index),
                        det_scores["validation"],
                        window,
                        policy,
                        reports_in["validation"],
                        **eval_kwargs,
                    )
                    candidates.append(Choice(name, params, q, threshold, k, res))
        best = min(candidates, key=lambda c: _selection_key(c.validation))
        choices.append(best)
        detector, scores = fitted[variants.index(best.params)]
        for split in SPLITS:
            scored[split][name] = scores[split]
        params = best.params
        policy = AlarmPolicy(threshold=best.threshold, persistence=best.persistence)
        test_res = evaluate(
            pd.DatetimeIndex(splits["test"].index),
            scores["test"],
            window,
            policy,
            reports_in["test"],
            **eval_kwargs,
        )
        calib_alarm_rate = float(np.mean(scores["calibration"] > best.threshold))
        results["detectors"][name] = {
            "params": params,
            "threshold_quantile": best.quantile,
            "threshold": best.threshold,
            "persistence": best.persistence,
            "calibration_window_alarm_rate": round(calib_alarm_rate, 5),
            "validation": best.validation.to_dict(),
            "test": test_res.to_dict(),
            "validation_grid": [
                {
                    "params": c.params,
                    "quantile": c.quantile,
                    "persistence": c.persistence,
                    "events_detected": c.validation.events_detected,
                    "false_alarms_per_day": c.validation.false_alarms_per_day,
                    "average_precision": c.validation.average_precision,
                }
                for c in candidates
            ],
        }
        bundles[name] = ModelBundle(
            version=f"{version}-{name}",
            detector_name=name,
            detector=detector,
            profile=profile,
            spec=spec,
            policy=policy,
            feature_names=FEATURE_NAMES,
        )

    selected = min(choices, key=lambda c: _selection_key(c.validation))
    results.update(
        {
            "experiment": config.experiment,
            "config_fingerprint": config.fingerprint(),
            "version": version,
            "selected_detector": selected.detector,
            "selection_rule": "validation only: max events detected, then min false alarms/day, "
            "then max average precision",
            "splits": {
                name: {
                    "range": list(config.splits[name]),
                    "valid_windows": len(splits[name]),
                    "reports": [r.report_id for r in reports_in[name]],
                }
                for name in SPLITS
            },
        }
    )
    for det_name, bundle in bundles.items():
        bundle.card = _model_card(config, results, det_name, bundle)
    return results, bundles, scored


def _model_card(
    config: ExperimentConfig, results: dict[str, Any], name: str, bundle: ModelBundle
) -> dict[str, Any]:
    det = results["detectors"][name]
    return {
        "model_version": bundle.version,
        "detector": name,
        "package_version": __version__,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": {
            "id": metropt3.DATASET_ID,
            "source": metropt3.SOURCE_PAGE,
            "license": metropt3.LICENSE,
            "archive_sha256": metropt3.ARCHIVE_SHA256,
            "data_kind": "real operational telemetry (public dataset)",
        },
        "training_period": list(config.splits["train"]),
        "calibration_period": list(config.splits["calibration"]),
        "window": asdict(bundle.spec),
        "threshold": det["threshold"],
        "threshold_quantile": det["threshold_quantile"],
        "persistence": det["persistence"],
        "feature_names": list(bundle.feature_names),
        "config": json.loads(config.model_dump_json()),
        "evaluation": {"validation": det["validation"], "test": det["test"]},
        "intended_use": "Advisory early warning of abnormal compressor behaviour for human review. "
        "Scores are anomaly scores, not failure probabilities or root causes.",
        "limitations": [
            "Trained and evaluated on one compressor (MetroPT-3); transfer to other assets is "
            "untested.",
            "Labels are four coarse air-leak reports; unreported anomalies count as false alarms.",
            "Report timestamps' timezone is not stated by the source.",
        ],
    }


def save(
    results: dict[str, Any],
    bundles: dict[str, ModelBundle],
    scored: dict[str, pd.DataFrame],
    artifacts_dir: Path,
    model_name: str,
) -> Path:
    """Persist experiment results, per-window scores and the model bundles; mark the selected
    detector as the current model."""
    run_dir = artifacts_dir / "experiments" / results["experiment"] / results["version"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), "utf-8")
    for split, frame in scored.items():
        frame.to_parquet(run_dir / f"scores_{split}.parquet")

    model_root = artifacts_dir / "models" / model_name
    for bundle in bundles.values():
        out = model_root / bundle.version
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out / "model.joblib")
        digest = hashlib.sha256((out / "model.joblib").read_bytes()).hexdigest()
        card = {**bundle.card, "artifact_sha256": digest}
        (out / "model_card.json").write_text(json.dumps(card, indent=2, default=str), "utf-8")
    selected: str = bundles[results["selected_detector"]].version
    (model_root / "CURRENT").write_text(selected + "\n", encoding="utf-8")
    return Path(run_dir)
