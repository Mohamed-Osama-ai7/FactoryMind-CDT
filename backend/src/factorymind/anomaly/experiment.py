"""Reproducible MetroPT-3 anomaly-detection experiment (protocol v2).

Protocol (chronological; replaces the single-validation-event protocol of M1):

1. ``train``: fit every detector on *operating*, non-extreme windows (no labels used).
2. ``calibration``: reference-normal windows after training; thresholds are quantiles of the
   detector's reference scores (label-free) or values on an absolute grid (duty rule).
3. ``evaluation``: everything after calibration, containing all four operator reports.
4. **Nested leave-one-event-out (LOEO):** for each report, the policy (variant, threshold,
   persistence) is chosen using only the *other* reports and then measured on the held-out one.
   Selection criterion, fixed in the config: events detected, then false alarms per day, then
   median absolute delay. Four events support per-event reporting, not statistics.
5. **Default detector rule** (`choose_default`): the best rule-based detector is the bar; a learned
   detector becomes the default only if it matches the rule's leak detection at no worse false
   alarms, or detects injected sensor faults (simulated) above a minimum rate at a low false-alarm
   rate without losing leak detection.

Disclosed limits: the normal time used to estimate false alarms is shared by selection and
evaluation, which favours every detector equally; the final deployed policy is selected on all four
events and is therefore in-sample.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from factorymind import __version__
from factorymind.anomaly.alarm import AlarmPolicy, AlarmTracker
from factorymind.anomaly.calibration import ExceedanceCalibrator
from factorymind.anomaly.detectors import Detector, DetectorName, FeatureProfile, build_detector
from factorymind.anomaly.evaluation import EvaluationResult, evaluate
from factorymind.anomaly.faults import FAULT_TYPES, FaultCase, build_cases
from factorymind.anomaly.features import FEATURE_NAMES, WindowSpec, window_features
from factorymind.data import metropt3

PERIODS = ("train", "calibration", "evaluation")
ML_DETECTORS: tuple[str, ...] = ("isolation_forest", "pca_spe")


class WindowConfig(BaseModel):
    length: str = "10min"
    min_coverage: float = Field(0.8, gt=0, le=1)
    max_gap_s: float = Field(60.0, gt=0)
    resume_gap_s: float = Field(600.0, gt=0)
    stopped_off_fraction: float = Field(0.9, gt=0, le=1)

    def spec(self) -> WindowSpec:
        return WindowSpec(**self.model_dump())


class ReferenceConfig(BaseModel):
    """Which windows may define "normal" for fitting and calibration."""

    exclude_stopped: bool = True
    max_load_duty: float | None = Field(0.9, gt=0, le=1)  # None disables the extreme-duty rule


class ThresholdConfig(BaseModel):
    mode: Literal["quantile", "absolute"]
    grid: list[float]

    @model_validator(mode="after")
    def _check(self) -> ThresholdConfig:
        if not self.grid:
            raise ValueError("threshold grid must not be empty")
        if self.mode == "quantile" and any(not 0.5 < q < 1 for q in self.grid):
            raise ValueError("quantiles must be in (0.5, 1)")
        return self


class DetectorConfig(BaseModel):
    variants: list[dict[str, Any]] = [{}]
    threshold: ThresholdConfig

    @field_validator("variants")
    @classmethod
    def _nonempty(cls, value: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not value:
            raise ValueError("each detector needs at least one parameter set")
        return value


class SelectionConfig(BaseModel):
    preference: list[DetectorName]  # tie-break order, simplest first
    rule_detectors: list[DetectorName] = ["lps_rule", "duty_rule"]
    ml_fault_detection_min: float = Field(0.9, ge=0, le=1)
    ml_fault_fa_max: float = Field(0.1, ge=0)


class FaultConfig(BaseModel):
    n_per_type: int = Field(12, ge=1)
    duration_hours: float = Field(3.0, gt=0)
    seed: int = 7
    min_gap_from_reports_hours: float = Field(24.0, ge=0)


class ExperimentConfig(BaseModel):
    experiment: str
    seed: int = 42
    window: WindowConfig = WindowConfig()
    periods: dict[str, tuple[str, str]]
    reference: ReferenceConfig = ReferenceConfig()
    early_horizon_hours: float = Field(6.0, ge=0)
    post_tolerance_hours: float = Field(24.0, ge=0)
    persistence_grid: list[int]
    detectors: dict[DetectorName, DetectorConfig]
    selection: SelectionConfig
    faults: FaultConfig = FaultConfig()
    exceedance_alpha: float = Field(0.01, gt=0, lt=1)

    @field_validator("periods")
    @classmethod
    def _chronological(cls, value: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
        if list(value) != list(PERIODS):
            raise ValueError(f"periods must be exactly {PERIODS} in order")
        bounds = [pd.Timestamp(b) for name in PERIODS for b in value[name]]
        if any(b2 < b1 for b1, b2 in pairwise(bounds)):
            raise ValueError("period boundaries must be chronological and non-overlapping")
        return value

    @field_validator("persistence_grid")
    @classmethod
    def _persistence(cls, value: list[int]) -> list[int]:
        if not value or any(k < 1 for k in value):
            raise ValueError("persistence values must be >= 1")
        return value

    @model_validator(mode="after")
    def _selection_known(self) -> ExperimentConfig:
        unknown = [d for d in self.selection.preference if d not in self.detectors]
        if unknown or set(self.selection.preference) != set(self.detectors):
            raise ValueError("selection.preference must list exactly the configured detectors")
        return self

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
    calibrator: ExceedanceCalibrator
    spec: WindowSpec
    policy: AlarmPolicy
    feature_names: tuple[str, ...]
    card: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    detector: str
    variant: int
    params: dict[str, Any]
    threshold_label: str
    threshold: float
    persistence: int
    result: EvaluationResult


@dataclass
class RunOutput:
    results: dict[str, Any]
    bundles: dict[str, ModelBundle]
    scored: dict[str, pd.DataFrame]


def _selection_key(c: Candidate, exclude_event: str | None) -> tuple[int, float, float]:
    events = [e for e in c.result.events if e.report_id != exclude_event]
    detected = sum(e.detected for e in events)
    delays = [abs(e.delay_minutes) for e in events if e.detected and e.delay_minutes is not None]
    median_delay = float(np.median(delays)) if delays else float("inf")
    return (-detected, c.result.false_alarms_per_day, median_delay)


def select_policy(candidates: list[Candidate], exclude_event: str | None = None) -> Candidate:
    """Best candidate by the pre-registered criterion, ignoring one report if given."""
    return min(candidates, key=lambda c: _selection_key(c, exclude_event))


def _median_abs_delay(delays: list[float]) -> float | None:
    return round(float(np.median([abs(d) for d in delays])), 1) if delays else None


def _loeo_summary(candidates: list[Candidate], report_ids: list[str]) -> dict[str, Any]:
    folds: list[dict[str, Any]] = []
    for rid in report_ids:
        chosen = select_policy(candidates, exclude_event=rid)
        held = next(e for e in chosen.result.events if e.report_id == rid)
        folds.append(
            {
                "held_out": rid,
                "variant": chosen.params,
                "threshold": chosen.threshold_label,
                "persistence": chosen.persistence,
                "detected": held.detected,
                "delay_minutes": held.delay_minutes,
                "false_alarms_per_day": chosen.result.false_alarms_per_day,
            }
        )
    delays = [f["delay_minutes"] for f in folds if f["detected"] and f["delay_minutes"] is not None]
    return {
        "folds": folds,
        "events_detected": sum(f["detected"] for f in folds),
        "events_total": len(folds),
        "mean_false_alarms_per_day": round(
            float(np.mean([f["false_alarms_per_day"] for f in folds])), 3
        ),
        "median_abs_delay_minutes": _median_abs_delay(delays),
    }


def _fault_detection(
    detector: Detector,
    policy: AlarmPolicy,
    cases: list[FaultCase],
    window: pd.Timedelta,
    spec: WindowSpec,
) -> dict[str, Any]:
    def raised(frame: pd.DataFrame) -> bool:
        usable = frame
        scores = detector.score(usable[list(FEATURE_NAMES)])
        tracker = AlarmTracker(window, policy)
        opened = False
        for ts, valid, stopped, s in zip(
            usable.index, usable["valid"], usable["stopped"], scores, strict=True
        ):
            events = tracker.push(
                pd.Timestamp(ts), float(s) if bool(valid) else None, explained=bool(stopped)
            )
            opened |= any(e.kind == "opened" for e in events)
        return opened

    per_type: dict[str, list[bool]] = {f: [] for f in FAULT_TYPES}
    control: list[bool] = []
    for case in cases:
        per_type[case.fault].append(raised(case.faulted))
        control.append(raised(case.control))
    total = sum(sum(v) for v in per_type.values())
    n = sum(len(v) for v in per_type.values())
    return {
        "detected": total,
        "cases": n,
        "rate": round(total / n, 3) if n else None,
        "per_type": {f: f"{sum(v)}/{len(v)}" for f, v in per_type.items()},
        "control_alarms": f"{sum(control)}/{len(control)}",
        "simulated": True,
        "min_window_spec": spec.length,
    }


def choose_default(
    summary: dict[str, dict[str, Any]], faults: dict[str, dict[str, Any]], cfg: SelectionConfig
) -> tuple[str, str]:
    """Apply the pre-declared default-detector rule; returns (detector, rationale)."""

    def rank(name: str) -> tuple[int, float, float, int]:
        s = summary[name]
        median = s["median_abs_delay_minutes"]
        return (
            -s["events_detected"],
            s["mean_false_alarms_per_day"],
            float("inf") if median is None else median,
            cfg.preference.index(name),  # type: ignore[arg-type]
        )

    rules = [d for d in cfg.rule_detectors if d in summary]
    best_rule = min(rules, key=rank)
    bar = summary[best_rule]
    eligible: list[tuple[str, str]] = []
    for name in (d for d in cfg.preference if d in ML_DETECTORS and d in summary):
        s = summary[name]
        fault = faults.get(name, {})
        matches_rule = (
            s["events_detected"] >= bar["events_detected"]
            and s["mean_false_alarms_per_day"] <= bar["mean_false_alarms_per_day"]
        )
        sees_faults = (
            (fault.get("rate") or 0.0) >= cfg.ml_fault_detection_min
            and s["mean_false_alarms_per_day"] <= cfg.ml_fault_fa_max
            and s["events_detected"] >= bar["events_detected"]
        )
        if matches_rule:
            eligible.append((name, f"matches {best_rule} on leaks at no worse false alarms"))
        elif sees_faults:
            eligible.append(
                (name, f"detects injected faults ({fault['rate']:.0%}) and matches leak detection")
            )
    if not eligible:
        return best_rule, f"no learned detector qualified; {best_rule} is the best rule baseline"
    best = min(eligible, key=lambda e: rank(e[0]))
    return best[0], best[1]


def _usable_masks(
    window_feats: pd.DataFrame, config: ExperimentConfig
) -> tuple[pd.DataFrame, pd.Series]:
    usable = window_feats[window_feats["valid"]]
    ref = config.reference
    fit_ok = pd.Series(True, index=usable.index)
    if ref.exclude_stopped:
        fit_ok &= ~usable["stopped"]
    if ref.max_load_duty is not None:
        fit_ok &= usable["load_duty"] <= ref.max_load_duty
    return usable, fit_ok


def run(
    config: ExperimentConfig,
    telemetry: pd.DataFrame,
    reports: tuple[metropt3.FailureReport, ...] = metropt3.FAILURE_REPORTS,
) -> RunOutput:
    spec = config.window.spec()
    window = pd.Timedelta(spec.length)
    usable, fit_ok = _usable_masks(window_features(telemetry, spec), config)

    def period(name: str) -> pd.DataFrame:
        lo, hi = (pd.Timestamp(b) for b in config.periods[name])
        out = usable[(usable.index >= lo) & (usable.index < hi)]
        if out.empty:
            raise ValueError(f"period {name!r} has no valid windows")
        return out

    train, cal, ev = period("train"), period("calibration"), period("evaluation")
    for name in ("train", "calibration"):
        lo, hi = (pd.Timestamp(b) for b in config.periods[name])
        if any(lo <= pd.Timestamp(r.start) < hi for r in reports):
            raise ValueError("training/calibration periods must not contain reported failures")
    train_fit, cal_fit = train[fit_ok.loc[train.index]], cal[fit_ok.loc[cal.index]]
    if train_fit.empty or cal_fit.empty:
        raise ValueError("no windows left to fit or calibrate after the reference exclusions")

    x_train = train_fit[list(FEATURE_NAMES)]
    profile = FeatureProfile().fit(x_train)
    early = pd.Timedelta(hours=config.early_horizon_hours)
    post = pd.Timedelta(hours=config.post_tolerance_hours)
    suppress = ev["stopped"].to_numpy(dtype=bool)
    ev_index = pd.DatetimeIndex(ev.index)
    report_ids = [r.report_id for r in reports]
    version = f"{datetime.now(UTC):%Y%m%d}-{config.fingerprint()}"

    results: dict[str, Any] = {"detectors": {}}
    bundles: dict[str, ModelBundle] = {}
    all_candidates: dict[str, list[Candidate]] = {}
    fitted: dict[str, tuple[Detector, np.ndarray, np.ndarray, np.ndarray, Candidate]] = {}
    cases: list[FaultCase] | None = None

    for name, dcfg in config.detectors.items():
        candidates: list[Candidate] = []
        variant_scores: list[tuple[Detector, np.ndarray, np.ndarray, np.ndarray]] = []
        for v, params in enumerate(dcfg.variants):
            det = build_detector(name, params, config.seed).fit(x_train)
            ref = det.score(cal_fit[list(FEATURE_NAMES)])
            tr = det.score(x_train)
            evs = det.score(ev[list(FEATURE_NAMES)])
            variant_scores.append((det, tr, ref, evs))
            if dcfg.threshold.mode == "quantile":
                thresholds = [(f"q{q}", float(np.quantile(ref, q))) for q in dcfg.threshold.grid]
            else:
                thresholds = [(f"{g}", float(g)) for g in dcfg.threshold.grid]
            for label, thr in thresholds:
                for k in config.persistence_grid:
                    res = evaluate(
                        ev_index,
                        evs,
                        window,
                        AlarmPolicy(thr, k),
                        reports,
                        early_horizon=early,
                        post_tolerance=post,
                        suppress=suppress,
                    )
                    candidates.append(Candidate(name, v, params, label, thr, k, res))
        all_candidates[name] = candidates
        final = select_policy(candidates)
        det, tr, ref, evs = variant_scores[final.variant]
        fitted[name] = (det, tr, ref, evs, final)

    summary: dict[str, dict[str, Any]] = {}
    fault_results: dict[str, dict[str, Any]] = {}
    for name, (det, _tr, _ref, _evs, final) in fitted.items():
        summary[name] = _loeo_summary(all_candidates[name], report_ids)
        if name in ML_DETECTORS or name == "duty_rule":
            if cases is None:
                cases = build_cases(
                    telemetry,
                    pd.concat([train, cal, ev]),
                    spec,
                    reports,
                    n_per_type=config.faults.n_per_type,
                    duration_hours=config.faults.duration_hours,
                    min_gap_from_reports_hours=config.faults.min_gap_from_reports_hours,
                    seed=config.faults.seed,
                    period=(
                        pd.Timestamp(config.periods["evaluation"][0]),
                        pd.Timestamp(config.periods["evaluation"][1]),
                    ),
                )
            fault_results[name] = _fault_detection(
                det, AlarmPolicy(final.threshold, final.persistence), cases, window, spec
            )

    default, rationale = choose_default(summary, fault_results, config.selection)
    far = _far_from_reports(ev_index, reports, window, config)
    scored: dict[str, pd.DataFrame] = {
        "train": train_fit[["n_samples", "stopped", "load_duty", "off_fraction"]].copy(),
        "calibration": cal_fit[["n_samples", "stopped", "load_duty", "off_fraction"]].copy(),
        "evaluation": ev[["n_samples", "stopped", "load_duty", "off_fraction"]].copy(),
    }
    for name, (det, tr, ref, evs, final) in fitted.items():
        calibrator = ExceedanceCalibrator.fit(ref)
        scored["train"][name] = tr
        scored["calibration"][name] = ref
        scored["evaluation"][name] = evs
        scored["evaluation"][f"p_{name}"] = calibrator.p_value(evs)
        results["detectors"][name] = {
            "final_policy": {
                "variant": final.params,
                "threshold": final.threshold,
                "threshold_label": final.threshold_label,
                "persistence": final.persistence,
                "exceedance_at_threshold": float(calibrator.p_value(final.threshold)[0]),
                "selected_on": "all four reports (in-sample)",
            },
            "loeo": summary[name],
            "in_sample": final.result.to_dict(),
            "fault_injection": fault_results.get(name),
            "reference_holdout_check": _holdout_check(ref, config.exceedance_alpha),
            "exceedance_check": _exceedance_check(
                calibrator, evs, ev_index, far, ev["stopped"].to_numpy(dtype=bool), config
            ),
            "grid_size": len(all_candidates[name]),
        }
        policy = AlarmPolicy(final.threshold, final.persistence)
        bundles[name] = ModelBundle(
            version=f"{version}-{name}",
            detector_name=name,
            detector=det,
            profile=profile,
            calibrator=calibrator,
            spec=spec,
            policy=policy,
            feature_names=FEATURE_NAMES,
        )
    results.update(
        {
            "experiment": config.experiment,
            "protocol": "v2: nested leave-one-event-out, label-free calibration, state-aware",
            "config_fingerprint": config.fingerprint(),
            "version": version,
            "selected_detector": default,
            "selection_rationale": rationale,
            "periods": {
                "train": {"range": list(config.periods["train"]), "windows": len(train_fit)},
                "calibration": {
                    "range": list(config.periods["calibration"]),
                    "windows": len(cal_fit),
                },
                "evaluation": {
                    "range": list(config.periods["evaluation"]),
                    "windows": len(ev),
                    "stopped_windows": int(ev["stopped"].sum()),
                },
            },
            "reference_exclusions": config.reference.model_dump(),
            "excluded_from_fit": {
                "train": int(len(train) - len(train_fit)),
                "calibration": int(len(cal) - len(cal_fit)),
            },
            "reports": report_ids,
        }
    )
    for det_name, bundle in bundles.items():
        bundle.card = _model_card(config, results, det_name, bundle)
    return RunOutput(results, bundles, scored)


def _far_from_reports(
    index: pd.DatetimeIndex,
    reports: tuple[metropt3.FailureReport, ...],
    window: pd.Timedelta,
    config: ExperimentConfig,
) -> np.ndarray:
    near = np.zeros(len(index), dtype=bool)
    margin = pd.Timedelta(hours=config.post_tolerance_hours)
    for r in reports:
        near |= (index >= pd.Timestamp(r.start) - margin) & (index <= pd.Timestamp(r.end) + margin)
    return ~near


def _holdout_check(ref: np.ndarray, alpha: float) -> dict[str, Any]:
    """Calibration mechanics: fit on the first half of the reference, test on the second half.

    Inside one stationary normal period the share at or below ``alpha`` should be close to
    ``alpha``. It cannot show robustness to drift; see ``exceedance_check`` for that.
    """
    half = len(ref) // 2
    if half < 20:
        return {"alpha": alpha, "share_at_or_below_alpha": None, "windows": len(ref) - half}
    calibrator = ExceedanceCalibrator.fit(ref[:half])
    share = float(np.mean(calibrator.p_value(ref[half:]) <= alpha))
    return {"alpha": alpha, "share_at_or_below_alpha": round(share, 4), "windows": len(ref) - half}


def _exceedance_check(
    calibrator: ExceedanceCalibrator,
    scores: np.ndarray,
    index: pd.DatetimeIndex,
    far: np.ndarray,
    stopped: np.ndarray,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Share of far-from-report operating windows with exceedance <= alpha, overall and by month.

    A calibrated score should land near ``alpha`` on genuinely normal data. Values well above it
    mean the reference period under-represents later normal operation or the later period
    contains unreported anomalies; the labels cannot say which.
    """
    mask = far & ~stopped
    p = calibrator.p_value(scores[mask])
    flagged = pd.Series(p <= config.exceedance_alpha, index=index[mask])
    by_month = flagged.groupby(pd.DatetimeIndex(flagged.index).to_period("M")).mean()
    return {
        "alpha": config.exceedance_alpha,
        "share_at_or_below_alpha": round(float(flagged.mean()), 4),
        "by_month": {str(k): round(float(v), 4) for k, v in by_month.items()},
        "windows": int(mask.sum()),
    }


def _model_card(
    config: ExperimentConfig, results: dict[str, Any], name: str, bundle: ModelBundle
) -> dict[str, Any]:
    det = results["detectors"][name]
    return {
        "schema_version": 2,
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
        "training_period": list(config.periods["train"]),
        "calibration_period": list(config.periods["calibration"]),
        "reference_exclusions": config.reference.model_dump(),
        "window": asdict(bundle.spec),
        "threshold": bundle.policy.threshold,
        "persistence": bundle.policy.persistence,
        "exceedance_at_threshold": det["final_policy"]["exceedance_at_threshold"],
        "feature_names": list(bundle.feature_names),
        "config": json.loads(config.model_dump_json()),
        "evaluation": {
            "protocol": results["protocol"],
            "loeo": det["loeo"],
            "fault_injection": det["fault_injection"],
            "selected_as_default": results["selected_detector"] == name,
            "selection_rationale": results["selection_rationale"],
        },
        "intended_use": "Advisory early warning of abnormal compressor behaviour for human "
        "review. Exceedance is the share of reference-normal windows scoring at least as high; it "
        "is not a failure probability or a root cause.",
        "limitations": [
            "Trained and evaluated on one compressor (MetroPT-3); transfer to other assets is "
            "untested.",
            "Four coarse air-leak reports are the only labels; unreported anomalies count as false "
            "alarms and the deployed policy is selected on all four reports (in-sample).",
            "Report timestamps' timezone is not stated by the source.",
            "Fault-injection results are simulated, not field evidence.",
        ],
    }


def save(
    output: RunOutput, artifacts_dir: Path, model_name: str, *, extra: dict[str, Any] | None = None
) -> Path:
    """Persist results, per-window scores and model bundles; mark the default as CURRENT."""
    results = output.results if extra is None else {**output.results, **extra}
    run_dir = artifacts_dir / "experiments" / str(results["experiment"]) / str(results["version"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, default=str), "utf-8")
    for split, frame in output.scored.items():
        frame.to_parquet(run_dir / f"scores_{split}.parquet")

    model_root = artifacts_dir / "models" / model_name
    for bundle in output.bundles.values():
        out = model_root / bundle.version
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, out / "model.joblib")
        digest = hashlib.sha256((out / "model.joblib").read_bytes()).hexdigest()
        card = {**bundle.card, "artifact_sha256": digest}
        (out / "model_card.json").write_text(json.dumps(card, indent=2, default=str), "utf-8")
    selected = str(output.bundles[str(results["selected_detector"])].version)
    (model_root / "CURRENT").write_text(selected + "\n", encoding="utf-8")
    return Path(run_dir)


def run_sensitivity(config: ExperimentConfig, telemetry: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Re-run without the extreme-duty reference exclusion to show how much it matters."""
    variant = config.model_copy(deep=True)
    variant.reference = ReferenceConfig(
        exclude_stopped=config.reference.exclude_stopped, max_load_duty=None
    )
    out = run(variant, telemetry).results
    return {
        "no_extreme_duty_exclusion": {
            "selected_detector": out["selected_detector"],
            "loeo": {
                d: {
                    "events_detected": v["loeo"]["events_detected"],
                    "mean_false_alarms_per_day": v["loeo"]["mean_false_alarms_per_day"],
                }
                for d, v in out["detectors"].items()
            },
        }
    }
