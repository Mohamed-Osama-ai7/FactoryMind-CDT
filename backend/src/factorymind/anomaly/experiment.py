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

Disclosed limits: LOEO policy selection is non-chronological (for earlier events such as F1,
selection uses later reports F2-F4); the normal time used to estimate false alarms is shared by
selection and evaluation and includes stopped windows in ``valid_days`` (``operating_days`` is
reported alongside); the final deployed policy is selected on all four events (in-sample); and
reference-normal windows drift within the two-week calibration window.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
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
from factorymind.anomaly.alarm import AlarmPolicy, AlarmTracker, Episode, episodes
from factorymind.anomaly.calibration import ExceedanceCalibrator
from factorymind.anomaly.detectors import Detector, DetectorName, FeatureProfile, build_detector
from factorymind.anomaly.evaluation import EvaluationResult, evaluate, is_outside_report_windows
from factorymind.anomaly.events import (
    build_warning_episodes,
    primary_leak_scorable,
    summarize_episode_regimes,
    warning_scorable,
)
from factorymind.anomaly.faults import FAULT_TYPES, FaultCase, build_cases
from factorymind.anomaly.features import FEATURE_NAMES, WindowSpec, window_features
from factorymind.anomaly.regimes import OperatingRegime, RegimeAssessment, classify_sequence
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
    """Best candidate by the pre-registered criterion, ignoring one report if given.

    Note: for leakage-free LOEO selection, callers must evaluate ``candidates`` with the held-out
    report omitted from ``reports`` so that it does not mask false alarms in
    ``false_alarms_per_day`` (relying only on ``exclude_event`` would still mask held-out alarms).
    """
    return min(candidates, key=lambda c: _selection_key(c, exclude_event))


def _median_abs_delay(delays: list[float]) -> float | None:
    return round(float(np.median([abs(d) for d in delays])), 1) if delays else None


def _candidate_key(c: Candidate) -> tuple[int, str, int]:
    return (c.variant, c.threshold_label, c.persistence)


def _loeo_summary(
    fold_candidates: dict[str, list[Candidate]],
    full_candidates: list[Candidate],
    reports: tuple[metropt3.FailureReport, ...],
    ev_index: pd.DatetimeIndex,
    variant_evs: list[np.ndarray],
    window: pd.Timedelta,
    early: pd.Timedelta,
    post: pd.Timedelta,
    suppress: np.ndarray,
    episodes_by_key: dict[tuple[int, str, int], list[Episode]],
) -> dict[str, Any]:
    full_by_key = {_candidate_key(c): c for c in full_candidates}
    folds: list[dict[str, Any]] = []
    for r in reports:
        # Policy selection sees ONLY other reports (both event detection and FA masking).
        chosen = select_policy(fold_candidates[r.report_id])
        key = _candidate_key(chosen)
        # Evaluate the selected policy on the held-out report separately.
        held_eval = evaluate(
            ev_index,
            variant_evs[chosen.variant],
            window,
            AlarmPolicy(chosen.threshold, chosen.persistence),
            (r,),
            early_horizon=early,
            post_tolerance=post,
            suppress=suppress,
            precomputed_episodes=episodes_by_key[key],
        )
        held = held_eval.events[0]
        full = full_by_key[key]
        folds.append(
            {
                "held_out": r.report_id,
                "variant": chosen.params,
                "threshold": chosen.threshold_label,
                "persistence": chosen.persistence,
                "detected": held.detected,
                "delay_minutes": held.delay_minutes,
                "false_alarms_per_day": full.result.false_alarms_per_day,
                "operating_false_alarms_per_day": full.result.operating_false_alarms_per_day,
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
        "mean_operating_false_alarms_per_day": round(
            float(np.mean([f["operating_false_alarms_per_day"] for f in folds])), 3
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
    faulted_hits: list[bool] = []
    for case in cases:
        hit = raised(case.faulted)
        ctrl = raised(case.control)
        per_type[case.fault].append(hit)
        faulted_hits.append(hit)
        control.append(ctrl)
    total = sum(faulted_hits)
    control_total = sum(control)
    n = len(cases)
    clean_cases = n - control_total
    clean_detected = sum((not c) and f for c, f in zip(control, faulted_hits, strict=True))
    return {
        "detected": total,
        "cases": n,
        "gross_rate": round(total / n, 3) if n else None,
        "control_detected": control_total,
        "control_alarm_rate": round(control_total / n, 3) if n else None,
        "control_alarms": f"{control_total}/{n}",
        "clean_control_cases": clean_cases,
        "clean_control_detected": clean_detected,
        "clean_control_sensitivity": round(clean_detected / clean_cases, 3)
        if clean_cases
        else None,
        "incremental_lift": round((total - control_total) / n, 3) if n else None,
        "per_type": {f: f"{sum(v)}/{len(v)}" for f, v in per_type.items()},
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
        clean_sensitivity = fault.get("clean_control_sensitivity")
        incremental_lift = fault.get("incremental_lift")
        sees_faults = (
            clean_sensitivity is not None
            and clean_sensitivity >= cfg.ml_fault_detection_min
            and (incremental_lift or 0.0) > 0.0
            and s["mean_false_alarms_per_day"] <= cfg.ml_fault_fa_max
            and s["events_detected"] >= bar["events_detected"]
        )
        if matches_rule:
            eligible.append((name, f"matches {best_rule} on leaks at no worse false alarms"))
        elif sees_faults:
            eligible.append(
                (
                    name,
                    f"detects injected faults ({clean_sensitivity:.0%} clean-control sensitivity) "
                    "and matches leak detection",
                )
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


_REGIME_KEYS: tuple[OperatingRegime, ...] = (
    "loaded",
    "unloaded",
    "starting",
    "stopped",
    "insufficient_evidence",
)
_INSUFFICIENT_REASONS: tuple[str, ...] = (
    "mixed_duty_cycle",
    "hysteresis_hold",
    "stale_telemetry",
    "signal_inconsistency",
    "incomplete_or_gapped",
)


def _evaluate_loeo_detector(
    name: str,
    dcfg: DetectorConfig,
    config: ExperimentConfig,
    ev_index: pd.DatetimeIndex,
    reports: tuple[metropt3.FailureReport, ...],
    window: pd.Timedelta,
    early: pd.Timedelta,
    post: pd.Timedelta,
    variant_scores: list[tuple[Detector, np.ndarray, np.ndarray, np.ndarray]],
    suppress: np.ndarray,
) -> tuple[
    list[Candidate],
    dict[str, list[Candidate]],
    dict[tuple[int, str, int], list[Episode]],
    dict[str, Any],
]:
    candidates: list[Candidate] = []
    fold_candidates: dict[str, list[Candidate]] = {r.report_id: [] for r in reports}
    episodes_by_key: dict[tuple[int, str, int], list[Episode]] = {}
    for v, params in enumerate(dcfg.variants):
        _det, _tr, ref, evs = variant_scores[v]
        if dcfg.threshold.mode == "quantile":
            thresholds = [(f"q{q}", float(np.quantile(ref, q))) for q in dcfg.threshold.grid]
        else:
            thresholds = [(f"{g}", float(g)) for g in dcfg.threshold.grid]
        for label, thr in thresholds:
            for k in config.persistence_grid:
                policy = AlarmPolicy(thr, k)
                eps = episodes(ev_index, evs, window, policy, suppress)
                episodes_by_key[(v, label, k)] = eps
                res = evaluate(
                    ev_index,
                    evs,
                    window,
                    policy,
                    reports,
                    early_horizon=early,
                    post_tolerance=post,
                    suppress=suppress,
                    precomputed_episodes=eps,
                )
                candidates.append(Candidate(name, v, params, label, thr, k, res))
                for r in reports:
                    other_reports = tuple(
                        other for other in reports if other.report_id != r.report_id
                    )
                    sel_res = evaluate(
                        ev_index,
                        evs,
                        window,
                        policy,
                        other_reports,
                        early_horizon=early,
                        post_tolerance=post,
                        suppress=suppress,
                        precomputed_episodes=eps,
                    )
                    fold_candidates[r.report_id].append(
                        Candidate(name, v, params, label, thr, k, sel_res)
                    )
    loeo = _loeo_summary(
        fold_candidates,
        candidates,
        reports,
        ev_index,
        [vs[3] for vs in variant_scores],
        window,
        early,
        post,
        suppress,
        episodes_by_key,
    )
    return candidates, fold_candidates, episodes_by_key, loeo


def _regime_split_summary(
    raw_period: pd.DataFrame,
    regimes: Sequence[RegimeAssessment],
) -> dict[str, Any]:
    counts: Counter[OperatingRegime] = Counter(r.regime for r in regimes)
    reasons: Counter[str] = Counter(
        r.reason for r in regimes if r.regime == "insufficient_evidence"
    )
    invalid_gapped = int((~raw_period["valid"]).sum())
    return {
        "valid_windows": len(regimes),
        "invalid_gapped_windows": invalid_gapped,
        "counts": {k: int(counts[k]) for k in _REGIME_KEYS},
        "insufficient_reasons": {r: int(reasons[r]) for r in _INSUFFICIENT_REASONS if reasons[r]},
    }


def _episodes_overlap(start_a: pd.Timestamp, end_a: pd.Timestamp, b: Sequence[Episode]) -> bool:
    return any(ep.start < end_a and ep.end > start_a for ep in b)


def _evaluate_wp2(
    *,
    config: ExperimentConfig,
    spec: WindowSpec,
    window: pd.Timedelta,
    early: pd.Timedelta,
    post: pd.Timedelta,
    raw_windows: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    regimes_by_split: dict[str, list[RegimeAssessment]],
    default: str,
    bundles: dict[str, ModelBundle],
    scored: dict[str, pd.DataFrame],
    all_variant_scores: dict[str, list[tuple[Detector, np.ndarray, np.ndarray, np.ndarray]]],
    wp0_summary: dict[str, dict[str, Any]],
    fault_results: dict[str, dict[str, Any]],
    reports: tuple[metropt3.FailureReport, ...],
) -> dict[str, Any]:
    ev = splits["evaluation"]
    ev_index = pd.DatetimeIndex(ev.index)
    eval_regimes = regimes_by_split["evaluation"]
    valid_arr = ev["valid"].to_numpy(dtype=bool)
    stopped_arr = ev["stopped"].to_numpy(dtype=bool)

    primary_bundle = bundles[default]
    primary_scores = scored["evaluation"][default].to_numpy(dtype=float)
    leak_scorable = np.array(
        [
            primary_leak_scorable(bool(v), bool(s), r)
            for v, s, r in zip(valid_arr, stopped_arr, eval_regimes, strict=True)
        ],
        dtype=bool,
    )
    integrity_suppress = ~leak_scorable

    wp0_eps = episodes(ev_index, primary_scores, window, primary_bundle.policy, stopped_arr)
    wp0_eval = evaluate(
        ev_index,
        primary_scores,
        window,
        primary_bundle.policy,
        reports,
        early_horizon=early,
        post_tolerance=post,
        suppress=stopped_arr,
        precomputed_episodes=wp0_eps,
    )
    gated_eps = episodes(
        ev_index, primary_scores, window, primary_bundle.policy, integrity_suppress
    )
    gated_eval = evaluate(
        ev_index,
        primary_scores,
        window,
        primary_bundle.policy,
        reports,
        early_horizon=early,
        post_tolerance=post,
        suppress=integrity_suppress,
        precomputed_episodes=gated_eps,
    )

    fa_leak_eps = [
        ep
        for ep in gated_eps
        if is_outside_report_windows(
            ep.start, ep.end, reports, early_horizon=early, post_tolerance=post
        )
    ]
    fa_dom_regimes: list[OperatingRegime] = []
    starting_overlap_fa = 0
    mixed_overlap_fa = 0
    for ep in fa_leak_eps:
        dom, _ = summarize_episode_regimes(eval_regimes, ep.start, ep.end)
        fa_dom_regimes.append(dom)
        ep_regs = [r for r in eval_regimes if ep.start <= r.window_start < ep.end]
        if any(r.regime == "starting" for r in ep_regs):
            starting_overlap_fa += 1
        if any(r.reason == "mixed_duty_cycle" for r in ep_regs):
            mixed_overlap_fa += 1
    fa_by_dom = Counter(fa_dom_regimes)

    warning_bundle = bundles.get("pca_spe")
    warn_scorable_flags = [
        warning_scorable(bool(v), bool(s), r)
        for v, s, r in zip(valid_arr, stopped_arr, eval_regimes, strict=True)
    ]
    warn_scores: list[float | None] = [None] * len(ev)
    warn_excs: list[float | None] = [None] * len(ev)
    if warning_bundle is not None:
        raw_warn = scored["evaluation"]["pca_spe"].to_numpy(dtype=float)
        raw_exc = scored["evaluation"]["p_pca_spe"].to_numpy(dtype=float)
        for i, can in enumerate(warn_scorable_flags):
            if can:
                warn_scores[i] = float(raw_warn[i])
                warn_excs[i] = float(raw_exc[i])

    ev_timestamps = [pd.Timestamp(str(ts)) for ts in ev_index]
    warn_eps = build_warning_episodes(
        window=window,
        timestamps=ev_timestamps,
        regimes=eval_regimes,
        warning_scores=warn_scores,
        warning_exceedances=warn_excs,
        warning_scorable_flags=warn_scorable_flags,
        warning_policy=warning_bundle.policy if warning_bundle else None,
        warning_detector_name=warning_bundle.detector_name if warning_bundle else None,
        range_start=ev_timestamps[0],
        range_end=ev_timestamps[-1] + window,
    )
    warn_kind_counts = Counter(w.warning_kind for w in warn_eps)
    warn_overlap_count = sum(1 for w in warn_eps if _episodes_overlap(w.start, w.end, gated_eps))
    warn_only_count = len(warn_eps) - warn_overlap_count
    valid_days = len(ev_index) * window.total_seconds() / 86400.0
    operating_days = int((~stopped_arr).sum()) * window.total_seconds() / 86400.0

    per_regime: dict[str, Any] = {}
    warn_scores_arr = np.array([s if s is not None else np.nan for s in warn_scores], dtype=float)
    warn_scorable_arr = np.asarray(warn_scorable_flags, dtype=bool)
    for reg_key in _REGIME_KEYS:
        reg_mask = np.array([r.regime == reg_key for r in eval_regimes], dtype=bool)
        p_mask = reg_mask & leak_scorable
        p_scored = int(p_mask.sum())
        p_exc = (
            int((primary_scores[p_mask] > primary_bundle.policy.threshold).sum()) if p_scored else 0
        )
        fa_overlap = sum(
            1
            for ep in fa_leak_eps
            if any(
                r.regime == reg_key and ep.start <= r.window_start < ep.end for r in eval_regimes
            )
        )
        warn_block: dict[str, Any] | None = None
        if warning_bundle is not None:
            w_mask = reg_mask & warn_scorable_arr
            w_scored = int(w_mask.sum())
            w_exc = (
                int((warn_scores_arr[w_mask] > warning_bundle.policy.threshold).sum())
                if w_scored
                else 0
            )
            warn_block = {
                "scored_windows": w_scored,
                "exceedance_windows": w_exc,
                "exceedance_rate": round(w_exc / w_scored, 4) if w_scored else None,
                "warning_episodes": sum(1 for w in warn_eps if w.dominant_regime == reg_key),
            }
        per_regime[reg_key] = {
            "windows": int(reg_mask.sum()),
            "share": round(float(reg_mask.mean()), 4) if len(eval_regimes) else 0.0,
            "primary": {
                "scored_windows": p_scored,
                "exceedance_windows": p_exc,
                "exceedance_rate": round(p_exc / p_scored, 4) if p_scored else None,
                "false_alarm_episodes": int(fa_by_dom[reg_key]),
                "overlapping_false_alarm_episodes": fa_overlap,
            },
            "warning": warn_block,
        }

    default_cfg = config.detectors[default]  # type: ignore[index]
    _, _, _, loeo_integrity = _evaluate_loeo_detector(
        default,
        default_cfg,
        config,
        ev_index,
        reports,
        window,
        early,
        post,
        all_variant_scores[default],
        integrity_suppress,
    )
    starting_mask = np.array([r.regime == "starting" for r in eval_regimes], dtype=bool)
    _, _, _, loeo_starting = _evaluate_loeo_detector(
        default,
        default_cfg,
        config,
        ev_index,
        reports,
        window,
        early,
        post,
        all_variant_scores[default],
        integrity_suppress | starting_mask,
    )

    regime_dist: dict[str, Any] = {}
    for split_name in splits:
        lo, hi = (pd.Timestamp(b) for b in config.periods[split_name])
        raw_slice = raw_windows[(raw_windows.index >= lo) & (raw_windows.index < hi)]
        regime_dist[split_name] = _regime_split_summary(raw_slice, regimes_by_split[split_name])

    sim_comparison: dict[str, Any] = {}
    for det_name in (default, "pca_spe"):
        if det_name in fault_results and fault_results[det_name] is not None:
            fr = fault_results[det_name]
            sim_comparison[det_name] = {
                "detected": fr["detected"],
                "cases": fr["cases"],
                "control_detected": fr["control_detected"],
                "clean_control_sensitivity": fr["clean_control_sensitivity"],
                "incremental_lift": fr["incremental_lift"],
                "per_type": fr["per_type"],
            }

    return {
        "primary_detector": default,
        "warning_detector": warning_bundle.detector_name if warning_bundle else None,
        "regime_distribution": regime_dist,
        "event_separation": {
            "baseline_leak_episodes_wp0": wp0_eval.alarm_episodes,
            "baseline_false_alarm_episodes_wp0": wp0_eval.false_alarm_episodes,
            "gated_leak_episodes_wp2": gated_eval.alarm_episodes,
            "gated_false_alarm_episodes_wp2": gated_eval.false_alarm_episodes,
            "integrity_suppressed_leak_episodes": (
                wp0_eval.false_alarm_episodes - gated_eval.false_alarm_episodes
            ),
            "events_detected_wp2": gated_eval.events_detected,
            "events_total": gated_eval.events_total,
            "false_alarms_per_day_wp2": gated_eval.false_alarms_per_day,
            "operating_false_alarms_per_day_wp2": gated_eval.operating_false_alarms_per_day,
            "sensor_consistency_warning_episodes": len(warn_eps),
            "warnings_per_day": round(len(warn_eps) / valid_days, 3) if valid_days else 0.0,
            "operating_warnings_per_day": (
                round(len(warn_eps) / operating_days, 3) if operating_days else 0.0
            ),
            "warning_only_episodes": warn_only_count,
            "warning_overlapping_leak_episodes": warn_overlap_count,
            "warning_kinds": {k: int(v) for k, v in sorted(warn_kind_counts.items())},
            "false_alarms_by_dominant_regime": {k: int(fa_by_dom[k]) for k in _REGIME_KEYS},
            "starting_overlapping_false_alarms": starting_overlap_fa,
            "mixed_cycle_overlapping_false_alarms": mixed_overlap_fa,
        },
        "per_regime_metrics": per_regime,
        "regime_gating_ablations": {
            "wp0_stopped_only": {
                **wp0_summary[default],
                "suppressed_states": ["stopped"],
                "note": "WP0 baseline: only stopped windows suppressed.",
            },
            "wp2_integrity_gated": {
                **loeo_integrity,
                "suppressed_states": ["stopped", "stale_telemetry", "signal_inconsistency"],
                "note": (
                    "WP2 integrity gating: abstains primary air-leak detector on frozen or "
                    "contradictory telemetry."
                ),
            },
            "starting_suppressed_ablation": {
                **loeo_starting,
                "suppressed_states": [
                    "stopped",
                    "stale_telemetry",
                    "signal_inconsistency",
                    "starting",
                ],
                "note": (
                    f"Starting-regime suppression changed LOEO operating FA/day from "
                    f"{loeo_integrity['mean_operating_false_alarms_per_day']:.3f} to "
                    f"{loeo_starting['mean_operating_false_alarms_per_day']:.3f} with "
                    f"{loeo_starting['events_detected']}/{loeo_starting['events_total']} events; "
                    "not enabled in default leak detection unless operating FA/day strictly "
                    "improves."
                ),
            },
        },
        "simulated_fault_separation": {
            "simulated": True,
            "real_operational_data": False,
            "note": (
                "Simulated single-sensor faults evaluate warning-tier separation; "
                "never count toward real air-leak failure metrics."
            ),
            "detectors": sim_comparison,
        },
    }


def run(
    config: ExperimentConfig,
    telemetry: pd.DataFrame,
    reports: tuple[metropt3.FailureReport, ...] = metropt3.FAILURE_REPORTS,
) -> RunOutput:
    spec = config.window.spec()
    window = pd.Timedelta(spec.length)
    raw_windows = window_features(telemetry, spec)
    usable, fit_ok = _usable_masks(raw_windows, config)

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
    all_variant_scores: dict[str, list[tuple[Detector, np.ndarray, np.ndarray, np.ndarray]]] = {}
    fitted: dict[str, tuple[Detector, np.ndarray, np.ndarray, np.ndarray, Candidate]] = {}
    cases: list[FaultCase] | None = None
    summary: dict[str, dict[str, Any]] = {}
    fault_results: dict[str, dict[str, Any]] = {}

    for name, dcfg in config.detectors.items():
        variant_scores: list[tuple[Detector, np.ndarray, np.ndarray, np.ndarray]] = []
        for params in dcfg.variants:
            det = build_detector(name, params, config.seed).fit(x_train)
            ref = det.score(cal_fit[list(FEATURE_NAMES)])
            tr = det.score(x_train)
            evs = det.score(ev[list(FEATURE_NAMES)])
            variant_scores.append((det, tr, ref, evs))
        all_variant_scores[name] = variant_scores
        candidates, _, _, loeo_dict = _evaluate_loeo_detector(
            name,
            dcfg,
            config,
            ev_index,
            reports,
            window,
            early,
            post,
            variant_scores,
            suppress,
        )
        all_candidates[name] = candidates
        summary[name] = loeo_dict
        final = select_policy(candidates)
        det, tr, ref, evs = variant_scores[final.variant]
        fitted[name] = (det, tr, ref, evs, final)

    for name, (det, _tr, _ref, _evs, final) in fitted.items():
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
    splits = {"train": train, "calibration": cal, "evaluation": ev}
    regimes_by_split = {k: classify_sequence(df, spec) for k, df in splits.items()}
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
            "wp2": _evaluate_wp2(
                config=config,
                spec=spec,
                window=window,
                early=early,
                post=post,
                raw_windows=raw_windows,
                splits=splits,
                regimes_by_split=regimes_by_split,
                default=default,
                bundles=bundles,
                scored=scored,
                all_variant_scores=all_variant_scores,
                wp0_summary=summary,
                fault_results=fault_results,
                reports=reports,
            ),
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
            "reference_holdout_check": det["reference_holdout_check"],
            "exceedance_check": det["exceedance_check"],
            "selected_as_default": results["selected_detector"] == name,
            "selection_rationale": results["selection_rationale"],
            "wp2": results.get("wp2"),
        },
        "intended_use": "Advisory early warning of abnormal compressor behaviour for human "
        "review. Exceedance is the share of reference-normal windows scoring at least as high; it "
        "is not a failure probability or a root cause.",
        "limitations": [
            "Trained and evaluated on one compressor (MetroPT-3); transfer to other assets is "
            "untested.",
            "Four coarse air-leak reports are the only labels; LOEO policy selection is "
            "non-chronological (later reports inform earlier fold choices), shares evaluation "
            "normal time across folds, and deploys the in-sample policy selected on all four "
            "reports.",
            "The calibration reference distribution is non-stationary (see reference holdout "
            "exceedance check); exceedance is relative to that reference, never a failure "
            "probability.",
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
