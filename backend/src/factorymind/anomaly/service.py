"""Inference service: loads the current model bundle and scores telemetry.

Advisory only: results are anomaly scores and evidence for human review. Nothing here writes to
industrial systems or triggers actions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from factorymind.anomaly.alarm import AlarmTracker, Episode
from factorymind.anomaly.events import (
    EventClass,
    WarningEpisode,
    WindowStatus,
    build_warning_episodes,
    classify_window_event,
    episode_hypothesis_and_uncertainty,
    primary_leak_scorable,
    summarize_episode_regimes,
    warning_scorable,
)
from factorymind.anomaly.evidence import EvidenceItem, build_evidence
from factorymind.anomaly.experiment import ModelBundle
from factorymind.anomaly.features import block_features, window_features
from factorymind.anomaly.regimes import (
    OperatingRegime,
    RegimeAssessment,
    classify_sequence,
    classify_window,
)
from factorymind.data import metropt3

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
    regime: RegimeAssessment
    event_class: EventClass
    event_meaning: str
    warning_active: bool = False
    warning_score: float | None = None
    warning_exceedance: float | None = None
    warning_detector: str | None = None
    warning_reasons: tuple[str, ...] = ()
    warning_evidence: list[EvidenceItem] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayEpisode:
    episode: Episode
    truncated: bool  # the run may have started before the scanned range
    ongoing: bool  # still anomalous at the end of the requested range
    event_class: EventClass = "air_leak_alarm"
    hypothesis: str = ""
    uncertainty_note: str = ""
    dominant_regime: OperatingRegime = "insufficient_evidence"
    regime_breakdown: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ReplayOutput:
    """Complete WP2 replay output separating air-leak alarms, warnings, and regimes."""

    windows: list[WindowScore]
    episodes: list[ReplayEpisode]
    warnings: list[WarningEpisode]
    reports: list[metropt3.FailureReport]


def _load_verified_bundle_dir(model_dir: Path, version: str) -> tuple[ModelBundle, dict[str, Any]]:
    artifact, card_path = model_dir / "model.joblib", model_dir / "model_card.json"
    if not artifact.exists() or not card_path.exists():
        raise ModelUnavailableError(f"Model {version} is incomplete")
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ModelUnavailableError(f"Model {version} has an unsupported format") from exc
    if not isinstance(card, dict):
        raise ModelUnavailableError(f"Model {version} has an unsupported format")
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


def load_bundle(models_dir: Path, model_name: str) -> tuple[ModelBundle, dict[str, Any]]:
    root = models_dir / model_name
    pointer = root / "CURRENT"
    if not pointer.exists():
        raise ModelUnavailableError("No current model is available; run the training pipeline")
    version = pointer.read_text(encoding="utf-8").strip()
    return _load_verified_bundle_dir(root / version, version)


def load_warning_bundle(
    models_dir: Path,
    model_name: str,
    primary_bundle: ModelBundle,
    warning_detector: str = "pca_spe",
) -> ModelBundle | None:
    """Load the sibling ``pca_spe`` warning bundle sharing ``primary_bundle``'s run prefix."""
    if primary_bundle.detector_name == warning_detector:
        return None
    suffix = f"-{primary_bundle.detector_name}"
    if not primary_bundle.version.endswith(suffix):
        return None
    prefix = primary_bundle.version[: -len(suffix)]
    sibling_version = f"{prefix}-{warning_detector}"
    sibling_dir = models_dir / model_name / sibling_version
    if not sibling_dir.exists():
        return None
    warning_bundle, _ = _load_verified_bundle_dir(sibling_dir, sibling_version)
    return warning_bundle


class AnomalyService:
    def __init__(
        self,
        bundle: ModelBundle,
        card: dict[str, Any],
        processed_dir: Path,
        reports: tuple[metropt3.FailureReport, ...] = metropt3.FAILURE_REPORTS,
        *,
        warning_bundle: ModelBundle | None = None,
    ) -> None:
        self.bundle = bundle
        self.card = card
        self._processed_dir = processed_dir
        self._reports = reports
        self._warning_bundle = warning_bundle

    @property
    def warning_bundle(self) -> ModelBundle | None:
        """Optional secondary detector bundle used strictly for consistency warnings."""
        return self._warning_bundle

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

    def _warning_evidence(self, row: pd.Series) -> list[EvidenceItem]:
        if self._warning_bundle is not None:
            return build_evidence(
                row,
                self._warning_bundle.profile,
                self._warning_bundle.feature_names,
            )
        return self._evidence(row)

    def _score_rows(self, rows: pd.DataFrame, bundle: ModelBundle | None = None) -> np.ndarray:
        target = bundle or self.bundle
        return target.detector.score(rows[list(target.feature_names)])

    def _window(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        row: pd.Series,
        regime: RegimeAssessment,
        score: float | None,
        warning_score: float | None,
        warning_exceedance: float | None,
    ) -> WindowScore:
        n = int(row["n_samples"])
        row_valid = bool(row["valid"])
        row_stopped = bool(row["stopped"])
        leak_scorable = primary_leak_scorable(row_valid, row_stopped, regime)

        exceedance: float | None = None
        primary_anomalous = False
        if leak_scorable:
            if score is None:
                raise ValueError("a valid operating window needs a score")
            exceedance = float(self.bundle.calibrator.p_value(score)[0])
            primary_anomalous = score > self.bundle.policy.threshold
        else:
            score = None

        warning_det_active = (
            warning_score is not None
            and self._warning_bundle is not None
            and warning_score > self._warning_bundle.policy.threshold
        )
        classified = classify_window_event(
            regime=regime,
            row_valid=row_valid,
            row_stopped=row_stopped,
            primary_anomalous=primary_anomalous,
            warning_detector_active=warning_det_active,
        )
        leak_evidence = self._evidence(row) if classified.status == "anomalous" else []
        warn_evidence = self._warning_evidence(row) if classified.warning_active else []
        warn_detector_label: str | None = None
        if warning_det_active and self._warning_bundle is not None and classified.integrity_warning:
            warn_detector_label = f"{self._warning_bundle.detector_name}+signal_integrity"
        elif warning_det_active and self._warning_bundle is not None:
            warn_detector_label = self._warning_bundle.detector_name
        elif classified.integrity_warning:
            warn_detector_label = "signal_integrity"
        elif warning_score is not None and self._warning_bundle is not None:
            warn_detector_label = self._warning_bundle.detector_name

        return WindowScore(
            window_start=start,
            window_end=end,
            n_samples=n,
            status=classified.status,
            score=score,
            exceedance=exceedance,
            evidence=leak_evidence,
            regime=regime,
            event_class=classified.event_class,
            event_meaning=classified.event_meaning,
            warning_active=classified.warning_active,
            warning_score=warning_score,
            warning_exceedance=warning_exceedance,
            warning_detector=warn_detector_label,
            warning_reasons=classified.warning_reasons,
            warning_evidence=warn_evidence,
        )

    def score_block(self, frame: pd.DataFrame) -> WindowScore:
        """Score one submitted block of samples as a single window."""
        row = block_features(frame, self.bundle.spec)
        start, end = pd.Timestamp(row["first_sample"]), pd.Timestamp(row["last_sample"])
        regime = classify_window(start, end, row, self.bundle.spec)
        row_valid = bool(row["valid"])
        row_stopped = bool(row["stopped"])
        row_df = row.to_frame().T

        score: float | None = None
        if primary_leak_scorable(row_valid, row_stopped, regime):
            score = float(self._score_rows(row_df)[0])

        warn_score: float | None = None
        warn_exc: float | None = None
        if self._warning_bundle is not None and warning_scorable(row_valid, row_stopped, regime):
            warn_score = float(self._score_rows(row_df, self._warning_bundle)[0])
            warn_exc = float(self._warning_bundle.calibrator.p_value(warn_score)[0])

        return self._window(start, end, row, regime, score, warn_score, warn_exc)

    def replay_details(self, start: pd.Timestamp, end: pd.Timestamp) -> ReplayOutput:
        """Score recorded telemetry in [start, end) with regime and warning separation.

        Alarm runs are tracked from ``start - REPLAY_LOOKBACK`` so an incident's start and
        raised-at time do not depend on where the requested range begins.
        """
        data = self.telemetry
        window = pd.Timedelta(self.bundle.spec.length)
        scan_start = start - REPLAY_LOOKBACK
        frame = data[(data.index >= scan_start) & (data.index < end)]
        if frame.empty:
            return ReplayOutput(windows=[], episodes=[], warnings=[], reports=[])

        features = window_features(frame, self.bundle.spec)
        regimes = classify_sequence(features, self.bundle.spec)
        valid_arr = features["valid"].to_numpy(dtype=bool)
        stopped_arr = features["stopped"].to_numpy(dtype=bool)

        leak_scorable = np.array(
            [
                primary_leak_scorable(bool(v), bool(s), r)
                for v, s, r in zip(valid_arr, stopped_arr, regimes, strict=True)
            ],
            dtype=bool,
        )
        scores = np.full(len(features), np.nan)
        if leak_scorable.any():
            scores[leak_scorable] = self._score_rows(features[leak_scorable])

        warn_scorable_flags = [
            warning_scorable(bool(v), bool(s), r)
            for v, s, r in zip(valid_arr, stopped_arr, regimes, strict=True)
        ]
        warn_scores: list[float | None] = [None] * len(features)
        warn_excs: list[float | None] = [None] * len(features)
        if self._warning_bundle is not None:
            warn_mask = np.asarray(warn_scorable_flags, dtype=bool)
            if warn_mask.any():
                raw_warn = self._score_rows(features[warn_mask], self._warning_bundle)
                raw_exc = self._warning_bundle.calibrator.p_value(raw_warn)
                for idx, w_sc, w_ex in zip(
                    np.flatnonzero(warn_mask), raw_warn, raw_exc, strict=True
                ):
                    warn_scores[int(idx)] = float(w_sc)
                    warn_excs[int(idx)] = float(w_ex)

        tracker = AlarmTracker(window, self.bundle.policy)
        closed: list[Episode] = []
        windows: list[WindowScore] = []
        timestamps = [pd.Timestamp(str(ts)) for ts in features.index]
        for i, ((_, row), stamp, regime) in enumerate(
            zip(features.iterrows(), timestamps, regimes, strict=True)
        ):
            score = None if np.isnan(scores[i]) else float(scores[i])
            events = tracker.push(stamp, score, explained=not bool(leak_scorable[i]))
            closed.extend(e.episode for e in events if e.kind == "closed")
            if stamp >= start:
                windows.append(
                    self._window(
                        stamp,
                        stamp + window,
                        row,
                        regime,
                        score,
                        warn_scores[i],
                        warn_excs[i],
                    )
                )
        ongoing = tracker.open_episode
        closed.extend(e.episode for e in tracker.finish() if e.kind == "closed")

        first_window = timestamps[0]
        episodes: list[ReplayEpisode] = []
        for ep in closed:
            if not (ep.end > start and ep.start < end):
                continue
            dominant, breakdown = summarize_episode_regimes(regimes, ep.start, ep.end)
            hypothesis, uncertainty = episode_hypothesis_and_uncertainty(
                regimes, ep.start, ep.end, event_class="air_leak_alarm"
            )
            episodes.append(
                ReplayEpisode(
                    ep,
                    truncated=ep.start <= first_window,
                    ongoing=ongoing is not None and ep.start == ongoing.start,
                    event_class="air_leak_alarm",
                    hypothesis=hypothesis,
                    uncertainty_note=uncertainty,
                    dominant_regime=dominant,
                    regime_breakdown=breakdown,
                )
            )

        warnings = build_warning_episodes(
            window=window,
            timestamps=timestamps,
            regimes=regimes,
            warning_scores=warn_scores,
            warning_exceedances=warn_excs,
            warning_scorable_flags=warn_scorable_flags,
            warning_policy=self._warning_bundle.policy if self._warning_bundle else None,
            warning_detector_name=(
                self._warning_bundle.detector_name if self._warning_bundle else None
            ),
            range_start=start,
            range_end=end,
        )
        reports = [
            r for r in self._reports if pd.Timestamp(r.start) < end and pd.Timestamp(r.end) >= start
        ]
        return ReplayOutput(
            windows=windows,
            episodes=episodes,
            warnings=warnings,
            reports=reports,
        )

    def replay(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[list[WindowScore], list[ReplayEpisode], list[metropt3.FailureReport]]:
        """Score recorded telemetry in [start, end) window by window."""
        out = self.replay_details(start, end)
        return out.windows, out.episodes, out.reports

    def data_range(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        try:
            index = pd.DatetimeIndex(self.telemetry.index)
        except DataUnavailableError:
            return None
        return pd.Timestamp(index.min()), pd.Timestamp(index.max())
