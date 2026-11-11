"""Event-centered multi-phase evidence packages, quality assessment, and timeline builder.

Builds structured, provenance-tagged :class:`~factorymind.contracts.EventEvidencePackage`
objects on top of processed telemetry and :class:`~factorymind.anomaly.service.AnomalyService`
outputs without duplicating scoring/regime logic or filling missing measurements.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from typing import Final, Literal

import numpy as np
import pandas as pd

from factorymind.anomaly.events import is_integrity_warning_regime
from factorymind.anomaly.features import (
    WindowSpec,
    anchored_window_features,
    window_features,
)
from factorymind.anomaly.regimes import OperatingRegime
from factorymind.anomaly.service import AnomalyService, ReplayOutput, WindowScore
from factorymind.contracts.events import (
    ContradictionRecord,
    DerivedFeatureSummary,
    EventEvidencePackage,
    EventPhaseName,
    EvidenceItemContract,
    EvidenceQualityAssessment,
    EvidenceQualityStatus,
    HumanObservationOut,
    NeighboringEventSummary,
    ObservedSignalSummary,
    PhaseEvidenceSummary,
    ProvenanceRecord,
    TimelineEntry,
)
from factorymind.data import metropt3

_EXPECTED_SAMPLES_PER_MIN: Final = 6  # ~0.1 Hz (10 s cadence)
_FREEZE_STD_TOLERANCE: Final = 1e-12
_KEY_ANALOG: Final[tuple[str, ...]] = (
    "TP3",
    "TP2",
    "Reservoirs",
    "Motor_current",
    "Oil_temperature",
    "DV_pressure",
    "H1",
)
_KEY_DIGITAL: Final[tuple[str, ...]] = (
    "COMP",
    "DV_eletric",
    "LPS",
    "MPG",
    "Oil_level",
    "Pressure_switch",
    "Caudal_impulses",
    "Towers",
)
_PHASE_LABELS: Final[dict[EventPhaseName, str]] = {
    "pre_event_baseline": "Pre-event baseline",
    "onset": "Event onset",
    "evolution": "Event evolution",
    "peak": "Peak behaviour",
    "recovery_or_persistence": "Recovery or persistence",
}


def unit_of_feature(name: str) -> str:
    """Return physical unit for a sensor channel or computed window feature."""
    if name in metropt3.ANALOG_SENSORS:
        return metropt3.ANALOG_SENSORS[name]
    if name in metropt3.DIGITAL_SENSORS:
        return "state (0/1)"
    if name in ("load_duty", "off_fraction", "unloaded_fraction"):
        return "fraction"
    if name == "load_starts":
        return "cycles/window"
    if name in ("unloaded_tp3_slope", "loaded_tp3_rise"):
        return "bar/min"
    if name == "tp3_reservoir_gap":
        return "bar"
    for sensor, unit in metropt3.ANALOG_SENSORS.items():
        if name.startswith(f"{sensor}_"):
            return unit
    for sensor in metropt3.DIGITAL_SENSORS:
        if name.startswith(f"{sensor}_"):
            return "fraction"
    return "unitless"


def _package_id(asset_id: str, model_version: str, start: pd.Timestamp, end: pd.Timestamp) -> str:
    raw = f"{asset_id}|{model_version}|{start.isoformat()}|{end.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _slice_raw(data: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return data[(data.index >= start) & (data.index < end)]


def _expected_window_count(start: pd.Timestamp, end: pd.Timestamp, window: pd.Timedelta) -> int:
    span = max(0.0, (end - start).total_seconds())
    w_s = max(1.0, window.total_seconds())
    return max(1, int(np.ceil(span / w_s)))


def _observed_signals_for_slice(
    raw_slice: pd.DataFrame,
    *,
    dataset_id: str,
) -> tuple[list[ObservedSignalSummary], list[str], list[str]]:
    """Aggregate direct telemetry observations without interpolating missing values."""
    items: list[ObservedSignalSummary] = []
    frozen: list[str] = []
    out_of_range: list[str] = []
    prov = f"dataset:{dataset_id}:processed_parquet"

    for sensor in _KEY_ANALOG:
        unit = metropt3.ANALOG_SENSORS[sensor]
        if raw_slice.empty or sensor not in raw_slice.columns:
            items.append(
                ObservedSignalSummary(
                    signal=sensor,
                    unit=unit,
                    sample_count=0,
                    missing_count=0,
                    out_of_range_count=0,
                    frozen=False,
                    provenance=prov,
                )
            )
            continue
        series = pd.to_numeric(raw_slice[sensor], errors="coerce")
        valid = series.dropna()
        missing_count = int(series.isna().sum())
        lo, hi = metropt3.VALID_RANGES[sensor]
        oor_count = int(((valid < lo) | (valid > hi)).sum()) if not valid.empty else 0
        if oor_count > 0:
            out_of_range.append(sensor)
        if valid.empty:
            items.append(
                ObservedSignalSummary(
                    signal=sensor,
                    unit=unit,
                    sample_count=0,
                    missing_count=missing_count,
                    out_of_range_count=oor_count,
                    frozen=False,
                    provenance=prov,
                )
            )
            continue
        std_val = float(valid.std(ddof=1)) if len(valid) >= 2 else 0.0
        is_frozen = len(valid) >= 2 and std_val <= _FREEZE_STD_TOLERANCE
        if is_frozen:
            frozen.append(sensor)
        items.append(
            ObservedSignalSummary(
                signal=sensor,
                unit=unit,
                sample_count=len(valid),
                missing_count=missing_count,
                out_of_range_count=oor_count,
                min=round(float(valid.min()), 4),
                mean=round(float(valid.mean()), 4),
                max=round(float(valid.max()), 4),
                std=round(std_val, 4),
                last=round(float(valid.iloc[-1]), 4),
                frozen=is_frozen,
                provenance=prov,
            )
        )

    for sensor in _KEY_DIGITAL:
        if raw_slice.empty or sensor not in raw_slice.columns:
            items.append(
                ObservedSignalSummary(
                    signal=sensor,
                    unit="state (0/1)",
                    sample_count=0,
                    missing_count=0,
                    provenance=prov,
                )
            )
            continue
        series = pd.to_numeric(raw_slice[sensor], errors="coerce")
        valid = series.dropna()
        oor_count = int((~valid.isin([0.0, 1.0])).sum()) if not valid.empty else 0
        if oor_count > 0:
            out_of_range.append(sensor)
        items.append(
            ObservedSignalSummary(
                signal=sensor,
                unit="state (0/1)",
                sample_count=len(valid),
                missing_count=int(series.isna().sum()),
                out_of_range_count=oor_count,
                active_fraction=round(float(valid.mean()), 4) if not valid.empty else None,
                last=round(float(valid.iloc[-1]), 4) if not valid.empty else None,
                provenance=prov,
            )
        )

    return items, frozen, out_of_range


def _derived_features_for_windows(
    win_features: pd.DataFrame,
    spec: WindowSpec,
) -> list[DerivedFeatureSummary]:
    """Summarize computed physical window features with explicit observability tracking."""
    prov = f"window_features:{spec.length}:min_coverage={spec.min_coverage}"
    out: list[DerivedFeatureSummary] = []
    specs = (
        ("load_duty", "fraction", "Mean fraction of samples with compressor loaded"),
        ("off_fraction", "fraction", "Mean fraction of samples with motor off"),
        ("unloaded_fraction", "fraction", "Mean fraction of samples with compressor unloaded"),
        ("load_starts", "cycles/window", "Total load cycle starts across valid windows"),
        ("inrush_samples", "samples", "Observed electrical startup inrush samples (> 8.0 A)"),
        ("tp3_reservoir_gap", "bar", "Mean panel-to-reservoir pressure differential"),
        ("loaded_tp3_rise", "bar/min", "Mean TP3 pressure rise rate while under load"),
        ("unloaded_tp3_slope", "bar/min", "Mean TP3 pressure decay slope while unloaded"),
    )
    if win_features.empty:
        return [
            DerivedFeatureSummary(
                name=name,
                unit=unit,
                value=None,
                observable=False,
                note="No valid telemetry windows available in this phase.",
                provenance=prov,
            )
            for name, unit, _ in specs
        ]

    valid_wf = win_features[win_features["valid"]]
    for name, unit, desc in specs:
        if valid_wf.empty or name not in valid_wf.columns:
            out.append(
                DerivedFeatureSummary(
                    name=name,
                    unit=unit,
                    value=None,
                    observable=False,
                    note="No valid 10-minute windows passed coverage/cadence checks.",
                    provenance=prov,
                )
            )
            continue
        col = pd.to_numeric(valid_wf[name], errors="coerce").dropna()
        if col.empty:
            out.append(
                DerivedFeatureSummary(
                    name=name,
                    unit=unit,
                    value=None,
                    observable=False,
                    note=f"{desc}; unobservable (requires active load/unload segments in phase).",
                    provenance=prov,
                )
            )
            continue
        agg = float(col.sum()) if name in ("load_starts", "inrush_samples") else float(col.mean())
        out.append(
            DerivedFeatureSummary(
                name=name,
                unit=unit,
                value=round(agg, 4),
                observable=True,
                note=desc,
                provenance=prov,
            )
        )
    return out


def _select_peak_window(
    event_windows: Sequence[WindowScore],
    *,
    event_start: pd.Timestamp,
) -> WindowScore | None:
    scored_primary = [w for w in event_windows if w.score is not None]
    if scored_primary:
        return max(scored_primary, key=lambda w: w.score if w.score is not None else float("-inf"))
    scored_warn = [w for w in event_windows if w.warning_score is not None]
    if scored_warn:
        return max(
            scored_warn,
            key=lambda w: w.warning_score if w.warning_score is not None else float("-inf"),
        )
    at_or_after = [w for w in event_windows if w.window_start >= event_start]
    return at_or_after[0] if at_or_after else (event_windows[0] if event_windows else None)


def _phase_features(
    raw_slice: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window: pd.Timedelta,
    spec: WindowSpec,
) -> pd.DataFrame:
    """Compute strictly contained derived features inside ``[start, end)`` without leakage."""
    if raw_slice.empty or (end - start) < window:
        return pd.DataFrame()
    aligned = start == start.floor(spec.length)
    wf = (
        window_features(raw_slice, spec)
        if aligned
        else anchored_window_features(raw_slice, spec, start)
    )
    win_starts = pd.DatetimeIndex(wf.index)
    return wf[(win_starts >= start) & (win_starts + window <= end)]


def _summarize_phase(
    *,
    phase: EventPhaseName,
    start: pd.Timestamp,
    end: pd.Timestamp,
    window: pd.Timedelta,
    spec: WindowSpec,
    raw_data: pd.DataFrame,
    windows: Sequence[WindowScore],
    dataset_id: str,
) -> tuple[PhaseEvidenceSummary, list[str], list[str]]:
    raw_slice = _slice_raw(raw_data, start, end)
    observed, frozen, oor = _observed_signals_for_slice(raw_slice, dataset_id=dataset_id)
    phase_wins = [w for w in windows if start <= w.window_start and w.window_end <= end]

    if end <= start:
        empty_reason = (
            "Single-window event has no distinct intermediate evolution span."
            if phase == "evolution"
            else "Phase disabled or zero duration requested (end <= start)."
        )
        return (
            PhaseEvidenceSummary(
                phase=phase,
                label=_PHASE_LABELS[phase],
                start=start.to_pydatetime(),
                end=end.to_pydatetime(),
                available=False,
                reason_unavailable=empty_reason,
                expected_windows=0,
                total_windows=0,
                valid_windows=0,
                sample_count=0,
                observed_signals=[],
                derived_features=[],
            ),
            [],
            [],
        )

    expected = _expected_window_count(start, end, window)
    wf = _phase_features(raw_slice, start=start, end=end, window=window, spec=spec)
    derived = _derived_features_for_windows(wf, spec)
    valid_wf_wins = int(wf["valid"].sum()) if not wf.empty else 0
    valid_replay_wins = sum(1 for w in phase_wins if w.regime.reason != "incomplete_or_gapped")
    total_wins = len(phase_wins) if phase_wins else len(wf)
    valid_wins = valid_replay_wins if phase_wins else valid_wf_wins
    counts: Counter[OperatingRegime] = Counter(w.regime.regime for w in phase_wins)
    dominant = counts.most_common(1)[0][0] if counts else None

    p_scores = [w.score for w in phase_wins if w.score is not None]
    p_excs = [w.exceedance for w in phase_wins if w.exceedance is not None]
    w_scores = [w.warning_score for w in phase_wins if w.warning_score is not None]
    w_excs = [w.warning_exceedance for w in phase_wins if w.warning_exceedance is not None]
    flags = list(
        dict.fromkeys(
            flag for w in phase_wins for flag in (*w.regime.quality_flags, *w.warning_reasons)
        )
    )
    available = not raw_slice.empty and valid_wins > 0
    reason_unavail: str | None = None
    if raw_slice.empty:
        reason_unavail = "No raw or processed telemetry recorded in this time interval."
    elif valid_wins == 0:
        reason_unavail = (
            "Samples present do not form a valid 10-minute 0.1 Hz window "
            "(coverage < 80% or internal gap > 60 s)."
        )

    summary = PhaseEvidenceSummary(
        phase=phase,
        label=_PHASE_LABELS[phase],
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        available=available,
        reason_unavailable=reason_unavail,
        expected_windows=expected,
        total_windows=total_wins,
        valid_windows=valid_wins,
        sample_count=len(raw_slice),
        dominant_regime=dominant,
        regime_counts={str(k): int(v) for k, v in counts.items()},
        primary_peak_score=round(max(p_scores), 4) if p_scores else None,
        primary_min_exceedance=round(min(p_excs), 6) if p_excs else None,
        warning_peak_score=round(max(w_scores), 4) if w_scores else None,
        warning_min_exceedance=round(min(w_excs), 6) if w_excs else None,
        warning_active_windows=sum(1 for w in phase_wins if w.warning_active),
        observed_signals=observed,
        derived_features=derived,
        quality_flags=flags,
    )
    return summary, frozen, oor


def _assess_quality(
    *,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    window: pd.Timedelta,
    phases: list[PhaseEvidenceSummary],
    event_windows: Sequence[WindowScore],
    recovery_windows: Sequence[WindowScore],
    raw_event: pd.DataFrame,
    frozen_sensors: list[str],
    out_of_range_signals: list[str],
) -> tuple[EvidenceQualityAssessment, list[str], list[str]]:
    by_phase = {p.phase: p for p in phases}
    baseline = by_phase["pre_event_baseline"]
    recovery = by_phase["recovery_or_persistence"]
    expected_ev_wins = _expected_window_count(event_start, event_end, window)
    valid_ev_wins = sum(1 for w in event_windows if w.regime.reason != "incomplete_or_gapped")
    coverage_ratio = min(1.0, valid_ev_wins / expected_ev_wins) if expected_ev_wins else 0.0
    missing_wins = max(0, expected_ev_wins - valid_ev_wins)

    expected_samples = int(max(1, round((event_end - event_start).total_seconds() / 10.0)))
    observed_samples = len(raw_event)
    missing_samples = max(0, expected_samples - observed_samples)

    stale_wins = sum(1 for w in event_windows if w.regime.stale_telemetry)
    contradictions: list[ContradictionRecord] = []
    alignment_issues: list[str] = []
    missing_measurements: list[str] = []
    conflicting: list[str] = []

    if not baseline.available:
        missing_measurements.append(
            f"Pre-event baseline ({baseline.start.isoformat()} to {baseline.end.isoformat()}): "
            f"{baseline.reason_unavailable or 'unavailable'}"
        )
        alignment_issues.append("Pre-event baseline coverage missing or invalid")
    if not recovery.available:
        rec_span = f"{recovery.start.isoformat()} to {recovery.end.isoformat()}"
        missing_measurements.append(
            f"Recovery/persistence tail ({rec_span}): "
            f"{recovery.reason_unavailable or 'unavailable'}"
        )
        alignment_issues.append("Recovery/persistence window missing telemetry")
    if missing_wins > 0:
        missing_measurements.append(
            f"{missing_wins} of {expected_ev_wins} expected 10-min event windows "
            "lacked valid coverage"
        )

    for w in event_windows:
        flags = set(w.regime.quality_flags)
        ts = w.window_start.to_pydatetime()
        if "incomplete_or_gapped" in flags or w.regime.reason == "incomplete_or_gapped":
            alignment_issues.append(
                f"Window {w.window_start.isoformat()} fails sample coverage or <= 60 s cadence"
            )
        if w.regime.stale_telemetry:
            conflicting.append(f"Stale telemetry freeze at {w.window_start.isoformat()}")
            contradictions.append(
                ContradictionRecord(
                    kind="sensor_freeze",
                    timestamp=ts,
                    severity="critical",
                    signals=["TP3", "TP2", "Motor_current"],
                    description=(
                        "TP3, TP2, and Motor_current stayed simultaneously constant; "
                        "primary air-leak evaluation abstained."
                    ),
                )
            )
        if "valve_contradiction" in flags:
            conflicting.append(f"COMP & DV_eletric active together at {w.window_start.isoformat()}")
            contradictions.append(
                ContradictionRecord(
                    kind="valve_contradiction",
                    timestamp=ts,
                    severity="critical",
                    signals=["COMP", "DV_eletric"],
                    description="Intake and outlet valves reported active simultaneously > 10%.",
                )
            )
        if "motor_valve_mismatch" in flags:
            conflicting.append(f"Motor current vs valve mismatch at {w.window_start.isoformat()}")
            contradictions.append(
                ContradictionRecord(
                    kind="motor_valve_mismatch",
                    timestamp=ts,
                    severity="warning",
                    signals=["Motor_current", "COMP", "DV_eletric"],
                    description="Motor current > 5.5 A disagrees with intake/outlet valve states.",
                )
            )

    for sensor in sorted(set(frozen_sensors)):
        if stale_wins > 0 and sensor in {"TP3", "TP2", "Motor_current"}:
            continue
        msg = f"Channel {sensor} remained constant (std <= 1e-12) across an event phase"
        if msg not in conflicting:
            conflicting.append(msg)
        contradictions.append(
            ContradictionRecord(
                kind="sensor_freeze",
                timestamp=event_start.to_pydatetime(),
                severity="warning",
                signals=[sensor],
                description=msg,
            )
        )

    for sensor in sorted(set(out_of_range_signals)):
        msg = f"Channel {sensor} contained out-of-range raw samples"
        if msg not in conflicting:
            conflicting.append(msg)
        contradictions.append(
            ContradictionRecord(
                kind="out_of_range",
                timestamp=event_start.to_pydatetime(),
                severity="critical",
                signals=[sensor],
                description=msg,
            )
        )

    dedup_contradictions: list[ContradictionRecord] = []
    seen_contradictions: set[tuple[str, object, tuple[str, ...]]] = set()
    for c in contradictions:
        key = (c.kind, c.timestamp, tuple(c.signals))
        if key not in seen_contradictions:
            seen_contradictions.add(key)
            dedup_contradictions.append(c)
    contradictions = dedup_contradictions

    recovery_status: Literal["recovered", "persisting", "stopped", "unverified"] = "unverified"
    if recovery.available:
        if any(w.status == "anomalous" or w.warning_active for w in recovery_windows):
            recovery_status = "persisting"
        elif all(w.status == "stopped" for w in recovery_windows):
            recovery_status = "stopped"
        else:
            recovery_status = "recovered"

    # Completeness weighs core event coverage (60%), baseline (20%), recovery (20%).
    completeness = (
        0.6 * coverage_ratio
        + (0.2 if baseline.available else 0.0)
        + (0.2 if recovery.available else 0.0)
    )
    completeness = round(min(1.0, max(0.0, completeness)), 4)

    # Confidence penalty reflects gaps, stale windows, freezes, contradictions, missing tails.
    penalty = completeness
    if stale_wins > 0:
        stale_frac = stale_wins / max(1, len(event_windows))
        penalty *= max(0.15, 1.0 - 0.75 * stale_frac)
    if contradictions:
        crit = sum(1 for c in contradictions if c.severity == "critical")
        warn = sum(1 for c in contradictions if c.severity == "warning")
        penalty *= max(0.2, 1.0 - 0.15 * crit - 0.08 * warn)
    if any(w.regime.regime == "starting" for w in event_windows):
        penalty *= 0.9
    penalty = round(min(1.0, max(0.0, penalty)), 4)

    warnings: list[str] = []
    if stale_wins > 0:
        warnings.append(
            f"{stale_wins} event window(s) exhibited frozen whole-feed telemetry (stale_telemetry)."
        )
    non_stale_frozen = [
        s
        for s in sorted(set(frozen_sensors))
        if stale_wins == 0 or s not in {"TP3", "TP2", "Motor_current"}
    ]
    if non_stale_frozen:
        warnings.append(
            f"Single-sensor constant reading observed on: {', '.join(non_stale_frozen)}."
        )
    if any(w.regime.regime == "starting" for w in event_windows):
        warnings.append(
            "Startup transition detected (~0.1 Hz sampling misses sub-10 s electrical inrush)."
        )
    if not baseline.available:
        warnings.append("Pre-event baseline telemetry unavailable; pre/post contrast unverified.")
    if not recovery.available:
        warnings.append("Post-event recovery telemetry unavailable; recovery remains unverified.")
    if valid_ev_wins == 0:
        warnings.append("Zero valid 10-minute windows inside event interval.")

    if (
        valid_ev_wins == 0
        or coverage_ratio < 0.5
        or completeness < 0.45
        or (stale_wins == len(event_windows) and len(event_windows) > 0)
    ):
        overall: EvidenceQualityStatus = "insufficient"
    elif (
        completeness < 0.85
        or contradictions
        or bool(frozen_sensors)
        or bool(out_of_range_signals)
        or bool(alignment_issues)
    ):
        overall = "degraded"
    else:
        overall = "sufficient"

    limitations = [
        (
            "MetroPT-3 operates at ~0.1 Hz (10 s); transient electrical inrush spikes (<10 s) "
            "are mostly missed."
        ),
        (
            "No independent pneumatic air-demand flowmeter or PLC mode tag exists; duty rises "
            "can reflect heavy demand as well as leaks."
        ),
        (
            "Missing or out-of-range windows are left uninterpolated so inferred states are "
            "never fabricated."
        ),
    ]

    return (
        EvidenceQualityAssessment(
            overall_status=overall,
            completeness_score=completeness,
            coverage_ratio=round(coverage_ratio, 4),
            baseline_available=baseline.available,
            recovery_available=recovery.available,
            recovery_status=recovery_status,
            expected_event_windows=expected_ev_wins,
            valid_event_windows=valid_ev_wins,
            missing_windows_count=missing_wins,
            missing_samples_count=missing_samples,
            stale_windows_count=stale_wins,
            frozen_sensors=sorted(set(frozen_sensors)),
            out_of_range_signals=sorted(set(out_of_range_signals)),
            contradictions=contradictions,
            alignment_issues=list(dict.fromkeys(alignment_issues)),
            confidence_penalty_factor=penalty,
            quality_warnings=warnings,
            limitations=limitations,
        ),
        list(dict.fromkeys(missing_measurements)),
        list(dict.fromkeys(conflicting)),
    )


def _neighbor_relationship(
    start_a: pd.Timestamp, end_a: pd.Timestamp, ev_start: pd.Timestamp, ev_end: pd.Timestamp
) -> Literal["concurrent", "preceding", "following"]:
    if end_a <= ev_start:
        return "preceding"
    if start_a >= ev_end:
        return "following"
    return "concurrent"


def _build_neighbors(
    *,
    replay: ReplayOutput,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    primary_detector: str,
) -> list[NeighboringEventSummary]:
    out: list[NeighboringEventSummary] = []
    for ep in replay.episodes:
        # Exclude exact event bounds when equal; keep overlapping neighboring runs.
        if ep.episode.start == event_start and ep.episode.end == event_end:
            continue
        out.append(
            NeighboringEventSummary(
                kind="air_leak_alarm",
                title=f"Air-leak alarm episode ({ep.dominant_regime})",
                start=ep.episode.start.to_pydatetime(),
                end=ep.episode.end.to_pydatetime(),
                raised_at=ep.episode.raised_at.to_pydatetime(),
                detector_or_source=primary_detector,
                relationship=_neighbor_relationship(
                    ep.episode.start, ep.episode.end, event_start, event_end
                ),
                origin="model_prediction",
            )
        )
    for w in replay.warnings:
        if w.start == event_start and w.end == event_end:
            continue
        out.append(
            NeighboringEventSummary(
                kind="sensor_consistency_warning",
                title=f"Consistency warning: {w.warning_kind}",
                start=w.start.to_pydatetime(),
                end=w.end.to_pydatetime(),
                raised_at=w.raised_at.to_pydatetime(),
                detector_or_source=w.detector_name,
                relationship=_neighbor_relationship(w.start, w.end, event_start, event_end),
                origin="statistical_inference" if "pca_spe" in w.sources else "derived",
            )
        )
    for rep in replay.reports:
        rs, re_ = pd.Timestamp(rep.start), pd.Timestamp(rep.end)
        out.append(
            NeighboringEventSummary(
                kind="operator_report",
                event_id=rep.report_id,
                title=f"Operator report {rep.report_id}: {rep.failure}",
                start=rs.to_pydatetime(),
                end=re_.to_pydatetime(),
                detector_or_source="operator_report_weak_label",
                relationship=_neighbor_relationship(rs, re_, event_start, event_end),
                origin="human_observation",
            )
        )
    out.sort(key=lambda item: (item.start, item.kind))
    return out


def _phase_of_timestamp(
    ts: pd.Timestamp, phases: list[PhaseEvidenceSummary]
) -> EventPhaseName | None:
    by_name = {p.phase: p for p in phases}
    peak = by_name.get("peak")
    if peak and peak.start <= ts.to_pydatetime() < peak.end:
        return "peak"
    for name in ("pre_event_baseline", "onset", "evolution", "recovery_or_persistence"):
        p = by_name.get(name)
        if p and p.start <= ts.to_pydatetime() < p.end:
            return p.phase
    return None


def _build_timeline(
    *,
    replay: ReplayOutput,
    phases: list[PhaseEvidenceSummary],
    peak_window: WindowScore | None,
    evidence_items: list[EvidenceItemContract],
    observations: Sequence[HumanObservationOut],
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    primary_detector: str,
    primary_version: str,
    window_length: str,
) -> list[TimelineEntry]:
    entries: list[TimelineEntry] = []
    source_id = f"window_features:{window_length}"
    prev_regime: str | None = None
    prev_reason: str | None = None

    lookback_start = phases[0].start
    recovery_end = phases[-1].end

    for idx, w in enumerate(replay.windows):
        w_start_dt = w.window_start.to_pydatetime()
        w_end_dt = w.window_end.to_pydatetime()
        if w_start_dt < lookback_start or w_end_dt > recovery_end:
            continue
        in_event = event_start <= w.window_start and w.window_end <= event_end
        phase = _phase_of_timestamp(w.window_start, phases)
        regime = w.regime
        regime_changed = regime.regime != prev_regime or (
            regime.regime == "insufficient_evidence" and regime.reason != prev_reason
        )
        is_special = w.status != "normal" or w.warning_active or regime_changed
        is_anchor = idx == 0 or idx == len(replay.windows) - 1 or idx % 6 == 0

        if is_special or in_event or is_anchor:
            entries.append(
                TimelineEntry(
                    timestamp=w.window_start.to_pydatetime(),
                    end_timestamp=w.window_end.to_pydatetime(),
                    category="telemetry_window",
                    origin="derived",
                    phase=phase,
                    title=f"Window kinematics ({regime.regime})",
                    summary=regime.summary,
                    source_id=source_id,
                    regime=regime.regime,
                    details={
                        "n_samples": w.n_samples,
                        "load_duty": regime.load_duty,
                        "off_fraction": regime.off_fraction,
                        "load_starts": regime.load_starts,
                        "inrush_samples": regime.inrush_samples,
                        "stale_telemetry": regime.stale_telemetry,
                    },
                )
            )
        if regime_changed:
            entries.append(
                TimelineEntry(
                    timestamp=w.window_start.to_pydatetime(),
                    end_timestamp=w.window_end.to_pydatetime(),
                    category="regime_change",
                    origin="derived",
                    phase=phase,
                    title=f"Regime: {regime.regime} ({regime.confidence})",
                    summary=f"{regime.reason}: {regime.summary}",
                    source_id=source_id,
                    regime=regime.regime,
                    details={
                        "reason": regime.reason,
                        "quality_flags": list(regime.quality_flags),
                        "missing_signals": list(regime.missing_signals),
                    },
                )
            )
            prev_regime = regime.regime
            prev_reason = regime.reason

    for ep in replay.episodes:
        entries.append(
            TimelineEntry(
                timestamp=ep.episode.start.to_pydatetime(),
                end_timestamp=ep.episode.end.to_pydatetime(),
                category="alarm_episode",
                origin="model_prediction",
                phase=_phase_of_timestamp(ep.episode.start, phases),
                title=f"Air-leak alarm episode ({ep.dominant_regime})",
                summary=ep.hypothesis or ep.uncertainty_note or "Sustained duty exceedance.",
                source_id=f"model:{primary_version}",
                detector=primary_detector,
                model_version=primary_version,
                regime=ep.dominant_regime,
                event_class=ep.event_class,
                details={
                    "raised_at": ep.episode.raised_at.isoformat(),
                    "peak_score": round(ep.episode.peak_score, 4),
                    "ongoing": ep.ongoing,
                    "truncated": ep.truncated,
                },
            )
        )

    for warn in replay.warnings:
        entries.append(
            TimelineEntry(
                timestamp=warn.start.to_pydatetime(),
                end_timestamp=warn.end.to_pydatetime(),
                category="consistency_warning",
                origin="statistical_inference" if "pca_spe" in warn.sources else "derived",
                phase=_phase_of_timestamp(warn.start, phases),
                title=f"Consistency warning ({warn.warning_kind})",
                summary=warn.hypothesis,
                source_id=f"warning:{warn.detector_name}",
                detector=warn.detector_name,
                regime=warn.dominant_regime,
                event_class=warn.event_class,
                details={
                    "raised_at": warn.raised_at.isoformat(),
                    "peak_score": warn.peak_score,
                    "peak_exceedance": warn.peak_exceedance,
                    "sources": list(warn.sources),
                },
            )
        )

    if peak_window is not None and (
        peak_window.score is not None or peak_window.warning_score is not None
    ):
        entries.append(
            TimelineEntry(
                timestamp=peak_window.window_start.to_pydatetime(),
                end_timestamp=peak_window.window_end.to_pydatetime(),
                category="model_output",
                origin="model_prediction",
                phase="peak",
                title="Peak model & exceedance readout",
                summary=(
                    f"Primary score={peak_window.score}, exceedance={peak_window.exceedance}; "
                    f"Warning score={peak_window.warning_score}, "
                    f"exceedance={peak_window.warning_exceedance}."
                ),
                source_id=f"model:{primary_version}",
                detector=primary_detector,
                model_version=primary_version,
                regime=peak_window.regime.regime,
                event_class=peak_window.event_class,
                details={
                    "score": peak_window.score,
                    "exceedance": peak_window.exceedance,
                    "warning_score": peak_window.warning_score,
                    "warning_exceedance": peak_window.warning_exceedance,
                },
            )
        )

    for ev_item in evidence_items:
        anchor_ts = (
            peak_window.window_start.to_pydatetime()
            if peak_window is not None
            else event_start.to_pydatetime()
        )
        entries.append(
            TimelineEntry(
                timestamp=anchor_ts,
                category="evidence_group",
                origin=ev_item.origin,
                phase=ev_item.phase,
                title=f"{ev_item.group_label}: {ev_item.feature}",
                summary=ev_item.meaning,
                source_id=ev_item.source_id,
                details={
                    "value": ev_item.value,
                    "unit": ev_item.unit,
                    "training_median": ev_item.training_median,
                    "robust_z": ev_item.robust_z,
                    "novel": ev_item.novel,
                },
            )
        )

    for rep in replay.reports:
        entries.append(
            TimelineEntry(
                timestamp=rep.start,
                end_timestamp=rep.end,
                category="operator_report",
                origin="human_observation",
                title=f"Operator report {rep.report_id}: {rep.failure}",
                summary=(
                    "Weak historical operator failure interval (exact maintenance log unverified)."
                ),
                source_id=f"operator_report:{rep.report_id}",
                details={"report_id": rep.report_id, "failure": rep.failure},
            )
        )

    for obs in observations:
        entries.append(
            TimelineEntry(
                timestamp=obs.observed_at or obs.created_at,
                category="human_observation",
                origin="human_observation",
                phase=_phase_of_timestamp(pd.Timestamp(obs.observed_at or obs.created_at), phases),
                title=f"Human observation ({obs.category}) by {obs.actor_id}",
                summary=obs.note,
                source_id=f"human_observation:{obs.id}",
                details={"actor_id": obs.actor_id, "category": obs.category},
            )
        )

    entries.sort(key=lambda e: (e.timestamp, e.category, e.phase or "", e.source_id, e.title))
    return entries


def build_event_evidence_package(
    service: AnomalyService,
    *,
    asset_id: str,
    event_start: pd.Timestamp,
    event_end: pd.Timestamp,
    raised_at: pd.Timestamp | None = None,
    incident_id: str | None = None,
    human_observations: Sequence[HumanObservationOut] = (),
    baseline_hours: float = 4.0,
    recovery_hours: float = 2.0,
) -> EventEvidencePackage:
    """Assemble a full WP-A Event Evidence Package for ``[event_start, event_end)``."""
    if event_end <= event_start:
        raise ValueError("event_end must be strictly after event_start")

    spec = service.bundle.spec
    window = pd.Timedelta(spec.length)
    lookback_start = event_start - pd.Timedelta(hours=baseline_hours)
    recovery_end = event_end + pd.Timedelta(hours=recovery_hours)

    replay = service.replay_details(lookback_start, recovery_end)
    raw_data = service.telemetry
    event_wins = [
        w for w in replay.windows if event_start <= w.window_start and w.window_end <= event_end
    ]
    recovery_wins = [
        w for w in replay.windows if event_end <= w.window_start and w.window_end <= recovery_end
    ]
    peak_win = _select_peak_window(event_wins, event_start=event_start)

    onset_end = min(event_end, event_start + window)
    evo_start = onset_end
    evo_end = event_end
    peak_start = peak_win.window_start if peak_win is not None else event_start
    peak_end = min(event_end, peak_start + window)

    card_dataset = service.card.get("dataset", {})
    dataset_id = str(card_dataset.get("id", metropt3.DATASET_ID))
    phase_bounds: list[tuple[EventPhaseName, pd.Timestamp, pd.Timestamp]] = [
        ("pre_event_baseline", lookback_start, event_start),
        ("onset", event_start, onset_end),
        ("evolution", evo_start, evo_end),
        ("peak", peak_start, peak_end),
        ("recovery_or_persistence", event_end, recovery_end),
    ]

    phases: list[PhaseEvidenceSummary] = []
    frozen_all: list[str] = []
    oor_all: list[str] = []
    for pname, lo, hi in phase_bounds:
        psummary, p_frozen, p_oor = _summarize_phase(
            phase=pname,
            start=lo,
            end=hi,
            window=window,
            spec=spec,
            raw_data=raw_data,
            windows=replay.windows,
            dataset_id=dataset_id,
        )
        phases.append(psummary)
        if pname in ("onset", "evolution", "peak"):
            frozen_all.extend(p_frozen)
            oor_all.extend(p_oor)
    frozen_all = sorted(set(frozen_all))
    oor_all = sorted(set(oor_all))

    evidence_items: list[EvidenceItemContract] = []
    if peak_win is not None:
        raw_items = peak_win.evidence or peak_win.warning_evidence
        use_warning_profile = not peak_win.evidence and bool(peak_win.warning_evidence)
        source_model = (
            service.warning_bundle.version
            if use_warning_profile and service.warning_bundle is not None
            else (
                peak_win.warning_detector or service.bundle.version
                if use_warning_profile
                else service.bundle.version
            )
        )
        for e in raw_items:
            evidence_items.append(
                EvidenceItemContract(
                    group=e.group,
                    group_label=e.group_label,
                    feature=e.feature,
                    value=e.value,
                    unit=unit_of_feature(e.feature),
                    training_median=e.training_median,
                    robust_z=e.robust_z,
                    novel=e.novel,
                    meaning=e.meaning,
                    origin="statistical_inference",
                    phase="peak",
                    source_id=f"feature_profile:{source_model}",
                )
            )

    quality, missing_meas, conflicting = _assess_quality(
        event_start=event_start,
        event_end=event_end,
        window=window,
        phases=phases,
        event_windows=event_wins,
        recovery_windows=recovery_wins,
        raw_event=_slice_raw(raw_data, event_start, event_end),
        frozen_sensors=frozen_all,
        out_of_range_signals=oor_all,
    )

    if (
        peak_win is not None
        and any(is_integrity_warning_regime(w.regime) for w in event_wins)
        and "Telemetry integrity warning active in event span" not in conflicting
    ):
        conflicting.append("Telemetry integrity warning active in event span")

    neighbors = _build_neighbors(
        replay=replay,
        event_start=event_start,
        event_end=event_end,
        primary_detector=service.bundle.detector_name,
    )
    observations = list(human_observations)
    timeline = _build_timeline(
        replay=replay,
        phases=phases,
        peak_window=peak_win,
        evidence_items=evidence_items,
        observations=observations,
        event_start=event_start,
        event_end=event_end,
        primary_detector=service.bundle.detector_name,
        primary_version=service.bundle.version,
        window_length=spec.length,
    )

    provenance = ProvenanceRecord(
        dataset_id=dataset_id,
        dataset_source=str(card_dataset.get("source", metropt3.SOURCE_PAGE)),
        dataset_license=str(card_dataset.get("license", metropt3.LICENSE)),
        archive_sha256=(
            str(card_dataset["archive_sha256"]) if card_dataset.get("archive_sha256") else None
        ),
        preprocessing_version="non_interpolated_parquet_v1",
        sampling_contract="~0.1 Hz nominal (5-15 s median, max internal gap <= 60 s)",
        window_length=spec.length,
        primary_detector=service.bundle.detector_name,
        primary_model_version=service.bundle.version,
        warning_detector=service.warning_bundle.detector_name if service.warning_bundle else None,
        warning_model_version=service.warning_bundle.version if service.warning_bundle else None,
    )

    return EventEvidencePackage(
        package_id=_package_id(asset_id, service.bundle.version, event_start, event_end),
        asset_id=asset_id,
        incident_id=incident_id,
        event_start=event_start.to_pydatetime(),
        event_end=event_end.to_pydatetime(),
        raised_at=raised_at.to_pydatetime() if raised_at is not None else None,
        lookback_start=lookback_start.to_pydatetime(),
        recovery_end=recovery_end.to_pydatetime(),
        provenance=provenance,
        phases=phases,
        evidence_items=evidence_items,
        quality=quality,
        missing_measurements=missing_meas,
        conflicting_measurements=conflicting,
        neighboring_events=neighbors,
        human_observations=observations,
        timeline=timeline,
    )
