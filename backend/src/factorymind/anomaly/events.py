"""Canonical event-class taxonomy and warning/alarm separation for MetroPT-3.

WP2 strictly separates four operator-facing event classes so instrumentation
faults, frozen telemetry, and cross-sensor decoupling are surfaced without
being mislabelled as air-leak alarms or inflating air-leak failure metrics:

- ``air_leak_alarm``: Sustained duty exceedance from the primary leak detector
  (``duty_rule``) on valid, non-stopped, telemetry-consistent operating windows.
- ``sensor_consistency_warning``: Multivariate ``pca_spe`` residual exceedance or
  explicit signal-integrity failure (``stale_telemetry`` / ``signal_inconsistency``).
- ``informational``: Normal operating or confirmed stopped state with no active
  leak alarm or consistency warning.
- ``insufficient_evidence``: Incomplete, gapped, or non-scorable telemetry where
  neither a leak nor an operating state can be reliably inferred.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd

from factorymind.anomaly.alarm import AlarmPolicy, AlarmTracker, Episode
from factorymind.anomaly.regimes import OperatingRegime, RegimeAssessment

WindowStatus = Literal["normal", "anomalous", "stopped", "insufficient_data"]
EventClass = Literal[
    "informational",
    "sensor_consistency_warning",
    "air_leak_alarm",
    "insufficient_evidence",
]

INTEGRITY_REASONS: Final[frozenset[str]] = frozenset({"stale_telemetry", "signal_inconsistency"})
NON_SCORABLE_REASONS: Final[frozenset[str]] = frozenset(
    {"incomplete_or_gapped", "stale_telemetry", "signal_inconsistency"}
)

EVENT_MEANINGS: Final[dict[EventClass, str]] = {
    "air_leak_alarm": (
        "Sustained excess compressor load duty consistent with an air leak hypothesis; verify "
        "downstream pneumatic demand and valve/line pressure decay before isolating equipment."
    ),
    "sensor_consistency_warning": (
        "Multivariate sensor residual (PCA SPE) or telemetry integrity check flagged frozen, "
        "contradictory, or decoupled signals; indicates instrumentation/telemetry fault rather "
        "than a confirmed air leak."
    ),
    "informational": (
        "Nominal operating or explained stopped telemetry with no active air-leak alarm or "
        "sensor-consistency warning."
    ),
    "insufficient_evidence": (
        "Telemetry window lacks valid sample coverage or cadence integrity; operating regime "
        "and air-leak hypotheses remain unverified."
    ),
}

_REGIME_TIE_ORDER: Final[tuple[OperatingRegime, ...]] = (
    "loaded",
    "unloaded",
    "starting",
    "stopped",
    "insufficient_evidence",
)


@dataclass(frozen=True)
class WindowEventClassification:
    """Window-level event taxonomy paired with primary-status and warning flags."""

    status: WindowStatus
    event_class: EventClass
    event_meaning: str
    warning_active: bool
    integrity_warning: bool
    warning_reasons: tuple[str, ...]


@dataclass(frozen=True)
class WarningEpisode:
    """Coherent sensor-consistency or telemetry-integrity warning episode."""

    start: pd.Timestamp
    end: pd.Timestamp
    raised_at: pd.Timestamp
    peak_score: float | None
    peak_exceedance: float | None
    detector_name: str
    warning_kind: str
    sources: tuple[str, ...]
    truncated: bool
    ongoing: bool
    event_class: EventClass
    hypothesis: str
    uncertainty_note: str
    dominant_regime: OperatingRegime
    regime_breakdown: dict[str, int]


def is_integrity_warning_regime(regime: RegimeAssessment) -> bool:
    """Return True when the regime assessment failed telemetry integrity checks."""
    return regime.reason in INTEGRITY_REASONS


def primary_leak_scorable(
    row_valid: bool,
    row_stopped: bool,
    regime: RegimeAssessment,
) -> bool:
    """Return True when the primary air-leak detector may safely score this window."""
    return (
        row_valid
        and not row_stopped
        and regime.regime != "stopped"
        and regime.reason not in NON_SCORABLE_REASONS
    )


def warning_scorable(
    row_valid: bool,
    row_stopped: bool,
    regime: RegimeAssessment,
) -> bool:
    """Return True when the multivariate consistency warning detector may score this window.

    Unlike ``primary_leak_scorable``, ``pca_spe`` intentionally still scores
    ``stale_telemetry`` and ``signal_inconsistency`` windows so sensor/telemetry
    anomalies are actively flagged rather than masked.
    """
    return (
        row_valid
        and not row_stopped
        and regime.regime != "stopped"
        and regime.reason != "incomplete_or_gapped"
    )


def classify_window_event(
    *,
    regime: RegimeAssessment,
    row_valid: bool,
    row_stopped: bool,
    primary_anomalous: bool,
    warning_detector_active: bool,
) -> WindowEventClassification:
    """Classify one window into its primary status and canonical WP2 event class."""
    if not row_valid or regime.reason == "incomplete_or_gapped":
        return WindowEventClassification(
            status="insufficient_data",
            event_class="insufficient_evidence",
            event_meaning=EVENT_MEANINGS["insufficient_evidence"],
            warning_active=False,
            integrity_warning=False,
            warning_reasons=(),
        )

    integrity = is_integrity_warning_regime(regime)
    reasons: list[str] = []
    if integrity:
        reasons.extend(regime.quality_flags or (regime.reason,))
    if warning_detector_active:
        reasons.append("pca_spe_exceedance")
    reason_tuple = tuple(dict.fromkeys(reasons))

    if integrity:
        return WindowEventClassification(
            status="insufficient_data",
            event_class="sensor_consistency_warning",
            event_meaning=EVENT_MEANINGS["sensor_consistency_warning"],
            warning_active=True,
            integrity_warning=True,
            warning_reasons=reason_tuple,
        )

    if row_stopped or regime.regime == "stopped":
        return WindowEventClassification(
            status="stopped",
            event_class="informational",
            event_meaning=EVENT_MEANINGS["informational"],
            warning_active=False,
            integrity_warning=False,
            warning_reasons=(),
        )

    if primary_anomalous:
        return WindowEventClassification(
            status="anomalous",
            event_class="air_leak_alarm",
            event_meaning=EVENT_MEANINGS["air_leak_alarm"],
            warning_active=warning_detector_active,
            integrity_warning=False,
            warning_reasons=reason_tuple,
        )

    if warning_detector_active:
        return WindowEventClassification(
            status="normal",
            event_class="sensor_consistency_warning",
            event_meaning=EVENT_MEANINGS["sensor_consistency_warning"],
            warning_active=True,
            integrity_warning=False,
            warning_reasons=reason_tuple,
        )

    return WindowEventClassification(
        status="normal",
        event_class="informational",
        event_meaning=EVENT_MEANINGS["informational"],
        warning_active=False,
        integrity_warning=False,
        warning_reasons=(),
    )


def summarize_episode_regimes(
    regimes: Sequence[RegimeAssessment],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[OperatingRegime, dict[str, int]]:
    """Compute dominant operating regime and per-regime window counts within ``[start, end)``."""
    counts: Counter[OperatingRegime] = Counter(
        r.regime for r in regimes if r.window_start >= start and r.window_start < end
    )
    if not counts:
        return "insufficient_evidence", {}
    breakdown: dict[str, int] = {key: counts[key] for key in _REGIME_TIE_ORDER if counts[key] > 0}
    dominant = max(
        counts,
        key=lambda key: (counts[key], -_REGIME_TIE_ORDER.index(key)),
    )
    return dominant, breakdown


def episode_hypothesis_and_uncertainty(
    regimes: Sequence[RegimeAssessment],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    event_class: EventClass,
    sources: tuple[str, ...] = (),
) -> tuple[str, str]:
    """Return physical hypothesis and explicit uncertainty statement for one episode."""
    in_ep = [r for r in regimes if r.window_start >= start and r.window_start < end]
    has_starting = any(r.regime == "starting" for r in in_ep)
    has_mixed = any(r.reason == "mixed_duty_cycle" for r in in_ep)
    has_stale = "stale_telemetry" in sources or any(r.reason == "stale_telemetry" for r in in_ep)
    has_incon = "signal_inconsistency" in sources or any(
        r.reason == "signal_inconsistency" for r in in_ep
    )

    if event_class == "air_leak_alarm":
        hypothesis = (
            "Sustained high compressor load duty above the calibrated baseline; consistent "
            "with a pneumatic air-leak hypothesis."
        )
        notes = [
            "Duty exceedance alone cannot rule out heavy downstream plant air demand without "
            "demand-flow telemetry.",
        ]
        if has_starting:
            notes.append(
                "Contains startup transition windows (~0.1 Hz cadence may miss sub-10 s "
                "electrical inrush)."
            )
        if has_mixed:
            notes.append(
                "Includes mixed 10-minute operating cycles where steady-regime dominance "
                "could not be established."
            )
        return hypothesis, " ".join(notes)

    if has_stale and "pca_spe" in sources:
        return (
            "Frozen sensor/PLC feed confirmed by simultaneous zero-variance TP3/TP2/Motor_current "
            "and PCA SPE residual departure.",
            "Telemetry integrity is compromised; air-leak evaluation abstained and this warning "
            "must not be counted as a leak.",
        )
    if has_stale:
        return (
            "Simultaneously constant TP3, TP2, and Motor_current across the window indicate "
            "frozen or stale telemetry.",
            "No independent PLC heartbeat or high-rate bus signal is present; primary air-leak "
            "scoring abstained.",
        )
    if has_incon:
        return (
            "Contradictory motor-current and COMP/DV_eletric valve states exceed the 10% "
            "consistency tolerance.",
            "Indicates valve limit-switch mismatch, stalled compressor instrumentation, or signal "
            "wiring fault rather than an air leak.",
        )
    notes = [
        "Multivariate feature structure departed from the normal operating subspace without "
        "proving a pneumatic air leak.",
        "Confirm single-sensor health (pressure transducers, oil temperature, current clamp) "
        "before scheduling leak maintenance.",
    ]
    if has_starting or has_mixed:
        notes.append("Co-occurs with non-steady operating transitions.")
    return (
        "Multivariate PCA SPE residual exceedance indicating potential cross-sensor decoupling "
        "or single-sensor fault.",
        " ".join(notes),
    )


@dataclass
class _RawWarningSpan:
    start: pd.Timestamp
    end: pd.Timestamp
    raised_at: pd.Timestamp
    peak_score: float | None
    peak_exceedance: float | None
    sources: set[str]
    truncated: bool
    ongoing: bool


def _collect_integrity_spans(
    *,
    timestamps: Sequence[pd.Timestamp],
    regimes: Sequence[RegimeAssessment],
    window: pd.Timedelta,
    first_window: pd.Timestamp,
) -> list[_RawWarningSpan]:
    spans: list[_RawWarningSpan] = []
    run_start: pd.Timestamp | None = None
    run_last: pd.Timestamp | None = None
    run_sources: set[str] = set()

    def flush(*, ongoing: bool) -> None:
        nonlocal run_start, run_last, run_sources
        if run_start is None or run_last is None:
            return
        spans.append(
            _RawWarningSpan(
                start=run_start,
                end=run_last + window,
                raised_at=run_start + window,
                peak_score=None,
                peak_exceedance=None,
                sources=set(run_sources),
                truncated=run_start <= first_window,
                ongoing=ongoing,
            )
        )
        run_start = run_last = None
        run_sources = set()

    last_index = len(timestamps) - 1
    for i, (ts, reg) in enumerate(zip(timestamps, regimes, strict=True)):
        is_integ = is_integrity_warning_regime(reg)
        contiguous = run_last is not None and (ts - run_last) == window
        if run_start is not None and not (is_integ and contiguous):
            flush(ongoing=False)
        if not is_integ:
            continue
        if run_start is None:
            run_start = ts
            run_sources = set()
        run_last = ts
        run_sources.add(reg.reason)
        if i == last_index:
            flush(ongoing=True)
    if run_start is not None:
        flush(ongoing=False)
    return spans


def build_warning_episodes(
    *,
    window: pd.Timedelta,
    timestamps: Sequence[pd.Timestamp],
    regimes: Sequence[RegimeAssessment],
    warning_scores: Sequence[float | None],
    warning_exceedances: Sequence[float | None],
    warning_scorable_flags: Sequence[bool],
    warning_policy: AlarmPolicy | None,
    warning_detector_name: str | None,
    range_start: pd.Timestamp,
    range_end: pd.Timestamp,
) -> list[WarningEpisode]:
    """Assemble persistent ``pca_spe`` warnings and telemetry-integrity warnings."""
    if not timestamps:
        return []
    first_window = timestamps[0]
    spans: list[_RawWarningSpan] = []

    if warning_policy is not None:
        tracker = AlarmTracker(window, warning_policy)
        closed: list[Episode] = []
        for ts, score, can_score in zip(
            timestamps, warning_scores, warning_scorable_flags, strict=True
        ):
            events = tracker.push(ts, score, explained=not can_score)
            closed.extend(e.episode for e in events if e.kind == "closed")
        ongoing_ep = tracker.open_episode
        closed.extend(e.episode for e in tracker.finish() if e.kind == "closed")

        score_by_ts = {ts: sc for ts, sc in zip(timestamps, warning_scores, strict=True)}
        exc_by_ts = {ts: ex for ts, ex in zip(timestamps, warning_exceedances, strict=True)}
        for ep in closed:
            ep_excs = [
                ex for ts, ex in exc_by_ts.items() if ep.start <= ts < ep.end and ex is not None
            ]
            ep_scores = [
                sc for ts, sc in score_by_ts.items() if ep.start <= ts < ep.end and sc is not None
            ]
            peak_exc = min(ep_excs) if ep_excs else None
            peak_sc = max(ep_scores) if ep_scores else ep.peak_score
            spans.append(
                _RawWarningSpan(
                    start=ep.start,
                    end=ep.end,
                    raised_at=ep.raised_at,
                    peak_score=peak_sc,
                    peak_exceedance=peak_exc,
                    sources={warning_detector_name or "pca_spe"},
                    truncated=ep.start <= first_window,
                    ongoing=ongoing_ep is not None and ep.start == ongoing_ep.start,
                )
            )

    spans.extend(
        _collect_integrity_spans(
            timestamps=timestamps,
            regimes=regimes,
            window=window,
            first_window=first_window,
        )
    )
    if not spans:
        return []

    spans.sort(key=lambda s: (s.start, s.end))
    merged: list[_RawWarningSpan] = [spans[0]]
    for span in spans[1:]:
        prev = merged[-1]
        if span.start <= prev.end:
            prev.end = max(prev.end, span.end)
            prev.raised_at = min(prev.raised_at, span.raised_at)
            if span.peak_score is not None:
                prev.peak_score = (
                    span.peak_score
                    if prev.peak_score is None
                    else max(prev.peak_score, span.peak_score)
                )
            if span.peak_exceedance is not None:
                prev.peak_exceedance = (
                    span.peak_exceedance
                    if prev.peak_exceedance is None
                    else min(prev.peak_exceedance, span.peak_exceedance)
                )
            prev.sources.update(span.sources)
            prev.truncated = prev.truncated or span.truncated
            prev.ongoing = prev.ongoing or span.ongoing
        else:
            merged.append(span)

    out: list[WarningEpisode] = []
    for span in merged:
        if not (span.end > range_start and span.start < range_end):
            continue
        ordered_sources = tuple(sorted(span.sources))
        has_pca = any(s not in INTEGRITY_REASONS for s in ordered_sources)
        has_stale = "stale_telemetry" in span.sources
        has_incon = "signal_inconsistency" in span.sources
        if has_pca and (has_stale or has_incon):
            kind = "mixed_consistency_warning"
            det_label = f"{warning_detector_name or 'pca_spe'}+signal_integrity"
        elif has_stale and has_incon:
            kind = "mixed_consistency_warning"
            det_label = "signal_integrity"
        elif has_stale:
            kind = "stale_telemetry"
            det_label = "signal_integrity"
        elif has_incon:
            kind = "signal_inconsistency"
            det_label = "signal_integrity"
        else:
            kind = "multivariate_sensor_drift"
            det_label = warning_detector_name or "pca_spe"

        dominant, breakdown = summarize_episode_regimes(regimes, span.start, span.end)
        hypothesis, uncertainty = episode_hypothesis_and_uncertainty(
            regimes,
            span.start,
            span.end,
            event_class="sensor_consistency_warning",
            sources=ordered_sources,
        )
        out.append(
            WarningEpisode(
                start=span.start,
                end=span.end,
                raised_at=span.raised_at,
                peak_score=span.peak_score,
                peak_exceedance=span.peak_exceedance,
                detector_name=det_label,
                warning_kind=kind,
                sources=ordered_sources,
                truncated=span.truncated,
                ongoing=span.ongoing,
                event_class="sensor_consistency_warning",
                hypothesis=hypothesis,
                uncertainty_note=uncertainty,
                dominant_regime=dominant,
                regime_breakdown=breakdown,
            )
        )
    return out


__all__ = [
    "EVENT_MEANINGS",
    "INTEGRITY_REASONS",
    "NON_SCORABLE_REASONS",
    "EventClass",
    "WarningEpisode",
    "WindowEventClassification",
    "WindowStatus",
    "build_warning_episodes",
    "classify_window_event",
    "episode_hypothesis_and_uncertainty",
    "is_integrity_warning_regime",
    "primary_leak_scorable",
    "summarize_episode_regimes",
    "warning_scorable",
]
