"""Causal, evidence-based operating-regime classification for MetroPT-3 APU windows.

Four physical operating regimes are defined from the documented APU signals:

- ``stopped``: motor off (``Motor_current < 1.0 A``) for at least ``stopped_off_fraction`` (0.9).
- ``unloaded``: motor running unloaded (``1.0 <= Motor_current <= 5.5 A``) for >= 0.9 of the window.
- ``loaded``: motor under compression load (``Motor_current > 5.5 A``) for >= 0.9 of the window.
- ``starting``: electrical startup inrush (``Motor_current > 8.0 A``) observed, or contiguous
  resumption out of ``stopped`` / a telemetry gap with active startup cycling.

Scientific honesty rules:

- Missing, gapped, frozen/stale (``TP3``, ``TP2`` and ``Motor_current`` simultaneously constant),
  or contradictory valve/current windows are never forced into an operating regime; they are
  classified as ``insufficient_evidence``.
- Mixed 10-minute operating cycles (e.g. normal alternating unloaded/stopped/loaded cycles where
  no single steady regime reaches dominance) are labelled ``insufficient_evidence`` with
  ``mixed_duty_cycle`` rather than rounded to an arbitrary majority label.
- ``starting`` transitions carry ``confidence="transitional"`` and explicitly record that ~0.1 Hz
  downsampling misses sub-10 s electrical inrush in most starts unless higher-rate signals exist.
- Sequence tracking applies left-to-right hysteresis without rewriting earlier observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from factorymind.anomaly.features import WindowSpec

OperatingRegime = Literal["stopped", "unloaded", "loaded", "starting", "insufficient_evidence"]
RegimeConfidence = Literal["confirmed", "transitional", "insufficient_evidence"]

DOMINANT_REGIME_FRACTION: Final = 0.9
HOLD_REGIME_FRACTION: Final = 0.8
MAX_INCONSISTENCY_FRACTION: Final = 0.1
HYSTERESIS_STOPPED_HOLD_FRACTION: Final = 0.85

MISSING_SIGNALS_TRANSITION: Final[tuple[str, ...]] = (
    "high_rate_electrical_current_gt_1hz",
    "plc_run_start_command",
)
MISSING_SIGNALS_MIXED: Final[tuple[str, ...]] = (
    "sub_minute_window_resolution",
    "plc_operating_mode_tag",
)
MISSING_SIGNALS_INTEGRITY: Final[tuple[str, ...]] = ("valid_non_stale_telemetry",)


@dataclass(frozen=True)
class RegimeAssessment:
    """Operating-regime decision paired with its physical evidence and uncertainty."""

    window_start: pd.Timestamp
    window_end: pd.Timestamp
    regime: OperatingRegime
    confidence: RegimeConfidence
    reason: str
    summary: str
    quality_flags: tuple[str, ...]
    missing_signals: tuple[str, ...]
    off_fraction: float | None
    unloaded_fraction: float | None
    load_duty: float | None
    load_starts: float | None
    inrush_samples: int
    valve_contradiction_fraction: float | None
    motor_valve_mismatch_fraction: float | None
    minutes_since_resume: float | None
    stale_telemetry: bool


def _finite(row: pd.Series, key: str) -> float | None:
    if key not in row:
        return None
    val = float(row[key])
    return val if np.isfinite(val) else None


def _make_assessment(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    regime: OperatingRegime,
    confidence: RegimeConfidence,
    reason: str,
    summary: str,
    quality_flags: tuple[str, ...],
    missing_signals: tuple[str, ...],
    off_fraction: float | None,
    unloaded_fraction: float | None,
    load_duty: float | None,
    load_starts: float | None,
    inrush_samples: int,
    valve_contra: float | None,
    mv_mismatch: float | None,
    minutes_since_resume: float | None,
    stale: bool,
) -> RegimeAssessment:
    return RegimeAssessment(
        window_start=start,
        window_end=end,
        regime=regime,
        confidence=confidence,
        reason=reason,
        summary=summary,
        quality_flags=quality_flags,
        missing_signals=missing_signals,
        off_fraction=off_fraction,
        unloaded_fraction=unloaded_fraction,
        load_duty=load_duty,
        load_starts=load_starts,
        inrush_samples=inrush_samples,
        valve_contradiction_fraction=valve_contra,
        motor_valve_mismatch_fraction=mv_mismatch,
        minutes_since_resume=minutes_since_resume,
        stale_telemetry=stale,
    )


class RegimeTracker:
    """Causal window-by-window regime classifier with anti-oscillation hysteresis."""

    def __init__(self, spec: WindowSpec) -> None:
        self._spec = spec
        self._window = pd.Timedelta(spec.length)
        self._has_history = False
        self._last_ts: pd.Timestamp | None = None
        self._last_regime: OperatingRegime | None = None

    def push(
        self,
        start: pd.Timestamp,
        row: pd.Series,
        *,
        end: pd.Timestamp | None = None,
    ) -> RegimeAssessment:
        window_end = end if end is not None else start + self._window
        contiguous = self._last_ts is not None and (start - self._last_ts) == self._window
        resumed_after_gap = self._has_history and (_finite(row, "minutes_since_resume") == 0.0)

        off_fraction = _finite(row, "off_fraction")
        unloaded_fraction = _finite(row, "unloaded_fraction")
        load_duty = _finite(row, "load_duty")
        load_starts = _finite(row, "load_starts")
        raw_inrush = _finite(row, "inrush_samples")
        inrush_samples = int(raw_inrush) if raw_inrush is not None else 0
        valve_contra = _finite(row, "valve_contradiction_fraction")
        mv_mismatch = _finite(row, "motor_valve_mismatch_fraction")
        minutes_since_resume = _finite(row, "minutes_since_resume")
        stale = bool(row.get("stale_telemetry", False))

        def emit(
            regime: OperatingRegime,
            confidence: RegimeConfidence,
            reason: str,
            summary: str,
            quality_flags: tuple[str, ...] = (),
            missing_signals: tuple[str, ...] = (),
        ) -> RegimeAssessment:
            out = _make_assessment(
                start=start,
                end=window_end,
                regime=regime,
                confidence=confidence,
                reason=reason,
                summary=summary,
                quality_flags=quality_flags,
                missing_signals=missing_signals,
                off_fraction=off_fraction,
                unloaded_fraction=unloaded_fraction,
                load_duty=load_duty,
                load_starts=load_starts,
                inrush_samples=inrush_samples,
                valve_contra=valve_contra,
                mv_mismatch=mv_mismatch,
                minutes_since_resume=minutes_since_resume,
                stale=stale,
            )
            self._has_history = True
            self._last_ts = start
            self._last_regime = regime
            return out

        if not bool(row.get("valid", False)):
            return emit(
                "insufficient_evidence",
                "insufficient_evidence",
                "incomplete_or_gapped",
                "Window fails sample coverage, internal gap, or 0.1 Hz cadence requirements.",
                quality_flags=("incomplete_or_gapped",),
                missing_signals=MISSING_SIGNALS_INTEGRITY,
            )
        if (
            off_fraction is None
            or unloaded_fraction is None
            or load_duty is None
            or valve_contra is None
            or mv_mismatch is None
        ):
            return emit(
                "insufficient_evidence",
                "insufficient_evidence",
                "incomplete_or_gapped",
                "Required motor-current or valve signals are missing or non-finite.",
                quality_flags=("missing_signals",),
                missing_signals=MISSING_SIGNALS_INTEGRITY,
            )
        if stale:
            return emit(
                "insufficient_evidence",
                "insufficient_evidence",
                "stale_telemetry",
                "TP3, TP2 and Motor_current are simultaneously constant (frozen telemetry); "
                "operating regime cannot be confirmed.",
                quality_flags=("stale_telemetry",),
                missing_signals=MISSING_SIGNALS_INTEGRITY,
            )
        incon_flags: list[str] = []
        if valve_contra > MAX_INCONSISTENCY_FRACTION:
            incon_flags.append("valve_contradiction")
        if mv_mismatch > MAX_INCONSISTENCY_FRACTION:
            incon_flags.append("motor_valve_mismatch")
        if incon_flags:
            return emit(
                "insufficient_evidence",
                "insufficient_evidence",
                "signal_inconsistency",
                "Motor current and COMP/DV_eletric valve states contradict each other across "
                "more than 10 % of the window.",
                quality_flags=tuple(incon_flags),
                missing_signals=("consistent_motor_valve_telemetry",),
            )

        starts = load_starts or 0.0
        if bool(row.get("stopped", False)) or off_fraction >= self._spec.stopped_off_fraction:
            return emit(
                "stopped",
                "confirmed",
                "confirmed_stopped",
                f"Motor off for {off_fraction:.0%} of window (explained stopped state).",
            )

        from_stopped = contiguous and self._last_regime == "stopped"
        inrush_observed = inrush_samples >= 1
        minor_stop_bounce = (
            from_stopped
            and off_fraction >= HYSTERESIS_STOPPED_HOLD_FRACTION
            and starts == 0.0
            and load_duty == 0.0
            and not inrush_observed
        )
        if minor_stop_bounce:
            return emit(
                "insufficient_evidence",
                "insufficient_evidence",
                "hysteresis_hold",
                "Brief boundary dip below 0.90 off fraction without load start or inrush; "
                "held out of starting to prevent stopped/starting chatter.",
                quality_flags=("transition_hysteresis_hold",),
                missing_signals=MISSING_SIGNALS_TRANSITION,
            )

        startup_transition = (
            (from_stopped or resumed_after_gap)
            and load_duty < DOMINANT_REGIME_FRACTION
            and unloaded_fraction < DOMINANT_REGIME_FRACTION
            and (starts >= 1.0 or load_duty > 0.0 or unloaded_fraction > 0.0)
        )
        if inrush_observed or startup_transition:
            reason = "startup_inrush" if inrush_observed else "startup_transition"
            flags: tuple[str, ...] = (
                ("electrical_inrush_observed",)
                if inrush_observed
                else ("coarse_sampling_inrush_limit",)
            )
            missing: tuple[str, ...] = () if inrush_observed else MISSING_SIGNALS_TRANSITION
            summary = (
                f"Startup inrush > 8.0 A captured ({inrush_samples} sample(s))."
                if inrush_observed
                else "Transitional resumption from stopped/gap with active startup cycling; "
                "0.1 Hz sampling misses sub-10 s electrical inrush in most starts."
            )
            return emit(
                "starting",
                "transitional",
                reason,
                summary,
                quality_flags=flags,
                missing_signals=missing,
            )

        hold_loaded = (
            contiguous and self._last_regime == "loaded" and load_duty >= HOLD_REGIME_FRACTION
        )
        if load_duty >= DOMINANT_REGIME_FRACTION or hold_loaded:
            confirmed = load_duty >= DOMINANT_REGIME_FRACTION
            return emit(
                "loaded",
                "confirmed",
                "confirmed_loaded" if confirmed else "hysteresis_hold",
                f"Compressor under load for {load_duty:.0%} of the window.",
                quality_flags=() if confirmed else ("transition_hysteresis_hold",),
            )

        hold_unloaded = (
            contiguous
            and self._last_regime == "unloaded"
            and unloaded_fraction >= HOLD_REGIME_FRACTION
        )
        if unloaded_fraction >= DOMINANT_REGIME_FRACTION or hold_unloaded:
            confirmed = unloaded_fraction >= DOMINANT_REGIME_FRACTION
            return emit(
                "unloaded",
                "confirmed",
                "confirmed_unloaded" if confirmed else "hysteresis_hold",
                f"Compressor running unloaded for {unloaded_fraction:.0%} of the window.",
                quality_flags=() if confirmed else ("transition_hysteresis_hold",),
            )

        return emit(
            "insufficient_evidence",
            "insufficient_evidence",
            "mixed_duty_cycle",
            f"Mixed 10-minute operating cycle (off {off_fraction:.0%}, unloaded "
            f"{unloaded_fraction:.0%}, loaded {load_duty:.0%}); no single steady regime dominates.",
            quality_flags=("mixed_duty_cycle",),
            missing_signals=MISSING_SIGNALS_MIXED,
        )


def classify_window(
    start: pd.Timestamp,
    end: pd.Timestamp,
    row: pd.Series,
    spec: WindowSpec,
) -> RegimeAssessment:
    """Stateless regime assessment for one isolated telemetry block."""
    return RegimeTracker(spec).push(start, row, end=end)


def classify_sequence(features: pd.DataFrame, spec: WindowSpec) -> list[RegimeAssessment]:
    """Chronological, non-retroactive operating-regime sequence over ``window_features`` rows."""
    tracker = RegimeTracker(spec)
    window = pd.Timedelta(spec.length)
    results: list[RegimeAssessment] = []
    for ts, row in features.iterrows():
        stamp = pd.Timestamp(str(ts))
        results.append(tracker.push(stamp, row, end=stamp + window))
    return results


__all__ = [
    "DOMINANT_REGIME_FRACTION",
    "HOLD_REGIME_FRACTION",
    "HYSTERESIS_STOPPED_HOLD_FRACTION",
    "MAX_INCONSISTENCY_FRACTION",
    "OperatingRegime",
    "RegimeAssessment",
    "RegimeConfidence",
    "RegimeTracker",
    "classify_sequence",
    "classify_window",
]
