"""Deterministic physical multi-hypothesis RCA engine grounded in EventEvidencePackage.

Separates physical causes (pneumatic air leakage, abnormal high downstream demand),
instrumentation faults (single-sensor freeze/drift/offset/out-of-range), telemetry
integrity failures (whole-feed freeze, valve contradictions), concurrent conditions,
and explicit abstention when evidence is insufficient.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Final

from factorymind.contracts.events import (
    DerivedFeatureSummary,
    EventEvidencePackage,
    EventPhaseName,
    ObservedSignalSummary,
    PhaseEvidenceSummary,
)
from factorymind.contracts.rca import (
    EvidenceReference,
    RankedHypothesis,
    RcaAssessment,
    RcaConfidenceBand,
    RelativeSupport,
    VerificationStep,
)

_HIGH_DUTY_THRESHOLD: Final = 0.45
_SEVERE_DUTY_THRESHOLD: Final = 0.65
_DUTY_RISE_THRESHOLD: Final = 0.15
_LEAK_DECAY_SLOPE_MAX: Final = -0.05  # bar/min when unloaded
_LOW_PRESSURE_BAR: Final = 8.0
_WARM_OIL_DELTA_C: Final = 4.0
_MIN_DIAGNOSTIC_SUPPORT: Final = 0.25

_CONTINUOUS_CORE_SENSORS: Final[frozenset[str]] = frozenset(
    {"TP3", "TP2", "Reservoirs", "Motor_current", "Oil_temperature"}
)
_NEGATION_RE: Final = re.compile(
    r"\b(?:no|not|without|never|neither|intact|normal|ok|cleared|negative|none|passed)\b"
)
_LEAK_POSITIVE_RE: Final = re.compile(
    r"\b(?:air\s*leak(?:age)?|leaking|leak|hiss(?:ing)?|purge\s+line|soap\s+test|blow[- ]?by)\b"
)
_SENSOR_POSITIVE_RE: Final = re.compile(
    r"\b(?:frozen|stuck|offset|drifted?|out[- ]of[- ]range|out\s+of\s+calibration|"
    r"calibration\s+(?:error|drift|fault)|wiring\s+fault|"
    r"(?:sensor|transducer|gauge|switch)\s+(?:fault|fail(?:ed|ure)?|stuck|frozen|bad|offline)|"
    r"bad\s+(?:sensor|transducer|gauge|switch))\b"
)


@dataclass(frozen=True)
class _ExtractedSignals:
    by_phase: dict[EventPhaseName, PhaseEvidenceSummary]
    baseline: PhaseEvidenceSummary | None
    focus: PhaseEvidenceSummary
    recovery: PhaseEvidenceSummary | None
    peak_duty: float | None
    base_duty: float | None
    duty_delta: float | None
    non_stale_valid_windows: int
    duty_trusted: bool
    peak_unloaded_slope: float | None
    base_unloaded_slope: float | None
    peak_loaded_rise: float | None
    base_loaded_rise: float | None
    peak_load_starts: float | None
    tp3_mean: float | None
    base_tp3_mean: float | None
    res_mean: float | None
    oil_mean: float | None
    base_oil_mean: float | None
    lps_active: float | None
    caudal_active: float | None
    primary_score: float | None
    warning_score: float | None
    warning_active_total: int
    has_primary_exceedance: bool
    has_concurrent_leak_alarm: bool
    has_concurrent_pca_warning: bool
    valve_contradiction_count: int
    motor_mismatch_count: int
    starting_present: bool


def _signal(phase: PhaseEvidenceSummary | None, name: str) -> ObservedSignalSummary | None:
    if phase is None:
        return None
    for item in phase.observed_signals:
        if item.signal == name:
            return item
    return None


def _feature(phase: PhaseEvidenceSummary | None, name: str) -> DerivedFeatureSummary | None:
    if phase is None:
        return None
    for item in phase.derived_features:
        if item.name == name:
            return item
    return None


def _feat_val(phase: PhaseEvidenceSummary | None, name: str) -> float | None:
    item = _feature(phase, name)
    if item is None or not item.observable:
        return None
    return item.value


def _sig_mean(phase: PhaseEvidenceSummary | None, name: str) -> float | None:
    item = _signal(phase, name)
    if item is None or item.sample_count == 0:
        return None
    return item.mean if item.mean is not None else item.active_fraction


def _sig_frozen(phase: PhaseEvidenceSummary | None, name: str) -> bool:
    item = _signal(phase, name)
    return bool(item and item.frozen)


def _best_focus_phase(by_phase: dict[EventPhaseName, PhaseEvidenceSummary]) -> PhaseEvidenceSummary:
    for pname in ("peak", "evolution", "onset"):
        p = by_phase.get(pname)
        if p is not None and p.available:
            return p
    return by_phase.get("peak") or by_phase["onset"]


def _extract_signals(pkg: EventEvidencePackage) -> _ExtractedSignals:
    by_phase: dict[EventPhaseName, PhaseEvidenceSummary] = {p.phase: p for p in pkg.phases}
    baseline = by_phase.get("pre_event_baseline")
    if baseline is not None and not baseline.available:
        baseline = None
    recovery = by_phase.get("recovery_or_persistence")
    focus = _best_focus_phase(by_phase)

    peak_duty = _feat_val(focus, "load_duty")
    base_duty = _feat_val(baseline, "load_duty")
    duty_delta = (
        round(peak_duty - base_duty, 4) if peak_duty is not None and base_duty is not None else None
    )

    peak_slope = _feat_val(focus, "unloaded_tp3_slope")
    base_slope = _feat_val(baseline, "unloaded_tp3_slope")
    peak_rise = _feat_val(focus, "loaded_tp3_rise")
    base_rise = _feat_val(baseline, "loaded_tp3_rise")
    peak_starts = _feat_val(focus, "load_starts")

    tp3_mean = _sig_mean(focus, "TP3")
    base_tp3_mean = _sig_mean(baseline, "TP3")
    res_mean = _sig_mean(focus, "Reservoirs")
    oil_mean = _sig_mean(focus, "Oil_temperature")
    base_oil_mean = _sig_mean(baseline, "Oil_temperature")
    lps_active = _sig_mean(focus, "LPS")
    caudal_active = _sig_mean(focus, "Caudal_impulses")

    core_phases = [by_phase[k] for k in ("onset", "evolution", "peak") if k in by_phase]
    p_scores = [p.primary_peak_score for p in core_phases if p.primary_peak_score is not None]
    w_scores = [p.warning_peak_score for p in core_phases if p.warning_peak_score is not None]
    warn_total = sum(p.warning_active_windows for p in core_phases)

    has_leak_alarm = any(
        n.kind == "air_leak_alarm" and n.relationship == "concurrent"
        for n in pkg.neighboring_events
    )
    has_pca_warn = any(
        n.kind == "sensor_consistency_warning"
        and n.relationship == "concurrent"
        and "pca_spe" in n.detector_or_source
        for n in pkg.neighboring_events
    ) or any(
        t.category == "consistency_warning"
        and "pca_spe" in t.details.get("sources", [])
        and t.phase in ("onset", "evolution", "peak")
        for t in pkg.timeline
    )
    non_stale_valid = max(0, pkg.quality.valid_event_windows - pkg.quality.stale_windows_count)
    focus_motor_frozen = _sig_frozen(focus, "Motor_current")
    duty_trusted = (
        non_stale_valid > 0
        and not focus_motor_frozen
        and "Motor_current" not in pkg.quality.out_of_range_signals
    )
    has_primary_exc = any(
        p.primary_min_exceedance is not None and p.primary_min_exceedance <= 0.05
        for p in core_phases
    )
    valve_cnt = sum(1 for c in pkg.quality.contradictions if c.kind == "valve_contradiction")
    motor_cnt = sum(1 for c in pkg.quality.contradictions if c.kind == "motor_valve_mismatch")
    starting_present = any("starting" in p.regime_counts for p in core_phases)

    return _ExtractedSignals(
        by_phase=by_phase,
        baseline=baseline,
        focus=focus,
        recovery=recovery,
        peak_duty=peak_duty,
        base_duty=base_duty,
        duty_delta=duty_delta,
        non_stale_valid_windows=non_stale_valid,
        duty_trusted=duty_trusted,
        peak_unloaded_slope=peak_slope,
        base_unloaded_slope=base_slope,
        peak_loaded_rise=peak_rise,
        base_loaded_rise=base_rise,
        peak_load_starts=peak_starts,
        tp3_mean=tp3_mean,
        base_tp3_mean=base_tp3_mean,
        res_mean=res_mean,
        oil_mean=oil_mean,
        base_oil_mean=base_oil_mean,
        lps_active=lps_active,
        caudal_active=caudal_active,
        primary_score=max(p_scores) if p_scores else None,
        warning_score=max(w_scores) if w_scores else None,
        warning_active_total=warn_total,
        has_primary_exceedance=has_primary_exc,
        has_concurrent_leak_alarm=has_leak_alarm,
        has_concurrent_pca_warning=has_pca_warn,
        valve_contradiction_count=valve_cnt,
        motor_mismatch_count=motor_cnt,
        starting_present=starting_present,
    )


def _confidence_band(
    score: float,
    pkg: EventEvidencePackage,
    *,
    abstained: bool,
    contradictions_count: int,
) -> RcaConfidenceBand:
    if abstained:
        return "abstained"
    effective = score * (0.5 + 0.5 * pkg.quality.confidence_penalty_factor)
    if (
        pkg.quality.overall_status == "degraded" and pkg.quality.confidence_penalty_factor < 0.75
    ) or contradictions_count >= 2:
        return "low"
    if (
        pkg.quality.overall_status == "sufficient"
        and pkg.quality.confidence_penalty_factor >= 0.9
        and effective >= 0.7
        and contradictions_count == 0
    ):
        return "high"
    if effective >= 0.42:
        return "moderate"
    return "low"


def _human_obs_signals(
    pkg: EventEvidencePackage,
) -> tuple[list[EvidenceReference], list[EvidenceReference]]:
    """Classify human observations using negation-aware clause matching."""
    leak_refs: list[EvidenceReference] = []
    sensor_refs: list[EvidenceReference] = []
    for idx, obs in enumerate(pkg.human_observations, start=1):
        clauses = [c.strip() for c in re.split(r"[,.;!?\n:]+", obs.note.lower()) if c.strip()]
        matched_leak = False
        matched_sensor = False
        for clause in clauses:
            if _NEGATION_RE.search(clause):
                continue
            if _LEAK_POSITIVE_RE.search(clause):
                matched_leak = True
            if _SENSOR_POSITIVE_RE.search(clause):
                matched_sensor = True
        ref = EvidenceReference(
            ref_id=f"human_obs:{idx}",
            origin="human_observation",
            phase=None,
            signal_or_feature=obs.category,
            statement=f"Operator/engineer note ({obs.actor_id}): {obs.note}",
        )
        if matched_leak:
            leak_refs.append(ref)
        if matched_sensor:
            sensor_refs.append(ref)
    return leak_refs, sensor_refs


def _phase_for_contradiction(
    pkg: EventEvidencePackage, kind: str, fallback: EventPhaseName
) -> EventPhaseName:
    for c in pkg.quality.contradictions:
        if c.kind == kind:
            for p in pkg.phases:
                if p.phase != "peak" and p.start <= c.timestamp < p.end:
                    return p.phase
            return fallback
    return fallback


def _common_missing_refs(
    pkg: EventEvidencePackage, ex: _ExtractedSignals
) -> list[EvidenceReference]:
    refs: list[EvidenceReference] = [
        EvidenceReference(
            ref_id="missing:flowmeter",
            origin="derived",
            phase=None,
            signal_or_feature="air_demand_flowmeter",
            statement=(
                "No independent pneumatic downstream air-demand flowmeter exists; elevated "
                "compressor duty can reflect either leakage or sustained high pneumatic demand."
            ),
        )
    ]
    if ex.baseline is None:
        refs.append(
            EvidenceReference(
                ref_id="missing:baseline",
                origin="derived",
                phase="pre_event_baseline",
                signal_or_feature="pre_event_baseline",
                statement=(
                    "Pre-event baseline telemetry unavailable; pre/post shift magnitude unverified."
                ),
            )
        )
    if ex.starting_present:
        refs.append(
            EvidenceReference(
                ref_id="missing:high_rate_current",
                origin="derived",
                phase=ex.focus.phase,
                signal_or_feature="high_rate_electrical_current_gt_1hz",
                statement=(
                    "~0.1 Hz sampling misses sub-10 s motor startup inrush and PLC run command."
                ),
            )
        )
    if pkg.quality.recovery_status == "unverified":
        refs.append(
            EvidenceReference(
                ref_id="missing:recovery_tail",
                origin="derived",
                phase="recovery_or_persistence",
                signal_or_feature="recovery_or_persistence",
                statement=(
                    "Post-event recovery telemetry missing; fault persistence vs clearance "
                    "unverified."
                ),
            )
        )
    return refs


def _duty_elevated(ex: _ExtractedSignals) -> bool:
    if not ex.duty_trusted or ex.peak_duty is None:
        return False
    if ex.peak_duty >= _HIGH_DUTY_THRESHOLD:
        return True
    return bool(
        ex.duty_delta is not None
        and ex.duty_delta >= _DUTY_RISE_THRESHOLD
        and (ex.has_concurrent_leak_alarm or ex.has_primary_exceedance)
    )


def _build_air_leakage_hypothesis(
    pkg: EventEvidencePackage,
    ex: _ExtractedSignals,
    human_leak_refs: list[EvidenceReference],
    human_sensor_refs: list[EvidenceReference],
    common_missing: list[EvidenceReference],
) -> RankedHypothesis:
    sup: list[EvidenceReference] = []
    con: list[EvidenceReference] = []
    missing: list[EvidenceReference] = list(common_missing)
    score = 0.0

    focus_phase = ex.focus.phase
    duty_elevated = _duty_elevated(ex)
    if duty_elevated and ex.peak_duty is not None:
        pts = 0.36 if ex.peak_duty >= _SEVERE_DUTY_THRESHOLD else 0.24
        score += pts
        sup.append(
            EvidenceReference(
                ref_id="leak:load_duty",
                origin="derived",
                phase=focus_phase,
                signal_or_feature="load_duty",
                statement=(
                    f"Sustained compressor load duty reached {ex.peak_duty:.3f}"
                    + (f" vs baseline {ex.base_duty:.3f}." if ex.base_duty is not None else ".")
                ),
                observed_value=ex.peak_duty,
                unit="fraction",
                baseline_value=ex.base_duty,
            )
        )
    elif not ex.duty_trusted and ex.peak_duty is not None and ex.peak_duty >= _HIGH_DUTY_THRESHOLD:
        con.append(
            EvidenceReference(
                ref_id="leak:untrusted_duty",
                origin="observed",
                phase=focus_phase,
                signal_or_feature="Motor_current",
                statement=(
                    f"Apparent load duty ({ex.peak_duty:.3f}) is untrusted because Motor_current "
                    "was frozen, out of range, or all event windows were stale."
                ),
                observed_value=ex.peak_duty,
                unit="fraction",
            )
        )
    elif ex.peak_duty is not None:
        con.append(
            EvidenceReference(
                ref_id="leak:low_load_duty",
                origin="derived",
                phase=focus_phase,
                signal_or_feature="load_duty",
                statement=(
                    f"Load duty remained {ex.peak_duty:.3f} (< {_HIGH_DUTY_THRESHOLD:.2f}), "
                    "inconsistent with sustained pneumatic air-leakage demand."
                ),
                observed_value=ex.peak_duty,
                unit="fraction",
                baseline_value=ex.base_duty,
            )
        )

    if ex.duty_trusted and (
        ex.has_concurrent_leak_alarm
        or ex.has_primary_exceedance
        or (ex.primary_score is not None and ex.primary_score >= 0.7)
    ):
        score += 0.16
        sup.append(
            EvidenceReference(
                ref_id="leak:primary_detector",
                origin="model_prediction",
                phase=focus_phase,
                signal_or_feature=pkg.provenance.primary_detector,
                statement=(
                    f"Primary detector {pkg.provenance.primary_detector} fired with peak score "
                    f"{ex.primary_score:.3f}."
                    if ex.primary_score is not None
                    else (
                        "Concurrent air_leak_alarm episode raised by "
                        f"{pkg.provenance.primary_detector}."
                    )
                ),
                observed_value=ex.primary_score,
            )
        )

    if ex.peak_unloaded_slope is not None:
        if ex.peak_unloaded_slope <= _LEAK_DECAY_SLOPE_MAX and (
            ex.base_unloaded_slope is None or ex.peak_unloaded_slope < ex.base_unloaded_slope - 0.02
        ):
            score += 0.14
            sup.append(
                EvidenceReference(
                    ref_id="leak:unloaded_decay",
                    origin="derived",
                    phase=focus_phase,
                    signal_or_feature="unloaded_tp3_slope",
                    statement=(
                        f"TP3 decayed at {ex.peak_unloaded_slope:.3f} bar/min while unloaded, "
                        "matching downstream pneumatic bleeding."
                    ),
                    observed_value=ex.peak_unloaded_slope,
                    unit="bar/min",
                    baseline_value=ex.base_unloaded_slope,
                )
            )
        elif ex.peak_unloaded_slope > _LEAK_DECAY_SLOPE_MAX:
            con.append(
                EvidenceReference(
                    ref_id="leak:stable_unloaded_slope",
                    origin="derived",
                    phase=focus_phase,
                    signal_or_feature="unloaded_tp3_slope",
                    statement=(
                        f"Unloaded TP3 slope ({ex.peak_unloaded_slope:.3f} bar/min) did not drop "
                        "rapidly, weighing against a major downstream line leak."
                    ),
                    observed_value=ex.peak_unloaded_slope,
                    unit="bar/min",
                    baseline_value=ex.base_unloaded_slope,
                )
            )
    elif duty_elevated and ex.peak_duty is not None:
        missing.append(
            EvidenceReference(
                ref_id="leak:unobservable_unloaded_slope",
                origin="derived",
                phase=focus_phase,
                signal_or_feature="unloaded_tp3_slope",
                statement=(
                    "unloaded_tp3_slope is unobservable because compressor stayed loaded cleanly; "
                    "leak decay rate cannot be separated from continuous heavy load purely via "
                    "decay slope."
                ),
            )
        )

    if ex.tp3_mean is not None and (
        ex.tp3_mean < _LOW_PRESSURE_BAR
        or (ex.base_tp3_mean is not None and ex.tp3_mean < ex.base_tp3_mean - 0.4)
    ):
        score += 0.1
        delta_txt = (
            f" ({ex.tp3_mean - ex.base_tp3_mean:+.2f} bar vs baseline)"
            if ex.base_tp3_mean is not None
            else " (below 8.0 bar nominal target)"
        )
        sup.append(
            EvidenceReference(
                ref_id="leak:tp3_depression",
                origin="observed",
                phase=focus_phase,
                signal_or_feature="TP3",
                statement=f"Mean panel pressure TP3 measured {ex.tp3_mean:.2f} bar{delta_txt}.",
                observed_value=ex.tp3_mean,
                unit="bar",
                baseline_value=ex.base_tp3_mean,
            )
        )

    if ex.lps_active is not None and ex.lps_active > 0.0:
        score += 0.08
        sup.append(
            EvidenceReference(
                ref_id="leak:lps_active",
                origin="observed",
                phase=focus_phase,
                signal_or_feature="LPS",
                statement=(
                    f"Low-pressure switch LPS activated ({ex.lps_active:.2f} active fraction)."
                ),
                observed_value=ex.lps_active,
                unit="state (0/1)",
            )
        )

    if (
        ex.oil_mean is not None
        and ex.base_oil_mean is not None
        and ex.oil_mean - ex.base_oil_mean >= _WARM_OIL_DELTA_C
    ):
        score += 0.05
        sup.append(
            EvidenceReference(
                ref_id="leak:oil_temp_rise",
                origin="observed",
                phase=focus_phase,
                signal_or_feature="Oil_temperature",
                statement=(
                    f"Compressor oil temperature rose +{ex.oil_mean - ex.base_oil_mean:.1f} degC "
                    "above pre-event baseline under sustained duty."
                ),
                observed_value=ex.oil_mean,
                unit="degC",
                baseline_value=ex.base_oil_mean,
            )
        )

    if pkg.quality.recovery_status == "persisting":
        score += 0.06
        sup.append(
            EvidenceReference(
                ref_id="leak:persisting_recovery",
                origin="derived",
                phase="recovery_or_persistence",
                signal_or_feature="recovery_or_persistence",
                statement="Elevated duty or anomaly state persisted after event_end.",
            )
        )

    if human_leak_refs:
        score += 0.08
        sup.extend(human_leak_refs)

    # Integrity contradictions erode pure air-leak attribution
    if (
        pkg.quality.stale_windows_count > 0
        or ex.valve_contradiction_count > 0
        or ex.motor_mismatch_count > 0
    ):
        score *= 0.45
        if pkg.quality.stale_windows_count > 0:
            conflict_sig = "stale_telemetry"
            conflict_phase = _phase_for_contradiction(pkg, "sensor_freeze", focus_phase)
        elif ex.valve_contradiction_count > 0:
            conflict_sig = "valve_contradiction"
            conflict_phase = _phase_for_contradiction(pkg, "valve_contradiction", focus_phase)
        else:
            conflict_sig = "motor_valve_mismatch"
            conflict_phase = _phase_for_contradiction(pkg, "motor_valve_mismatch", focus_phase)
        con.append(
            EvidenceReference(
                ref_id="leak:integrity_conflict",
                origin="derived",
                phase=conflict_phase,
                signal_or_feature=conflict_sig,
                statement=(
                    "Co-occurring stale/frozen telemetry or valve contradictions weaken isolated "
                    "pneumatic air-leak attribution."
                ),
            )
        )
    if pkg.quality.frozen_sensors:
        frozen_core = [
            s
            for s in pkg.quality.frozen_sensors
            if s in {"TP3", "TP2", "Reservoirs", "Motor_current"}
        ]
        if frozen_core and pkg.quality.stale_windows_count == 0:
            score *= 0.45 if "Motor_current" in frozen_core else 0.6
            con.append(
                EvidenceReference(
                    ref_id="leak:frozen_channel",
                    origin="observed",
                    phase=_phase_for_contradiction(pkg, "sensor_freeze", focus_phase),
                    signal_or_feature=frozen_core[0],
                    statement=(
                        f"Channel(s) {', '.join(frozen_core)} remained constant, preventing "
                        "reliable physical corroboration for an air leak."
                    ),
                )
            )
    if pkg.quality.out_of_range_signals:
        score *= 0.4
        con.append(
            EvidenceReference(
                ref_id="leak:oor_channel",
                origin="observed",
                phase=_phase_for_contradiction(pkg, "out_of_range", focus_phase),
                signal_or_feature=pkg.quality.out_of_range_signals[0],
                statement=(
                    f"Channel(s) {', '.join(pkg.quality.out_of_range_signals)} reported "
                    "out-of-range raw telemetry."
                ),
            )
        )
    if human_sensor_refs:
        con.extend(human_sensor_refs)

    # If peak load duty was not reliably elevated, cap raw air_leakage low
    if not duty_elevated:
        score = min(score, 0.2)

    score = round(
        min(1.0, max(0.0, score * (0.65 + 0.35 * pkg.quality.confidence_penalty_factor))), 4
    )
    return RankedHypothesis(
        hypothesis_id="hyp:air_leakage",
        category="air_leakage",
        title="Pneumatic air leakage downstream or at APU fittings",
        rank=1,
        relative_support="weak",
        support_score=score,
        confidence_band="low",
        physical_mechanism=(
            "Continuous downstream air escape prevents panel/reservoir stabilization, forcing "
            "prolonged compressor load duty, faster unloaded pressure decay, and potential "
            "thermal rise."
        ),
        supporting_evidence=sup,
        contradicting_evidence=con,
        missing_evidence=missing,
        correlation_vs_causation_note=(
            "High load_duty and pressure deviations statistically correlate with leaks, but duty "
            "exceedance alone does not prove an orifice leak without flow isolation or physical "
            "check."
        ),
        verification_steps=[
            VerificationStep(
                step_id="ver:leak_acoustic",
                priority="immediate",
                action=(
                    "During safe stationary inspection, check flexible hoses, air-dryer purge "
                    "line, check valves, and bogie suspension couplings with ultrasonic/soap "
                    "leak test."
                ),
                expected_if_confirmed=(
                    "Audible/ultrasonic hiss or rapid TP3 pressure drop when isolated."
                ),
                expected_if_refuted=(
                    "No external pressure bleed during isolated pressure-hold check."
                ),
            ),
            VerificationStep(
                step_id="ver:leak_duty_hold",
                priority="high",
                action=(
                    "Inspect train pneumatic demand log (doors, brakes, suspension) against "
                    "duty bursts."
                ),
                expected_if_confirmed=(
                    "Compressor stays loaded even when auxiliary pneumatic loads are idle."
                ),
                expected_if_refuted=(
                    "Load duty drops cleanly once downstream auxiliary cycling ceases."
                ),
            ),
        ],
    )


def _build_sensor_malfunction_hypothesis(
    pkg: EventEvidencePackage,
    ex: _ExtractedSignals,
    human_sensor_refs: list[EvidenceReference],
    common_missing: list[EvidenceReference],
) -> RankedHypothesis:
    sup: list[EvidenceReference] = []
    con: list[EvidenceReference] = []
    missing: list[EvidenceReference] = list(common_missing)
    score = 0.0
    focus_phase = ex.focus.phase

    if pkg.quality.frozen_sensors and pkg.quality.stale_windows_count == 0:
        core_frozen = [s for s in pkg.quality.frozen_sensors if s in _CONTINUOUS_CORE_SENSORS]
        score += 0.52 if core_frozen else 0.34
        freeze_phase = _phase_for_contradiction(pkg, "sensor_freeze", focus_phase)
        for s in pkg.quality.frozen_sensors[:4]:
            sup.append(
                EvidenceReference(
                    ref_id=f"sensor:freeze:{s}",
                    origin="observed",
                    phase=freeze_phase,
                    signal_or_feature=s,
                    statement=f"Channel {s} stayed constant (std <= 1e-12) across an event phase.",
                )
            )
    elif pkg.quality.frozen_sensors:
        extra = [s for s in pkg.quality.frozen_sensors if s not in {"TP3", "TP2", "Motor_current"}]
        if extra:
            score += 0.22
            freeze_phase = _phase_for_contradiction(pkg, "sensor_freeze", focus_phase)
            for s in extra[:3]:
                sup.append(
                    EvidenceReference(
                        ref_id=f"sensor:extra_freeze:{s}",
                        origin="observed",
                        phase=freeze_phase,
                        signal_or_feature=s,
                        statement=f"Channel {s} remained constant alongside whole-feed freeze.",
                    )
                )

    if pkg.quality.out_of_range_signals:
        score += 0.45
        oor_phase = _phase_for_contradiction(pkg, "out_of_range", focus_phase)
        for s in pkg.quality.out_of_range_signals:
            sup.append(
                EvidenceReference(
                    ref_id=f"sensor:oor:{s}",
                    origin="observed",
                    phase=oor_phase,
                    signal_or_feature=s,
                    statement=f"Channel {s} breached physical VALID_RANGES limits.",
                )
            )

    if ex.has_concurrent_pca_warning and pkg.provenance.warning_detector is not None:
        pts = (
            0.38
            if (not ex.duty_trusted or ex.peak_duty is None or ex.peak_duty < _HIGH_DUTY_THRESHOLD)
            else 0.14
        )
        score += pts
        sup.append(
            EvidenceReference(
                ref_id="sensor:pca_spe",
                origin="statistical_inference",
                phase=focus_phase,
                signal_or_feature=pkg.provenance.warning_detector,
                statement=(
                    f"Multivariate {pkg.provenance.warning_detector} detected "
                    f"cross-sensor structure departure ({ex.warning_active_total} active "
                    "warning window(s))."
                ),
                observed_value=ex.warning_score,
            )
        )
        for item in pkg.evidence_items[:3]:
            if (item.robust_z is not None and abs(item.robust_z) >= 3.0) or item.novel:
                sup.append(
                    EvidenceReference(
                        ref_id=f"sensor:item:{item.feature}",
                        origin="statistical_inference",
                        phase=item.phase,
                        signal_or_feature=item.feature,
                        statement=(
                            f"Feature {item.feature} deviated from normal profile ({item.meaning})."
                        ),
                        observed_value=item.value,
                        unit=item.unit,
                        baseline_value=item.training_median,
                    )
                )

    if human_sensor_refs:
        score += 0.15
        sup.extend(human_sensor_refs)

    if (
        ex.duty_trusted
        and ex.peak_duty is not None
        and ex.peak_duty >= _SEVERE_DUTY_THRESHOLD
        and ex.has_concurrent_leak_alarm
    ):
        score *= 0.65
        con.append(
            EvidenceReference(
                ref_id="sensor:systemic_load",
                origin="derived",
                phase=focus_phase,
                signal_or_feature="load_duty",
                statement=(
                    f"Motor load duty ({ex.peak_duty:.3f}) also shifted systemically, arguing "
                    "against an isolated single-channel instrumentation glitch alone."
                ),
                observed_value=ex.peak_duty,
                unit="fraction",
            )
        )

    score = round(
        min(1.0, max(0.0, score * (0.7 + 0.3 * pkg.quality.confidence_penalty_factor))), 4
    )
    return RankedHypothesis(
        hypothesis_id="hyp:sensor_malfunction",
        category="sensor_malfunction",
        title="Single-sensor malfunction, bias offset, or transducer freeze",
        rank=2,
        relative_support="weak",
        support_score=score,
        confidence_band="low",
        physical_mechanism=(
            "Faulty pressure transducer, oil thermistor, or current clamp produces frozen "
            "readings, calibration offset, drift, or physical cross-sensor decoupling flagged "
            "by PCA SPE."
        ),
        supporting_evidence=sup,
        contradicting_evidence=con,
        missing_evidence=missing,
        correlation_vs_causation_note=(
            "Multivariate PCA SPE departure indicates unusual channel correlation relative to "
            "training normal windows; physical calibration check is needed to prove transducer "
            "failure."
        ),
        verification_steps=[
            VerificationStep(
                step_id="ver:sensor_cross_check",
                priority="high",
                action=(
                    "Compare TP3, TP2, and Reservoirs readings against local mechanical "
                    "manifold gauges during static hold and verify oil thermistor resistance."
                ),
                expected_if_confirmed=(
                    "Panel reading stays fixed or offset from mechanical manifold gauge."
                ),
                expected_if_refuted=(
                    "Electronic channels track mechanical reference gauges cleanly."
                ),
            )
        ],
    )


def _build_telemetry_inconsistency_hypothesis(
    pkg: EventEvidencePackage,
    ex: _ExtractedSignals,
    common_missing: list[EvidenceReference],
) -> RankedHypothesis:
    sup: list[EvidenceReference] = []
    con: list[EvidenceReference] = []
    score = 0.0
    focus_phase = ex.focus.phase

    if pkg.quality.stale_windows_count > 0:
        score += 0.55
        sup.append(
            EvidenceReference(
                ref_id="integ:stale",
                origin="derived",
                phase=focus_phase,
                signal_or_feature="stale_telemetry",
                statement=(
                    f"{pkg.quality.stale_windows_count} event window(s) had simultaneously frozen "
                    "TP3, TP2, and Motor_current (stale_telemetry)."
                ),
                observed_value=float(pkg.quality.stale_windows_count),
            )
        )
    if ex.valve_contradiction_count > 0:
        score += 0.45
        sup.append(
            EvidenceReference(
                ref_id="integ:valve_contra",
                origin="derived",
                phase=focus_phase,
                signal_or_feature="valve_contradiction",
                statement=(
                    "COMP & DV_eletric were simultaneously active in "
                    f"{ex.valve_contradiction_count} contradiction record(s)."
                ),
                observed_value=float(ex.valve_contradiction_count),
            )
        )
    if ex.motor_mismatch_count > 0:
        score += 0.35
        sup.append(
            EvidenceReference(
                ref_id="integ:motor_valve",
                origin="derived",
                phase=focus_phase,
                signal_or_feature="motor_valve_mismatch",
                statement=(
                    "Motor current vs intake/outlet valve disagreement observed "
                    f"({ex.motor_mismatch_count} window(s))."
                ),
                observed_value=float(ex.motor_mismatch_count),
            )
        )
    if (
        pkg.quality.stale_windows_count == 0
        and ex.valve_contradiction_count == 0
        and ex.motor_mismatch_count == 0
    ):
        con.append(
            EvidenceReference(
                ref_id="integ:clean_checks",
                origin="derived",
                phase=focus_phase,
                signal_or_feature="coverage_ratio",
                statement="No whole-feed freeze or valve/motor physical contradictions detected.",
                observed_value=pkg.quality.coverage_ratio,
            )
        )

    score = round(min(1.0, max(0.0, score)), 4)
    return RankedHypothesis(
        hypothesis_id="hyp:telemetry_inconsistency",
        category="telemetry_inconsistency",
        title="PLC logger stall, stale telemetry buffer, or valve-state contradiction",
        rank=3,
        relative_support="weak",
        support_score=score,
        confidence_band="low",
        physical_mechanism=(
            "Data acquisition stall freezes unrelated channels simultaneously, or discrete valve "
            "limit-switch wiring contradicts electrical motor current state."
        ),
        supporting_evidence=sup,
        contradicting_evidence=con,
        missing_evidence=list(common_missing),
        correlation_vs_causation_note=(
            "Zero-variance channels across independent sensors indicate a telemetry-pipeline stall "
            "rather than physical compressor thermodynamics."
        ),
        verification_steps=[
            VerificationStep(
                step_id="ver:integ_daq",
                priority="immediate",
                action=(
                    "Inspect onboard train DAQ buffer timestamping, CAN/MVB link logs, and "
                    "valve limit-switch wiring."
                ),
                expected_if_confirmed=(
                    "PLC sample frame repeated or discrete switch input stuck high/low."
                ),
                expected_if_refuted=(
                    "Raw 1 Hz PLC buffer shows distinct dynamic samples matching physical "
                    "switching."
                ),
            )
        ],
    )


def _build_high_demand_hypothesis(
    pkg: EventEvidencePackage,
    ex: _ExtractedSignals,
    human_leak_refs: list[EvidenceReference],
    common_missing: list[EvidenceReference],
) -> RankedHypothesis:
    sup: list[EvidenceReference] = []
    con: list[EvidenceReference] = []
    score = 0.0
    focus_phase = ex.focus.phase

    if _duty_elevated(ex) and ex.peak_duty is not None:
        score += 0.25
        sup.append(
            EvidenceReference(
                ref_id="demand:flow_missing",
                origin="derived",
                phase=focus_phase,
                signal_or_feature="load_duty",
                statement=(
                    f"Elevated load duty ({ex.peak_duty:.3f}) can arise from heavy train braking, "
                    "door cycling, or suspension demand because no air-demand flowmeter exists."
                ),
                observed_value=ex.peak_duty,
                unit="fraction",
                baseline_value=ex.base_duty,
            )
        )
        if ex.peak_load_starts is not None and ex.peak_load_starts >= 2.0:
            score += 0.1
            sup.append(
                EvidenceReference(
                    ref_id="demand:starts",
                    origin="derived",
                    phase=focus_phase,
                    signal_or_feature="load_starts",
                    statement=(
                        f"Frequent compressor load starts ({ex.peak_load_starts:.1f}) match "
                        "intermittent consumer demand."
                    ),
                    observed_value=ex.peak_load_starts,
                    unit="cycles/window",
                )
            )
        if pkg.quality.recovery_status == "recovered":
            score += 0.12
            sup.append(
                EvidenceReference(
                    ref_id="demand:recovered",
                    origin="derived",
                    phase="recovery_or_persistence",
                    signal_or_feature="recovery_or_persistence",
                    statement=(
                        "Duty returned to normal post-event, consistent with temporary peak "
                        "passenger/operational load."
                    ),
                )
            )
    if ex.peak_unloaded_slope is not None and ex.peak_unloaded_slope <= _LEAK_DECAY_SLOPE_MAX:
        score -= 0.08
        con.append(
            EvidenceReference(
                ref_id="demand:unloaded_decay",
                origin="derived",
                phase=focus_phase,
                signal_or_feature="unloaded_tp3_slope",
                statement=(
                    f"Rapid unloaded TP3 decay ({ex.peak_unloaded_slope:.3f} bar/min) during idle "
                    "intervals favors leakage over normal service consumption."
                ),
                observed_value=ex.peak_unloaded_slope,
                unit="bar/min",
            )
        )
    if ex.lps_active is not None and ex.lps_active > 0.0:
        score -= 0.06
        con.append(
            EvidenceReference(
                ref_id="demand:lps_drop",
                origin="observed",
                phase=focus_phase,
                signal_or_feature="LPS",
                statement=(
                    "Low-pressure switch LPS activated (< 7 bar), indicating pressure "
                    "depletion exceeding normal duty."
                ),
                observed_value=ex.lps_active,
                unit="state (0/1)",
            )
        )
    if human_leak_refs:
        score -= 0.1
        con.extend(human_leak_refs)

    if pkg.quality.stale_windows_count > 0 and ex.peak_duty is None:
        score = 0.05

    score = round(
        min(0.85, max(0.0, score * (0.7 + 0.3 * pkg.quality.confidence_penalty_factor))), 4
    )
    return RankedHypothesis(
        hypothesis_id="hyp:abnormal_high_demand",
        category="abnormal_high_demand",
        title="Sustained downstream pneumatic demand (without leak)",
        rank=4,
        relative_support="weak",
        support_score=score,
        confidence_band="low",
        physical_mechanism=(
            "High passenger-door frequency, heavy grade braking, or air-suspension leveling raises "
            "legitimate APU airflow demand without a mechanical leak in the compressor skid."
        ),
        supporting_evidence=sup,
        contradicting_evidence=con,
        missing_evidence=list(common_missing),
        correlation_vs_causation_note=(
            "Because MetroPT-3 lacks a downstream flowmeter and train-door/brake duty tags, "
            "heavy operational demand stays an explicit alternative whenever leaks cannot be "
            "isolated."
        ),
        verification_steps=[
            VerificationStep(
                step_id="ver:demand_correlate",
                priority="standard",
                action=(
                    "Cross-check train operational duty (station dwell frequency, brake "
                    "applications, car loading) against event timing."
                ),
                expected_if_confirmed=(
                    "Elevated compressor duty stops immediately during off-peak/stationary "
                    "layovers."
                ),
                expected_if_refuted="Duty remains high during stationary/no-passenger intervals.",
            )
        ],
    )


def _build_concurrent_hypothesis(
    pkg: EventEvidencePackage,
    ex: _ExtractedSignals,
    leak_hyp: RankedHypothesis,
    sensor_hyp: RankedHypothesis,
    integ_hyp: RankedHypothesis,
    human_sensor_refs: list[EvidenceReference],
    common_missing: list[EvidenceReference],
) -> RankedHypothesis:
    sup: list[EvidenceReference] = []
    con: list[EvidenceReference] = []
    score = 0.0
    leak_present = (
        _duty_elevated(ex)
        and ex.peak_duty is not None
        and (
            ex.has_concurrent_leak_alarm
            or ex.has_primary_exceedance
            or ex.peak_duty >= _SEVERE_DUTY_THRESHOLD
        )
    )
    # Only genuine instrumentation/integrity contradictions (excluding naturally flat
    # 0-bar DP sensors like DV_pressure/H1 under continuous load) constitute a second fault.
    integrity_present = (
        pkg.quality.stale_windows_count > 0
        or ex.valve_contradiction_count > 0
        or ex.motor_mismatch_count > 0
        or any(s in _CONTINUOUS_CORE_SENSORS for s in pkg.quality.frozen_sensors)
        or len(pkg.quality.out_of_range_signals) > 0
        or len(human_sensor_refs) > 0
    )

    if leak_present and integrity_present:
        score = min(
            0.92,
            0.42
            + 0.45
            * min(
                1.0,
                (leak_hyp.support_score + max(integ_hyp.support_score, sensor_hyp.support_score))
                / 1.15,
            ),
        )
        raw_sup = [
            *leak_hyp.supporting_evidence[:2],
            *(integ_hyp.supporting_evidence or sensor_hyp.supporting_evidence)[:2],
        ]
        seen_refs: set[tuple[str, str, str]] = set()
        for ref in raw_sup:
            key = (ref.ref_id, ref.signal_or_feature, ref.statement)
            if key not in seen_refs:
                seen_refs.add(key)
                sup.append(ref)
    else:
        con.append(
            EvidenceReference(
                ref_id="concurrent:absent",
                origin="derived",
                phase=ex.focus.phase,
                signal_or_feature="load_duty",
                statement=(
                    "No simultaneous pneumatic overload plus instrumentation/integrity "
                    "contradiction detected."
                ),
            )
        )

    score = round(min(1.0, max(0.0, score)), 4)
    return RankedHypothesis(
        hypothesis_id="hyp:concurrent_conditions",
        category="concurrent_conditions",
        title="Concurrent pneumatic load/leak event and telemetry integrity anomaly",
        rank=5,
        relative_support="weak",
        support_score=score,
        confidence_band="low",
        physical_mechanism=(
            "A genuine air leak or heavy demand period overlaps with a transient DAQ freeze, "
            "frozen transducer channel, or discrete valve wiring fault."
        ),
        supporting_evidence=sup,
        contradicting_evidence=con,
        missing_evidence=list(common_missing),
        correlation_vs_causation_note=(
            "Co-occurrence requires isolating instrumentation faults before quantifying "
            "physical leak severity."
        ),
        verification_steps=[
            VerificationStep(
                step_id="ver:concurrent_split",
                priority="immediate",
                action=(
                    "Inspect timestamps where telemetry integrity clears and evaluate "
                    "sustained load_duty in unaffected windows separately."
                ),
                expected_if_confirmed=(
                    "Both a discrete sensor/DAQ glitch and lingering pneumatic load decay "
                    "exist on non-frozen segments."
                ),
                expected_if_refuted=(
                    "Apparent duty anomaly vanishes once corrupted/frozen intervals are removed."
                ),
            )
        ],
    )


def analyze_event_evidence(pkg: EventEvidencePackage) -> RcaAssessment:
    """Compute a deterministic, grounded multi-hypothesis RCA report from an evidence package."""
    ex = _extract_signals(pkg)
    human_leak_refs, human_sensor_refs = _human_obs_signals(pkg)
    common_missing = _common_missing_refs(pkg, ex)

    quality_abstained = (
        pkg.quality.overall_status == "insufficient"
        or ex.non_stale_valid_windows == 0
        or all(not p.available for k in ("onset", "evolution", "peak") if (p := ex.by_phase.get(k)))
    )

    leak_hyp = _build_air_leakage_hypothesis(
        pkg, ex, human_leak_refs, human_sensor_refs, common_missing
    )
    sensor_hyp = _build_sensor_malfunction_hypothesis(pkg, ex, human_sensor_refs, common_missing)
    integ_hyp = _build_telemetry_inconsistency_hypothesis(pkg, ex, common_missing)
    demand_hyp = _build_high_demand_hypothesis(pkg, ex, human_leak_refs, common_missing)
    concurrent_hyp = _build_concurrent_hypothesis(
        pkg, ex, leak_hyp, sensor_hyp, integ_hyp, human_sensor_refs, common_missing
    )

    candidates = [leak_hyp, sensor_hyp, integ_hyp, demand_hyp, concurrent_hyp]
    max_fault_score = max(h.support_score for h in candidates)
    nominal_abstained = (not quality_abstained) and (max_fault_score < _MIN_DIAGNOSTIC_SUPPORT)
    abstained = quality_abstained or nominal_abstained

    abstention_reason: str | None = None
    insufficient_sup: list[EvidenceReference] = []
    if quality_abstained:
        abstention_reason = (
            "Insufficient evidence for reliable root-cause ranking: "
            f"overall_status={pkg.quality.overall_status}, "
            f"non_stale_valid_windows={ex.non_stale_valid_windows}/"
            f"{pkg.quality.expected_event_windows}, "
            f"coverage_ratio={pkg.quality.coverage_ratio:.2f}."
        )
        insufficient_sup.append(
            EvidenceReference(
                ref_id="abstain:coverage",
                origin="derived",
                phase=ex.focus.phase,
                signal_or_feature="coverage_ratio",
                statement=(
                    f"Event had {ex.non_stale_valid_windows}/"
                    f"{pkg.quality.expected_event_windows} non-stale valid windows "
                    f"(coverage={pkg.quality.coverage_ratio:.2f}, stale_windows="
                    f"{pkg.quality.stale_windows_count}), failing reliability requirements."
                ),
                observed_value=pkg.quality.coverage_ratio,
            )
        )
    elif nominal_abstained:
        abstention_reason = (
            "No active fault or anomalous physical deviation reached diagnostic threshold "
            f"(top support={max_fault_score:.2f} < {_MIN_DIAGNOSTIC_SUPPORT:.2f}); telemetry "
            "reflects nominal operation or lacks identifiable fault evidence."
        )
        insufficient_sup.append(
            EvidenceReference(
                ref_id="abstain:nominal",
                origin="derived",
                phase=ex.focus.phase,
                signal_or_feature="coverage_ratio",
                statement=abstention_reason,
                observed_value=pkg.quality.coverage_ratio,
            )
        )

    insuff_score = (
        round(min(1.0, max(0.78, 1.0 - pkg.quality.completeness_score + 0.35)), 4)
        if quality_abstained
        else (
            round(min(0.85, max(0.6, 1.0 - max_fault_score)), 4)
            if nominal_abstained
            else round(min(0.24, max(0.0, 0.3 * (1.0 - pkg.quality.completeness_score))), 4)
        )
    )
    insuff_hyp = RankedHypothesis(
        hypothesis_id="hyp:insufficient_evidence",
        category="insufficient_evidence",
        title=(
            "No actionable root cause identified (nominal or insufficient signal)"
            if nominal_abstained
            else "Insufficient evidence to establish root cause"
        ),
        rank=1 if abstained else 6,
        relative_support="primary" if abstained else "ruled_out",
        support_score=insuff_score,
        confidence_band="abstained" if abstained else "low",
        physical_mechanism=(
            "Gaps, missing phase windows, pervasive stale telemetry, or absence of physical "
            "fault signatures prevent attributing compressor faults."
        ),
        supporting_evidence=insufficient_sup,
        contradicting_evidence=[],
        missing_evidence=list(common_missing),
        correlation_vs_causation_note=(
            "Zero extrapolation applied to missing data; abstaining prevents false maintenance "
            "diagnosis."
        ),
        verification_steps=[
            VerificationStep(
                step_id="ver:restore_telemetry",
                priority="immediate",
                action=(
                    "Recover un-downsampled DAQ logs for missing intervals before "
                    "diagnosing compressor health."
                ),
                expected_if_confirmed=(
                    "Full 1 Hz/0.1 Hz logs close coverage gaps and unblock RCA scoring."
                ),
                expected_if_refuted=(
                    "Telemetry outage spans physical hardware shutdown rather than sensor failure."
                ),
            )
        ],
    )

    if abstained:
        cap = round(max(0.0, insuff_hyp.support_score - 0.05), 4)
        capped = [
            h.model_copy(update={"support_score": min(h.support_score, cap)}) for h in candidates
        ]
        sorted_rest = sorted(capped, key=lambda h: (-h.support_score, h.hypothesis_id))
        ordered = [insuff_hyp, *sorted_rest]
    else:
        pool = list(candidates)
        if pkg.quality.overall_status == "degraded":
            pool.append(insuff_hyp)
        ordered = sorted(pool, key=lambda h: (-h.support_score, h.hypothesis_id))

    ranked: list[RankedHypothesis] = []
    top_score = ordered[0].support_score
    for idx, hyp in enumerate(ordered, start=1):
        if abstained:
            rel: RelativeSupport = "primary" if idx == 1 else "weak"
        elif idx == 1:
            rel = "primary"
        elif hyp.support_score >= max(0.28, 0.55 * top_score):
            rel = "plausible_alternative"
        elif hyp.support_score >= 0.12:
            rel = "weak"
        else:
            rel = "ruled_out"
        band = _confidence_band(
            hyp.support_score,
            pkg,
            abstained=abstained,
            contradictions_count=len(hyp.contradicting_evidence),
        )
        ranked.append(
            hyp.model_copy(
                update={
                    "rank": idx,
                    "relative_support": rel,
                    "confidence_band": band,
                }
            )
        )

    primary = ranked[0]
    raw_id = f"{pkg.package_id}|{pkg.provenance.primary_model_version}|{primary.category}"
    rca_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:24]

    if abstained:
        summary = abstention_reason or "Abstained due to insufficient telemetry coverage."
    else:
        alts = [h.category for h in ranked[1:] if h.relative_support == "plausible_alternative"]
        alt_txt = f" Plausible competing explanations preserved: {', '.join(alts)}." if alts else ""
        summary = (
            f"Top ranked hypothesis: {primary.title} "
            f"(support_score={primary.support_score:.2f}, confidence={primary.confidence_band})."
            + alt_txt
        )

    uncertainty_parts = [
        "Ranked scores index consistency with available telemetry; they are not calibrated "
        "probabilities."
    ]
    if pkg.quality.overall_status == "degraded":
        uncertainty_parts.append(
            f"Evidence quality is degraded (penalty={pkg.quality.confidence_penalty_factor:.2f}); "
            "confidence band is capped accordingly."
        )
    if ex.peak_duty is not None and ex.peak_duty >= _HIGH_DUTY_THRESHOLD:
        uncertainty_parts.append(
            "Absent a downstream air-flow sensor, abnormal high pneumatic demand cannot be "
            "excluded automatically."
        )

    return RcaAssessment(
        rca_id=rca_id,
        package_id=pkg.package_id,
        asset_id=pkg.asset_id,
        incident_id=pkg.incident_id,
        event_start=pkg.event_start,
        event_end=pkg.event_end,
        abstained=abstained,
        abstention_reason=abstention_reason,
        primary_category=primary.category,
        primary_title=primary.title,
        summary=summary,
        uncertainty_note=" ".join(uncertainty_parts),
        quality_status=pkg.quality.overall_status,
        confidence_penalty_factor=pkg.quality.confidence_penalty_factor,
        ranked_hypotheses=ranked,
        correlation_vs_causation_boundary=[
            "Detector scores and robust z-scores reflect distributional departure, not causal "
            "proof.",
            "MetroPT-3 (~0.1 Hz) lacks a downstream demand flowmeter and high-rate PLC command "
            "tags.",
            "Conflicting valve signals or frozen sensors weaken air-leak attribution until "
            "field-verified.",
        ],
        provenance=pkg.provenance,
    )
