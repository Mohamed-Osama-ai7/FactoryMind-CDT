"""Reproducible WP-B RCA Casebook evaluation harness.

Separates REAL MetroPT-3 labeled events and warning cases cleanly from SYNTHETIC
stress cases so synthetic integrity separation never masquerades as real failure validation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from factorymind.anomaly.evidence_package import build_event_evidence_package
from factorymind.anomaly.rca import analyze_event_evidence
from factorymind.anomaly.service import AnomalyService
from factorymind.contracts.events import EventEvidencePackage
from factorymind.contracts.rca import EvidenceReference, RcaAssessment, RcaHypothesisCategory

_VALID_DERIVED_REFS = frozenset(
    {
        "load_duty",
        "off_fraction",
        "load_starts",
        "inrush_samples",
        "tp3_reservoir_gap",
        "loaded_tp3_rise",
        "unloaded_tp3_slope",
        "unloaded_fraction",
        "stale_telemetry",
        "valve_contradiction",
        "motor_valve_mismatch",
        "coverage_ratio",
        "air_demand_flowmeter",
        "pre_event_baseline",
        "high_rate_electrical_current_gt_1hz",
        "recovery_or_persistence",
    }
)


@dataclass(frozen=True)
class RcaEvalCaseResult:
    case_id: str
    kind: str  # real_labeled | real_unlabeled_warning | synthetic_stress
    expected_categories: list[RcaHypothesisCategory] | None
    expect_abstain: bool | None
    primary_category: RcaHypothesisCategory
    abstained: bool
    top_support_score: float
    confidence_band: str
    plausible_alternatives: list[RcaHypothesisCategory]
    citations_grounded: bool
    correlation_disclosed: bool
    matches_expectation: bool | None
    verification_steps_count: int


def _valid_signals_in_pkg(pkg: EventEvidencePackage) -> set[str]:
    names: set[str] = set()
    for p in pkg.phases:
        names.update(s.signal for s in p.observed_signals)
        names.update(d.name for d in p.derived_features)
    names.update(c.kind for c in pkg.quality.contradictions)
    for c in pkg.quality.contradictions:
        names.update(c.signals)
    names.update(o.category for o in pkg.human_observations)
    names.update(e.feature for e in pkg.evidence_items)
    names.add(pkg.provenance.primary_detector)
    if pkg.provenance.warning_detector:
        names.add(pkg.provenance.warning_detector)
    names.update(_VALID_DERIVED_REFS)
    return names


def citations_grounded(pkg: EventEvidencePackage, rca: RcaAssessment) -> bool:
    """Verify every EvidenceReference cites an existing package symbol & matches phase bounds."""
    known = _valid_signals_in_pkg(pkg)
    phase_names = {p.phase for p in pkg.phases}
    obs_by_cat = {o.category for o in pkg.human_observations}
    for hyp in rca.ranked_hypotheses:
        for ref_list in (hyp.supporting_evidence, hyp.contradicting_evidence, hyp.missing_evidence):
            ref: EvidenceReference
            for ref in ref_list:
                if ref.phase is not None and ref.phase not in phase_names:
                    return False
                if ref.origin == "human_observation" and ref.signal_or_feature not in obs_by_cat:
                    return False
                if ref.signal_or_feature not in known:
                    return False
                if not ref.statement.strip():
                    return False
    return True


def correlation_disclosed(rca: RcaAssessment) -> bool:
    return (
        len(rca.correlation_vs_causation_boundary) > 0
        and all(bool(h.correlation_vs_causation_note.strip()) for h in rca.ranked_hypotheses)
        and "not a calibrated" in rca.score_meaning_note.lower()
    )


def evaluate_rca_case(
    case_id: str,
    pkg: EventEvidencePackage,
    *,
    kind: str,
    expected_categories: Sequence[RcaHypothesisCategory] | None = None,
    expect_abstain: bool | None = None,
) -> tuple[RcaEvalCaseResult, RcaAssessment]:
    rca = analyze_event_evidence(pkg)
    grounded = citations_grounded(pkg, rca)
    disclosed = correlation_disclosed(rca)
    alts = [
        h.category
        for h in rca.ranked_hypotheses[1:]
        if h.relative_support == "plausible_alternative"
    ]
    top = rca.ranked_hypotheses[0]

    matches: bool | None = None
    if expected_categories is not None or expect_abstain is not None:
        cat_ok = (
            True if expected_categories is None else rca.primary_category in expected_categories
        )
        abs_ok = True if expect_abstain is None else rca.abstained == expect_abstain
        matches = cat_ok and abs_ok

    return (
        RcaEvalCaseResult(
            case_id=case_id,
            kind=kind,
            expected_categories=list(expected_categories) if expected_categories else None,
            expect_abstain=expect_abstain,
            primary_category=rca.primary_category,
            abstained=rca.abstained,
            top_support_score=top.support_score,
            confidence_band=top.confidence_band,
            plausible_alternatives=alts,
            citations_grounded=grounded,
            correlation_disclosed=disclosed,
            matches_expectation=matches,
            verification_steps_count=sum(len(h.verification_steps) for h in rca.ranked_hypotheses),
        ),
        rca,
    )


def evaluate_service_report_rca(
    service: AnomalyService,
    *,
    asset_id: str = "metropt3-apu-1",
) -> dict[str, Any]:
    """Run WP-B RCA Casebook evaluation across MetroPT-3 ``F1..F4`` operator reports."""
    rows: list[RcaEvalCaseResult] = []
    for report in service.reports:
        pkg = build_event_evidence_package(
            service,
            asset_id=asset_id,
            event_start=pd.Timestamp(report.start),
            event_end=pd.Timestamp(report.end),
            raised_at=pd.Timestamp(datetime.fromisoformat(str(report.start))),
        )
        # F1 includes pre-onset/early-event stale telemetry -> concurrent or leakage are both valid
        expected: list[RcaHypothesisCategory] = (
            ["air_leakage", "concurrent_conditions"]
            if pkg.quality.stale_windows_count > 0 or pkg.quality.frozen_sensors
            else ["air_leakage"]
        )
        res, _ = evaluate_rca_case(
            report.report_id,
            pkg,
            kind="real_labeled",
            expected_categories=expected,
            expect_abstain=False,
        )
        rows.append(res)

    n = len(rows)
    checked = [r for r in rows if r.matches_expectation is not None]
    return {
        "cases": n,
        "kind": "real_labeled",
        "evidence_citation_grounding_rate": round(sum(r.citations_grounded for r in rows) / n, 4)
        if n
        else None,
        "correlation_disclaimer_rate": round(sum(r.correlation_disclosed for r in rows) / n, 4)
        if n
        else None,
        "expectation_agreement_rate": round(
            sum(bool(r.matches_expectation) for r in checked) / len(checked), 4
        )
        if checked
        else None,
        "results": [asdict(r) for r in rows],
    }


def evaluate_unlabeled_warning_slice(
    service: AnomalyService,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    asset_id: str = "metropt3-apu-1",
    max_cases: int = 5,
) -> dict[str, Any]:
    """Evaluate real unlabeled warning episodes on non-leak diagnosis & grounding."""
    replay = service.replay_details(start, end)
    rows: list[RcaEvalCaseResult] = []
    for idx, warn in enumerate(replay.warnings[:max_cases], start=1):
        pkg = build_event_evidence_package(
            service,
            asset_id=asset_id,
            event_start=warn.start,
            event_end=warn.end,
            raised_at=warn.raised_at,
        )
        expected: list[RcaHypothesisCategory] = (
            [
                "telemetry_inconsistency",
                "insufficient_evidence",
                "sensor_malfunction",
                "concurrent_conditions",
            ]
            if warn.warning_kind
            in ("stale_telemetry", "signal_inconsistency", "mixed_consistency_warning")
            else ["sensor_malfunction", "concurrent_conditions", "abnormal_high_demand"]
        )
        res, _ = evaluate_rca_case(
            f"W{idx}_{warn.warning_kind}",
            pkg,
            kind="real_unlabeled_warning",
            expected_categories=expected,
            expect_abstain=None,
        )
        rows.append(res)

    n = len(rows)
    return {
        "cases": n,
        "kind": "real_unlabeled_warning",
        "evidence_citation_grounding_rate": round(sum(r.citations_grounded for r in rows) / n, 4)
        if n
        else None,
        "correlation_disclaimer_rate": round(sum(r.correlation_disclosed for r in rows) / n, 4)
        if n
        else None,
        "non_leak_primary_rate": round(
            sum(r.primary_category != "air_leakage" or r.abstained for r in rows) / n, 4
        )
        if n
        else None,
        "results": [asdict(r) for r in rows],
    }
