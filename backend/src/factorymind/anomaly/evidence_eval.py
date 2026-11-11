"""Reproducible WP-A evaluation suite for Event Evidence Packages and Timelines.

Measures objective engineering properties across synthetic stress cases and (when local
processed data exists) real MetroPT-3 failure intervals without mixing their metrics:

- Temporal alignment of 5 event phases (10-minute window boundaries)
- Provenance and origin-tag separation (observed vs derived vs inference vs model vs human)
- Evidence completeness and missing-data honesty (no silent interpolation)
- Single-sensor freeze, whole-feed stale telemetry, and valve contradiction detection
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from factorymind.anomaly.evidence_package import build_event_evidence_package
from factorymind.anomaly.service import AnomalyService
from factorymind.contracts.events import EventEvidencePackage


@dataclass(frozen=True)
class EvidenceEvalCaseResult:
    case_id: str
    kind: str  # real_labeled | synthetic_stress
    overall_status: str
    completeness_score: float
    coverage_ratio: float
    confidence_penalty_factor: float
    boundary_aligned: bool
    origins_pure: bool
    missing_data_honest: bool
    contradictions_detected: int
    frozen_sensors: list[str]
    stale_windows_count: int
    timeline_entries: int
    neighbors_count: int


def _boundary_aligned(pkg: EventEvidencePackage) -> bool:
    by_phase = {p.phase: p for p in pkg.phases}
    base = by_phase["pre_event_baseline"]
    onset = by_phase["onset"]
    evo = by_phase["evolution"]
    peak = by_phase["peak"]
    rec = by_phase["recovery_or_persistence"]
    phases_ok = (
        base.start == pkg.lookback_start
        and base.end == pkg.event_start
        and onset.start == pkg.event_start
        and onset.end == evo.start
        and evo.end == pkg.event_end
        and rec.start == pkg.event_end
        and rec.end == pkg.recovery_end
        and pkg.event_start <= peak.start <= peak.end <= pkg.event_end
    )
    if not phases_ok:
        return False
    # Verify non-empty model outputs / warnings inside the timeline do not sit past recovery_end.
    return all(
        entry.end_timestamp is None or entry.end_timestamp <= pkg.recovery_end
        for entry in pkg.timeline
        if entry.category in ("telemetry_window", "regime_change")
    )


def _origins_pure(pkg: EventEvidencePackage) -> bool:
    for p in pkg.phases:
        if any(s.origin != "observed" for s in p.observed_signals):
            return False
        if any(d.origin != "derived" for d in p.derived_features):
            return False
        if p.primary_model_origin != "model_prediction":
            return False
        if p.exceedance_origin != "statistical_inference":
            return False
    if any(item.origin not in ("statistical_inference", "derived") for item in pkg.evidence_items):
        return False
    for n in pkg.neighboring_events:
        if n.kind == "air_leak_alarm" and n.origin != "model_prediction":
            return False
        if n.kind == "operator_report" and n.origin != "human_observation":
            return False
        if n.kind == "sensor_consistency_warning":
            expected = (
                "derived" if n.detector_or_source == "signal_integrity" else "statistical_inference"
            )
            if n.origin != expected:
                return False
    for entry in pkg.timeline:
        if entry.category in ("telemetry_window", "regime_change"):
            if (
                entry.origin != "derived"
                or entry.detector is not None
                or entry.model_version is not None
            ):
                return False
            if any(k in entry.details for k in ("score", "exceedance", "warning_score")):
                return False
        elif entry.category in ("alarm_episode", "model_output"):
            if entry.origin != "model_prediction":
                return False
        elif entry.category == "evidence_group":
            if entry.origin not in ("statistical_inference", "derived"):
                return False
        elif entry.category in ("human_observation", "operator_report"):
            if entry.origin != "human_observation":
                return False
        elif entry.category == "consistency_warning":
            sources = entry.details.get("sources", [])
            expected = "statistical_inference" if "pca_spe" in sources else "derived"
            if entry.origin != expected:
                return False
    return all(c.origin in ("derived", "statistical_inference") for c in pkg.quality.contradictions)


def _missing_data_honest(pkg: EventEvidencePackage) -> bool:
    for p in pkg.phases:
        if p.sample_count == 0 and any(s.mean is not None for s in p.observed_signals):
            return False
        if not p.available and p.reason_unavailable is None:
            return False
    return True


def evaluate_package(
    case_id: str,
    pkg: EventEvidencePackage,
    *,
    kind: str,
) -> EvidenceEvalCaseResult:
    """Inspect structural, provenance, alignment, and quality properties of one package."""
    return EvidenceEvalCaseResult(
        case_id=case_id,
        kind=kind,
        overall_status=pkg.quality.overall_status,
        completeness_score=pkg.quality.completeness_score,
        coverage_ratio=pkg.quality.coverage_ratio,
        confidence_penalty_factor=pkg.quality.confidence_penalty_factor,
        boundary_aligned=_boundary_aligned(pkg),
        origins_pure=_origins_pure(pkg),
        missing_data_honest=_missing_data_honest(pkg),
        contradictions_detected=len(pkg.quality.contradictions),
        frozen_sensors=list(pkg.quality.frozen_sensors),
        stale_windows_count=pkg.quality.stale_windows_count,
        timeline_entries=len(pkg.timeline),
        neighbors_count=len(pkg.neighboring_events),
    )


def evaluate_service_report_events(
    service: AnomalyService,
    *,
    asset_id: str = "metropt3-apu-1",
    kind: str = "real_labeled",
) -> dict[str, Any]:
    """Build and score WP-A packages across all configured failure reports for ``service``."""
    results: list[EvidenceEvalCaseResult] = []
    for report in service.reports:
        pkg = build_event_evidence_package(
            service,
            asset_id=asset_id,
            event_start=pd.Timestamp(report.start),
            event_end=pd.Timestamp(report.end),
            raised_at=pd.Timestamp(datetime.fromisoformat(str(report.start))),
        )
        results.append(evaluate_package(report.report_id, pkg, kind=kind))
    n = len(results)
    return {
        "cases": n,
        "kind": kind,
        "boundary_alignment_rate": round(sum(r.boundary_aligned for r in results) / n, 4)
        if n
        else None,
        "origin_purity_rate": round(sum(r.origins_pure for r in results) / n, 4) if n else None,
        "missing_data_honesty_rate": round(sum(r.missing_data_honest for r in results) / n, 4)
        if n
        else None,
        "mean_completeness_score": round(sum(r.completeness_score for r in results) / n, 4)
        if n
        else None,
        "results": [asdict(r) for r in results],
    }
