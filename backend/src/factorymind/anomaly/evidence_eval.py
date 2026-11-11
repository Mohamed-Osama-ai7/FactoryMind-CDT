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
    rec = by_phase["recovery_or_persistence"]
    return (
        base.end == pkg.event_start
        and onset.start == pkg.event_start
        and onset.end == evo.start
        and evo.end == pkg.event_end
        and rec.start == pkg.event_end
    )


def _origins_pure(pkg: EventEvidencePackage) -> bool:
    # Observed signal summaries must always carry origin="observed"
    for p in pkg.phases:
        if any(s.origin != "observed" for s in p.observed_signals):
            return False
        if any(d.origin != "derived" for d in p.derived_features):
            return False
        if p.primary_model_origin != "model_prediction":
            return False
        if p.exceedance_origin != "statistical_inference":
            return False
    # Inference evidence items must never masquerade as direct sensor observations
    return all(item.origin in ("statistical_inference", "derived") for item in pkg.evidence_items)


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
