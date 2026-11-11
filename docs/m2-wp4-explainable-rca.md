# M2-WP4 (`WP-B`) — Explainable Root-Cause Analysis & Casebook Evaluation

M2-WP4 adds a deterministic, evidence-grounded, uncertainty-aware physical multi-hypothesis **Root-Cause Analysis (RCA)** engine and reproducible **RCA Casebook** evaluation harness on top of the WP-A `EventEvidencePackage` foundation. Every hypothesis preserves explicit evidence origins (`observed`, `derived`, `statistical_inference`, `model_prediction`, `human_observation`), strictly separates correlation from causation, abstains when evidence is insufficient, and recommends advisory-only field inspection checks without ever triggering automated actuation.

---

## 1. Domain Contracts & Multi-Hypothesis Ranking (`factorymind.contracts.rca`)

`backend/src/factorymind/contracts/rca.py` defines strongly typed Pydantic v2 schemas:

- **Hypothesis categories (`RcaHypothesisCategory`):**
  - `air_leakage`
  - `sensor_malfunction`
  - `telemetry_inconsistency`
  - `abnormal_high_demand`
  - `concurrent_conditions`
  - `insufficient_evidence`
- **Honest ranking & uncertainty (`RankedHypothesis`, `RcaAssessment`):**
  - Ranks candidates (`1..N`) with `relative_support` (`primary`, `plausible_alternative`, `weak`, `ruled_out`), `support_score` (`0..1` physical-evidence consistency index — explicitly documented as **not** a calibrated failure probability), and `confidence_band` (`high`, `moderate`, `low`, `abstained`).
  - Separates `supporting_evidence`, `contradicting_evidence`, and `missing_evidence` as explicit `EvidenceReference` citations pointing directly to phases/signals/features inside the source `EventEvidencePackage`.
  - Attaches explicit `correlation_vs_causation_note`, global `correlation_vs_causation_boundary`, and non-actuating `VerificationStep` checklists (`safety_note` enforcing advisory-only physical inspection).

---

## 2. Deterministic Physical RCA Engine (`factorymind.anomaly.rca`)

`analyze_event_evidence(pkg: EventEvidencePackage) -> RcaAssessment` reasons strictly from the provenance-tagged evidence package (never bypassing `pkg` to inspect raw Parquet files):

1. **Mandatory abstention gate:** Sets `abstained=True` (`primary_category="insufficient_evidence"`, all confidence bands `"abstained"`) whenever `quality.overall_status == "insufficient"`, `valid_event_windows == 0`, or all core event phases lack valid windows.
2. **Pneumatic air leak vs downstream demand:** Weighs `load_duty`, primary `duty_rule` episode exceedance, `unloaded_tp3_slope` decay, panel pressure `TP3` depression, `LPS` activation, oil thermal rise, and `recovery_or_persistence`. Because MetroPT-3 telemetry has no downstream air-demand flowmeter or train brake/door duty tag, `abnormal_high_demand` remains an explicit competing explanation unless leak decay/pressure drop/human observations separate them.
3. **Instrumentation & integrity separation:** Separates `stale_telemetry`, `valve_contradiction`, `motor_valve_mismatch`, single-channel freezes, `out_of_range` samples, and non-leak `pca_spe` departures into `telemetry_inconsistency` / `sensor_malfunction`. Valid `pca_spe` co-firing during consistent high-duty leaks is not misclassified as an integrity conflict, while genuine co-occurring integrity faults (e.g., `F1`) activate `concurrent_conditions` and cap `confidence_band="low"`.
4. **Human observations:** Incorporates append-only `human_observations` transparently under explicit `origin="human_observation"`.

---

## 3. Reproducible Casebook Evaluation & Live MetroPT-3 Verification (`rca_eval.py`)

`backend/src/factorymind/anomaly/rca_eval.py` verifies `citations_grounded`, `correlation_disclosed`, abstention, and integrity separation across separate real and synthetic slices:

### Live MetroPT-3 Failure Reports (`evaluate_service_report_rca` — Real Operational Telemetry)

Executed on real processed MetroPT-3 data with trained `20261111-f069042373` bundles (`F1..F4`):

- Aggregate metrics:
  - `evidence_citation_grounding_rate`: `1.0` (`100%` of citations resolve cleanly to `EventEvidencePackage` items)
  - `correlation_disclaimer_rate`: `1.0`
  - `expectation_agreement_rate`: `1.0` (`4/4` real failure events)
- Per-event outcomes:
  - `F1`: `concurrent_conditions` (`top_support_score=0.7217`, `confidence_band=low`, alternative `telemetry_inconsistency`) — surfaces both early `stale_telemetry`/constant channels and sustained air leak without overclaiming confidence.
  - `F2`: `air_leakage` (`top_support_score=0.67`, `confidence_band=moderate`, alternative `abnormal_high_demand`).
  - `F3`: `air_leakage` (`top_support_score=0.6093`, `confidence_band=low` due to degraded coverage across multi-day gapped telemetry).
  - `F4`: `air_leakage` (`top_support_score=0.5656`, `confidence_band=moderate`, alternative `abnormal_high_demand`).

### Real Unlabeled Warning & Nominal Evaluation (`2020-04-01` to `2020-09-02`)

- **Full evaluation period (`34` real warning episodes via `evaluate_unlabeled_warning_slice`):**
  - `evidence_citation_grounding_rate`: `1.0`
  - `correlation_disclaimer_rate`: `1.0`
  - `non_leak_primary_rate`: `0.6765` (`23/34`; the remaining `11` multivariate `pca_spe` warnings co-occurred with elevated high-duty intervals / sustained leaks where `duty_rule` and pressure indicators supported `air_leakage`, never fabricated `sensor_malfunction` without channel isolation).
- **April sample highlights (`2020-04-10` to `2020-04-17`) & nominal baseline check (`2020-04-05 00:00..03:00`):**
  - `W2_stale_telemetry`: `abstained=True` (`primary_category=insufficient_evidence`, `confidence_band=abstained`, `citations_grounded=True`).
  - `W1_multivariate_sensor_drift`: surfaced `air_leakage` (`0.57`, `moderate`) alongside `abnormal_high_demand` (`citations_grounded=True`).
  - `2020-04-05` normal operating slice: `abstained=True` (`primary_category=insufficient_evidence`, refusing to promote low/nominal scores as a fault cause).

### Synthetic & API Verification

- **Backend:** `ruff format --check .` (`68` files), `ruff check .`, `mypy` (`0` issues across `39` source files), `pytest -q` -> **`451 passed`** (`test_rca_foundation.py`, `test_auth_routes.py`, `test_safety_boundary.py`, `test_openapi_contract.py`).
- **Frontend:** `npm run generate:api` cleanly synced, `npm run lint` (`biome check .`, `42` files), `npm run typecheck` (`tsc -b`), `npm run test` -> **`8` files / `32` passed** (including `RcaPanel.test.tsx` and `EventEvidencePackageView.test.tsx`), `npm run build` (`tsc -b && vite build`).
- **Synthetic stress & regression cases (`test_rca_foundation.py`):** Clean synthetic leak, single-sensor `TP3` freeze + `COMP`/`DV_eletric` valve contradiction, single-sensor `Motor_current` freeze at `7.0 A` (`sensor_malfunction` primary rather than spurious leak), mixed stale + gapped zero non-stale windows (mandatory abstention + monotonic ranks), `warning_bundle=None` non-fabrication, negated human inspection notes (`no leak found, hose intact`), nominal abstention, and human-observation corroboration.
- **API & UI integration:**
  - `GET /api/v1/incidents/{incident_id}/rca` (`incident:read`, `cost=5`, tenant & asset isolated)
  - `GET /api/v1/anomaly/rca` (`anomaly:read`, `cost=5`, naive timestamps)
  - React `RcaPanel.tsx` integrated on `/incidents/$incidentId` (with `ObservationForm` invalidating `getGetIncidentRcaQueryKey`) and on-demand inside `ReplayPanel` event inspection; OpenAPI & Orval client synchronized.

