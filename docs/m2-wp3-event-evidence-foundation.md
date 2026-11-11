# M2-WP3 (`WP-A`) — Event Evidence Foundation, Quality & Unified Timeline

M2-WP3 establishes the shared origin-tagged domain contracts, multi-phase `EventEvidencePackage` builder, explicit data-quality & contradiction assessment, unified chronological timeline, and append-only `human_observation` persistence layer required before automated root-cause reasoning (`WP-B`).

All telemetry analytics and packages remain strictly advisory and read-only toward industrial assets (`test_safety_boundary.py` enforced).

---

## 1. Origin-Tagged Domain Contracts (`factorymind.contracts.events`)

To prevent downstream reasoning or operators from conflating raw sensor measurements with derived features, statistical thresholds, detector predictions, or human notes, every piece of evidence carries an explicit `EvidenceOrigin`:

- `observed`: Direct physical sensor aggregates per phase (`ObservedSignalSummary` with explicit unit, min/mean/max/std/last or digital active fraction, missing & out-of-range counts, and `frozen` indicator; never interpolated when absent).
- `derived`: Windowed feature summaries (`DerivedFeatureSummary`) mapped to engineering units (`unit_of_feature`).
- `statistical_inference`: Calibration exceedances (`exceedance_origin`) and grouped `EvidenceItemContract` items (`robust_z`, `novel`, training median).
- `model_prediction`: Detector peak/phase outputs (`primary_model_origin`) alongside sibling `pca_spe` warning detector activity.
- `human_observation`: Append-only operator/engineer notes persisted in SQLite (`HumanObservationOut`).

---

## 2. Five-Phase `EventEvidencePackage` & Quality Assessment

`build_event_evidence_package` (`backend/src/factorymind/anomaly/evidence_package.py`) reuses `AnomalyService.replay_details` without duplicating scoring or regime classification logic and structures every event across five chronological phases (`EventPhaseName`):

1. `pre_event_baseline` (configurable `baseline_hours`, default `4.0 h` before `event_start`)
2. `onset` (first active window / onset interval from `event_start`)
3. `evolution` (progression across `[onset.end, event_end)`)
4. `peak` (maximum `duty_rule` primary detector activity window inside the event)
5. `recovery_or_persistence` (configurable `recovery_hours`, default `2.0 h` after `event_end`, classifying `recovery_status` as `recovered`, `persisting`, `stopped`, or `unverified`)

### Explicit Quality, Contradictions & Missing Data Honesty

`EvidenceQualityAssessment` makes evidence limitations first-class fields:

- Computes `completeness_score` (`0..1`), `coverage_ratio` (`0..1`), `confidence_penalty_factor` (`0..1`), `missing_windows_count`, `missing_samples_count`, `stale_windows_count`, `frozen_sensors`, `out_of_range_signals`, `alignment_issues`, and explicit `overall_status` (`sufficient`, `degraded`, `insufficient`).
- Surfaces explicit `ContradictionRecord` entries (`valve_contradiction`, `motor_valve_mismatch`, `sensor_freeze`, `out_of_range`, `model_divergence`).
- Leaves absent or invalid windows uninterpolated (`sample_count == 0`, `mean == None`, explicit `available=False` and `reason_unavailable`).

---

## 3. Append-Only Persistence, RBAC, Audit & APIs

- **Database schema (`SCHEMA_VERSION = "2"`):** Added `human_observation` table to `backend/src/factorymind/db/models.py` and `APPEND_ONLY_TABLES`, guarded by SQLite `BEFORE UPDATE` / `BEFORE DELETE` abort triggers in `db/session.py` (`IncidentRepo.observations` and `IncidentRepo.add_observation`).
- **RBAC (`auth/principal.py`):** Added `Permission.OBSERVATION_CREATE = "observation:create"` granted to `Role.ENGINEER`, `Role.APPROVER`, and `Role.ADMIN`; `VIEWER` and `AUDITOR` remain strictly read-only.
- **Endpoints:**
  - `GET /api/v1/incidents/{incident_id}/evidence-package` (`incident:read` + tenant & data-source asset scoping; enriches package with persisted `human_observations`).
  - `POST /api/v1/incidents/{incident_id}/observations` (`observation:create`, CSRF/write check, append-only SQLite + hash-chained `observation.create` audit event).
  - `GET /api/v1/anomaly/event-evidence` (`anomaly:read` + data-source authorization, inspectable directly from Replay UI).
  - Extended `IncidentDetail` (`GET /api/v1/incidents/{id}`) with `observations`.
- **Frontend UI & OpenAPI:** Regenerated `frontend/openapi.json` and Orval client; added `/incidents/` list route, `/incidents/$incidentId` investigation route, `EventEvidencePackageView` (quality banner, timeline category filter, phase drilldown, baseline/peak contrasts, contradictions, neighbor events), `ObservationForm` (permission-gated), and inline event package drawer inside `ReplayPanel`.

---

## 4. Verification & Reproducible Evaluation

### Synthetic & Integration Test Suites

- **Backend:** `ruff format --check src tests` (`63 files`), `ruff check src tests`, `mypy src` (`0` errors across `36` source files), `pytest -q` -> **`436 passed`**. Covers WP-A package alignment, origin purity, synthetic missing-baseline/gapped/frozen-`TP3`/valve-contradiction stress tests, RBAC matrix, tenant isolation, SQLite append-only trigger enforcement, and read-only safety boundaries.
- **Frontend:** `npm run generate:api` synced with `openapi.json`, `npm run lint` (`biome check .`, `40` files), `npm run typecheck` (`tsc -b`), `npm run test` -> **`7` files / `31` passed** (including `EventEvidencePackageView.test.tsx`), `npm run build` (`tsc -b && vite build`).

### Live MetroPT-3 Failure Report Evaluation (Real Operational Telemetry)

Executed directly against processed MetroPT-3 telemetry and trained `20261111-f069042373` bundles via `evaluate_service_report_events(service)` (`backend/src/factorymind/anomaly/evidence_eval.py`):

- Aggregate across real failure intervals `F1..F4`:
  - `boundary_alignment_rate`: `1.0`
  - `origin_purity_rate`: `1.0`
  - `missing_data_honesty_rate`: `1.0`
  - `mean_completeness_score`: `0.9276`
- Per-event findings (preserving missing/stale telemetry honestly rather than hiding gaps):
  - `F1`: `degraded` (`completeness_score=0.9917`, `coverage_ratio=0.9861`, `confidence_penalty_factor=0.687`, `contradictions_detected=2`, `frozen_sensors=7`, `stale_windows_count=2`, `timeline_entries=203`, `neighbors_count=4`) — surfaces the actual pre-onset telemetry freeze window during `F1`.
  - `F2`: `sufficient` (`completeness_score=1.0`, `coverage_ratio=1.0`, `confidence_penalty_factor=1.0`, `timeline_entries=95`, `neighbors_count=3`).
  - `F3`: `degraded` (`completeness_score=0.741`, `coverage_ratio=0.9016`, `confidence_penalty_factor=0.741`, `timeline_entries=343`, `neighbors_count=12`).
  - `F4`: `degraded` (`completeness_score=0.9778`, `coverage_ratio=0.963`, `confidence_penalty_factor=0.9778`, `timeline_entries=62`, `neighbors_count=4`).
