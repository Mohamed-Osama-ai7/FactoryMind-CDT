# FactoryMind CDT — Authoritative Roadmap & Execution Tracker

> **Purpose:** Single authoritative roadmap, work-package tracker, and reconciliation record on `main`.
> Preserves historical milestone docs ([M1](m1-anomaly-detection.md), [M2-WP0](m2-wp0-detection-hardening.md),
> [M2-WP1](m2-wp1-auth-persistence.md), [M2-WP2](m2-wp2-operating-regime-warning-separation.md)) without
> modifying or relying on external open PR #3 (`docs/post-m1-review-rebased`).

## 1. Phase 0 Reconciliation Summary

Verified baseline on `main`: `03c3fd53f45c1256333d5b5425c7cced0a2562d8` (PR #6 merged 2026-11-11).

### Completed Milestones

| Milestone | Scope | PR | Verified Merge / Head SHA | Key Verified Metrics / Baseline |
| --- | --- | --- | --- | --- |
| **M0** | Engineering foundation, FastAPI modular monolith, React/TS UI, CI, read-only safety boundary | [#1](https://github.com/Mohamed-Osama-ai7/FactoryMind-CDT/pull/1) | `666a956` (`5824e16`) | Green backend/frontend CI, `test_safety_boundary.py` active |
| **M1** | Initial MetroPT-3 compressor anomaly detection pipeline, API (`/model`, `/score`, `/replay`), React `/anomaly` | [#2](https://github.com/Mohamed-Osama-ai7/FactoryMind-CDT/pull/2) | `1f9a893` (`2e95e90`) | Single validation event (`isolation_forest` val `1/1`, test `2/3`; protocol superseded by WP0) |
| **M2-WP0** | Nested LOEO hardening, `duty_rule` physics baseline, `ExceedanceCalibrator`, stateful `AlarmTracker`, grouped evidence, simulated sensor-fault check | [#4](https://github.com/Mohamed-Osama-ai7/FactoryMind-CDT/pull/4) | `e4e17f9` (`721949d`) | `duty_rule` LOEO `4/4` (`0.034` FA/valid d, `0.044` FA/op d, `50 min` `|delay|`); `pca_spe` `4/4` (`0.057`/`0.072`, `+48.3 pp` simulated clean lift) |
| **M2-WP1** | OIDC resource-server auth, 5-role RBAC + separation of duties, SQLite append-only incidents/evidence/decisions, SHA-256 audit hash chain, PKCE UI & `/incidents` list | [#5](https://github.com/Mohamed-Osama-ai7/FactoryMind-CDT/pull/5) | `ba7ac37` (`e5815c9`) | `404` backend tests, `27` frontend tests; local JWKS + SQLite verified |
| **M2-WP2** | Causal 4-regime classifier (`stopped`, `unloaded`, `loaded`, `starting`, `insufficient_evidence`), 4-class event taxonomy, integrity gating (`stale_telemetry`, `signal_inconsistency`), separated `pca_spe` warning tier | [#6](https://github.com/Mohamed-Osama-ai7/FactoryMind-CDT/pull/6) | `03c3fd5` (`5abfaa4`) | `417` backend tests, `29` frontend tests; WP2 gated `duty_rule` LOEO `4/4` (`0.026` FA/valid d, `0.035` FA/op d, `50 min`; 1 frozen false leak prevented; `34` separated warnings) |

### Documented Discrepancies Reconciled in Phase 0

1. **`docs/reference/FactoryMind_CDT_Project_Documentation_with_UI_UX.pdf`**: Initial 7-page academic submission + illustrative pump `M-004` UI mockup; lacks detailed specs for M2–M5 components. Retained strictly as initial requirements/UX direction; real operational claims apply solely to evaluated pipelines in this repo.
2. **Feature count**: `features.py` exports **42** `FEATURE_NAMES` (`7 * 4 + 8 + 6`, accurately documented in `m2-wp2-operating-regime-warning-separation.md`), correcting an inherited `"46"` note in `m1-anomaly-detection.md`.
3. **Training vs inference gating**: `experiment.py` preserves WP0 `duty_rule` LOEO numbers alongside `wp2.regime_gating_ablations["wp2_integrity_gated"]` (`0.026` / `0.035`), while runtime `AnomalyService` enforces WP2 integrity gating (`stale_telemetry` / `signal_inconsistency` abstention).
4. **Repository governance items noted**: No root `LICENSE` file yet (though `pyproject.toml` declares `MIT`); `.github/workflows/ci.yml` uses version tags rather than pinned commit SHAs; SQLite schema updated to `SCHEMA_VERSION="3"` in WP-F1 (adding append-only `outcome_verification`); PostgreSQL/Alembic and IdP verification remain deferred.

---

## 2. Planned Work Packages & Dependencies

Work packages map sequentially across **M2 (completion)** → **M3 (Asset, Multi-Model ML, RAG, KG)** → **M4 (Digital Twin, Impact, Decision Support, Copilot)** → **M5 (Read-Only OT/Edge & Final Validation)**. Each package ships on its own branch, requires independent subagent review, passes full backend/frontend/OpenAPI/security CI, and merges cleanly before the next starts.

| Work Package | Target Milestone | Dependencies | Status | Deliverables & Acceptance Criteria | Branch / PR / SHA |
| --- | --- | --- | --- | --- | --- |
| **WP-A (`M2-WP3`)** | M2 | M2-WP0..WP2 | **Complete** | Shared `factorymind.contracts` (origin-tagged `observed`/`derived`/`statistical_inference`/`model_prediction`/`human_observation`); multi-phase `EventEvidencePackage` (pre-event baseline, onset, evolution, peak, recovery/persistence, units, provenance); explicit `EvidenceQualityAssessment`; unified chronological timeline; append-only `HumanObservation` DB table + triggers + `POST /incidents/{id}/observations` + `/incidents/{id}/evidence-package` + `/anomaly/event-evidence`; evidence evaluation suite; `/incidents/$incidentId` investigation UI ([doc](m2-wp3-event-evidence-foundation.md); `437` backend / `31` frontend tests; MetroPT-3 `F1..F4` `mean_completeness_score=0.9266`, `1.0` alignment/origin/honesty) | [#7](https://github.com/Mohamed-Osama-ai7/FactoryMind-CDT/pull/7) · `8ad761a` (`2c6b7d9`) |
| **WP-B (`M2-WP4`)** | M2 | WP-A | **Complete** | Deterministic physics-based multi-hypothesis RCA engine (`air_leakage`, `sensor_malfunction`, `telemetry_inconsistency`, `abnormal_high_demand`, `concurrent_conditions`, `insufficient_evidence`), supporting/contradicting/missing evidence, correlation vs causation boundary, verification steps, abstention logic, RCA Casebook evaluation + API/UI panel ([doc](m2-wp4-explainable-rca.md); `451` backend / `32` frontend tests; MetroPT-3 `F1..F4` `1.0` grounding/agreement/disclaimer, `34` warnings `1.0` grounding / `0.6765` non-leak rate) | [#8](https://github.com/Mohamed-Osama-ai7/FactoryMind-CDT/pull/8) · `004cb9a` (`5e4d6c9`) |
| **WP-F1 (`M2-WP5`)** | M2 | WP-A, WP-B | **Complete** | Complete Human Decision Support & Investigation Workspace UI on `/incidents/$incidentId` + `/incidents`: evidence timeline, RCA review, `confirm`/`dismiss`/`defer` actions with SoD enforcement, append-only `OutcomeVerification` record (`SCHEMA_VERSION="3"`, `POST /incidents/{id}/outcomes`), `/audit` hash-chain viewer UI ([doc](m2-wp5-decision-workspace.md); `458` backend / `39` frontend tests) | [#9](https://github.com/Mohamed-Osama-ai7/FactoryMind-CDT/pull/9) · `aa2a50b` (`a9821b3`) |
| **WP-C1 (`M3-WP0`)** | M3 | M2 complete | **In Review** | Industrial Asset Digital Passport (`asset:read`, physical specs, sensor limits/units, failure modes, model applicability boundaries, unobserved quantities) + non-destructive Data Foundation (`SourceDatasetRecord`, SHA-256 checksum & Parquet schema verification, uninterpolated gap honesty, lineage) + `/assets/` overview & `/assets/$assetId` detail UI ([doc](m3-wp0-asset-digital-passport.md); `479` backend / `44` frontend tests; live MetroPT-3 `verified`) | `feat/m3-wp0-asset-digital-passport` |
| **WP-C2 (`M3-WP1`)** | M3 | WP-C1 | **Planned** | Multi-Model ML Engine & Uncertainty Quantification: specialized model orchestrator, OOD/Mahalanobis/applicability gating, data-quality-degraded abstention (`"Insufficient evidence to make a reliable diagnosis"`), model promotion contracts + **AI4I 2020** controlled classification & calibration benchmark | — |
| **WP-C3 (`M3-WP2`)** | M3 | WP-C1, WP-C2 | **Planned** | Multimodal/Multi-Model RAG & Knowledge Graph: verified technical corpus (MetroPT-3 PDF + pneumatic/maintenance notes with license/provenance), BM25 + TF-IDF/SVD hybrid + Reciprocal Rank Fusion + KG traversal (assets/components/sensors/regimes/events/failure modes/docs with causal vs correlation typing), retrieval eval (`Recall@K`, `MRR`, `nDCG`), tenant-scoped search API & Document/KG UI | — |
| **WP-D (`M4-WP0`)** | M4 | WP-B, WP-C1 | **Planned** | Physics-Informed Compressor & Reservoir Digital Twin (normalized reservoir mass-balance + duty dynamics fitted strictly on training/cal normal segments, explicit unidentifiable parameters), held-out trajectory validation (MAE/RMSE/constraint violations/stability), controlled scenario simulator (baseline, leak, continued operation, maintenance repair) + Simulation UI | — |
| **WP-E (`M4-WP1`)** | M4 | WP-D, WP-F1 | **Planned** | Impact Estimation & Bounded Maintenance Optimization: kWh/load proxy & downtime exposure separated cleanly into measured, engineering estimate, configurable economic assumption, and simulated projection; constrained option comparison (repair now vs schedule vs monitor) integrated into incident decision comparison UI | — |
| **WP-G (`M4-WP2`)** | M4 | WP-C3, WP-D, WP-E | **Planned** | Controlled Industrial Copilot: permission-scoped read-only orchestration tools (`get_event_evidence`, `get_rca_hypotheses`, `search_docs`, `query_kg`, `simulate_scenario`, `compare_options`), strict input/output schema validation, offline deterministic synthesis fallback + optional configured LLM adapter, citation grounding & prompt-injection/safety red-team suite | — |
| **WP-H (`M5-WP0`)** | M5 | WP-A..G | **Planned** | Read-Only OT & Edge Connector Foundations (`OPC-UA`, `MQTT`, `Historian` read-only adapters with explicit source allowlist, SSRF/host hygiene, reconnect/buffer handling, cadence validation, zero PLC writes enforced via `test_safety_boundary.py`), supply-chain hardening (`LICENSE`, SHA-pinned CI actions, signed non-pickle or hash-attested model metadata option), final cross-system ablations, performance & security verification | — |

---

## 3. Open Technical Decisions & Limitations

- **D2 (Production DB / Alembic / PostgreSQL):** SQLite with triggers used locally; PostgreSQL role-enforced immutability & Alembic migrations remain open before multi-instance production deployment.
- **D4 (MetroPT-3 maintenance-date ambiguity):** Source failure report table lacks exact verified maintenance repair timestamps; M5/WP-D outcome verification remains bounded against reported event end windows and synthetic/user-recorded test cases rather than unverified repair dates.
- **D5 (Licensed external corpus):** Only repo-verified technical documents and explicitly attributed UCI/MetroPT-3/NIST ISO 23247 public materials are indexed cleanly in RAG; proprietary manuals require explicit owner ingestion.
- **D6 (External LLM provider):** Copilot uses deterministic grounded synthesis by default and optionally calls an external LLM API only when explicitly configured with provider credentials and strict schema/citation checks.
- **S0 Governance (`LICENSE`, CI SHA pinning):** Root `LICENSE` (`MIT` matching `pyproject.toml`) and CI action pinning will be reconciled during WP-H security hardening (keeping workflow tests green across `main`).
- **PR #3 Boundary:** Branch `docs/post-m1-review-rebased` (PR #3) remains open and untouched throughout.

