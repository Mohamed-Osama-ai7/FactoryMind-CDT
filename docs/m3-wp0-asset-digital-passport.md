# M3-WP0 (`WP-C1`) — Industrial Asset Digital Passport & Data Foundation

M3-WP0 establishes the physical **Asset Digital Passport** registry and **non-destructive Data Foundation** layer connecting FactoryMind CDT's telemetry contracts, operating envelopes, failure modes, model applicability boundaries, and dataset provenance before multi-model ML and RAG/KG integration.

---

## 1. Domain Contracts (`factorymind.contracts.assets`)

- **`AssetPassport` & `AssetPassportSummary`:** Physical identity (`asset_id`, `asset_type`, `system_context`, `site_or_domain`), nominal sampling cadence (`10.0s`), `operating_regimes` (`stopped`, `unloaded`, `loaded`, `starting`, `insufficient_evidence`), regime physical envelopes, `sensors`, `failure_modes`, `connected_dataset_ids`, `model_applicability`, and explicit `unobserved_quantities` (such as downstream air-demand flowmeter and maintenance timestamps).
- **`SensorPassport`:** Links every canonical MetroPT-3 channel (`7` analog sensors with `ANALOG_SENSORS` units and `VALID_RANGES`, `8` digital binary signals with `[0.0, 1.0]` ranges) to its physical role, nominal operating envelope, and channel-specific unobservability notes (`H1`, `DV_pressure` excluded from univariate WP4 RCA sensor-freeze heuristics while `pca_spe` still uses all `42` window features).
- **`FailureModeSpec` & `ModelApplicabilityEntry`:** Explicitly pairs failure modes (`air_leakage`, `abnormal_high_demand`, `sensor_malfunction`, `telemetry_inconsistency`) with competing explanations and marks `cross_asset_transfer` as `not_calibrated`.
- **`SourceDatasetRecord`, `DatasetQualitySummary`, `QualityCheckItem`, `DataLineageStep`:** Track license (`CC BY 4.0`), expected archive SHA-256 (`aab991a9...`), non-destructive raw source preservation, uninterpolated gap policy (`preserve_gaps_uninterpolated`), automated checks, and 4-step preparation lineage.

---

## 2. Registry & Non-Destructive Data Foundation

- **`factorymind.assets.passport`:**
  - `get_asset_passport`, `list_asset_passports`, and safe JSON-only `read_active_model_version` (blocking `CURRENT` pointer path traversal and avoiding deserializing pickled models).
- **`factorymind.data.foundation`:**
  - `inspect_metropt3_dataset` inspects `raw/metropt3/metropt3.zip`, `processed/metropt3/manifest.json`, and `processed/metropt3/telemetry.parquet` using `pyarrow.parquet` footer metadata and schema checks in O(1) without loading or mutating files.
  - `verify_dataset_integrity` verifies full raw archive SHA-256 while preserving source bytes verbatim.
  - Live MetroPT-3 foundation verification run confirmed `artifact_status = "verified"`, raw zip SHA-256 match, `1,516,948` cleaned rows across all `15` canonical sensor columns, and explicit honest reporting of `331` gaps > 60s (`909.52 h` preserved uninterpolated).

---

## 3. Read-Only API & `/assets` Frontend UI

- **RBAC (`factorymind.auth.principal`):** Added `Permission.ASSET_READ = "asset:read"` for `viewer`, `engineer`, `approver`, and `admin` (`auditor` remains scoped to incidents and audit verification).
- **Read-Only Endpoints (`factorymind.api.assets`):**
  - `GET /api/v1/assets`
  - `GET /api/v1/assets/{asset_id}` (`404` on unknown or ungranted `asset_id`)
  - `GET /api/v1/datasets` (optional `asset_id` filter scoped to caller grants)
  - All three endpoints enforce `require_data_source(Permission.ASSET_READ)` and add zero mutating routes (`test_safety_boundary.py` unchanged).
- **Frontend UI (`AssetsOverview.tsx`, `AssetPassportDetail.tsx`, `SourceDatasetCard.tsx`, `/assets/`, `/assets/$assetId`):**
  - Displays asset passports, sensor filterable catalog (`all` / `analog` / `digital`), unobserved-quantity disclosures, model applicability matrix, and non-destructive dataset provenance & lineage cards.

