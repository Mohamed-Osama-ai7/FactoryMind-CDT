# M2-WP5 (`WP-F1`) — Human Decision Support, Outcome Verification & Audit Workspace

M2-WP5 completes Milestone M2 by closing the advisory human-in-the-loop workflow on top of M2-WP1 authentication/persistence, WP-A multi-phase `EventEvidencePackage`, and WP-B explainable `RcaAssessment`:

1. **Append-only Outcome Verification persistence (`SCHEMA_VERSION = "3"`):** Records verified post-incident outcomes (`confirmed_leak`, `false_positive`, `sensor_fault_confirmed`, `high_demand_confirmed`, `inconclusive`) separately from model predictions and incident creation decisions.
2. **Decision Workspace & Separation-of-Duties (SoD) UI (`DecisionPanel.tsx`):** Lets dedicated `approver` callers (`decision:create` — explicitly excluded from `admin`) review incident state and record `confirm`, `dismiss`, or `defer` decisions with rationale, while surfacing SoD enforcement when `actor_id === incident.created_by`, displaying recorded history, and refreshing incident state on `409 Conflict`.
3. **Human Outcome Verification UI (`OutcomeVerificationPanel.tsx`):** Lets `engineer`, `approver`, and `admin` callers (`outcome:create`) log append-only field verification records marked explicitly as `origin="human_observation"`.
4. **Tenant-Scoped Audit Trail & Hash-Chain Verification UI (`AuditViewer.tsx`, `/audit`):** Gives `auditor` and `admin` callers (`audit:read`) action filtering, pagination, and one-click SHA-256 hash-chain verification via `/api/v1/audit/events` and `/api/v1/audit/verify`.

---

## 1. Backend Changes

- **Permission & RBAC (`factorymind.auth.principal`):**
  - Added `Permission.OUTCOME_CREATE = "outcome:create"`, granted to `ENGINEER`, `APPROVER`, and `ADMIN` roles (`VIEWER` and `AUDITOR` remain strictly read-only; `ADMIN` remains excluded from `Permission.DECISION_CREATE`).
- **Schema & Append-Only Persistence (`factorymind.db.models`, `factorymind.db.repository`):**
  - Bumped `SCHEMA_VERSION` from `"2"` to `"3"` in `models.py` (refusing unmigrated v2 files via `SchemaMismatchError`).
  - Added `outcome_verification` to `APPEND_ONLY_TABLES` (enforcing SQLite `BEFORE UPDATE` and `BEFORE DELETE` immutability triggers).
  - Added `OutcomeVerification` ORM model (`id`, `tenant_id`, `asset_id`, `incident_id`, `actor_id`, `outcome`, `maintenance_action`, `verified_at`, `notes`, `created_at`).
  - Added `OutcomeKind` (`Literal["confirmed_leak", "false_positive", "sensor_fault_confirmed", "high_demand_confirmed", "inconclusive"]`), `IncidentRepo.outcomes()`, and `IncidentRepo.add_outcome()` with tenant and asset scoping.
- **API Endpoints & Contracts (`factorymind.api.incidents`):**
  - Added `OutcomeVerificationCreate`, `OutcomeVerificationOut` (fixing `origin: Literal["human_observation"] = "human_observation"`), and `IncidentDetail.outcomes`.
  - Added `POST /api/v1/incidents/{incident_id}/outcomes` protected by `Permission.OUTCOME_CREATE`, tenant/asset isolation, write capability enforcement, and hash-chained `"outcome.create"` audit logging.

---

## 2. Frontend & OpenAPI Integration

- Exported updated `frontend/openapi.json` and regenerated Orval TypeScript hooks/types (`OutcomeVerificationCreate`, `OutcomeVerificationOut`, `IncidentDetail.outcomes`, `useCreateOutcomeVerification`).
- **`DecisionPanel.tsx` & `OutcomeVerificationPanel.tsx` (`routes/incidents.$incidentId.tsx`):**
  - Mounted alongside `EventEvidencePackageView`, `ObservationForm`, and `RcaPanel` on `/incidents/$incidentId`.
  - Surfaces UI hints matching server RBAC and SoD rules (server remains authoritative), displays recorded `decisions` and `outcomes`, recovers cleanly on `409 Conflict`, and invalidates incident, list, and audit queries after submission.
- **`AuditViewer.tsx` (`routes/audit.tsx`, `__root.tsx`):**
  - Added permission-gated `Audit trail` link in the navigation bar (`audit:read`).
  - Supports tenant-scoped action filters (`incident.create`, `decision.create`, `observation.create`, `outcome.create`, `authz.denied`), pagination capped at `MAX_OFFSET = 10_000`, and SHA-256 chain integrity verification.

---

## 3. Verification

- **Safety Boundary (`test_safety_boundary.py`):** Preserved strictly advisory, non-actuating design (`POST /incidents/{incident_id}/outcomes` added to explicit human-workflow write allowlist).
- **Backend (`test_decision_workspace_api.py`, `test_auth_routes.py`, `test_safety_boundary.py`, `test_openapi_contract.py` + full suite):**
  - Verified RBAC, separation of duties (`creator != decision approver`, including `409 Conflict` & optimistic `version` concurrency control), same-tenant cross-asset `404` isolation, `defer` transitions, append-only SQLite `UPDATE`/`DELETE` trigger refusal on `outcome_verification`, SHA-256 audit hash-chain integrity across `outcome.create`, and `SCHEMA_VERSION` `"2"` -> `"3"` mismatch refusal.
- **Frontend (`DecisionOutcomeWorkspace.test.tsx`, `AuditViewer.test.tsx` + full suite):**
  - Verified SoD UI lock, `confirm` / `defer` submission, `409 Conflict` refresh recovery, append-only outcome submission/rendering, action filter query omission (`all` without `?action=null`), pagination controls, permission error gating, and broken/intact hash-chain verification states.
