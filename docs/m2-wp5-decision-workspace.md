# M2-WP5 (`WP-F1`) — Human Decision Support, Outcome Verification & Audit Workspace

M2-WP5 completes Milestone M2 by closing the advisory human-in-the-loop workflow on top of M2-WP1 authentication/persistence, WP-A multi-phase `EventEvidencePackage`, and WP-B explainable `RcaAssessment`:

1. **Append-only Outcome Verification persistence (`SCHEMA_VERSION = "3"`):** Records verified post-incident outcomes (`confirmed_leak`, `confirmed_other_fault`, `false_alarm`, `inconclusive`, `not_inspected`) separately from model predictions and incident creation decisions.
2. **Decision Workspace & Separation-of-Duties (SoD) UI (`DecisionPanel.tsx`):** Lets approvers (`approver`, `admin`) review incident state and record `confirm`, `dismiss`, or `defer` decisions with rationale, while clearly surfacing SoD enforcement when `actor_id === incident.created_by` and checking `defer` workflow transitions cleanly.
3. **Human Outcome Verification UI (`OutcomeVerificationPanel.tsx`):** Lets engineers/approvers/admins (`outcome:create`) log append-only field verification records marked explicitly as `origin="human_observation"`.
4. **Tenant-Scoped Audit Trail & Hash-Chain Verification UI (`AuditViewer.tsx`, `/audit`):** Gives `auditor` / `admin` callers (`audit:read`) action filtering, pagination, and one-click SHA-256 hash-chain verification via `/api/v1/audit/events` and `/api/v1/audit/verify`.

---

## 1. Backend Changes

- **Permission & RBAC (`factorymind.auth.principal`):**
  - Added `Permission.OUTCOME_CREATE = "outcome:create"`, granted to `ENGINEER`, `APPROVER`, and `ADMIN` roles (`VIEWER` and `AUDITOR` remain strictly read-only).
- **Schema & Append-Only Persistence (`factorymind.db.models`, `factorymind.db.repository`):**
  - Bumped `SCHEMA_VERSION` from `"2"` to `"3"` in `models.py` (refusing unmigrated v2 files via `SchemaMismatchError`).
  - Added `outcome_verification` to `APPEND_ONLY_TABLES` (enforcing SQLite `BEFORE UPDATE` and `BEFORE DELETE` immutability triggers).
  - Added `OutcomeVerification` ORM model (`id`, `tenant_id`, `asset_id`, `incident_id`, `actor_id`, `outcome`, `summary`, `observed_at`, `created_at`).
  - Added `OutcomeKind` (`Literal["confirmed_leak", "confirmed_other_fault", "false_alarm", "inconclusive", "not_inspected"]`), `IncidentRepo.outcomes()`, and `IncidentRepo.add_outcome()` with tenant and asset scoping.
- **API Endpoints & Contracts (`factorymind.api.incidents`):**
  - Added `OutcomeVerificationCreate`, `OutcomeVerificationOut` (fixing `origin: Literal["human_observation"] = "human_observation"`), and `IncidentDetail.outcomes`.
  - Added `POST /api/v1/incidents/{incident_id}/outcomes` protected by `Permission.OUTCOME_CREATE`, tenant/asset isolation, write capability enforcement, and hash-chained `"outcome.create"` audit logging.

---

## 2. Frontend & OpenAPI Integration

- Regenerated `frontend/openapi.json` and Orval TypeScript hooks (`OutcomeVerificationCreate`, `OutcomeVerificationOut`, `IncidentDetail.outcomes`, `useCreateOutcomeVerification`).
- **`DecisionPanel.tsx` & `OutcomeVerificationPanel.tsx` (`routes/incidents.$incidentId.tsx`):**
  - Mounted alongside `EventEvidencePackageView`, `ObservationForm`, and `RcaPanel` on `/incidents/$incidentId`.
  - Enforces UI hints matching server RBAC and SoD rules (server remains authoritative), displays recorded `decisions` and `outcomes`, and invalidates incident + incident-list queries after submission.
- **`AuditViewer.tsx` (`routes/audit.tsx`, `__root.tsx`):**
  - Added permission-gated `Audit trail` link in the navigation bar (`audit:read`).
  - Supports action filtering (`incident.create`, `decision.create`, `observation.create`, `outcome.create`, `authz.denied`, `authn.failure`), pagination, and SHA-256 chain integrity verification.

---

## 3. Verification

- **Safety Boundary (`test_safety_boundary.py`):** Preserved strictly advisory, non-actuating design (`POST /incidents/{incident_id}/outcomes` added to explicit human-workflow write allowlist).
- **Backend (`test_decision_workspace_api.py`, `test_auth_routes.py`, `test_safety_boundary.py`, `test_openapi_contract.py` + full suite):**
  - Verified RBAC, separation of duties (`creator != decision approver`, including `409 Conflict` & optimistic `version` concurrency control), `defer` transitions, append-only SQLite `UPDATE`/`DELETE` trigger refusal on `outcome_verification`, SHA-256 audit hash-chain integrity across `outcome.create`, and `SCHEMA_VERSION` `"2"` -> `"3"` mismatch refusal.
  - Full backend run: `uv run ruff format --check .` (`69 files`), `uv run ruff check .`, `uv run mypy` (`39 source files`), and `uv run pytest -q` (**458 passed**).
- **Frontend (`DecisionOutcomeWorkspace.test.tsx`, `AuditViewer.test.tsx` + full suite):**
  - Full frontend run: `biome check .` (`48 files`), `tsc -b`, `vitest run` (**10 test files / 36 tests passed**), and `vite build`.
