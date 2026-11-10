# M2-WP1 — Minimal OIDC authentication, RBAC and secure persistence foundations

Implements ADR-012 (authentication and roles before anything is persisted). Advisory and read-only
toward OT: nothing here writes to or controls industrial systems; a "decision" is a human review
record about an alarm, not an action.

## Scope

- FactoryMind is an OIDC **resource server**: it validates access tokens issued by an external
  identity provider (no passwords, no login UI in the backend). The React app signs users in with
  the authorization-code flow with PKCE.
- Single-tenant MVP with the tenant model in place: every record and contract carries `tenant_id`,
  `asset_id` and `actor_id`; every query is scoped by the caller's tenant and asset grants.
- Persistence for incidents, evidence snapshots, review decisions and a hash-chained audit log
  (SQLAlchemy; SQLite file under `FACTORYMIND_DATA_DIR`, dialect-neutral models).
- Deny by default: all routes except `GET /api/v1/health` require a valid token and an explicit
  permission.

Out of scope (deferred): enterprise SSO/SCIM, per-tenant keys, database-enforced row-level
security, PostgreSQL deployment and migrations (ADR-010), MFA enforcement (IdP policy), external
rate limiting/WAF, anything that acts on equipment.

## Roles and permissions

| Role | Permissions |
|---|---|
| viewer | `anomaly:read`, `incident:read` |
| engineer | viewer + `incident:create` |
| approver | viewer + `decision:create` |
| auditor | `incident:read`, `audit:read` |
| admin | everything except `decision:create` (separation of duties) |

An incident cannot be decided by the actor who created it (configurable, on by default). Unknown
role strings in a token are ignored; a token with no recognised role gets no permission.

## Acceptance criteria

1. Every route except `GET /api/v1/health` answers 401 without a valid token (checked by enumerating
   the application's routes).
2. Token validation rejects: expired, not-yet-valid, wrong issuer, wrong audience, `alg=none`,
   HS256 signed with the public key, unknown `kid`, tampered signature, missing `sub`/`exp`,
   oversized tokens. Failures return a generic body and never echo the token.
3. A role × route matrix test pins the exact allowed/forbidden outcome for every role (403 on
   missing permission, written to the audit log).
4. Tenant isolation: a principal from another tenant gets 404 (no existence leak) for read and
   decision on any incident; asset grants restrict by asset; a token without a tenant claim and
   without a configured default tenant is refused.
5. Persistence integrity: incident creation is idempotent per detected episode and its evidence is
   computed server-side (the client only points at an episode); evidence is immutable; decisions
   are append-only; stale `expected_version` → 409; decided incidents refuse further decisions;
   separation of duties enforced.
6. Audit: every authentication failure, authorization denial, incident creation and decision writes
   an event with actor, tenant, action, outcome and request ID; per-tenant hash chain verifies and
   a tampered or deleted row makes verification fail; audit read is tenant-scoped.
7. Rate limiting: per-principal read/write limits and per-client authentication-failure throttling
   return 429 with `Retry-After`; limits are independent per principal.
8. Input limits: request bodies over the cap → 413; unknown fields and out-of-range values → 422;
   pagination capped; time ranges capped.
9. Hygiene: no stack traces, paths or token contents in error bodies; `X-Request-ID` and security
   headers on every response; interactive API docs disabled in `production`.
10. Safety: no route name or path targets control; the state-changing route allowlist is exact and
    tested.
11. Frontend: no API call without a token; sign-in redirect, callback and sign-out; role-aware UI;
    401 handling; tests with a mocked OIDC client; lint, typecheck and build pass.
12. Backend ruff, format, strict mypy and all tests pass; frontend lint, typecheck, tests and build
    pass; generated API client has no drift.

## Results

Verified locally on Windows 11 (Python 3.12 venv, Node via npm):

| Gate | Result |
|---|---|
| Backend tests | 374 passed |
| Backend ruff check / format / mypy | clean |
| Frontend biome / tsc | clean |
| Frontend vitest | 25 passed |
| Frontend production build | succeeds |
| OpenAPI client regeneration | no drift after regeneration |

Evidence is from automated tests with locally generated RSA keys and a static JWKS. Not verified:
behaviour against a real identity provider (issuer metadata, key rotation, refresh), the browser
sign-in redirect end to end, PostgreSQL (tests use SQLite), and multi-process deployment (the rate
limiter is in-process, so limits are per worker). No production-readiness claim is made.

Remaining risks and unfinished items:

- Access tokens are held in memory only, so a page reload re-runs sign-in; silent renew is off.
- Roles/tenant/asset claim names are configurable but untested against a specific IdP.
- Rate limiting is single-process; a shared store is needed before scaling out.
- Audit hash chain detects tampering after the fact; it does not prevent a database owner rewriting it.
- No UI yet for recording decisions or reading the audit log (API only).

