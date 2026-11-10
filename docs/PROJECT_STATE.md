# Project state

Read this first in every session; update it at the end of every substantial session. Do not treat
roadmap items as implemented.

_Last updated: 2026-11-10 (post-M1 review)._

## Current milestone

**M1 merged (PR 2, squash `1f9a893`).** Next: M2, beginning with WP0 hardening. Owner approved the WP0-first plan (D1)
and the authentication-first revision of D3 (SD-1) on 2026-11-10. The docs PR (PR 3) is open. Review and plan:
[architecture-review-post-m1.md](architecture-review-post-m1.md).

## Verified completed work

| Item | State | Evidence |
|---|---|---|
| M0 foundation (backend, React, CI) | Merged to `main` (PR 1, `666a956`) | CI green |
| M1 detection pipeline, API, React monitor, docs | **Merged to `main`** (PR 2, squash `1f9a893`) | CI green; backend 60 tests, frontend 12 tests |
| Post-M1 review, ADR log, this record | On `docs/post-m1-review-rebased`, PR 3 open | This commit |
| Security audit (read-only) | Report outside the repo: `D:\FactoryMind\security-audit\FactoryMind-security-audit-2026-11-10.md`; findings F-01–F-13, recommendations REC-01–30, decisions SD-1–12 | No code or settings changed by the audit |

## Branches

`main` = `1f9a893`. `docs/post-m1-review-rebased` = `main` plus the review commits (PR 3). The older
`docs/post-m1-review` and `feat/m1-ml-pipeline` branches are superseded.

## Active architecture decisions

See [decisions.md](decisions.md). Open: ADR-007 (under revision), ADR-010 (persistence, owner
decision D2). ADR-012 revised and accepted (authentication before persistence).

## Evaluation status (real MetroPT-3 data, four operator air-leak reports)

Final M1 comparison and protocol: [m1-anomaly-detection.md](m1-anomaly-detection.md). Test period
(82 days, three reports): Isolation Forest 2/3 at 0 false alarms/day (selected on validation), PCA
SPE 3/3 at 0.085/day, plant low-pressure switch 1/3. The post-M1 review found the selection fragile,
a one-feature duty-cycle rule competitive, a saturated Isolation Forest score, and a contaminated
"normal" calibration period. No model is accepted as the product default until WP0 passes.

## Known limitations

One compressor and one failure type; four coarse labels; report timezone and maintenance dates
unverified; scores are uncalibrated; no operating-state model; evidence is a z-score table; alarm
persistence is range-bound; no persistence layer, authentication, audit or observability; joblib
artifacts require trusted storage; nothing deployed or validated in a plant.

## Remaining dependencies

- PR 3 (docs) review and merge. The GitHub CLI is not signed in on the development machine; use the
  GitHub fallback agent for GitHub operations.
- Owner decisions D2, D4–D6 in the review (persistence, maintenance dates, corpus, LLM); D1 and D3 are
  approved. WP0 does not need them.
- Security decisions SD-2–SD-12 (personal data in the public repo, license, GitHub settings, IdP,
  deployment model, LLM policy, artifact format, compliance, ownership, workstation, residency).
  Priority P0 settings (REC-01: `main` ruleset, SHA-pinned actions, Dependabot, secret scanning,
  CodeQL) require repository admin action.
- Docker/WSL is not set up; C: has about 2 GB free (data, caches and the repository are on D:).
- Retrieval corpus with usable licences (needed for M3).

## Immediate next task

M2-WP0 per review §10 (branch from `main`) (duty-cycle baseline,
leave-one-event-out protocol, exceedance calibration, alarm tracker, deduplicated evidence,
operating-state features).

## Environment notes

Repository at `D:\FactoryMind\FactoryMind-CDT`; data in `D:\FactoryMind\data`, artifacts in
`D:\FactoryMind\artifacts`. Commit email is the GitHub noreply address (repo-local). Stop running
dev servers before `npm ci` on Windows (native modules are locked). `UV_SYSTEM_CERTS=1` is needed
for uv behind the machine's HTTPS interception.
