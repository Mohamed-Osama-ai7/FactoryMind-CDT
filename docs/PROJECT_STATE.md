# Project state

Read this first in every session; update it at the end of every substantial session. Do not treat
roadmap items as implemented.

_Last updated: 2026-11-10 (post-M1 review)._

## Current milestone

**M1 complete in code, awaiting PR review.** Next: M2, beginning with WP0 hardening. M2 must not
start until the M1 PR is opened, reviewed and merged. Review and plan:
[architecture-review-post-m1.md](architecture-review-post-m1.md).

## Verified completed work

| Item | State | Evidence |
|---|---|---|
| M0 foundation (backend, React, CI) | Merged to `main` (PR 1, `666a956`) | CI green |
| M1 detection pipeline, API, React monitor, docs | Pushed on `feat/m1-ml-pipeline` (`2ccd434`, `6ef0810`, `2e95e90`); **no PR exists yet** | CI green on all three commits; backend 60 tests, frontend 12 tests |
| Post-M1 review, ADR log, this record | On `docs/post-m1-review` (based on the M1 branch) | This commit |

## Branches

`main` = `666a956`. `feat/m1-ml-pipeline` = `2e95e90`. `docs/post-m1-review` = M1 plus the review commit
(merge M1 first; its diff against `main` then contains only the review).

## Active architecture decisions

See [decisions.md](decisions.md). Open: ADR-007 (under revision), ADR-010 (persistence, owner
decision D2), ADR-012 (tenancy, D3).

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

- M1 PR opened, reviewed and merged (the GitHub CLI is not signed in on the development machine;
  hand this to the GitHub fallback agent or open it in the browser).
- Owner decisions D1–D6 in the review (persistence, tenancy, corpus, maintenance dates, LLM).
- Docker/WSL is not set up; C: has about 2 GB free (data, caches and the repository are on D:).
- Retrieval corpus with usable licences (needed for M3).

## Immediate next task

After M1 is merged and decisions D1–D2 are answered: M2-WP0 per review §10 (duty-cycle baseline,
leave-one-event-out protocol, exceedance calibration, alarm tracker, deduplicated evidence,
operating-state features).

## Environment notes

Repository at `D:\FactoryMind\FactoryMind-CDT`; data in `D:\FactoryMind\data`, artifacts in
`D:\FactoryMind\artifacts`. Commit email is the GitHub noreply address (repo-local). Stop running
dev servers before `npm ci` on Windows (native modules are locked). `UV_SYSTEM_CERTS=1` is needed
for uv behind the machine's HTTPS interception.
