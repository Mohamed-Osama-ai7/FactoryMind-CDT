# Post-M1 architecture and product review

Scope: M1 as pushed on `feat/m1-ml-pipeline` (commits `2ccd434`, `6ef0810`, `2e95e90`; CI green on
all three). Reviewed against the project vision (System Analysis, Playbook, Project Documentation,
owner direction of 2026-11-09). The M1 PR has not been opened or merged as of this review, and M2
has not been started.

Evidence comes from `docs/analysis/post_m1_review.py` (reproducible, read-only). It is
**exploratory**: the test period has already been seen, so these numbers justify what to fix, not
which model to ship.

## 1. Verdict

1. **MetroPT-3 is the right first subsystem.** It is real, permissively licensed, physically
   interpretable, and supports SENSE → VALIDATE → UNDERSTAND with honest labels. It supports
   *detection and evidence generation*, but **cannot measure root-cause accuracy**: every report is
   the same failure type (air leak).
2. **The M1 plumbing is sound and worth keeping**: shared train/inference features, checksummed data
   and models, leakage-safe splits, typed API, generated client, safety test.
3. **The M1 modelling conclusion is weaker than the write-up suggests.** Selection rests on one
   validation event; the only rule baseline was the plant's low-pressure switch, which barely
   responds to leaks, while the strongest simple baseline (compressor load duty) was missing; and the selected detector's score is saturated.
4. **Four architectural gaps will cause rework in M2 if not closed first**: stateful alarm/incident
   handling, domain contracts, persistence, and operating-state awareness.
5. **Recommendation:** do not start the planned M2 directly. Run a short hardening work package
   (WP0, below) as the first part of M2, because the later workflow (diagnosis, uncertainty,
   simulation) inherits the semantics of the M1 score and alarm.

## 2. Evidence

| # | Finding | Evidence |
|---|---|---|
| E1 | Both learned models separate every report well; failures are in thresholds/policy, not representation | AUROC in-report vs far-normal: Isolation Forest 0.994 / 0.988 / 0.997 / 0.950 (F1–F4); PCA SPE 0.989 / 0.968 / 0.992 / 0.949; robust z 0.80; plant LPS switch ≈ 0.50 |
| E2 | The F4 miss is a threshold artefact | IF in-report window recall at its threshold: F1 0.46, F2 0.90, F3 0.89, **F4 0.00** (F4 median score 0.710 < threshold 0.726 although AUROC is 0.95). PCA: 1.00 / 0.97 / 1.00 / 0.93 |
| E3 | The Isolation Forest score is saturated and cannot express severity | Median in-report score 0.726–0.730 against threshold 0.726, far-normal p99 0.723, global max 0.740. PCA SPE grades severity (F1–F3 median ≈ 200–250, F4 ≈ 56, threshold 3.8) |
| E4 | Selection is fragile | Validation holds one event (F1); IF beat PCA by one false-alarm episode. Pooled post-hoc, same policies: IF 3/4 at 0.024 FA/day, PCA 4/4 at 0.089 FA/day. Neither result is unbiased |
| E5 | **The models do not demonstrably beat a one-feature physics rule** | Load duty (share of a window with motor current > 5.5 A): AUROC 0.967 / 0.952 / 0.966 / 0.770. Median 0.733 in-report vs 0.066 normal; 85 % of in-report windows exceed 0.5 vs 3.5 % of normal. Rule "duty > 0.5 for 1 h": 4/4 at 0.032 FA/day. **Threshold chosen post hoc**, so optimistic; the calibration-quantile version of the same rule fails (0–1/4) |
| E6 | The "normal" calibration/training periods are not clean | 30 of 1,912 calibration windows (2020-03-26/27/29) and 45 of 5,400 training windows have duty > 0.9, versus a normal median of 0.07. Thresholds that assume "no anomalies here" are biased |
| E7 | Presented evidence is correct but shallow and redundant | #1 evidence feature is `DV_pressure_min/mean` in 352 of 354 alarmed in-report windows; top-5 spans ~4 sensors that all restate "compressor is loaded". It describes the *state*, not a leak |
| E8 | A designed physics feature is unobservable when it matters | `unloaded_tp3_slope` is defined in 97 % of normal windows but 1 % of in-report windows: during a leak the compressor never unloads |
| E9 | Alarms confuse different explanations | PCA false alarms include a 20-hour episode with load duty 0.0 (machine stopped) and 12 h at 0.89; IF's three false alarms are 2020-05-19/20 at duty 0.50–0.76. "Anomalous" ≠ "leak": stop, high demand and leak look different only with operating-state context |
| E10 | Possible unreported degradation | Far-from-report windows score high in April–June (PCA p95 ≈ 54 / 223 / 50) and quiet in July–August (≈ 1.3). "False alarms" before F2/F3 may be precursors; the labels cannot say |

## 3. Is MetroPT-3 → anomaly detection the right foundation?

**Yes, with explicit limits.**

- **Fit with the product:** it feeds the whole lifecycle for one asset: telemetry replay (SENSE),
  schema/range/gap checks (VALIDATE), windowed features and detectors (UNDERSTAND). Maintenance
  dates in the report table enable VERIFY later (after they are disambiguated; see D4 in §9).
- **What it cannot do:** multi-class RCA evaluation (one failure type), quality intelligence (no
  quality data), retrieval (no documents). These need other data, sourced deliberately:
  controlled fault injection on real telemetry (clearly labelled *simulated*) for hypothesis
  discrimination; AI4I 2020 (known failure rules) for RCA/uncertainty benchmarking; a licensed
  compressor/air-brake document corpus for retrieval.
- **Risk to the product claim:** the platform thesis is that each component must earn its
  complexity (System Analysis §24). E5 says detection alone is not where the ML earns it on this
  dataset. Its justified roles are (a) multi-signal anomalies a single rule cannot see (sensor
  faults, stopped machine), (b) severity/OOD scoring, (c) feeding evidence. These must be shown by
  experiment, not asserted.

## 4. Requirement coverage after M1 (System Analysis §16, subset)

| Area | Status |
|---|---|
| FR-007 machine monitoring | **Partial**: score timeline only; no raw-signal trends |
| FR-008 read-only integration | **Partial**: file replay, no `TelemetrySource` abstraction |
| FR-009 validation/quarantine | **Partial**: batch cleaning + request validation; no quarantine store |
| FR-010 freshness | Not applicable to replay; not built |
| FR-013 anomaly detection | **Done** for one asset (baselines + 3 detectors, model card) |
| FR-012 incidents, FR-025 evidence, FR-026/027 RCA + uncertainty, FR-032–035 recommendation/approval/verification/memory, FR-037 audit | **Not started**; alarm episodes are not persisted as incidents |
| FR-015–023 documents/retrieval/graph, FR-028–031 twin/simulation/impact | Not started (as planned) |
| FR-001–003 identity/tenancy | Not started; scope decision open (§9) |
| NFR-09 reproducibility, NFR-17 model governance | **Partial** (seeded config fingerprint, checksums, model card; no registry/rollback beyond a `CURRENT` pointer) |
| NFR-16 safety | **Done for current routes** (allowlist test); must extend with every new route |
| NFR-08 observability | Not started (no structured logs, metrics, correlation IDs) |

## 5. How M1 maps onto the lifecycle

| Stage | Provided by M1 | Gap |
|---|---|---|
| SENSE | CSV/Parquet replay; `POST /score` accepts samples | No source abstraction, asset IDs, freshness, live feed |
| VALIDATE | Contract checks, cleaning, `insufficient_data` status | No quarantine, no data-quality flags passed downstream |
| UNDERSTAND | Windows, features, 4 detectors, score + status | No operating state, no calibrated score, no severity semantics (E3) |
| RETRIEVE | – | No corpus, index or evidence store |
| DIAGNOSE | Feature z-score table | Not evidence items; no hypotheses (E7, E9) |
| ASSESS UNCERTAINTY | Window validity only | Scores are uncalibrated; no OOD/abstention logic |
| SIMULATE / ESTIMATE IMPACT | – | Needs a validated process model and assumptions |
| HUMAN REVIEW / VERIFY / LEARN | – | Needs incidents, decisions, audit, maintenance records |

## 6. Keep, change, defer

**Keep**
- Package/layer structure, shared `features.py` for training and inference, gap-aware windows
  (never interpolated).
- Chronological splits, calibration-from-normal threshold idea, model card with provenance and
  checksum, `/score` + `/replay` + `/model` shape, generated client with drift checks, safety test.
- All four detectors in code (PCA SPE and IF both have a role, see §8), and the honest write-up.

**Change now (before M2 builds on it)**
1. **Add the duty-cycle physics baseline** and report all detectors against it (E5). The "plant
   switch" remains a baseline but is not the bar.
2. **Replace single-validation-event selection** with leave-one-event-out over the four reports,
   thresholds fitted per fold from that fold's earlier data, per-event reporting, and a selection
   rule fixed in config before running. Do not claim statistics from n = 4.
3. **Stop assuming clean normal data** (E6): exclude or flag windows with extreme duty from the
   calibration/training fit using an explicit, documented rule, and report the sensitivity.
4. **Calibrate scores to exceedance** (share of reference-normal windows with a higher score) so
   every detector emits a comparable, honestly-defined number; stop exposing raw IF scores as
   severity (E3). Use PCA SPE as the severity/OOD signal.
5. **Deduplicate evidence by physical signal** and label it with its operating meaning (E7); add
   features observable during leaks (e.g. TP3 rise rate while loading) in place of the unobservable
   slope (E8).
6. **Introduce a stateful alarm tracker** (windows in order → incident open/update/close events).
   Today persistence is computed only within the requested replay range, so incident boundaries
   depend on the query.
7. **Operating-state segmentation** (stopped / unloaded / loaded / starting) and state-aware status
   (a stopped machine is *explained*, not anomalous) (E9).

**Defer**
- Full multi-tenancy and enterprise SSO (authentication itself is **not** deferred; see D3 as revised), model registry/MLflow, drift monitoring, streaming
  ingestion, async job queue, deep-learning detectors, any LLM.
- Replacing joblib model artifacts (pickle). Accepted risk while artifacts are produced locally;
  record in ADR-009 and revisit before any shared deployment.

## 7. Architectural gaps that would cause rework

| Gap | Why it matters | Resolution |
|---|---|---|
| No domain contracts (`Case`, `AnomalyResult`, `EvidenceItem`, `Hypothesis`, `UncertaintyAssessment`, `Recommendation`, `DecisionRecord`) separate from API schemas; no `asset_id`, `case_id` or `schema_version` | Every later stage needs a stable shared vocabulary | `factorymind.contracts` package, versioned, API schemas derived from it |
| No persistence | Incidents, evidence, decisions and audit cannot exist without it | Decision D2 |
| Alarm state is range-bound | Incident identity is undefined | Alarm tracker (§6.6); property test: chunked replay equals single pass |
| Score has no defined meaning | Uncertainty/abstention cannot be built on an arbitrary number | Exceedance calibration (§6.4) |
| Evidence is not an object | RCA, retrieval and review all attach to evidence | `EvidenceItem` with id, source, polarity, physical meaning |
| Model loaded once (`lru_cache`) | A new `CURRENT` needs a restart | Acceptable now; revisit with registry |
| Telemetry fully in memory (~180 MB) | Fine for one asset | Revisit for multi-asset (DuckDB/Parquet partitioning) |
| Raw artifact trust (joblib) | Arbitrary code on load if the artifact directory is writable by others | Local-only for now (ADR-009) |

## 8. Isolation Forest vs PCA SPE

Keep the pre-declared selection (Isolation Forest) **for M1 reporting**: switching to PCA because it
looks better on test would be tuning on test. But do not treat either as final:

- IF: very few false alarms, saturated score (E3), misses the mildest report at its threshold (E2).
- PCA SPE: detects all four with delays of 0–90 min after report start and has real
  severity range, at a higher false-alarm rate that is partly stopped-machine and high-demand
  periods (E9).
- The plausible product design is a **PCA-style severity/OOD signal with calibrated exceedance,
  an operating-state-aware status, and the duty-cycle rule as the explicit baseline**, decided by
  the leave-one-event-out protocol, not by this review.

## 9. Roadmap review and dependencies

Revised sequence (earlier roadmap M0–M6, now version-controlled here):

| Step | Content | Depends on |
|---|---|---|
| M1 (done, PR pending) | Detection pipeline | – |
| **M2-WP0** | Hardening in §6 (baseline, LOEO, calibration, tracker, evidence, state) | M1 merged |
| M2-WP1 | Contracts, persistence, incident lifecycle, audit | Decision D2 |
| M2-WP2 | Evidence items and hypotheses (leak, high demand, sensor fault, stopped/maintenance) with supporting/contradicting/missing evidence and abstention; evaluation by labelled fault injection on real telemetry + the four real leaks | WP0, WP1 |
| M2-WP3 | React incident workspace: review, approve/reject, audit trail | WP1–2 |
| M3 | Retrieval: licensed corpus, BM25 vs dense vs hybrid, citations inside evidence | Corpus sourcing (start early) |
| M4 | Uncertainty/OOD/conformal; AI4I RCA benchmark; optional grounded LLM report writer | M2, M3 |
| M5 | Reservoir mass-balance twin, repair-now-vs-defer, impact ledger, verification with maintenance dates | Clarified maintenance dates; M2 |
| M6 | Ablations A0–A9, red-team/safety/performance tests, final evidence | All |

Dependencies on the owner or outside the repo: maintenance-date clarification (the source table is
ambiguous; the dataset authors are listed as contacts), a retrieval corpus with usable licences,
and the persistence decision (Docker/WSL is not set up and C: has ~2 GB free).

## 10. Recommended next milestone and acceptance criteria

**M2 "Trusted detection → incident → evidence-backed hypotheses".** Begin with WP0; the rest follows
only after WP0 passes.

WP0 acceptance (all must hold):
1. Leave-one-event-out table for `lps_rule`, `duty_rule`, `robust_z`, `isolation_forest`, `pca_spe`
   with per-fold thresholds from earlier data only; events detected (/4), per-event delay, false
   alarms/day on non-event days; selection rule committed in config before the run; rerun
   reproduces identical numbers (test).
2. An ML detector is kept as the product default only if it matches the duty rule's detection at
   no worse false alarms, **or** it detects labelled injected faults the rule cannot (stuck sensor,
   offset, flatline) at ≥ 90 % at ≤ 0.1 false alarms/day. Targets are provisional until measured;
   otherwise the rule stays default and the ML detector is documented as secondary.
3. Calibration-window exceedance rate at p < 0.01 lies within 0.5–2 % on held-out normal data
   after the documented exclusion of extreme-duty windows.
4. Property test: replaying any range split at arbitrary boundaries yields the same incidents as one
   pass (≥ 100 random splits).
5. Evidence: at most one item per physical signal, each with a documented physical meaning; no
   unobservable feature is presented as evidence.

M2 completion adds: every API response carries `asset_id`, `schema_version`, `model_version` and
the data window; an incident is persisted with an audit event per transition; a human decision is
recorded; every evidence ID resolves; fault-injected cases yield the correct hypothesis or
abstention at rates reported per class; the route-allowlist safety test still passes; the full
workflow is exercised by an end-to-end test through the real API and a Playwright run of the React
workspace.

## 11. Decisions needed from the owner

| ID | Decision | Recommendation |
|---|---|---|
| D1 | Adopt the WP0-first sequencing and the "ML must earn its place" rule (§10.2) | Accept |
| D2 | Persistence: PostgreSQL via Docker/WSL (needs admin, reboot, D: disk image) vs SQLite for local development with SQLAlchemy models plus a PostgreSQL service container in GitHub CI | Revise: SQLite locally, PostgreSQL in CI, avoiding PostgreSQL-only features; revisit before multi-user use |
| D3 | Authentication/tenancy (System Analysis marks it Must; MVP scope documents do not) | **Revised and approved 2026-11-10 (security audit SD-1):** minimal OIDC authentication with roles and audit before M2-WP1 persists anything; full tenancy deferred; `tenant_id`, `asset_id`, `actor_id` in every contract. Originally recommended deferral; reversed because approvals need authenticated actors |
| D4 | Maintenance-date ambiguity in the source report table | Ask the dataset authors or treat dates as unverified; do not build VERIFY on them until resolved |
| D5 | Retrieval corpus: which public documents are legally usable | Owner to nominate sources; Claude verifies licences |
| D6 | LLM provider and budget | Defer to M4; structured outputs and template report come first |
