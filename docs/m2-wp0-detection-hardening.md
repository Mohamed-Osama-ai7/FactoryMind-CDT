# M2-WP0 — Detection hardening

Scope: the "change now" items from the post-M1 review (§6, §10 of the review): fair baselines, a
leakage-aware protocol, honest score semantics, a single alarm implementation, grouped evidence and
operating-state awareness. This supersedes the single-validation-event protocol described in
[m1-anomaly-detection.md](m1-anomaly-detection.md); that document remains the record of M1.

## What changed

| Area | Change |
|---|---|
| Operating state | Motor current is trimodal (about 0.04 A off, 4 A unloaded, 7 A loaded; 55 % of the time off). Windows are `stopped` (explained, never an alarm) when the machine was off ≥ 90 % of the window; stopped and extreme-duty (> 0.9) windows are excluded from fitting and calibration |
| Features | `off_fraction`, `loaded_tp3_rise` (pressure rise while loading; slower during a leak, observable when `unloaded_tp3_slope` is not) |
| Baseline | `duty_rule`: compressor load duty alone, the physics baseline |
| Protocol | Train / calibration / evaluation; thresholds are quantiles of reference-normal scores (label-free) or an absolute grid (duty rule); **nested leave-one-event-out**: the policy (variant, threshold, persistence) is chosen on the other reports and measured on the held-out one |
| Default rule | Pre-declared in config: the best rule baseline is the bar. A learned detector becomes the default only if it matches the rule's leak detection at no worse false alarms, or detects simulated sensor faults at ≥ 90 % with ≤ 0.1 false alarms/day without losing leak detection |
| Simulated faults | Stuck, offset, flatline, drift and noise faults injected into 60 real normal operating intervals (3 h each), with unmodified control intervals. Always labelled simulated |
| Calibration | `ExceedanceCalibrator`: share of reference-normal windows scoring at least as high (smoothed conformal p-value). Not a failure probability |
| Alarm tracking | `AlarmTracker` is the one implementation of "k consecutive anomalous windows", used by evaluation and replay. Replay scans a 24 h look-back, so an incident's start and raised-at do not depend on the requested range |
| Evidence | One item per physical signal group (operating state, pressure level, pressure dynamics, oil temperature, protection switches) with a stated meaning; features that never varied in training are reported as *novel*, not with an absurd z-score; unobservable features are never reported |
| API and UI | `stopped` status, exceedance, grouped evidence, `truncated` / `ongoing` episodes, LOEO and fault summaries; error messages no longer disclose server paths. Model format `schema_version` 2 (older cards are refused) |

## Results (real MetroPT-3, four operator air-leak reports; evaluation period 2020-04-01 to 09-01)

Reproduce with `uv run python -m factorymind.anomaly.cli train` (about 35 s after `prepare`).

| Detector | Leaks detected (LOEO) | False alarms/day | Median \|delay\| | Simulated faults detected | Control intervals alarmed | Holdout exceedance at 1 % |
|---|---|---|---|---|---|---|
| lps_rule (plant switch) | 1/4 | 0.000 | 240 min | – | – | 0.7 % |
| **duty_rule (default)** | **4/4** | **0.032** | **50 min** | 6/60 (10 %) | 6/60 | 27.5 % |
| robust_z | 4/4 | 0.860 | 70 min | – | – | 22.8 % |
| isolation_forest | 3/4 | 0.043 | 80 min | 14/60 (23 %) | 11/60 | 27.8 % |
| pca_spe | 4/4 | 0.057 | 45 min | 44/60 (73 %) | 15/60 | 3.3 % |

- **Selection was stable.** The duty rule chose threshold 0.5 with persistence 6 in all four folds;
  PCA SPE chose quantile 0.99 with persistence 6 in all four. Isolation Forest was not stable: it
  missed F1 when the other three reports chose a stricter policy.
- **Per-event (held-out):** duty rule +90 / +50 / +50 / −20 min against F1–F4; PCA SPE +90 / +40 /
  +50 / −40 min. Negative means the alarm was raised before the reported start.
- **PCA SPE and faults:** stuck, offset and drift faults 12/12 each; flatline and noise 4/12 each.
  But a quarter of unmodified control intervals also alarm, so the net lift is smaller than the
  headline rate.
- **Duty rule and faults:** 6/60 detected against 6/60 control alarms: it cannot see sensor faults,
  as expected.
- **Sensitivity to the extreme-duty exclusion:** without it the duty rule and PCA SPE are unchanged
  in leak detection, Isolation Forest drops from 3/4 to 2/4, and the default stays `duty_rule`.

## Acceptance against the review criteria

| # | Criterion | Result |
|---|---|---|
| 1 | Leave-one-event-out table for all five detectors, selection rule fixed before the run, rerun reproduces numbers | **Met with a deviation.** Reproducibility is tested. Thresholds come from the earlier calibration period, but the LOEO policy choice uses the other reports, including later ones; "earlier data only" is therefore not literally satisfied (the identical choices across folds limit the effect) |
| 2 | A learned detector is default only if it matches the rule or detects injected faults ≥ 90 % at ≤ 0.1 false alarms/day | **Rule applied; no learned detector qualified.** PCA SPE reaches 73 %, Isolation Forest 23 %. The default is `duty_rule` |
| 3 | Held-out exceedance rate near 1 % after the documented exclusion | **Not met.** Fit on the first half of the calibration period and tested on the second half, the exceedance rate is 22.8–27.8 % for duty rule, robust z and Isolation Forest (PCA SPE 3.3 %). The "normal" reference is not stationary even within two weeks |
| 4 | Alarm incidents identical across chunkings | **Met.** 100 random chunkings of the tracker and 12 random replay ranges reproduce the single-pass episodes |
| 5 | At most one evidence item per physical signal; no unobservable feature shown | **Met** (tested) |

## What the evidence says

1. **A physics rule is the right default today.** Load duty over one hour detects all four reports,
   including F4, at 0.03 false alarms/day, chosen without peeking at the held-out report. The
   learned detectors did not beat it on leaks.
2. **PCA SPE has a different, real job.** It catches stuck, offset and drift sensor faults that the
   rule cannot, but at an operating point that alarms too often to be the leak alarm. The better
   design is two alarm classes: a leak alarm (duty rule) and a *consistency/sensor warning* (PCA
   SPE) evaluated on its own terms.
3. **"False alarms" against leak labels include genuine abnormal states.** In April–June, PCA SPE
   flags windows whose largest deviation is the oil-level switch (never active in training), plus
   unusual load cycling and pressure-switch activity; 61 % of its flagged far-from-report windows
   have load duty above 0.5. These are real unusual conditions that are not air leaks, which is why
   the product needs hypotheses and evidence, not a single alarm.
4. **Exceedance is relative to a reference that drifts**, so it is reported as "share of reference
   windows", never as a probability, and the check above is shown to users in the model card.

## Known limitations

- Four coarse reports; LOEO supports per-event reporting, not statistics.
- The normal time used for false-alarm estimates is shared by selection and evaluation, which
  favours all detectors equally; the deployed policy is selected on all four reports (in-sample).
- Fault magnitudes (0.5 bar offset, 1 bar drift, 0.3 bar noise, frozen readings) are fixed
  engineering choices, not field statistics; results are simulated evidence only.
- Control-interval alarms show that "normal operating" time in this dataset is not clean.
- Stopped windows are explained, not analysed: a very long stop is itself information (maintenance,
  fault) that is not yet surfaced.
- Replay incidents longer than the 24 h look-back are reported as `truncated`.

## Next

WP0 closes the detection-side prerequisites for M2-WP1: contracts with `asset_id`, `tenant_id`,
`actor_id`, authentication and roles, persistence and audit (ADR-012, ADR-010). The consistency
warning class and a lower-false-alarm operating point for PCA SPE belong to M2-WP2.
