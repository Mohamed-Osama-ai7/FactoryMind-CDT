# M2-WP0 — Detection hardening

Scope: the "change now" items from the post-M1 review (§6, §10 of the review): fair baselines, a
leakage-aware protocol, honest score semantics, a single alarm implementation, grouped evidence and
operating-state awareness. This supersedes the single-validation-event protocol described in
[m1-anomaly-detection.md](m1-anomaly-detection.md); that document remains the record of M1.

## What changed

| Area | Change |
|---|---|
| Operating state | Motor current is trimodal (about 0.04 A off, 4 A unloaded, 7 A loaded; 55 % of the time off). Windows are `stopped` (explained, never an alarm) when the machine was off ≥ 90 % of the window; stopped and extreme-duty (> 0.9) windows are excluded from fitting and calibration. (Discrete 4-state `stopped`/`unloaded`/`loaded`/`starting` regime tagging remains for M2-WP2) |
| Features | `off_fraction`, `loaded_tp3_rise` (pressure rise while loading; slower during a leak, observable when `unloaded_tp3_slope` is not) |
| Baseline | `duty_rule`: compressor load duty alone, the physics baseline |
| Protocol | Train / calibration / evaluation; thresholds are quantiles of reference-normal scores (label-free) or an absolute grid (duty rule); **non-chronological nested leave-one-event-out**: the policy (variant, threshold, persistence) is chosen on the other reports (including later ones) and measured on the held-out one. Operating-day false alarms/day (excluding stopped days) are reported alongside all-valid-day false alarms/day |
| Default rule | Pre-declared in config: the best rule baseline is the bar. A learned detector becomes the default only if it matches the rule's leak detection at no worse false alarms, or detects simulated sensor faults on clean control intervals at ≥ 90 % with positive incremental lift and ≤ 0.1 false alarms/day without losing leak detection |
| Simulated faults | Stuck, offset, flatline, drift and noise faults injected into 60 real normal operating intervals (3 h each), paired with unmodified control intervals. Reports gross rate, control alarm rate, sensitivity on clean control intervals, and incremental lift separately. Always labelled simulated |
| Calibration | `ExceedanceCalibrator`: share of reference-normal windows scoring at least as high (smoothed conformal p-value). Not a failure probability |
| Alarm tracking | `AlarmTracker` is the one implementation of "k consecutive anomalous windows", used by evaluation and replay. Replay scans a 24 h look-back, so an incident's start and raised-at do not depend on the requested range |
| Evidence | One item per physical signal group (operating state, pressure level, pressure dynamics, oil temperature, protection switches) with a stated meaning; features that never varied in training are reported as *novel*, not with an absurd z-score; unobservable features are never reported |
| API and UI | `stopped` status, exceedance, grouped evidence, `truncated` / `ongoing` episodes, LOEO (with operating-day FA/day), separated fault lift metrics, and reference holdout/exceedance calibration checks in `model_card.json`, `/model`, OpenAPI, generated client, and `ModelCard` UI. `POST /score` enforces the 10-minute scoring window (60 s tolerance, rejecting spans > 11 min with 422; internal gaps > 60 s return `insufficient_data`). Error messages do not disclose server paths; `schema_version` 2 required |

## Results (real MetroPT-3, four operator air-leak reports; evaluation period 2020-04-01 to 09-02)

Reproduce with `uv run python -m factorymind.anomaly.cli train` (about 35 s after `prepare`).

| Detector | Leaks (LOEO) | FA/day (all valid) | FA/day (operating) | Median \|delay\| | Faults (gross) | Control alarms | Clean-ctrl sens. | Lift | Holdout exc. (at 1 %) |
|---|---|---|---|---|---|---|---|---|---|
| lps_rule (plant switch) | 1/4 | 0.000 | 0.000 | 240 min | – | – | – | – | 0.7 % |
| **duty_rule (default)** | **4/4** | **0.032** | **0.041** | **50 min** | 6/60 (10.0 %) | 6/60 (10.0 %) | 0/54 (0.0 %) | +0.0 pp | 27.5 % |
| robust_z | 4/4 | 0.860 | 1.095 | 70 min | – | – | – | – | 22.8 % |
| isolation_forest | 3/4 | 0.043 | 0.054 | 80 min | 14/60 (23.3 %) | 11/60 (18.3 %) | 3/49 (6.1 %) | +5.0 pp | 27.8 % |
| pca_spe | 4/4 | 0.057 | 0.072 | 45 min | 44/60 (73.3 %) | 15/60 (25.0 %) | 29/45 (64.4 %) | +48.3 pp | 3.3 % |

- **Selection was stable.** The duty rule chose threshold 0.5 with persistence 6 in all four folds;
  PCA SPE chose quantile 0.99 with persistence 6 in all four. Isolation Forest was not stable: it
  missed F1 when the other three reports chose a stricter policy.
- **Per-event (held-out):** duty rule +90 / +50 / +50 / −20 min against F1–F4; PCA SPE +90 / +40 /
  +50 / −40 min. Negative means the alarm was raised before the reported start.
- **PCA SPE and faults:** stuck, offset and drift faults 12/12 each; flatline and noise 4/12 each.
  Separating control-interval alarms (`15/60`, `25.0 %`) shows `29/45` (`64.4 %`) sensitivity on
  clean control intervals and `+48.3 pp` incremental lift (`0.733 - 0.250`).
- **Duty rule and faults:** `6/60` gross detections come entirely from `6/60` control alarms
  (`0/54` clean-control sensitivity, `+0.0 pp` lift): it cannot see sensor faults, as expected.
- **Sensitivity to the extreme-duty exclusion:** without it the duty rule and PCA SPE are unchanged
  in leak detection, Isolation Forest drops from 3/4 to 2/4, and the default stays `duty_rule`.

## Acceptance against the review criteria

| # | Criterion | Result |
|---|---|---|
| 1 | Leave-one-event-out table for all five detectors, selection rule fixed before the run, rerun reproduces numbers | **Met with a deviation.** Reproducibility is tested. Quantile thresholds come from the earlier calibration period, but LOEO policy choice uses the other reports, including chronologically later ones, and shares normal evaluation time across folds; this is **not** strict chronological validation (identical choices across folds for `duty_rule` and `pca_spe` limit the effect) |
| 2 | A learned detector is default only if it matches the rule or detects injected faults ≥ 90 % at ≤ 0.1 false alarms/day | **Rule applied; no learned detector qualified.** Using clean-control sensitivity (`64.4 %` PCA SPE, `6.1 %` Isolation Forest; gross `73.3 %` and `23.3 %`), neither learned detector meets the 90 % bar, and `choose_default` rejects any detector without positive incremental lift. The default stays `duty_rule` |
| 3 | Held-out exceedance rate near 1 % after the documented exclusion | **Not met.** Fit on the first half of the calibration period and tested on the second half, the exceedance rate is 22.8–27.8 % for duty rule, robust z and Isolation Forest (PCA SPE 3.3 %). The "normal" reference is not stationary even within two weeks |
| 4 | Alarm incidents identical across chunkings | **Met.** 100 random chunkings of the tracker and 12 random replay ranges reproduce the single-pass episodes |
| 5 | At most one evidence item per physical signal; no unobservable feature shown | **Met** (tested) |

## What the evidence says

1. **A physics rule is the right default today.** Load duty over one hour detects all four reports,
   including F4, at 0.032 all-valid-day / 0.041 operating-day false alarms/day, chosen without
   peeking at the held-out report. The learned detectors did not beat it on leaks.
2. **PCA SPE has a different, real job.** It catches stuck, offset and drift sensor faults that the
   rule cannot, with `+48.3 pp` incremental lift (`64.4 %` sensitivity on clean controls), but at an
   operating point that alarms too often (`0.057` / `0.072` operating-day FA/day) to be the leak
   alarm. The better design is two alarm classes: a leak alarm (duty rule) and a *consistency/sensor
   warning* (PCA SPE) evaluated on its own terms.
3. **"False alarms" against leak labels include genuine abnormal states.** In April–June, PCA SPE
   flags windows whose largest deviation is the oil-level switch (never active in training), plus
   unusual load cycling and pressure-switch activity; 61 % of its flagged far-from-report windows
   have load duty above 0.5. These are real unusual conditions that are not air leaks, which is why
   the product needs hypotheses and evidence, not a single alarm.
4. **Exceedance is relative to a reference that drifts**, so it is reported as "share of reference
   windows", never as a probability, and both `reference_holdout_check` and `exceedance_check` are
   included in `model_card.json`, `/api/v1/anomaly/model`, and the `ModelCard` UI with an explicit
   calibration limitation notice.

## Known limitations

- Four coarse reports on one compressor and one failure mode; LOEO supports per-event reporting, not statistical generalization.
- LOEO policy selection is **non-chronological** (later reports inform earlier fold policy choices); the normal evaluation time used for false alarms is shared by selection and evaluation, and the deployed policy is selected on all four reports (in-sample).
- `duty_rule`'s absolute threshold grid `[0.3..0.7]` follows prior post-M1 review observation (E5).
- Fault magnitudes (0.5 bar offset, 1 bar drift, 0.3 bar noise, frozen readings) are fixed engineering choices, not field statistics; results are simulated evidence only.
- Control-interval alarms (`25 %` for PCA SPE) show that "normal operating" time in this dataset is not clean.
- Stopped windows are explained, not analysed: a very long stop is itself information (maintenance, fault) that is not yet surfaced.
- Replay incidents longer than the 24 h look-back are reported as `truncated`.

## Next

WP0 closes the detection-side prerequisites for M2-WP1: contracts with `asset_id`, `tenant_id`,
`actor_id`, authentication and roles, persistence and audit (ADR-012, ADR-010). The consistency
warning class and a lower-false-alarm operating point for PCA SPE belong to M2-WP2.
