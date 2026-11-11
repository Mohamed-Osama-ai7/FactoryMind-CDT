# M2-WP2 — Operating Regime Classification & Warning/Alarm Separation

M2-WP2 separates physical compressor operating regimes, telemetry-integrity checks, and multivariate sensor-consistency warnings (`pca_spe`) from primary pneumatic air-leak alarms (`duty_rule`) while keeping all WP0 baseline detector results intact and reproducibly measurable on the real MetroPT-3 dataset.

## 1. Discrete Operating-Regime Model

`backend/src/factorymind/anomaly/regimes.py` classifies each 10-minute telemetry window causally from left to right (`RegimeTracker`, `classify_window`, `classify_sequence`) into four physical regimes or explicit `insufficient_evidence`:

| Regime | Confidence | Operational Rule |
| --- | --- | --- |
| `stopped` | `confirmed` | Motor off (`Motor_current < 1.0 A`) for `>= 90 %` of the window (`off_fraction >= stopped_off_fraction`). |
| `loaded` | `confirmed` | Compressor under compression load (`Motor_current > 5.5 A`) for `>= 90 %` of the window (`>= 80 %` hysteresis hold when continuing from `loaded`). |
| `unloaded` | `confirmed` | Compressor running unloaded (`1.0 <= Motor_current <= 5.5 A`) for `>= 90 %` of the window (`>= 80 %` hysteresis hold when continuing from `unloaded`). |
| `starting` | `transitional` | Startup electrical inrush (`Motor_current > 8.0 A`, `>= 1` sample) or contiguous resumption from `stopped` / `> 600 s` gap with active cycling. |
| `insufficient_evidence` | `insufficient_evidence` | Window fails coverage/cadence (`incomplete_or_gapped`), has simultaneously zero-spread `TP3`/`TP2`/`Motor_current` (`stale_telemetry`), contradicts valve/motor physics across `> 10 %` of samples (`signal_inconsistency`), holds off a minor `< 90 %` boundary dip out of `stopped` (`hysteresis_hold`), or mixes duty phases without steady dominance (`mixed_duty_cycle`). |

Why the 46 `FEATURE_NAMES` vector was left unchanged: existing `ModelBundle` artifacts (`FeatureProfile`, `IsolationForestDetector`, `PcaSpe`) expect the exact 46 feature columns. New regime/integrity features (`unloaded_fraction`, `inrush_samples`, `valve_contradiction_fraction`, `motor_valve_mismatch_fraction`, `stale_telemetry`) live strictly in non-ML metadata columns of `window_features()` in `backend/src/factorymind/anomaly/features.py`.

### Telemetry sampling constraint (~0.1 Hz)

MetroPT-3 is downsampled to `~0.1 Hz` (`10 s` nominal period). Empirical inspection across `1,516,948` processed rows (`9,312` `off -> on` starts) shows only `44` samples above `8.0 A`, because sub-10 s motor startup inrush falls between samples in most starts. `starting` assessments therefore carry `confidence="transitional"` and record `missing_signals=("high_rate_electrical_current_gt_1hz", "plc_run_start_command")` whenever electrical inrush is not directly observed.

## 2. Canonical Event Taxonomy & Warning/Alarm Separation

`backend/src/factorymind/anomaly/events.py` defines four canonical operator-facing classes (`EventClass`):

- `air_leak_alarm`: Sustained elevated `duty_rule` load duty on valid, non-stopped, telemetry-consistent windows (`primary_leak_scorable`). Presented as a leak hypothesis with explicit downstream demand uncertainty; eligible for incident creation.
- `sensor_consistency_warning`: Multivariate `pca_spe` residual exceedance (scoring all valid operating, stale, and valve-inconsistent windows via `warning_scorable`) and/or deterministic signal-integrity failure (`stale_telemetry`, `signal_inconsistency`). On integrity failures, the primary air-leak detector abstains (`status="insufficient_data"`, `score=None`), so frozen or contradictory telemetry cannot raise a false air-leak alarm and incident creation is never offered.
- `informational`: Normal operation or explained `stopped` state with neither leak alarm nor consistency warning.
- `insufficient_evidence`: Gapped, incomplete, or missing telemetry where neither leak nor consistency scoring is possible.

`AnomalyService` (`backend/src/factorymind/anomaly/service.py`) loads the sibling `pca_spe` bundle (`load_warning_bundle`) using the same SHA-256 + `schema_version == 2` verification as the primary bundle, degrading cleanly if no sibling warning bundle exists while raising `ModelUnavailableError` on integrity tampering. Replay maintains backward compatibility (`replay()` returns the existing 3-tuple for incidents) while `replay_details()` exposes separate `episodes` (`air_leak_alarm`), `warnings` (`WarningEpisode` of class `sensor_consistency_warning`), and per-window `regime` + `event_class`.

## 3. Empirical MetroPT-3 Results (Reproduced Run `20261111-f069042373`)

Reproduced directly on real processed MetroPT-3 telemetry via `uv run --no-sync python -m factorymind.anomaly.cli train` (`D:\FactoryMind\artifacts\experiments\metropt3-anomaly\20261111-f069042373\results.json`).

### Preserved WP0 baseline detectors (real operational telemetry)

| Detector | LOEO leaks | FA / valid day | FA / operating day | Median \|delay\| | Simulated faults (`clean` / `lift`) | `p <= alpha` (far normal) |
| --- | --- | --- | --- | --- | --- | --- |
| `lps_rule` | `1/4` | `0.000` | `0.000` | `240 min` | `n/a` | `0.005` |
| `duty_rule` (default) | `4/4` | `0.034` | `0.044` | `50 min` | `6/60` (`0/54`, `+0.0 pp`) | `0.034` |
| `robust_z` | `4/4` | `0.860` | `1.095` | `70 min` | `n/a` | `0.210` |
| `isolation_forest` | `3/4` | `0.043` | `0.054` | `80 min` | `14/60` (`3/49`, `+5.0 pp`) | `0.049` |
| `pca_spe` (warning tier) | `4/4` | `0.057` | `0.072` | `55 min` | `44/60` (`29/45`, `+48.3 pp`) | `0.071` |

### WP2 Evaluation Regime Distribution

Across `17,744` valid evaluation windows (`2020-04-01` to `2020-09-02`, plus `355` invalid/gapped windows):

| Regime | Windows | Share | Notes |
| --- | --- | --- | --- |
| `loaded` | `186` | `1.05 %` | `duty_rule` exceedances: `186/186`, `2` non-report FA episodes; `pca_spe`: `186/186`, `3` warning episodes |
| `unloaded` | `66` | `0.37 %` | `duty_rule` exceedances: `0/66`, `0` FA episodes; `pca_spe`: `65/66`, `2` warning episodes |
| `starting` | `2,865` | `16.15 %` | `duty_rule` exceedances: `11/2865` (`0.38 %`), `0` FA episodes; `pca_spe`: `28/2865` (`0.98 %`), `0` warning episodes |
| `stopped` | `3,297` | `18.58 %` | Explained motor-off state: suppressed by both primary and warning detectors |
| `insufficient_evidence` | `11,330` | `63.85 %` | Breakdown: `mixed_duty_cycle=10,280`, `stale_telemetry=1,010`, `signal_inconsistency=40` |

### WP2 Event Separation & Telemetry-Integrity Gating

- **In-sample WP2 evaluation (`duty_rule` with integrity gating):** `14` air-leak episodes (`3` non-report false alarms, `0.024` FA/valid day, `0.032` FA/operating day) vs WP0 stopped-only (`15` episodes, `4` non-report false alarms, `0.032` FA/valid day, `0.041` FA/operating day). Integrity gating eliminated `1` false air-leak alarm episode caused by frozen telemetry while keeping `4/4` leak detection unchanged (`F1 +90 min`, `F2 +50 min`, `F3 +50 min`, `F4 -20 min`).
- **Separated warnings:** `34` `sensor_consistency_warning` episodes (`0.276` warnings/valid day, `0.351` warnings/operating day): `25` standalone warnings + `9` overlapping leak windows; kinds: `signal_inconsistency=13`, `multivariate_sensor_drift=12`, `stale_telemetry=6`, `mixed_consistency_warning=3`. None are mislabelled or counted as air-leak detections.
- **Transition attribution:** `0` non-report `air_leak_alarm` false alarms overlapped `starting` windows; `2` overlapped `mixed_duty_cycle` windows (`loaded` dominant: `2`, `insufficient_evidence` dominant: `1`).

### LOEO Regime-Gating Ablations (`duty_rule`)

| Ablation policy | Suppressed states | LOEO leaks | Mean FA / valid day | Mean FA / operating day | Median \|delay\| |
| --- | --- | --- | --- | --- | --- |
| `wp0_stopped_only` | `stopped` | `4/4` | `0.034` | `0.044` | `50 min` |
| `wp2_integrity_gated` | `stopped`, `stale_telemetry`, `signal_inconsistency` | `4/4` | `0.026` | `0.035` | `50 min` |
| `starting_suppressed_ablation` | `stopped`, `stale_telemetry`, `signal_inconsistency`, `starting` | `4/4` | `0.026` | `0.045` | `50 min` |

Result: gating `stale_telemetry` and `signal_inconsistency` strictly reduces both valid-day (`0.034 -> 0.026`) and operating-day (`0.044 -> 0.035`) false-alarm rates with zero loss in leak detection or delay. Suppressing `starting` windows gives zero valid-day reduction (`0.026` unchanged) and worsens operating-day false alarms (`0.035 -> 0.045` by shrinking operating-day denominator), so `starting` suppression is kept disabled in default air-leak detection.

### Simulated Single-Sensor Fault Separation (Strictly Non-Operational)

Reported independently (`simulated=true`, `real_operational_data=false`) to verify warning-tier instrumentation coverage:

- `duty_rule`: `6/60` gross (`6/60` control, `0/54 = 0.0 %` clean-control sensitivity, `+0.0 pp` incremental lift)
- `pca_spe`: `44/60` gross (`15/60` control, `29/45 = 64.4 %` clean-control sensitivity, `+48.3 pp` incremental lift; `stuck=12/12`, `offset=12/12`, `flatline=4/12`, `drift=12/12`, `noise=4/12`)

## 4. API, UI, Safety Boundary, and Limitations

- **API (`/api/v1/anomaly/model`, `/score`, `/replay`):** Exposes structured `RegimeInfo`, `event_class`, `event_meaning`, `warning_active`/`warning_score`/`warning_exceedance`/`warning_detector`/`warning_reasons`, separate `warnings` (`WarningEpisode`) alongside `episodes` (`AlarmEpisode` with `hypothesis`, `uncertainty_note`, `dominant_regime`, `regime_breakdown`), and typed `ModelInfo.wp2`. All routes retain WP1 OIDC/RBAC + tenant/asset/data-source guards (`require_data_source`), and the mutating route allowlist remains read-only toward plant systems (`backend/tests/test_safety_boundary.py`).
- **UI (`ModelCard`, `ReplayPanel`, `ScoreChart`):** Renders regime pill counts and insufficient-evidence causes, distinguishes `air_leak_alarm` episodes (red band + `CreateIncidentButton`) from `sensor_consistency_warning` episodes (cyan/amber visual separation, no leak label, no incident button), and surfaces WP2 per-regime, ablation, and simulated fault summaries.
- **Explicit remaining limitations:** MetroPT-3 remains a single compressor with four coarse leak reports and no PLC mode tag or high-rate (`> 1 Hz`) startup current tag; LOEO policy selection remains non-chronological across the four events; reference drift remains documented via `reference_holdout_check`; M2 remains open for subsequent RCA/evidence, simulation, and decision-support work packages; and no production-readiness claim is made.

