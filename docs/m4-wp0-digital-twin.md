# M4-WP0 (`WP-D` & `WP-P0C`) — Physics-Informed Compressor & Reservoir Digital Twin

M4-WP0 introduces a calibrated physics-informed digital twin (`CompressorReservoirTwin`) modeling the MetroPT-3 APU screw compressor, pneumatic panel (`TP3`), coupled reservoirs, oil thermal lag, and governor hysteresis (`MPG` / cut-out), paired with controlled counterfactual simulation and strict parameter-identifiability honesty.

`WP-P0C` (`apu-mass-balance-v2`) removes all silent hardcoded calibration/validation fallbacks (`TwinCalibrationError` / `HTTP 503` fail-closed when calibration telemetry is missing or invalid), separates teacher-forced vs. closed-loop rollout evaluation per horizon against Persistence and strictly pre-evaluation Constant-Training-Mean baselines, reports duty-cycle/load-cycle accuracy and pre-evaluation parameter sensitivity, and explicitly marks unsupported forecast horizons.

---

## 1. Governing Physics & Pre-Evaluation Calibration (`factorymind.twin.physics`)

- **Strict Pre-Evaluation Fit (`fit_from_telemetry`, `< 2020-04-01`, `165,547` active finite samples):**
  - Normalized mass balance: `dP_TP3/dt = k_charge * u_load(t) * eff(P) - (k_drain * demand_factor + k_leak(t))`
  - Oil thermal ODE: `dT_oil/dt = alpha_heat * u_load(t) - alpha_cool * (T_oil(t) - T_ambient_ref)`
  - Fitted rate & boundary parameters (no silent hardcoded fallback):
    - `k_charge = 0.948 bar/min`, `k_drain = 0.084 bar/min`
    - `cut_in = 8.09 bar`, `cut_out = 10.078 bar`, `charge_overshoot_bar = 0.158 bar` (pre-evaluation 99th-percentile loaded overshoot), `coupling_bias_bar = 0.0 bar`
    - `alpha_heat = 2.1217 degC/min`, `alpha_cool = 0.05398 1/min`, `T_ambient_ref = 63.04 C` (thermal OLS intercept proxy)
  - Strictly pre-evaluation constant baselines: `TP3 = 9.003 bar`, `Oil = 59.34 C`, `Duty = 0.1185`.
  - Pre-evaluation sensitivity checks (`77` pre-evaluation `10m` segments, baseline `0.3463 bar` TP3 MAE / `0.0569` duty MAE): `±15%` perturbations on `k_charge` and `k_drain` and `±0.15 bar` perturbations on `cut_in`/`cut_out` stay within `-0.0107..+0.0279 bar` TP3 delta and `-0.0052..+0.0182` duty delta.
- **Explicitly Unidentifiable Parameters:**
  - `reservoir_volume_m3` (expressed in normalized capacitance rates `bar/min`),
  - `downstream_demand_vs_internal_leak_split` (ambiguous under stationary steady loss without a flowmeter),
  - `ambient_air_temperature_c` (effective reference proxy `T_ambient_ref = 63.04 C` fitted from thermal OLS),
  - `sub_10s_inrush_transient` (~0.1 Hz cadence limit).

---

## 2. Out-of-Sample Held-Out Trajectory Validation (`>= 2020-04-01`)

Verified directly against contiguous held-out normal MetroPT-3 segments outside `±24h` around reported `F1..F4` failure windows, evaluating **Teacher-Forced (`TF`)** (`u_load` observed) separately from **Closed-Loop (`CL`)** (`u_load` generated autonomously via hysteresis governor):

| Horizon | Segs | TF TP3 MAE (Med) | CL TP3 MAE (Med) | Persistence MAE | Train-Mean MAE | Duty CL MAE vs Mean | Cycles/Seg Obs vs Sim | CL Oil MAE vs Mean | Supported (`TP3`) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `10m` | `227` | `0.3325` (`0.2345`) | `0.3280` (`0.2597`) | `1.5354` | `1.0014` | `0.0599` vs `0.2482` | `0.304` vs `0.194` | `2.721 C` vs `7.124 C` | `TF` & `CL` supported |
| `30m` | `191` | `0.4210` (`0.3388`) | `0.4918` (`0.5005`) | `1.8343` | `0.6659` | `0.0532` vs `0.0918` | `1.330` vs `0.801` | `3.984 C` vs `6.250 C` | `TF` & `CL` supported |
| `60m` | `168` | `0.5030` (`0.4242`) | `0.6008` (`0.6266`) | `1.8765` | `0.5821` | `0.0496` vs `0.0568` (`unsup.`) | `2.946` vs `2.024` | `4.753 C` vs `5.898 C` | `TF` only (`CL` unsupported) |
| `120m` | `146` | `0.6412` (`0.4857`) | `0.6584` (`0.6811`) | `1.9219` | `0.5418` | `0.0433` vs `0.0433` (`unsup.`) | `6.500` vs `4.226` | `5.210 C` vs `5.867 C` | **Unsupported** (`TF` & `CL`) |
| `240m` | `136` | `0.8473` (`0.4812`) | `0.6688` (`0.6721`) | `1.9531` | `0.5280` | `0.0443` vs `0.0388` (`unsup.`) | `13.618` vs `8.603` | `5.248 C` vs `5.843 C` | **Unsupported** (`TF` & `CL`) |

- **Empirical Scope Limits & Findings:**
  - **Closed-loop `TP3` and duty-cycle support stops at `30 min` (`max_supported_closed_loop_horizon_minutes = 30.0`, `max_supported_duty_horizon_minutes = 30.0`); `teacher_forced` beats Train-Mean up to `60 min` (`max_supported_teacher_forced_horizon_minutes = 60.0`).**
  - At `>= 60m` closed-loop and `>= 120m` teacher-forced, phase drift and load-cycle underprediction (`2.946` observed vs `2.024` simulated cycles/h at `60m`) cause trajectory error to underperform the stationary constant-training-mean baseline (`9.003 bar`); counterfactual simulations beyond `30 min` carry an explicit `horizon_supported=False` warning.
  - **Constraint violations (`1,711` total in TF only):** Teacher-forced integration during long train-off parking windows (`>= 120m`) subtracts unloaded drain `k_drain` when observed `u_load = 0`, hitting the `[0, 15] bar` pressure clamp (`237` clamps at `120m`, `1,474` at `240m`). Closed-loop rollouts have `0` violations across `10m..240m`, and both modes have `0` violations at `10m`, `30m`, and `60m`.
  - **Step-halving numerical diff (`10s` vs `5s`):** `0.395 bar` (governor discrete hysteresis step jitter).
  - **Failure-vs-normal residual magnification (`F1..F4` gap-free windows vs `60m` TF):** `2.914x`.

---

## 3. API, Decision Integration & `/twin` Counterfactual Explorer

- **API (`factorymind.api.twin`, `Permission.ASSET_READ`):**
  - `GET /api/v1/twin/model`: calibrated parameters, governing equations, unidentifiable parameters, per-horizon baseline comparison & pre-eval sensitivity validation (`503 Service Unavailable` when pre-eval calibration telemetry is unavailable or invalid — no silent fallback).
  - `POST /api/v1/twin/simulate`: stateless counterfactual rollout for `baseline`, `leak`, `continued_operation`, and `maintenance_repair` (`provenance_origin="simulated_projection"`, allowlisted in `test_safety_boundary.py`), returning explicit `horizon_supported`, `horizon_support_status`, and `horizon_warning`.
- **Decision Optimizer & Copilot (`optimizer.py`, `api/copilot.py`):** Removed silent `CompressorReservoirTwin()` construction fallback; when counterfactual horizons exceed closed-loop support (`>30m`), optimizer outputs carry explicit horizon warnings (`uncertainty_warnings`), and Copilot degrades Twin/Optimizer tool calls to `status="unavailable"` cleanly without blocking RAG/KG queries or silently inventing physics outputs.
- **Frontend (`/twin`, `DigitalTwinLab.tsx`):** Interactive scenario simulator, explicit fail-closed unavailable alert when backend calibration is missing (`503`), horizon-support banner (`horizon_warning`), summary KPI cards, per-horizon Teacher-Forced vs Closed-Loop baseline comparison table, pre-eval sensitivity table, trajectory table, and unidentifiable-parameter disclosure.
