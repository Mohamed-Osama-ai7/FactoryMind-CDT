# M4-WP0 (`WP-D`) — Physics-Informed Compressor & Reservoir Digital Twin

M4-WP0 introduces a calibrated physics-informed digital twin (`CompressorReservoirTwin`) modeling the MetroPT-3 APU screw compressor, pneumatic panel (`TP3`), coupled reservoirs, oil thermal lag, and governor hysteresis (`MPG` / cut-out), paired with controlled counterfactual simulation and strict parameter-identifiability honesty.

---

## 1. Governing Physics & Pre-Evaluation Calibration (`factorymind.twin.physics`)

- **Strict Pre-Evaluation Fit (`fit_from_telemetry`, `< 2020-04-01`, `165,547` active samples):**
  - Normalized mass balance: `dP_TP3/dt = k_charge * u_load(t) * eff(P) - (k_drain * demand_factor + k_leak(t))`
  - Fitted rate constants: `k_charge = 0.948 bar/min`, `k_drain = 0.084 bar/min`, `cut_in = 8.09 bar`, `cut_out = 10.078 bar`, `alpha_heat = 2.30353 degC/min`, `alpha_cool = 0.01734 1/min`.
- **Explicitly Unidentifiable Parameters:**
  - `reservoir_volume_m3` (expressed in normalized capacitance rates `bar/min`),
  - `downstream_demand_vs_internal_leak_split` (ambiguous under stationary steady loss without a flowmeter),
  - `ambient_air_temperature_c` (`T_ambient_ref = 50.0 C` proxy),
  - `sub_10s_inrush_transient` (~0.1 Hz cadence limit).

---

## 2. Out-of-Sample Held-Out Trajectory Validation (`>= 2020-04-01`)

Verified directly against live MetroPT-3 evaluation segments outside operator failure windows:

- **Held-out normal 1-hour trajectories:** `168` contiguous segments
- **10-min open-loop rollout MAE:** `0.2526 bar`
- **60-min open-loop rollout MAE / RMSE:** `0.5218 bar` / `0.5853 bar`
- **Physical constraint violations:** `0`
- **Step-halving numerical diff (`10s` vs `5s`):** `0.395 bar` (governor discrete hysteresis step jitter)
- **Failure-vs-normal residual magnification (`F1..F3`):** `3.78x`

---

## 3. API & `/twin` Counterfactual Explorer

- **API (`factorymind.api.twin`, `Permission.ASSET_READ`):**
  - `GET /api/v1/twin/model`: calibrated parameters, governing equations, unidentifiable parameters, held-out validation.
  - `POST /api/v1/twin/simulate`: stateless counterfactual rollout for `baseline`, `leak`, `continued_operation`, and `maintenance_repair` (`provenance_origin="simulated_projection"`, allowlisted in `test_safety_boundary.py`).
- **Frontend (`/twin`, `DigitalTwinLab.tsx`):** Interactive scenario simulator, summary KPI cards, trajectory table, held-out validation metrics, and explicit unidentifiable-parameter disclosure.

