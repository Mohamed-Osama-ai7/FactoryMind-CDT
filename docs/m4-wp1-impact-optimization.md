# M4-WP1 (`WP-E`) — Impact Estimation & Bounded Maintenance Optimization

M4-WP1 adds a physics-grounded impact estimator and constrained maintenance decision comparison engine (`repair_now` vs `schedule` vs `monitor`) integrated directly into the incident decision workflow.

---

## 1. Clean 4-Tier Provenance Stratification (`factorymind.decision.optimizer`)

Every impact quantity (`ImpactQuantity`) explicitly carries one of four provenance tiers:

- `measured`: Observed telemetry load duty (`Motor_current > 5.5 A` or stored peak feature when present), motor current (`A`), oil temperature (`degC`). Fallback defaults when omitted or absent from peak evidence are never labeled `measured`.
- `engineering_estimate`: Rated compressor power proxy (`kW` / `kWh`) derived from observed/simulated duty against `rated_apu_power_kw` (disclosing that MetroPT-3 lacks voltage/power-factor meters).
- `configurable_economic_assumption`: Caller-configurable tariffs (`electricity_eur_per_kwh`), depot planned maintenance (`planned_depot_maintenance_eur`), monitoring check cost, and unplanned disruption rates.
- `simulated_projection`: Counterfactual trajectories rolled out by `CompressorReservoirTwin`, continuing post-repair intervals in repaired nominal baseline rather than scaling pre-repair LPS excursions across long planning horizons.

---

## 2. Constrained Option Evaluation & Constraint-Dominance

- Evaluates all three candidates (`repair_now`, `schedule`, `monitor`) across hard operational constraints (`require_zero_lps_violation`, `max_allowable_duty`, and trajectory-peak `max_allowable_oil_temp_c`).
- Uses constraint-dominance across `(expected_total_cost_eur, projected_lps_risk_minutes, projected_unplanned_disruption_hours)`: infeasible options never mark feasible ones optimal, and utility ties resolve deterministically toward non-dominated lower-cost options.
- Links **WP-B RCA uncertainty**: `sensor_malfunction` / `telemetry_inconsistency` / `insufficient_evidence` defer invasive overhaul in favor of monitoring checks, while `abnormal_high_demand` simulates high downstream pneumatic load without internal leak escalation.

---

## 3. API & Incident Comparison UI

- `POST /api/v1/optimization/compare`: Stateless bounded maintenance trade-off comparison (`test_safety_boundary.py` allowlisted, `Permission.ASSET_READ`, `404` on non-APU assets, global safe `422` handling for non-finite `NaN`/`Infinity`).
- `GET /api/v1/incidents/{incident_id}/decision-comparison`: Incident-grounded impact comparison pulling stored finite evidence rows from SQLite (`Permission.INCIDENT_READ`, `cost=2`, tenant-isolated).
- **Frontend (`DecisionComparisonPanel.tsx`, `routes/incidents.$incidentId.tsx`):** Gated by `rcaPending` to avoid duplicate mount fetches, displaying 4 provenance tiers with `basis_note`, feasibility, Pareto status, excess kWh, and `(Recommended)` markers.
