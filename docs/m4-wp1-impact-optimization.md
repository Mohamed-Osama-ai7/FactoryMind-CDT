# M4-WP1 (`WP-E`) — Impact Estimation & Bounded Maintenance Optimization

M4-WP1 adds a physics-grounded impact estimator and constrained maintenance decision comparison engine (`repair_now` vs `schedule` vs `monitor`) integrated directly into the incident decision workflow.

---

## 1. Clean 4-Tier Provenance Stratification (`factorymind.decision.optimizer`)

Every impact quantity (`ImpactQuantity`) explicitly carries one of four provenance tiers:

- `measured`: Observed telemetry load duty (`Motor_current > 5.5 A` ratio), motor current (`A`), oil temperature (`C`).
- `engineering_estimate`: Rated compressor power proxy (`kW` / `kWh`) derived from observed/simulated duty against `rated_apu_power_kw` (disclosing that MetroPT-3 lacks voltage/power-factor meters).
- `configurable_economic_assumption`: Caller-configurable tariffs (`electricity_eur_per_kwh`), depot planned maintenance (`planned_depot_maintenance_eur`), monitoring check cost, and unplanned disruption rates.
- `simulated_projection`: Counterfactual trajectories rolled out by `CompressorReservoirTwin` over the configured planning horizon.

---

## 2. Constrained Option Evaluation & Pareto Dominance

- Evaluates `repair_now`, `schedule`, and `monitor` across hard operational constraints (`require_zero_lps_violation`, `max_allowable_duty`, `max_allowable_oil_temp_c`).
- Detects Pareto dominance across `(expected_total_cost_eur, projected_lps_risk_minutes, projected_unplanned_disruption_hours)`.
- Links **WP-B RCA uncertainty** into maintenance comparison: when the dominant RCA hypothesis is `sensor_malfunction`, `telemetry_inconsistency`, `abnormal_high_demand`, or `insufficient_evidence`, warns explicitly against invasive compressor teardown prior to verification and favors `monitor`.

---

## 3. API & Incident Comparison UI

- `POST /api/v1/optimization/compare`: Stateless bounded maintenance trade-off comparison (`test_safety_boundary.py` allowlisted, `Permission.ASSET_READ`, `404` on non-APU assets).
- `GET /api/v1/incidents/{incident_id}/decision-comparison`: Incident-grounded impact comparison pulling measured evidence rows from SQLite (`Permission.INCIDENT_READ`, tenant-isolated).
- **Frontend (`DecisionComparisonPanel.tsx`, `routes/incidents.$incidentId.tsx`):** Integrated directly above `DecisionPanel` with live configurable EUR/leak/horizon assumptions and explicit feasibility/Pareto badges.

