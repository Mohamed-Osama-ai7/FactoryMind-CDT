"""Impact estimation & bounded maintenance optimization (repair_now vs schedule vs monitor).

Strictly separates every quantity into four provenance tiers:
- ``measured``: observed sensor/feature measurements (duty, motor current, oil temperature)
- ``engineering_estimate``: rated electrical kW power proxy (~0.1 Hz current without voltage sensor)
- ``configurable_economic_assumption``: user-supplied tariffs and outage/depot cost rates
- ``simulated_projection``: counterfactual rollouts from the calibrated Physics Digital Twin

All outputs are strictly advisory comparisons for human engineering review.
"""

from __future__ import annotations

from factorymind.contracts.digital_twin import TwinSimulationRequest
from factorymind.contracts.optimization import (
    ImpactQuantity,
    MaintenanceDecisionComparisonResponse,
    MaintenanceDecisionRequest,
    MaintenanceOptionEvaluation,
    MaintenanceOptionKind,
)
from factorymind.twin.physics import CompressorReservoirTwin

_OPTION_TITLES: dict[MaintenanceOptionKind, str] = {
    "repair_now": "Immediate Depot Hold & Repair",
    "schedule": "Schedule Next Planned Turnaround Maintenance",
    "monitor": "Continue Service Under Enhanced Monitoring",
}


def _expected_kw(load_duty: float, *, rated_kw: float, unloaded_frac: float) -> float:
    """Engineering estimate of mean compressor power draw across loaded vs unloaded fraction."""
    idle_frac = max(0.0, 1.0 - load_duty)
    return rated_kw * (load_duty + idle_frac * unloaded_frac)


def _pareto_dominates(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> bool:
    """True if option ``a`` is <= ``b`` on all objectives (cost, lps, downtime) and < on >=1."""
    return all(x <= y + 1e-9 for x, y in zip(a, b, strict=True)) and any(
        x < y - 1e-9 for x, y in zip(a, b, strict=True)
    )


def compare_maintenance_options(
    req: MaintenanceDecisionRequest,
    *,
    twin: CompressorReservoirTwin | None = None,
    incident_id: str | None = None,
) -> MaintenanceDecisionComparisonResponse:
    """Compare ``repair_now``, ``schedule``, and ``monitor`` under physical & cost bounds."""
    active_twin = twin or CompressorReservoirTwin()
    a = req.assumptions
    c = req.constraints
    w = req.weights

    w_sum = w.cost_weight + w.reliability_weight + w.downtime_weight
    w_cost = w.cost_weight / w_sum
    w_rel = w.reliability_weight / w_sum
    w_down = w.downtime_weight / w_sum

    baseline_kw = _expected_kw(
        a.nominal_baseline_duty,
        rated_kw=a.rated_apu_power_kw,
        unloaded_frac=a.unloaded_power_fraction,
    )
    observed_kw = _expected_kw(
        req.measured_load_duty,
        rated_kw=a.rated_apu_power_kw,
        unloaded_frac=a.unloaded_power_fraction,
    )
    baseline_kwh_horizon = baseline_kw * a.planning_horizon_hours

    # Bound twin simulation to 240 min window, scaling horizon energy/excursion honestly
    sim_horizon_min = min(240.0, a.planning_horizon_hours * 60.0)
    horizon_scale = (a.planning_horizon_hours * 60.0) / sim_horizon_min
    is_non_leak_uncertainty = req.dominant_hypothesis in (
        "sensor_malfunction",
        "telemetry_inconsistency",
        "insufficient_evidence",
        "abnormal_high_demand",
    )

    # Simulate immediate depot hold (repair at 2 min) vs next planned slot (60% horizon) vs monitor
    sim_repair_now = active_twin.simulate(
        TwinSimulationRequest(
            scenario="maintenance_repair",
            horizon_minutes=sim_horizon_min,
            initial_oil_temp_c=max(20.0, min(115.0, req.measured_oil_temp_c)),
            leak_rate_bar_per_min=a.leak_rate_bar_per_min,
            repair_at_minute=2.0,
        )
    )
    sim_schedule = active_twin.simulate(
        TwinSimulationRequest(
            scenario="maintenance_repair",
            horizon_minutes=sim_horizon_min,
            initial_oil_temp_c=max(20.0, min(115.0, req.measured_oil_temp_c)),
            leak_rate_bar_per_min=a.leak_rate_bar_per_min,
            repair_at_minute=max(10.0, round(0.60 * sim_horizon_min, 1)),
        )
    )
    # Under sensor uncertainty without physical confirmation, monitor avoids progressive leak bloat
    monitor_scenario = "baseline" if is_non_leak_uncertainty else "continued_operation"
    sim_monitor = active_twin.simulate(
        TwinSimulationRequest(
            scenario=monitor_scenario,
            horizon_minutes=sim_horizon_min,
            initial_oil_temp_c=max(20.0, min(115.0, req.measured_oil_temp_c)),
            leak_rate_bar_per_min=0.0 if is_non_leak_uncertainty else a.leak_rate_bar_per_min,
        )
    )

    raw_specs = [
        (
            "repair_now",
            sim_repair_now.summary,
            a.planned_depot_maintenance_eur * 1.25,  # immediate expedite surcharge
            0.25,  # bounded immediate turnout check window (hr)
            (
                "Intervenes early via twin-modeled repair to halt pressure loss; adds expedite "
                "pull-in cost and brief scheduled turnaround window."
            ),
        ),
        (
            "schedule",
            sim_schedule.summary,
            a.planned_depot_maintenance_eur,
            0.10,  # piggybacked onto planned depot slot
            (
                "Schedules inspection/repair into the next planned depot slot; saves expedite "
                "overhead when intermediate LPS margin holds."
            ),
        ),
        (
            "monitor",
            sim_monitor.summary,
            a.monitoring_check_eur,
            0.0,
            (
                "Avoids premature compressor disassembly while tracking telemetry trends; carries "
                "unplanned disruption risk if an active leak degrades further."
            ),
        ),
    ]

    candidate_metrics: list[
        tuple[
            MaintenanceOptionKind,
            list[str],
            float,
            float,
            float,
            float,
            float,
            str,
        ]
    ] = []

    for opt_name, summary, direct_cost, planned_down_hr, rationale in raw_specs:
        opt_kind: MaintenanceOptionKind = opt_name  # type: ignore[assignment]
        violations: list[str] = []
        lps_mins = round(summary.lps_excursion_minutes * horizon_scale, 2)
        if c.require_zero_lps_violation and lps_mins > 0.0:
            violations.append(
                f"LPS excursion projected ({lps_mins:.1f} min < 7.0 bar safety switch)."
            )
        if summary.peak_load_duty > c.max_allowable_duty and opt_kind == "monitor":
            violations.append(
                f"Peak rolling duty ({summary.peak_load_duty:.2f}) exceeds "
                f"max_allowable_duty ({c.max_allowable_duty:.2f})."
            )
        if summary.final_oil_temp_c > c.max_allowable_oil_temp_c:
            violations.append(
                f"Projected oil temp ({summary.final_oil_temp_c:.1f} C) exceeds limit "
                f"({c.max_allowable_oil_temp_c:.1f} C)."
            )

        mean_kw = _expected_kw(
            summary.mean_load_duty,
            rated_kw=a.rated_apu_power_kw,
            unloaded_frac=a.unloaded_power_fraction,
        )
        proj_kwh = round(mean_kw * a.planning_horizon_hours, 2)
        excess_kwh = round(proj_kwh - baseline_kwh_horizon, 2)

        # Unplanned disruption exposure proportional to severe LPS / unplanned operational loss
        lps_hours = lps_mins / 60.0
        unplanned_hr = round(lps_hours * (1.2 if opt_kind == "monitor" else 0.5), 3)
        if opt_kind == "monitor" and not is_non_leak_uncertainty and a.leak_rate_bar_per_min >= 0.4:
            unplanned_hr = round(max(unplanned_hr, 0.35 * (a.planning_horizon_hours / 8.0)), 3)

        energy_cost = proj_kwh * a.electricity_eur_per_kwh
        disruption_cost = (
            unplanned_hr + planned_down_hr * 0.25
        ) * a.unplanned_service_disruption_eur_per_hr
        total_cost = round(direct_cost + energy_cost + disruption_cost, 2)

        candidate_metrics.append(
            (
                opt_kind,
                violations,
                proj_kwh,
                excess_kwh,
                lps_mins,
                round(unplanned_hr + planned_down_hr, 3),
                total_cost,
                rationale,
            )
        )

    max_cost = max(m[6] for m in candidate_metrics) or 1.0
    max_lps = max(m[4] for m in candidate_metrics) or 1.0
    max_down = max(m[5] for m in candidate_metrics) or 1.0

    # Determine Pareto dominance across all options on (cost, lps_mins, downtime_hr)
    vectors = {m[0]: (m[6], m[4], m[5]) for m in candidate_metrics}
    options_out: list[MaintenanceOptionEvaluation] = []
    for (
        opt_kind,
        violations,
        proj_kwh,
        excess_kwh,
        lps_mins,
        total_down_hr,
        total_cost,
        rationale,
    ) in candidate_metrics:
        feasible = len(violations) == 0
        dominated = any(
            other_k != opt_kind and _pareto_dominates(other_vec, vectors[opt_kind])
            for other_k, other_vec in vectors.items()
        )
        penalty = (
            w_cost * (total_cost / max_cost)
            + w_rel * (lps_mins / max_lps if lps_mins > 0 else 0.0)
            + w_down * (total_down_hr / max_down)
        )
        utility = round(max(0.0, min(1.0, (1.0 - penalty) * (1.0 if feasible else 0.25))), 4)
        options_out.append(
            MaintenanceOptionEvaluation(
                option=opt_kind,
                title=_OPTION_TITLES[opt_kind],
                feasible=feasible,
                constraint_violations=violations,
                pareto_dominated=dominated,
                projected_energy_kwh=proj_kwh,
                projected_excess_energy_kwh=excess_kwh,
                projected_lps_risk_minutes=lps_mins,
                projected_unplanned_disruption_hours=total_down_hr,
                expected_total_cost_eur=total_cost,
                bounded_utility_score=utility,
                rationale=rationale,
            )
        )

    feasible_opts = [o for o in options_out if o.feasible]
    best = max(
        feasible_opts if feasible_opts else options_out,
        key=lambda o: o.bounded_utility_score,
    )

    uncertainty_warnings = [
        (
            "Electrical power (kW) and energy (kWh) are engineering estimates derived from "
            "duty fraction and rated APU power assumptions; MetroPT-3 lacks voltage meters."
        ),
        (
            "Economic totals reflect configurable EUR tariffs and downtime assumptions, not "
            "audited depot accounting invoices."
        ),
    ]
    if is_non_leak_uncertainty:
        uncertainty_warnings.append(
            f"Dominant hypothesis is '{req.dominant_hypothesis}': verify instrumentation/demand "
            "before executing invasive pneumatic overhaul."
        )
    if not feasible_opts:
        uncertainty_warnings.append(
            "No candidate satisfies every hard operational constraint; ranked by least penalty."
        )

    impact_quantities = [
        ImpactQuantity(
            key="measured_load_duty",
            label="Measured Loaded Duty Fraction",
            value=round(req.measured_load_duty, 4),
            unit="fraction",
            tier="measured",
            basis_note="Observed ratio of active compression samples (Motor_current > 5.5 A).",
        ),
        ImpactQuantity(
            key="measured_motor_current_a",
            label="Measured Motor Phase Current",
            value=round(req.measured_motor_current_a, 2),
            unit="A",
            tier="measured",
            basis_note="Direct MetroPT-3 clamp sensor measurement.",
        ),
        ImpactQuantity(
            key="estimated_power_kw",
            label="Estimated Mean APU Power Draw",
            value=round(observed_kw, 3),
            unit="kW",
            tier="engineering_estimate",
            basis_note="Linear duty proxy scaling rated_apu_power_kw (voltage/PF unmeasured).",
        ),
        ImpactQuantity(
            key="assumed_unplanned_rate",
            label="Configured Unplanned Disruption Rate",
            value=round(a.unplanned_service_disruption_eur_per_hr, 2),
            unit="EUR/hr",
            tier="configurable_economic_assumption",
            basis_note="Caller-supplied operational disruption cost assumption.",
        ),
        ImpactQuantity(
            key="projected_monitor_lps_min",
            label="Simulated Monitor LPS Excursion",
            value=next(o.projected_lps_risk_minutes for o in options_out if o.option == "monitor"),
            unit="min",
            tier="simulated_projection",
            basis_note="Counterfactual digital-twin rollout scaled over planning_horizon_hours.",
        ),
    ]

    note = (
        f"Advisory ranking favors '{best.option}' (utility {best.bounded_utility_score:.3f}, "
        f"expected total cost EUR {best.expected_total_cost_eur:.2f}, feasible={best.feasible}). "
        "Final authority rests with the human approver."
    )
    return MaintenanceDecisionComparisonResponse(
        asset_id=req.asset_id,
        incident_id=incident_id,
        impact_quantities=impact_quantities,
        options=options_out,
        recommended_option=best.option,
        recommendation_note=note,
        uncertainty_warnings=uncertainty_warnings,
    )
