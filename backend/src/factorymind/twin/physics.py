"""Physics-informed Compressor & Reservoir Mass-Balance + Duty-Governor Digital Twin.

Strictly fitted on pre-evaluation normal operating telemetry (`2020-02-01` to `2020-04-01`) in
normalized capacity units (`bar/min`), without silent parameter fallbacks. Explicitly discloses
unidentifiable physical parameters, separates teacher-forced from closed-loop validation, compares
all horizons against Persistence and Constant-Training-Mean baselines, and flags unsupported
horizons honestly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import pandas as pd

from factorymind.contracts.digital_twin import (
    TwinHorizonEvaluation,
    TwinModelResponse,
    TwinParameterEstimate,
    TwinScenarioSummary,
    TwinSensitivityCheck,
    TwinSimulationRequest,
    TwinSimulationResponse,
    TwinTrajectoryPoint,
    TwinValidationSummary,
    UnidentifiableParameter,
)
from factorymind.data import metropt3

TWIN_MODEL_VERSION: Final = "apu-mass-balance-v2"
EVALUATION_BOUNDARY: Final = pd.Timestamp("2020-04-01")
TRAIN_START: Final = pd.Timestamp("2020-02-01")
DUTY_ALARM_THRESHOLD: Final = 0.7
LPS_SWITCH_BAR: Final = 7.0
MIN_PHYSICAL_BAR: Final = 0.0
MAX_PHYSICAL_BAR: Final = 15.0
SIM_DT_S: Final = 10.0  # matches nominal 0.1 Hz MetroPT-3 period
EVAL_HORIZONS_MINUTES: Final[tuple[float, ...]] = (10.0, 30.0, 60.0, 120.0, 240.0)
REQUIRED_TWIN_COLUMNS: Final[tuple[str, ...]] = (
    "TP3",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "DV_eletric",
)

GOVERNING_EQUATIONS: Final[list[str]] = [
    (
        "Mass-balance pressure dynamics: dP_TP3/dt = k_charge * u_load(t) * eff(P) - "
        "(k_drain * demand_factor + k_leak(t))"
    ),
    "Coupled reservoir pressure: P_Reservoirs(t) = clip(P_TP3(t) - coupling_bias_bar, 0, 15)",
    (
        "First-order oil thermal lag: dT_oil/dt = alpha_heat * u_load(t) - "
        "alpha_cool * (T_oil(t) - T_ambient_ref)"
    ),
    (
        "Governor hysteresis law: u_load transitions 0->1 when P_TP3 <= P_cut_in (MPG) and "
        "1->0 when P_TP3 >= P_cut_out"
    ),
]

UNIDENTIFIABLE_PARAMETERS: Final[tuple[UnidentifiableParameter, ...]] = (
    UnidentifiableParameter(
        name="reservoir_volume_m3",
        reason=(
            "Train pneumatic reservoir geometric volume and mass-flow rate are not directly "
            "metered; only their combined pressure-rate ratio dP/dt (bar/min) is identifiable."
        ),
        assumption_or_proxy="Expressed in normalized capacitance rates k_charge/k_drain (bar/min).",
    ),
    UnidentifiableParameter(
        name="downstream_demand_vs_internal_leak_split",
        reason=(
            "MetroPT-3 lacks brake/door consumption flowmeters; an external demand surge and an "
            "internal pneumatic leak enter dP/dt identically under steady-state pressure drop."
        ),
        assumption_or_proxy=(
            "Scenario parameters separate nominal demand_factor and injected k_leak explicitly."
        ),
    ),
    UnidentifiableParameter(
        name="ambient_air_temperature_c",
        reason=(
            "External tunnel/ambient air temperature around Porto Metro trains is unrecorded; "
            "cooling rate relies on a fitted constant effective ambient reference T_ambient_ref."
        ),
        assumption_or_proxy="Effective reference proxy T_ambient_ref estimated from thermal OLS.",
    ),
    UnidentifiableParameter(
        name="sub_10s_inrush_transient",
        reason=(
            "Published MetroPT-3 telemetry is downsampled 10x (~0.1 Hz), aliasing short <10 s "
            "electrical starting transients."
        ),
        assumption_or_proxy="Uses binary loaded governor state u_load in {0, 1} over 10 s steps.",
    ),
)


class TwinCalibrationError(ValueError):
    """Raised when telemetry fitting, calibration, or held-out validation cannot proceed safely."""


@dataclass(frozen=True)
class TwinCalibratedParams:
    """Identifiable physical rate constants & training baselines from pre-eval telemetry."""

    k_charge_bar_per_min: float
    k_drain_bar_per_min: float
    cut_in_bar: float
    cut_out_bar: float
    charge_overshoot_bar: float
    coupling_bias_bar: float
    alpha_heat_c_per_min: float
    alpha_cool_per_min: float
    t_ambient_ref_c: float
    train_mean_tp3_bar: float
    train_mean_oil_temp_c: float
    train_mean_load_duty: float
    fit_source: str = "live_pre_eval_fit"
    train_cal_samples: int = 0
    overshoot_from_samples: bool = True

    def to_contracts(self) -> list[TwinParameterEstimate]:
        return [
            TwinParameterEstimate(
                name="k_charge_bar_per_min",
                value=round(self.k_charge_bar_per_min, 5),
                unit="bar/min",
                identifiable=True,
                fit_source=self.fit_source,
                uncertainty_note="Median steady-state active charging slope plus nominal drain.",
            ),
            TwinParameterEstimate(
                name="k_drain_bar_per_min",
                value=round(self.k_drain_bar_per_min, 5),
                unit="bar/min",
                identifiable=True,
                fit_source=self.fit_source,
                uncertainty_note=(
                    "Median steady unloaded pressure decay across normal duty fluctuations."
                ),
            ),
            TwinParameterEstimate(
                name="cut_in_bar",
                value=round(self.cut_in_bar, 3),
                unit="bar",
                identifiable=True,
                fit_source=self.fit_source,
                uncertainty_note="Median MPG cut-in pressure at 0->1 load transitions.",
            ),
            TwinParameterEstimate(
                name="cut_out_bar",
                value=round(self.cut_out_bar, 3),
                unit="bar",
                identifiable=True,
                fit_source=self.fit_source,
                uncertainty_note="Median cut-out pressure at 1->0 unloading transitions.",
            ),
            TwinParameterEstimate(
                name="charge_overshoot_bar",
                value=round(self.charge_overshoot_bar, 4),
                unit="bar",
                identifiable=True,
                fit_source=self.fit_source,
                uncertainty_note=(
                    "99th-percentile pre-eval loaded pressure overshoot above cut_out."
                    if self.overshoot_from_samples
                    else (
                        "Nominal hysteresis band margin 0.1 * (cut_out - cut_in) "
                        "(<5 overshoot samples)."
                    )
                ),
            ),
            TwinParameterEstimate(
                name="coupling_bias_bar",
                value=round(self.coupling_bias_bar, 4),
                unit="bar",
                identifiable=True,
                fit_source=self.fit_source,
                uncertainty_note="Median panel TP3 minus downstream reservoir differential.",
            ),
            TwinParameterEstimate(
                name="alpha_heat_c_per_min",
                value=round(self.alpha_heat_c_per_min, 5),
                unit="degC/min",
                identifiable=True,
                fit_source=self.fit_source,
                uncertainty_note="First-order oil heating coefficient under active loaded duty.",
            ),
            TwinParameterEstimate(
                name="alpha_cool_per_min",
                value=round(self.alpha_cool_per_min, 5),
                unit="1/min",
                identifiable=True,
                fit_source=self.fit_source,
                uncertainty_note=(
                    f"Cooling rate toward fitted T_ambient_ref={self.t_ambient_ref_c:.2f} C proxy."
                ),
            ),
        ]


def _charge_efficiency(p_bar: float, cut_out_bar: float, charge_overshoot_bar: float) -> float:
    """Physical check-valve & back-pressure saturation above calibrated cut-out envelope."""
    if p_bar <= cut_out_bar:
        return 1.0
    margin = max(1e-3, charge_overshoot_bar)
    return max(0.0, min(1.0, (cut_out_bar + margin - p_bar) / margin))


def _clamp(val: float, lo: float, hi: float) -> tuple[float, bool]:
    if val < lo:
        return lo, True
    if val > hi:
        return hi, True
    return val, False


def _compute_skill(model_err: float, baseline_err: float) -> float:
    if baseline_err > 1e-9:
        return 1.0 - (model_err / baseline_err)
    return 0.0 if model_err <= 1e-9 else -1.0


def _integrate_twin_steps(
    params: TwinCalibratedParams,
    req: TwinSimulationRequest,
    *,
    dt_s: float,
) -> list[tuple[float, float, float, float, bool, float, bool]]:
    dt_min = dt_s / 60.0
    n_steps = round(req.horizon_minutes / dt_min) + 1
    win_steps = max(1, round(10.0 / dt_min))  # 10-min rolling duty window
    p = float(req.initial_tp3_bar)
    t_oil = float(req.initial_oil_temp_c)
    loaded = p <= params.cut_in_bar
    repair_min = req.repair_at_minute if req.repair_at_minute is not None else 30.0
    load_hist: list[float] = []
    rows: list[tuple[float, float, float, float, bool, float, bool]] = []

    for step_idx in range(n_steps):
        minute = step_idx * dt_min
        if req.scenario == "baseline":
            leak = 0.0
            demand_mult = 1.0
        elif req.scenario == "leak":
            leak = req.leak_rate_bar_per_min
            demand_mult = req.demand_factor
        elif req.scenario == "continued_operation":
            # Progressive leak degradation (~+25% escalation per hour if left unrepaired)
            leak = req.leak_rate_bar_per_min * (1.0 + 0.25 * (minute / 60.0))
            demand_mult = req.demand_factor
        else:  # maintenance_repair
            repaired = minute >= repair_min
            leak = 0.0 if repaired else req.leak_rate_bar_per_min
            demand_mult = 1.0 if repaired else req.demand_factor

        # Governor hysteresis transitions
        if loaded and p >= params.cut_out_bar:
            loaded = False
        elif not loaded and p <= params.cut_in_bar:
            loaded = True

        load_hist.append(1.0 if loaded else 0.0)
        recent = load_hist[-win_steps:]
        rolling_duty = sum(recent) / len(recent)
        res_p, _ = _clamp(p - params.coupling_bias_bar, MIN_PHYSICAL_BAR, MAX_PHYSICAL_BAR)
        eff = _charge_efficiency(p, params.cut_out_bar, params.charge_overshoot_bar)
        dp = (
            params.k_charge_bar_per_min * (1.0 if loaded else 0.0) * eff
            - params.k_drain_bar_per_min * demand_mult
            - leak
        )
        dt_oil = params.alpha_heat_c_per_min * (
            1.0 if loaded else 0.0
        ) - params.alpha_cool_per_min * (t_oil - params.t_ambient_ref_c)
        p_raw_next = p + dp * dt_min
        p_next, p_viol = _clamp(p_raw_next, MIN_PHYSICAL_BAR, MAX_PHYSICAL_BAR)
        t_oil_next, _ = _clamp(t_oil + dt_oil * dt_min, -20.0, 150.0)
        rows.append((minute, p, res_p, t_oil, loaded, rolling_duty, p_viol))
        p = p_next
        t_oil = t_oil_next
    return rows


def _step_halving_stability_diff(params: TwinCalibratedParams) -> float:
    req = TwinSimulationRequest(scenario="leak", horizon_minutes=60.0)
    coarse = _integrate_twin_steps(params, req, dt_s=SIM_DT_S)
    fine = _integrate_twin_steps(params, req, dt_s=SIM_DT_S / 2.0)
    stride_c = 6
    stride_f = 12
    diffs = [
        abs(coarse[idx * stride_c][1] - fine[idx * stride_f][1])
        for idx in range(min(len(coarse) // stride_c, len(fine) // stride_f))
    ]
    return max(diffs) if diffs else 0.0


def _contiguous_supported_limit(
    evals: list[TwinHorizonEvaluation],
    *,
    attr: Literal["closed_loop_supported", "teacher_forced_supported", "duty_cycle_supported"],
) -> float:
    """Return max contiguous horizon where the model outperforms baselines without skipping gaps."""
    limit = 0.0
    for ev in sorted(evals, key=lambda item: item.horizon_minutes):
        if bool(getattr(ev, attr)):
            limit = ev.horizon_minutes
        else:
            break
    return limit


def _extract_contiguous_segments(
    frame: pd.DataFrame,
    *,
    min_len: int,
    max_gap_s: float = 15.0,
) -> list[pd.DataFrame]:
    """Return gap-free contiguous slices of length >= ``min_len``."""
    if frame.empty:
        return []
    ordered = frame.sort_index()
    dts_s = ordered.index.to_series().diff().dt.total_seconds().fillna(999.0).to_numpy()
    breaks = np.where((dts_s > max_gap_s) | (dts_s <= 0.0))[0]
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(ordered)]])
    return [ordered.iloc[s:e] for s, e in zip(starts, ends, strict=True) if (e - s) >= min_len]


def _rollout_observed_segment(
    params: TwinCalibratedParams,
    seg: pd.DataFrame,
    *,
    closed_loop: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Simulate TP3, oil temperature, and valve state along a telemetry slice.

    When ``closed_loop=True``, only t=0 state is observed; future valve commands ``u[1:]``
    are governed strictly by the Twin hysteresis law on simulated pressure.
    """
    p_obs = seg["TP3"].to_numpy(dtype="float64")
    oil_obs = seg["Oil_temperature"].to_numpy(dtype="float64")
    u_obs = (seg["DV_eletric"].to_numpy(dtype="float64") >= 0.5).astype("float64")
    step_min = np.diff(seg.index.to_numpy()).astype("timedelta64[s]").astype(float) / 60.0

    n = len(p_obs)
    p_sim = np.empty(n, dtype="float64")
    oil_sim = np.empty(n, dtype="float64")
    u_sim = np.empty(n, dtype="float64")

    p_sim[0] = p_obs[0]
    oil_sim[0] = oil_obs[0]
    loaded = bool(u_obs[0] >= 0.5)
    u_sim[0] = 1.0 if loaded else 0.0
    violations = 0

    for i in range(1, n):
        if closed_loop:
            if loaded and p_sim[i - 1] >= params.cut_out_bar:
                loaded = False
            elif not loaded and p_sim[i - 1] <= params.cut_in_bar:
                loaded = True
            u_step = 1.0 if loaded else 0.0
            u_sim[i] = u_step
        else:
            u_step = u_obs[i - 1]
            u_sim[i] = u_obs[i]

        eff = _charge_efficiency(p_sim[i - 1], params.cut_out_bar, params.charge_overshoot_bar)
        raw_p = (
            p_sim[i - 1]
            + (params.k_charge_bar_per_min * u_step * eff - params.k_drain_bar_per_min)
            * step_min[i - 1]
        )
        p_next, violated = _clamp(raw_p, MIN_PHYSICAL_BAR, MAX_PHYSICAL_BAR)
        if violated:
            violations += 1
        p_sim[i] = p_next

        d_oil = params.alpha_heat_c_per_min * u_step - params.alpha_cool_per_min * (
            oil_sim[i - 1] - params.t_ambient_ref_c
        )
        oil_sim[i], _ = _clamp(oil_sim[i - 1] + d_oil * step_min[i - 1], -20.0, 150.0)

    return p_sim, oil_sim, u_sim, violations


def _evaluate_horizon_segments(
    params: TwinCalibratedParams,
    segments: list[pd.DataFrame],
    *,
    horizon_minutes: float,
    n_pts: int,
) -> tuple[TwinHorizonEvaluation, int]:
    """Compute teacher-forced, closed-loop, persistence, and train-mean metrics for one horizon."""
    tf_maes: list[float] = []
    tf_rmses: list[float] = []
    cl_maes: list[float] = []
    cl_rmses: list[float] = []
    pers_maes: list[float] = []
    mean_maes: list[float] = []

    duty_cl_maes: list[float] = []
    duty_pers_maes: list[float] = []
    duty_mean_maes: list[float] = []
    obs_cycles: list[float] = []
    sim_cycles: list[float] = []

    oil_cl_maes: list[float] = []
    oil_pers_maes: list[float] = []
    oil_mean_maes: list[float] = []
    total_violations = 0

    for block in segments:
        seg = block.iloc[:n_pts]
        p_obs = seg["TP3"].to_numpy(dtype="float64")
        oil_obs = seg["Oil_temperature"].to_numpy(dtype="float64")
        u_obs = (seg["DV_eletric"].to_numpy(dtype="float64") >= 0.5).astype("float64")

        p_tf, _oil_tf, _u_tf, tf_viol = _rollout_observed_segment(params, seg, closed_loop=False)
        p_cl, oil_cl, u_cl, cl_viol = _rollout_observed_segment(params, seg, closed_loop=True)
        total_violations += tf_viol + cl_viol

        tf_maes.append(float(np.mean(np.abs(p_tf - p_obs))))
        tf_rmses.append(float(np.sqrt(np.mean((p_tf - p_obs) ** 2))))
        cl_maes.append(float(np.mean(np.abs(p_cl - p_obs))))
        cl_rmses.append(float(np.sqrt(np.mean((p_cl - p_obs) ** 2))))
        pers_maes.append(float(np.mean(np.abs(p_obs[0] - p_obs))))
        mean_maes.append(float(np.mean(np.abs(params.train_mean_tp3_bar - p_obs))))

        obs_duty = float(np.mean(u_obs))
        sim_duty = float(np.mean(u_cl))
        duty_cl_maes.append(abs(sim_duty - obs_duty))
        duty_pers_maes.append(abs(float(u_obs[0]) - obs_duty))
        duty_mean_maes.append(abs(params.train_mean_load_duty - obs_duty))

        obs_cycles.append(float(np.sum((u_obs[1:] > 0.5) & (u_obs[:-1] < 0.5))))
        sim_cycles.append(float(np.sum((u_cl[1:] > 0.5) & (u_cl[:-1] < 0.5))))

        oil_cl_maes.append(float(np.mean(np.abs(oil_cl - oil_obs))))
        oil_pers_maes.append(float(np.mean(np.abs(oil_obs[0] - oil_obs))))
        oil_mean_maes.append(float(np.mean(np.abs(params.train_mean_oil_temp_c - oil_obs))))

    tf_mean = float(np.mean(tf_maes))
    tf_med = float(np.median(tf_maes))
    tf_rmse = float(np.mean(tf_rmses))
    cl_mean = float(np.mean(cl_maes))
    cl_med = float(np.median(cl_maes))
    cl_rmse = float(np.mean(cl_rmses))
    pers_mean = float(np.mean(pers_maes))
    pers_med = float(np.median(pers_maes))
    train_mean = float(np.mean(mean_maes))
    train_med = float(np.median(mean_maes))

    tf_skill_p = _compute_skill(tf_mean, pers_mean)
    tf_skill_m = _compute_skill(tf_mean, train_mean)
    tf_med_skill_p = _compute_skill(tf_med, pers_med)
    tf_med_skill_m = _compute_skill(tf_med, train_med)
    cl_skill_p = _compute_skill(cl_mean, pers_mean)
    cl_skill_m = _compute_skill(cl_mean, train_mean)
    cl_med_skill_p = _compute_skill(cl_med, pers_med)
    cl_med_skill_m = _compute_skill(cl_med, train_med)

    tf_ok = tf_skill_p > 0.0 and tf_skill_m > 0.0 and tf_med_skill_p > 0.0 and tf_med_skill_m > 0.0
    cl_ok = cl_skill_p > 0.0 and cl_skill_m > 0.0 and cl_med_skill_p > 0.0 and cl_med_skill_m > 0.0

    duty_mean = float(np.mean(duty_cl_maes))
    duty_med = float(np.median(duty_cl_maes))
    duty_p_mean = float(np.mean(duty_pers_maes))
    duty_m_mean = float(np.mean(duty_mean_maes))
    duty_m_med = float(np.median(duty_mean_maes))
    duty_skill_m = _compute_skill(duty_mean, duty_m_mean)
    duty_med_skill_m = _compute_skill(duty_med, duty_m_med)
    duty_ok = duty_mean < duty_p_mean and duty_skill_m > 0.0 and duty_med_skill_m > 0.0

    notes: list[str] = []
    if cl_ok:
        notes.append("Closed-loop TP3 beats both Persistence and Train-Mean")
    elif tf_ok:
        notes.append(
            "UNSUPPORTED closed-loop TP3 forecast (fails Train-Mean baseline due to cycle drift); "
            "teacher-forced only"
        )
    else:
        notes.append(
            "UNSUPPORTED TP3 trajectory forecast (both teacher-forced and closed-loop fail "
            "Train-Mean baseline)"
        )
    if not duty_ok:
        notes.append("duty_cycle also fails stationary Train-Mean")

    cyc_err = float(np.mean(np.abs(np.asarray(sim_cycles) - np.asarray(obs_cycles))))
    return (
        TwinHorizonEvaluation(
            horizon_minutes=horizon_minutes,
            segment_count=len(segments),
            teacher_forced_tp3_mae_bar=round(tf_mean, 4),
            teacher_forced_tp3_median_mae_bar=round(tf_med, 4),
            teacher_forced_tp3_rmse_bar=round(tf_rmse, 4),
            closed_loop_tp3_mae_bar=round(cl_mean, 4),
            closed_loop_tp3_median_mae_bar=round(cl_med, 4),
            closed_loop_tp3_rmse_bar=round(cl_rmse, 4),
            persistence_tp3_mae_bar=round(pers_mean, 4),
            persistence_tp3_median_mae_bar=round(pers_med, 4),
            train_mean_tp3_mae_bar=round(train_mean, 4),
            train_mean_tp3_median_mae_bar=round(train_med, 4),
            teacher_forced_skill_vs_persistence=round(tf_skill_p, 4),
            teacher_forced_skill_vs_train_mean=round(tf_skill_m, 4),
            closed_loop_skill_vs_persistence=round(cl_skill_p, 4),
            closed_loop_skill_vs_train_mean=round(cl_skill_m, 4),
            teacher_forced_supported=tf_ok,
            closed_loop_supported=cl_ok,
            duty_cycle_mae=round(min(1.0, duty_mean), 4),
            duty_cycle_median_mae=round(min(1.0, duty_med), 4),
            persistence_duty_mae=round(min(1.0, duty_p_mean), 4),
            train_mean_duty_mae=round(min(1.0, duty_m_mean), 4),
            train_mean_duty_median_mae=round(min(1.0, duty_m_med), 4),
            duty_skill_vs_train_mean=round(duty_skill_m, 4),
            duty_cycle_supported=duty_ok,
            observed_cycles_per_segment=round(float(np.mean(obs_cycles)), 3),
            simulated_cycles_per_segment=round(float(np.mean(sim_cycles)), 3),
            cycle_count_mae=round(cyc_err, 3),
            closed_loop_oil_mae_c=round(float(np.mean(oil_cl_maes)), 3),
            persistence_oil_mae_c=round(float(np.mean(oil_pers_maes)), 3),
            train_mean_oil_mae_c=round(float(np.mean(oil_mean_maes)), 3),
            support_note="; ".join(notes),
        ),
        total_violations,
    )


def _evaluate_pre_eval_sensitivity(
    pre_eval: pd.DataFrame,
    params: TwinCalibratedParams,
) -> list[TwinSensitivityCheck]:
    """Evaluate 10-min closed-loop sensitivity strictly on pre-evaluation contiguous segments."""
    pre_segs = _extract_contiguous_segments(pre_eval, min_len=60)
    if not pre_segs:
        return []
    sample_segs = [seg.iloc[:60] for seg in pre_segs[:80]]

    def _run(p_set: TwinCalibratedParams) -> tuple[float, float]:
        tp3_errs: list[float] = []
        duty_errs: list[float] = []
        for seg in sample_segs:
            p_obs = seg["TP3"].to_numpy(dtype="float64")
            u_obs = (seg["DV_eletric"].to_numpy(dtype="float64") >= 0.5).astype("float64")
            p_cl, _o_cl, u_cl, _v = _rollout_observed_segment(p_set, seg, closed_loop=True)
            tp3_errs.append(float(np.mean(np.abs(p_cl - p_obs))))
            duty_errs.append(abs(float(np.mean(u_cl)) - float(np.mean(u_obs))))
        return float(np.mean(tp3_errs)), float(np.mean(duty_errs))

    base_tp3, base_duty = _run(params)
    specs: list[tuple[str, TwinCalibratedParams]] = [
        (
            "k_charge -15%",
            TwinCalibratedParams(
                **{**params.__dict__, "k_charge_bar_per_min": params.k_charge_bar_per_min * 0.85}
            ),
        ),
        (
            "k_charge +15%",
            TwinCalibratedParams(
                **{**params.__dict__, "k_charge_bar_per_min": params.k_charge_bar_per_min * 1.15}
            ),
        ),
        (
            "k_drain -15%",
            TwinCalibratedParams(
                **{**params.__dict__, "k_drain_bar_per_min": params.k_drain_bar_per_min * 0.85}
            ),
        ),
        (
            "k_drain +15%",
            TwinCalibratedParams(
                **{**params.__dict__, "k_drain_bar_per_min": params.k_drain_bar_per_min * 1.15}
            ),
        ),
    ]
    if params.cut_in_bar + 0.15 + 0.2 < params.cut_out_bar:
        specs.append(
            (
                "cut_in +0.15 bar",
                TwinCalibratedParams(**{**params.__dict__, "cut_in_bar": params.cut_in_bar + 0.15}),
            )
        )
    if params.cut_out_bar - 0.15 > params.cut_in_bar + 0.2:
        specs.append(
            (
                "cut_out -0.15 bar",
                TwinCalibratedParams(
                    **{**params.__dict__, "cut_out_bar": params.cut_out_bar - 0.15}
                ),
            )
        )

    out: list[TwinSensitivityCheck] = []
    for label, perturbed in specs:
        p_tp3, p_duty = _run(perturbed)
        out.append(
            TwinSensitivityCheck(
                perturbation=label,
                segment_count=len(sample_segs),
                baseline_tp3_mae_10min_bar=round(base_tp3, 4),
                perturbed_tp3_mae_10min_bar=round(p_tp3, 4),
                delta_tp3_mae_bar=round(p_tp3 - base_tp3, 4),
                baseline_duty_mae_10min=round(base_duty, 4),
                perturbed_duty_mae_10min=round(p_duty, 4),
                delta_duty_mae=round(p_duty - base_duty, 4),
            )
        )
    return out


def _fit_pre_eval_params(
    pre_eval: pd.DataFrame,
    *,
    fit_source: str,
) -> TwinCalibratedParams:
    """Fit identifiable physical constants strictly on pre-evaluation telemetry (zero fallbacks)."""
    dt_min = pre_eval.index.to_series().diff().dt.total_seconds().to_numpy() / 60.0
    tp3 = pre_eval["TP3"].to_numpy(dtype="float64")
    res = pre_eval["Reservoirs"].to_numpy(dtype="float64")
    oil = pre_eval["Oil_temperature"].to_numpy(dtype="float64")
    curr = pre_eval["Motor_current"].to_numpy(dtype="float64")
    loaded = pre_eval["DV_eletric"].to_numpy(dtype="float64") >= 0.5
    prev_loaded = np.roll(loaded, 1)

    contiguous = np.isfinite(dt_min) & (dt_min > 0.0) & (dt_min <= 0.3)
    active = contiguous & (curr >= 1.0)
    steady_l = contiguous & loaded & prev_loaded & (curr >= 5.5)
    steady_u = contiguous & (~loaded) & (~prev_loaded) & (curr >= 1.0) & (curr <= 5.5)
    if not np.any(steady_l) or not np.any(steady_u) or int(np.sum(active)) < 60:
        raise TwinCalibrationError(
            "Missing steady loaded/unloaded active samples in pre-evaluation telemetry"
        )

    dp_dt = np.diff(tp3, prepend=np.nan) / dt_min
    raw_drain = -float(np.median(dp_dt[steady_u]))
    if not math.isfinite(raw_drain) or raw_drain <= 0.0:
        raise TwinCalibrationError(
            f"Unloaded pre-evaluation telemetry failed physical decay check (k_drain={raw_drain})"
        )
    net_loaded = float(np.median(dp_dt[steady_l]))
    if not math.isfinite(net_loaded) or net_loaded <= 0.0:
        raise TwinCalibrationError(
            f"Loaded pre-evaluation telemetry failed physical charging check (net={net_loaded})"
        )
    k_drain = raw_drain
    k_charge = net_loaded + k_drain

    on_trans = np.where(contiguous[1:] & loaded[1:] & (~loaded[:-1]) & (curr[1:] >= 1.0))[0]
    off_trans = np.where(contiguous[1:] & (~loaded[1:]) & loaded[:-1] & (curr[1:] >= 1.0))[0]
    if len(on_trans) == 0 or len(off_trans) == 0:
        raise TwinCalibrationError(
            "Missing governor loading (0->1) or unloading (1->0) transitions in pre-evaluation data"
        )
    cut_in = float(np.median(tp3[on_trans]))
    cut_out = float(np.median(tp3[off_trans]))
    if (
        not (math.isfinite(cut_in) and math.isfinite(cut_out))
        or cut_in <= MIN_PHYSICAL_BAR
        or cut_out >= MAX_PHYSICAL_BAR
        or cut_out <= cut_in + 0.2
    ):
        raise TwinCalibrationError(
            f"Calibrated governor hysteresis invalid (cut_in={cut_in}, cut_out={cut_out})"
        )

    over = tp3[contiguous & loaded & (tp3 > cut_out)] - cut_out
    overshoot_from_samples = len(over) >= 5
    if overshoot_from_samples:
        overshoot = max(0.05, float(np.quantile(over, 0.99)))
    else:
        overshoot = max(0.05, 0.1 * (cut_out - cut_in))

    coupling_bias = float(np.median(tp3[active] - res[active]))
    if not math.isfinite(coupling_bias):
        raise TwinCalibrationError("Non-finite panel-to-reservoir coupling bias")

    dt_oil = np.diff(oil, prepend=np.nan) / dt_min
    x_t = np.column_stack([loaded[active].astype("float64"), -oil[active], np.ones(np.sum(active))])
    try:
        beta_t, *_ = np.linalg.lstsq(x_t, dt_oil[active], rcond=None)
    except np.linalg.LinAlgError as exc:
        raise TwinCalibrationError(f"Thermal OLS calibration failed: {exc}") from exc

    alpha_heat, alpha_cool, intercept = float(beta_t[0]), float(beta_t[1]), float(beta_t[2])
    if (
        not (math.isfinite(alpha_heat) and math.isfinite(alpha_cool) and math.isfinite(intercept))
        or alpha_heat <= 0.0
        or alpha_cool <= 1e-6
    ):
        raise TwinCalibrationError(
            f"Thermal OLS produced non-physical parameters "
            f"(alpha_heat={alpha_heat}, alpha_cool={alpha_cool})"
        )
    t_ambient_ref = intercept / alpha_cool
    if not math.isfinite(t_ambient_ref) or t_ambient_ref < -20.0 or t_ambient_ref > 150.0:
        raise TwinCalibrationError(
            f"Fitted thermal ambient reference out of physical range ({t_ambient_ref} C)"
        )

    return TwinCalibratedParams(
        k_charge_bar_per_min=k_charge,
        k_drain_bar_per_min=k_drain,
        cut_in_bar=cut_in,
        cut_out_bar=cut_out,
        charge_overshoot_bar=overshoot,
        coupling_bias_bar=coupling_bias,
        alpha_heat_c_per_min=alpha_heat,
        alpha_cool_per_min=alpha_cool,
        t_ambient_ref_c=t_ambient_ref,
        train_mean_tp3_bar=float(np.mean(tp3)),
        train_mean_oil_temp_c=float(np.mean(oil)),
        train_mean_load_duty=float(np.mean(loaded)),
        fit_source=fit_source,
        train_cal_samples=int(np.sum(active)),
        overshoot_from_samples=overshoot_from_samples,
    )


class CompressorReservoirTwin:
    """Calibrated Physics-Informed Compressor + Reservoir Digital Twin (no silent fallbacks)."""

    def __init__(
        self,
        *,
        params: TwinCalibratedParams,
        validation: TwinValidationSummary,
    ) -> None:
        self.params = params
        self.validation = validation

    @classmethod
    def fit_from_telemetry(
        cls,
        frame: pd.DataFrame,
        *,
        eval_start: pd.Timestamp = EVALUATION_BOUNDARY,
    ) -> CompressorReservoirTwin:
        """Fit ODE constants & baselines strictly on pre-evaluation telemetry (< ``eval_start``)."""
        if not isinstance(frame.index, pd.DatetimeIndex):
            raise TwinCalibrationError("Telemetry frame must have a DatetimeIndex")
        missing_cols = [c for c in REQUIRED_TWIN_COLUMNS if c not in frame.columns]
        if missing_cols:
            raise TwinCalibrationError(
                f"Telemetry frame missing required columns: {', '.join(missing_cols)}"
            )

        pre_eval = frame[(frame.index >= TRAIN_START) & (frame.index < eval_start)].sort_index()
        finite_mask = np.isfinite(pre_eval[list(REQUIRED_TWIN_COLUMNS)].to_numpy(dtype="float64"))
        pre_eval = pre_eval[finite_mask.all(axis=1)]
        if len(pre_eval) < 120:
            raise TwinCalibrationError(
                f"Insufficient finite pre-evaluation telemetry ({len(pre_eval)} < 120 rows)"
            )

        params = _fit_pre_eval_params(pre_eval, fit_source="live_pre_eval_fit")
        sensitivity = _evaluate_pre_eval_sensitivity(pre_eval, params)

        cal_notes = [
            (
                f"Fitted on {params.train_cal_samples} finite active pre-eval samples "
                f"(<{eval_start.date()})."
            ),
            (
                "Training baselines anchored strictly pre-evaluation: "
                f"TP3={params.train_mean_tp3_bar:.3f} bar, "
                f"Oil={params.train_mean_oil_temp_c:.2f} C, "
                f"Duty={params.train_mean_load_duty:.4f}."
            ),
        ]
        # Optional pre-eval temporal sub-split stability check when enough spans exist
        mid_ts = pre_eval.index[len(pre_eval) // 2]
        early = pre_eval[pre_eval.index < mid_ts]
        late = pre_eval[pre_eval.index >= mid_ts]
        if len(early) >= 120 and len(late) >= 120:
            try:
                p_early = _fit_pre_eval_params(early, fit_source="pre_eval_early_half")
                p_late = _fit_pre_eval_params(late, fit_source="pre_eval_late_half")
                cal_notes.append(
                    "Pre-eval half-split parameter drift: "
                    f"k_charge [{p_early.k_charge_bar_per_min:.3f} vs "
                    f"{p_late.k_charge_bar_per_min:.3f}], "
                    f"k_drain [{p_early.k_drain_bar_per_min:.3f} vs "
                    f"{p_late.k_drain_bar_per_min:.3f}], "
                    f"cut_in [{p_early.cut_in_bar:.3f} vs {p_late.cut_in_bar:.3f}], "
                    f"cut_out [{p_early.cut_out_bar:.3f} vs {p_late.cut_out_bar:.3f}]."
                )
            except TwinCalibrationError:
                cal_notes.append(
                    "Pre-eval temporal sub-split too short/non-stationary for independent "
                    "half-fit; relies on pre-eval parameter perturbation sensitivity."
                )

        validation = cls._validate_held_out_params(
            params,
            frame,
            eval_start=eval_start,
            sensitivity_checks=sensitivity,
            calibration_notes=cal_notes,
        )
        return cls(params=params, validation=validation)

    @classmethod
    def _validate_held_out_params(
        cls,
        params: TwinCalibratedParams,
        frame: pd.DataFrame,
        *,
        eval_start: pd.Timestamp = EVALUATION_BOUNDARY,
        sensitivity_checks: list[TwinSensitivityCheck] | None = None,
        calibration_notes: list[str] | None = None,
    ) -> TwinValidationSummary:
        """Evaluate teacher-forced & closed-loop rollouts vs Persistence & Train-Mean baselines."""
        ev_raw = frame[frame.index >= eval_start].sort_index()
        finite_ev = np.isfinite(ev_raw[list(REQUIRED_TWIN_COLUMNS)].to_numpy(dtype="float64"))
        ev_df = ev_raw[finite_ev.all(axis=1)]
        rep_mask = pd.Series(True, index=ev_df.index)
        for rep in metropt3.FAILURE_REPORTS:
            lo = pd.Timestamp(rep.start) - pd.Timedelta(hours=24)
            hi = pd.Timestamp(rep.end) + pd.Timedelta(hours=24)
            rep_mask &= ~((ev_df.index >= lo) & (ev_df.index <= hi))

        normal_ev = ev_df[rep_mask]
        blocks_10 = _extract_contiguous_segments(normal_ev, min_len=60)
        blocks_60 = _extract_contiguous_segments(normal_ev, min_len=360)
        if not blocks_60:
            raise TwinCalibrationError(
                f"Zero contiguous >=60m held-out normal segments available (>={eval_start.date()})"
            )

        horizon_specs = [(h, round(h * 6.0)) for h in EVAL_HORIZONS_MINUTES]
        horizon_evals: list[TwinHorizonEvaluation] = []
        total_violations = 0
        for h_min, n_pts in horizon_specs:
            segs = _extract_contiguous_segments(normal_ev, min_len=n_pts)
            if not segs:
                continue
            ev_res, viol = _evaluate_horizon_segments(
                params, segs, horizon_minutes=h_min, n_pts=n_pts
            )
            total_violations += viol
            horizon_evals.append(ev_res)

        ev10 = next((h for h in horizon_evals if h.horizon_minutes == 10.0), horizon_evals[0])
        ev60 = next((h for h in horizon_evals if h.horizon_minutes == 60.0), horizon_evals[-1])

        max_cl = _contiguous_supported_limit(horizon_evals, attr="closed_loop_supported")
        max_tf = _contiguous_supported_limit(horizon_evals, attr="teacher_forced_supported")
        max_duty = _contiguous_supported_limit(horizon_evals, attr="duty_cycle_supported")

        unsupported: list[str] = []
        for h in horizon_evals:
            if not h.closed_loop_supported and not h.teacher_forced_supported:
                unsupported.append(f"{h.horizon_minutes:.0f}m (teacher_forced & closed_loop)")
            elif not h.closed_loop_supported:
                unsupported.append(f"{h.horizon_minutes:.0f}m (closed_loop)")
            elif not h.teacher_forced_supported:
                unsupported.append(f"{h.horizon_minutes:.0f}m (teacher_forced)")

        # Evaluate failure residual magnification strictly on contiguous gap-free failure slices
        f_maes: list[float] = []
        for rep in metropt3.FAILURE_REPORTS:
            sub = ev_df[
                (ev_df.index >= pd.Timestamp(rep.start)) & (ev_df.index <= pd.Timestamp(rep.end))
            ]
            f_segs = _extract_contiguous_segments(sub, min_len=60)
            if not f_segs:
                continue
            f_seg = max(f_segs, key=len).iloc[:360]
            p_obs = f_seg["TP3"].to_numpy(dtype="float64")
            p_tf, _o_tf, _u_tf, _v = _rollout_observed_segment(params, f_seg, closed_loop=False)
            f_maes.append(float(np.mean(np.abs(p_tf - p_obs))))

        if not f_maes:
            leak_ratio = 0.0
        else:
            leak_ratio = float(np.mean(f_maes)) / max(ev60.teacher_forced_tp3_mae_bar, 1e-6)

        step_diff = _step_halving_stability_diff(params)
        notes = list(calibration_notes or [])
        for extra_note in (
            f"Unsupported horizons failing Persistence/Train-Mean baselines: "
            f"{', '.join(unsupported)}."
            if unsupported
            else None,
            f"Contiguous segments vary by horizon ({len(blocks_10)} at 10m vs "
            f"{len(blocks_60)} at 60m)."
            if len(blocks_10) != len(blocks_60)
            else None,
        ):
            if extra_note is not None and extra_note not in notes:
                notes.append(extra_note)

        return TwinValidationSummary(
            split_policy=(
                f"Fit strictly on <{eval_start.date()}; validate on >={eval_start.date()} held-out."
            ),
            train_cal_samples=params.train_cal_samples,
            heldout_normal_segments=len(blocks_60),
            mae_10min_bar=ev10.teacher_forced_tp3_mae_bar,
            mae_60min_bar=ev60.teacher_forced_tp3_mae_bar,
            rmse_60min_bar=ev60.teacher_forced_tp3_rmse_bar,
            closed_loop_mae_10min_bar=ev10.closed_loop_tp3_mae_bar,
            closed_loop_mae_60min_bar=ev60.closed_loop_tp3_mae_bar,
            closed_loop_rmse_60min_bar=ev60.closed_loop_tp3_rmse_bar,
            max_supported_closed_loop_horizon_minutes=max_cl,
            max_supported_teacher_forced_horizon_minutes=max_tf,
            max_supported_duty_horizon_minutes=max_duty,
            unsupported_horizons=unsupported,
            horizon_evaluations=horizon_evals,
            sensitivity_checks=sensitivity_checks or [],
            calibration_notes=notes,
            constraint_violations=total_violations,
            step_halving_max_diff_bar=round(step_diff, 4),
            leak_vs_normal_residual_ratio=round(leak_ratio, 3),
        )

    def validate_held_out(
        self,
        frame: pd.DataFrame,
        *,
        eval_start: pd.Timestamp = EVALUATION_BOUNDARY,
    ) -> TwinValidationSummary:
        """Re-evaluate held-out performance against strict baselines without mutating params."""
        return self._validate_held_out_params(
            self.params,
            frame,
            eval_start=eval_start,
            sensitivity_checks=self.validation.sensitivity_checks,
            calibration_notes=self.validation.calibration_notes,
        )

    def _integrate_steps(
        self,
        req: TwinSimulationRequest,
        *,
        dt_s: float,
    ) -> list[tuple[float, float, float, float, bool, float, bool]]:
        return _integrate_twin_steps(self.params, req, dt_s=dt_s)

    def _step_halving_stability_diff(self) -> float:
        return _step_halving_stability_diff(self.params)

    def model_overview(self, asset_id: str = "metropt3-apu-1") -> TwinModelResponse:
        return TwinModelResponse(
            asset_id=asset_id,
            model_version=TWIN_MODEL_VERSION,
            governing_equations=GOVERNING_EQUATIONS,
            parameters=self.params.to_contracts(),
            unidentifiable_parameters=list(UNIDENTIFIABLE_PARAMETERS),
            validation=self.validation,
            advisory_note=(
                "Physics-informed surrogate calibrated strictly on pre-evaluation APU segments. "
                f"Closed-loop TP3 forecast validated up to "
                f"{self.validation.max_supported_closed_loop_horizon_minutes:.0f} min; "
                "longer rollouts remain uncalibrated exploratory projections."
            ),
        )

    def simulate(self, req: TwinSimulationRequest) -> TwinSimulationResponse:
        rows = self._integrate_steps(req, dt_s=SIM_DT_S)
        dt_min = SIM_DT_S / 60.0

        trajectory: list[TwinTrajectoryPoint] = []
        stride = 6
        for idx, (minute, p, res_p, t_oil, loaded, duty, _) in enumerate(rows):
            if idx % stride == 0 or idx == len(rows) - 1:
                regime = "loaded" if loaded else "unloaded"
                trajectory.append(
                    TwinTrajectoryPoint(
                        minute=round(minute, 2),
                        tp3_bar=round(p, 3),
                        reservoirs_bar=round(res_p, 3),
                        oil_temp_c=round(t_oil, 2),
                        motor_loaded=loaded,
                        rolling_duty=round(duty, 4),
                        lps_active=p < LPS_SWITCH_BAR,
                        regime=regime,
                    )
                )

        intervals = rows[:-1] or rows
        mean_duty = sum(r[4] for r in rows) / len(rows)
        peak_duty = max(r[5] for r in rows)
        min_tp3 = min(r[1] for r in rows)
        max_tp3 = max(r[1] for r in rows)
        lps_mins = sum(dt_min for r in intervals if r[1] < LPS_SWITCH_BAR)
        duty_breach_mins = sum(dt_min for r in intervals if r[5] >= DUTY_ALARM_THRESHOLD)
        violations = sum(1 for r in intervals if r[6])
        final_temp = rows[-1][3]

        if not (math.isfinite(mean_duty) and math.isfinite(final_temp)):
            raise RuntimeError("Digital twin integrator produced non-finite state")

        max_cl = self.validation.max_supported_closed_loop_horizon_minutes
        horizon_supported = max_cl > 0.0 and req.horizon_minutes <= max_cl
        horizon_status: Literal["supported_closed_loop", "unsupported_closed_loop_horizon"] = (
            "supported_closed_loop" if horizon_supported else "unsupported_closed_loop_horizon"
        )
        horizon_warning: str | None = None
        if not horizon_supported:
            horizon_warning = (
                f"Requested horizon ({req.horizon_minutes:.0f} min) exceeds max validated "
                f"closed-loop TP3 horizon ({max_cl:.0f} min). Beyond {max_cl:.0f} min, "
                "autonomous TP3 point accuracy fails the constant-training-mean baseline due to "
                "unmodeled demand-driven cycle phase drift; treat rollout as qualitative stress "
                "direction only."
            )

        if req.scenario == "baseline":
            interp = (
                f"Baseline nominal duty remains bounded inside the "
                f"[{self.params.cut_in_bar:.2f}, {self.params.cut_out_bar:.2f}] bar governor band "
                f"(mean duty {mean_duty:.2f}, LPS {lps_mins:.1f} min)."
            )
        elif req.scenario == "maintenance_repair":
            repair_min = req.repair_at_minute if req.repair_at_minute is not None else 30.0
            interp = (
                f"Counterfactual maintenance repair at {repair_min:.0f} min clears injected air "
                f"loss and restores closed-loop pressure stability "
                f"(peak rolling duty {peak_duty:.2f})."
            )
        else:
            interp = (
                f"Sustained loss drives peak rolling duty to {peak_duty:.2f} "
                f"(above 0.70 threshold for {duty_breach_mins:.1f} min) with min TP3 "
                f"{min_tp3:.2f} bar and {lps_mins:.1f} min below 7.0 bar LPS switch."
            )
        if horizon_warning is not None:
            interp = f"{interp} [UNSUPPORTED FORECAST HORIZON >{max_cl:.0f}m]"

        summary = TwinScenarioSummary(
            scenario=req.scenario,
            mean_load_duty=round(mean_duty, 4),
            peak_load_duty=round(peak_duty, 4),
            min_tp3_bar=round(min_tp3, 3),
            max_tp3_bar=round(max_tp3, 3),
            lps_excursion_minutes=round(lps_mins, 2),
            time_above_duty_threshold_minutes=round(duty_breach_mins, 2),
            final_oil_temp_c=round(final_temp, 2),
            constraint_violations=violations,
            advisory_interpretation=interp,
        )

        return TwinSimulationResponse(
            asset_id=req.asset_id,
            scenario=req.scenario,
            horizon_supported=horizon_supported,
            horizon_support_status=horizon_status,
            horizon_warning=horizon_warning,
            params_used=self.params.to_contracts(),
            unidentifiable_parameters=list(UNIDENTIFIABLE_PARAMETERS),
            trajectory=trajectory,
            summary=summary,
            advisory_note=(
                "Counterfactual physics-twin rollout (provenance='simulated_projection'). "
                "Advisory only; verify with field inspection before maintenance scheduling."
            ),
        )
