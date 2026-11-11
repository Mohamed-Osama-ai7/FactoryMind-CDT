"""Asset Digital Passport definitions backed by repository domain constants."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from factorymind.contracts.assets import (
    AssetPassport,
    AssetPassportSummary,
    FailureModeSpec,
    ModelApplicabilityEntry,
    SensorPassport,
)
from factorymind.data.foundation import inspect_connected_datasets
from factorymind.data.metropt3 import (
    ANALOG_SENSORS,
    DIGITAL_SENSORS,
    NOMINAL_PERIOD_S,
    VALID_RANGES,
)

REGISTERED_ASSET_IDS: Final[tuple[str, ...]] = ("metropt3-apu-1",)

_ANALOG_DETAILS: Final[dict[str, tuple[str, str, str, str | None]]] = {
    "TP2": (
        "Compressor pressure (TP2)",
        "Pressure upstream at compressor stage.",
        "~0 bar stopped/unloaded; ~8.2-10.2 bar under loaded compression",
        None,
    ),
    "TP3": (
        "Pneumatic panel pressure (TP3)",
        "Main pneumatic circuit control panel pressure.",
        "Nominal ~8.2-10.2 bar; triggers cut-in <8.2 bar and cut-out ~10.2 bar",
        None,
    ),
    "H1": (
        "Cyclonic separator filter drop (H1)",
        "Pressure drop across cyclonic filter discharge separator.",
        "Near 0 bar during active compression (loaded); tracks panel pressure (TP3) when "
        "unloaded or stopped",
        "Flat near 0 during continuous compression or purge holds; excluded from univariate "
        "RCA sensor-freeze heuristics (pca_spe still includes all 42 window features).",
    ),
    "DV_pressure": (
        "Air-dryer tower discharge drop (DV_pressure)",
        "Pressure drop across air-dryer discharge tower.",
        "Reads ~0 bar when tower discharges cleanly under load",
        "DV_pressure sits near 0 during loaded periods; flat readings alone are not an outage.",
    ),
    "Reservoirs": (
        "Downstream reservoir pressure",
        "Pressure inside train air reservoirs feeding brakes, doors, and suspension.",
        "Tracks panel envelope (~8.2-10.2 bar) during normal duty",
        None,
    ),
    "Oil_temperature": (
        "Compressor oil temperature",
        "Thermal state of the screw compressor lubrication circuit.",
        "Gradual thermal rise with duty cycle and ambient temperature",
        None,
    ),
    "Motor_current": (
        "Compressor motor phase current",
        "Electrical draw distinguishing compressor operating states.",
        "~0 A stopped; ~4 A unloaded; ~7 A loaded compression; ~9 A starting peak",
        None,
    ),
}

_DIGITAL_DETAILS: Final[dict[str, tuple[str, str, str]]] = {
    "COMP": (
        "Intake valve command (COMP)",
        "Electrical signal on air-intake valve (active=1 when no air intake: off/unloaded).",
        "0 under loaded air intake; 1 when stopped or unloaded",
    ),
    "DV_eletric": (
        "Solenoid outlet valve (DV_eletric)",
        "Outlet solenoid valve signal active during loaded compressor delivery.",
        "1 when working under load; 0 otherwise (retains source dataset spelling)",
    ),
    "Towers": (
        "Active drying tower selector (Towers)",
        "Defines which air-dryer twin tower is drying vs regenerating.",
        "Alternates 0 (tower one) and 1 (tower two) across purge cycles",
    ),
    "MPG": (
        "Low-pressure cut-in governor (MPG)",
        "Starts compressor under load whenever pneumatic panel drops below ~8.2 bar.",
        "1 when cut-in load start is requested (<8.2 bar); 0 otherwise",
    ),
    "LPS": (
        "Low-pressure emergency switch (LPS)",
        "Safety pressure switch active when system pressure drops below 7 bar.",
        "0 under healthy pressure (>7 bar); 1 indicates low-pressure excursion (<7 bar)",
    ),
    "Pressure_switch": (
        "Dryer tower discharge switch (Pressure_switch)",
        "Detects pressure-controlled air-drying tower discharge events.",
        "0 nominal; 1 on dryer tower discharge transition",
    ),
    "Oil_level": (
        "Low oil level switch (Oil_level)",
        "Detects compressor oil level dropping below expected minimum threshold.",
        "0 nominal oil level; 1 when oil falls below threshold",
    ),
    "Caudal_impulses": (
        "Airflow caudal pulse output (Caudal_impulses)",
        "Discrete airflow pulse train emitted by air production assembly.",
        "0/1 discrete pulse transitions during airflow delivery",
    ),
}


def _metropt3_sensors() -> list[SensorPassport]:
    sensors: list[SensorPassport] = []
    for name, unit in ANALOG_SENSORS.items():
        display, role, env, unobs = _ANALOG_DETAILS[name]
        sensors.append(
            SensorPassport(
                sensor_id=name,
                display_name=display,
                kind="analog",
                unit=unit,
                physical_role=role,
                valid_range=VALID_RANGES[name],
                nominal_envelope=env,
                unobservability_note=unobs,
            )
        )
    for name in DIGITAL_SENSORS:
        display, role, env = _DIGITAL_DETAILS[name]
        sensors.append(
            SensorPassport(
                sensor_id=name,
                display_name=display,
                kind="digital",
                unit="state (0/1)",
                physical_role=role,
                valid_range=(0.0, 1.0),
                nominal_envelope=env,
                unobservability_note=None,
            )
        )
    return sensors


def _build_metropt3_passport() -> AssetPassport:
    return AssetPassport(
        asset_id="metropt3-apu-1",
        name="Porto Metro Train APU Screw Compressor #1",
        asset_type="Air Production Unit (APU) Rotary Screw Compressor",
        system_context=(
            "Railway auxiliary pneumatic supply compressing air for braking, suspension, and "
            "passenger doors; continuously monitored at ~0.1 Hz in UCI MetroPT-3."
        ),
        site_or_domain=(
            "Metro do Porto — Fleet Auxiliary Air Production Unit (Public Dataset #791)"
        ),
        deployment_scope_note=(
            "Public retrospective benchmark telemetry (Feb-Aug 2020); advisory support only."
        ),
        nominal_sampling_period_s=NOMINAL_PERIOD_S,
        operating_regimes=[
            "stopped",
            "unloaded",
            "loaded",
            "starting",
            "insufficient_evidence",
        ],
        operating_envelopes={
            "stopped": (
                "Motor_current < 1.0 A (OFF_CURRENT_A) for >= 0.9 of the 10-min window, "
                "COMP == 1, DV_eletric == 0 (no active compression)"
            ),
            "unloaded": (
                "1.0 <= Motor_current <= 5.5 A for >= 0.9 of window, COMP == 1, "
                "DV_eletric == 0 (idle running)"
            ),
            "loaded": (
                "Motor_current > 5.5 A (LOADED_CURRENT_A) for >= 0.9 of window, COMP == 0, "
                "DV_eletric == 1 (active compression)"
            ),
            "starting": (
                "Contiguous resumption out of stopped or a >600 s gap with active startup cycling, "
                "or observed Motor_current > 8.0 A inrush (~0.1 Hz often misses sub-10 s inrush)"
            ),
            "insufficient_evidence": (
                "Mixed operating duty transitions (mixed_duty_cycle), hysteresis holds "
                "(hysteresis_hold), stale channels, valve/motor contradictions, or insufficient "
                "valid samples preventing single-regime attribution"
            ),
        },
        sensors=_metropt3_sensors(),
        failure_modes=[
            FailureModeSpec(
                mode_id="air_leakage",
                display_name="Pneumatic Air Leakage",
                description=(
                    "Loss of compressed air across fittings, hoses, dryer purge assemblies, "
                    "or downstream train circuits forcing elevated loaded duty and rapid "
                    "unloaded pressure decay."
                ),
                primary_signatures=[
                    "Sustained loaded duty exceedance relative to normal cut-in/cut-out baseline",
                    "Negative unloaded TP3/Reservoirs pressure slope outside nominal hold",
                    "Secondary oil temperature elevation and LPS activation under severe loss",
                ],
                competing_explanations=["abnormal_high_demand", "concurrent_conditions"],
                recommended_checks=[
                    "Isolate compressor circuit safely and inspect purge/dryer couplings and hoses",
                    "Perform stationary pressure-hold leak-down test during depot turnaround",
                ],
            ),
            FailureModeSpec(
                mode_id="abnormal_high_demand",
                display_name="Abnormal Downstream Air Demand",
                description=(
                    "Sustained heavy braking, passenger door actuation, or auxiliary pneumatic "
                    "load increasing compressor duty without an internal APU leak."
                ),
                primary_signatures=[
                    "Elevated loaded duty cycle with nominal stationary pressure-hold slope",
                    "Recovery back to normal cycling after peak demand period subsides",
                ],
                competing_explanations=["air_leakage"],
                recommended_checks=[
                    "Correlate episode interval with train duty logs and brake/door demand reports",
                ],
            ),
            FailureModeSpec(
                mode_id="sensor_malfunction",
                display_name="Sensor Freeze / Scaling Drift / Channel Fault",
                description=(
                    "Single analog sensor flatline, out-of-range corruption, or localized PCA "
                    "departure unsupported by physical compressor duty signals."
                ),
                primary_signatures=[
                    "Near-zero channel variance on core sensors (e.g., TP2, TP3, Motor_current)",
                    "Isolated pca_spe warning excursion without duty_rule exceedance",
                ],
                competing_explanations=["telemetry_inconsistency"],
                recommended_checks=[
                    "Inspect transducer harness, ADC loop wiring, and zero/span calibration",
                ],
            ),
            FailureModeSpec(
                mode_id="telemetry_inconsistency",
                display_name="Control / Telemetry State Contradiction",
                description=(
                    "Mutual contradiction between digital valve signals (COMP vs DV_eletric) or "
                    "between motor current and intake/outlet command states."
                ),
                primary_signatures=[
                    "valve_contradiction (COMP and DV_eletric inconsistent in window)",
                    "motor_valve_mismatch across operating regime transition",
                ],
                competing_explanations=["sensor_malfunction", "concurrent_conditions"],
                recommended_checks=[
                    "Check PLC digital input/output debounce and limit-switch continuity",
                ],
            ),
        ],
        connected_dataset_ids=["metropt3"],
        model_applicability=[
            ModelApplicabilityEntry(
                model_id="duty_rule",
                status="active",
                scope_note=(
                    "Primary calibrated air-leak detector scoring 10-min loaded duty (load_duty) "
                    "with binomial exceedance calibration and stateful persistence (LOEO 4/4)."
                ),
                limitations=[
                    "Scores load_duty only; unloaded_tp3_slope pressure decay is evaluated "
                    "downstream by the WP4 RCA engine rather than duty_rule.",
                    "Calibrated specifically on MetroPT-3 2020 telemetry windows; requires "
                    "re-calibration before deployment on any different compressor unit.",
                ],
            ),
            ModelApplicabilityEntry(
                model_id="pca_spe",
                status="active",
                scope_note=(
                    "Standardized PCA squared prediction error fitted across all 42 window "
                    "features (7 analog stats, 8 digital means, 6 derived) routed exclusively to "
                    "the non-leak warning tier."
                ),
                limitations=[
                    "Measures statistical projection residual across all 42 window features; "
                    "univariate WP4 sensor-freeze checks still exclude flat/purge channels "
                    "H1 and DV_pressure.",
                    "Indicates statistical departure only; does not diagnose leak root cause.",
                ],
            ),
            ModelApplicabilityEntry(
                model_id="wp2_regime_classifier",
                status="active",
                scope_note=(
                    "Causal window regime & integrity gate enforcing stale/contradictory "
                    "telemetry abstention before primary alarm scoring."
                ),
                limitations=[
                    "Abstains conservatively under valve contradiction or low valid sample counts.",
                ],
            ),
            ModelApplicabilityEntry(
                model_id="wp4_rca_engine",
                status="active",
                scope_note=(
                    "Deterministic multi-hypothesis physical RCA engine grounded solely in "
                    "provenance-tagged EventEvidencePackage records."
                ),
                limitations=[
                    "Support index (0-1) reflects physical evidence consistency, not a calibrated "
                    "posterior probability.",
                ],
            ),
            ModelApplicabilityEntry(
                model_id="cross_asset_transfer",
                status="not_calibrated",
                scope_note="Cross-asset or unseen compressor generalization is disabled.",
                limitations=[
                    "Trained bundles apply exclusively to metropt3-apu-1 unless OOD/calibration "
                    "checks are verified on target asset data.",
                ],
            ),
        ],
        unobserved_quantities=[
            "Downstream air-demand flowmeter (mass flow to train brakes, suspension, and doors)",
            "High-rate (>1 Hz) electrical motor current inrush waveforms (source is ~0.1 Hz)",
            "Direct PLC start/stop command and operating-mode control tags",
            "Exact depot maintenance repair and parts-replacement timestamps",
            "Compressor oil pressure gauge and oil moisture telemetry",
            "Ambient air temperature and train passenger occupancy logs",
        ],
    )


_PASSPORT_BUILDERS = {
    "metropt3-apu-1": _build_metropt3_passport,
}


def read_active_model_version(models_dir: Path, model_name: str) -> str | None:
    """Safely inspect the configured model version without loading pickled artifacts."""
    root = (models_dir / model_name).resolve()
    pointer = root / "CURRENT"
    if not pointer.is_file():
        return None
    try:
        version = pointer.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not version or "\x00" in version or Path(version).name != version:
        return None
    try:
        target_dir = (root / version).resolve()
        if not target_dir.is_relative_to(root):
            return None
        card_path = (target_dir / "model_card.json").resolve()
        if not card_path.is_relative_to(root) or not card_path.is_file():
            return None
        card = json.loads(card_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(card, dict):
        return None
    card_ver = card.get("model_version")
    return str(card_ver) if isinstance(card_ver, str) and card_ver else version


def get_asset_passport(asset_id: str) -> AssetPassport | None:
    builder = _PASSPORT_BUILDERS.get(asset_id)
    return builder() if builder is not None else None


def list_asset_passports(
    *,
    allowed_asset_ids: Sequence[str],
    raw_dir: Path,
    processed_dir: Path,
    models_dir: Path,
    model_name: str,
) -> list[AssetPassportSummary]:
    """Build Digital Passport summaries filtered strictly to caller/deployment asset IDs."""
    datasets = {
        d.dataset_id: d
        for d in inspect_connected_datasets(raw_dir, processed_dir, verify_raw_hash=False)
    }
    runtime_version = read_active_model_version(models_dir, model_name)
    summaries: list[AssetPassportSummary] = []
    seen: set[str] = set()
    for asset_id in allowed_asset_ids:
        if asset_id in seen:
            continue
        seen.add(asset_id)
        passport = get_asset_passport(asset_id)
        if passport is None:
            continue
        telemetry_ok = any(
            (ds := datasets.get(did)) is not None
            and ds.telemetry_present
            and any(
                c.check_id == "parquet_schema_and_rows" and c.status == "pass"
                for c in ds.quality_checks
            )
            for did in passport.connected_dataset_ids
        )
        summaries.append(
            AssetPassportSummary(
                asset_id=passport.asset_id,
                name=passport.name,
                asset_type=passport.asset_type,
                sensor_count=len(passport.sensors),
                failure_mode_count=len(passport.failure_modes),
                connected_dataset_ids=passport.connected_dataset_ids,
                active_models=[
                    m.model_id for m in passport.model_applicability if m.status == "active"
                ],
                telemetry_available=telemetry_ok,
                runtime_model_version=runtime_version,
            )
        )
    return summaries
