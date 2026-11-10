"""Evidence for an anomalous window: deviations grouped by physical signal.

Raw feature z-scores are redundant (the compressor being loaded shows up in a dozen valve and
pressure features). Evidence is therefore reported **once per physical group**, using the most
deviating feature in the group, with a plain statement of what the signal means. A deviation is
evidence of an *unusual state*, never of a cause.

Features that never varied in training (for example the oil-level switch) have no meaningful
z-score; when they differ from their training value they are reported as *novel* instead of with
an absurdly large z. Features that cannot be observed in the window (NaN) are never reported.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from factorymind.anomaly.detectors import FeatureProfile

GROUP_LABELS: Final[dict[str, str]] = {
    "operating_state": "Operating state",
    "pressure_level": "Pressure level",
    "pressure_dynamics": "Pressure dynamics",
    "thermal": "Oil temperature",
    "protection_switches": "Protection switches",
}

_EXACT_GROUP: Final[dict[str, str]] = {
    "load_duty": "operating_state",
    "off_fraction": "operating_state",
    "load_starts": "operating_state",
    "COMP_mean": "operating_state",
    "DV_eletric_mean": "operating_state",
    "MPG_mean": "operating_state",
    "Caudal_impulses_mean": "operating_state",
    "tp3_reservoir_gap": "pressure_level",
    "unloaded_tp3_slope": "pressure_dynamics",
    "loaded_tp3_rise": "pressure_dynamics",
    "LPS_mean": "protection_switches",
    "Pressure_switch_mean": "protection_switches",
    "Oil_level_mean": "protection_switches",
    "Towers_mean": "protection_switches",
}
_PREFIX_GROUP: Final[tuple[tuple[str, str], ...]] = (
    ("Motor_current_", "operating_state"),
    ("DV_pressure_", "operating_state"),
    ("TP2_", "pressure_level"),
    ("TP3_", "pressure_level"),
    ("H1_", "pressure_level"),
    ("Reservoirs_", "pressure_level"),
    ("Oil_temperature_", "thermal"),
)

_EXACT_MEANING: Final[dict[str, str]] = {
    "load_duty": "Share of the window the compressor spent under load. A leak makes it run "
    "loaded far more often.",
    "off_fraction": "Share of the window the motor was off.",
    "load_starts": "Load cycles started in the window.",
    "COMP_mean": "Intake-valve signal: active when the compressor is off or unloaded.",
    "DV_eletric_mean": "Outlet-valve signal: active when the compressor is under load.",
    "MPG_mean": "Signal that starts the compressor under load when pressure falls below 8.2 bar.",
    "Caudal_impulses_mean": "Air-flow pulses from the compressor to the reservoirs.",
    "tp3_reservoir_gap": "Panel (TP3) minus reservoir pressure; a large gap points to a "
    "restriction or an inconsistent sensor.",
    "unloaded_tp3_slope": "Pressure decay rate while unloaded (bar/min); a leak makes it "
    "steeper. Observable only when the compressor actually unloads.",
    "loaded_tp3_rise": "Pressure rise rate while loading (bar/min); a leak slows it.",
    "LPS_mean": "Low-pressure switch, active below 7 bar.",
    "Pressure_switch_mean": "Air-dryer discharge detection switch.",
    "Oil_level_mean": "Oil-level switch, active when the oil is below the expected level.",
    "Towers_mean": "Which air-drying tower is in service.",
}
_PREFIX_MEANING: Final[tuple[tuple[str, str], ...]] = (
    ("Motor_current_", "Motor current: about 0 A off, 4 A unloaded, 7 A under load."),
    (
        "DV_pressure_",
        "Air-dryer discharge pressure drop; about 0 bar means the compressor is under load.",
    ),
    ("TP2_", "Compressor outlet pressure (TP2)."),
    ("TP3_", "Pneumatic panel pressure (TP3)."),
    ("H1_", "Pressure drop when the cyclonic separator filter discharges."),
    ("Reservoirs_", "Reservoir pressure; should track the panel pressure (TP3)."),
    ("Oil_temperature_", "Compressor oil temperature."),
)

_CONSTANT_SCALE: Final = 1.0001e-6  # FeatureProfile floors the scale of constant features at 1e-6


@dataclass(frozen=True)
class EvidenceItem:
    group: str
    group_label: str
    feature: str
    value: float
    training_median: float
    robust_z: float | None  # None when the feature never varied in training
    novel: bool
    meaning: str


def group_of(feature: str) -> str:
    if feature in _EXACT_GROUP:
        return _EXACT_GROUP[feature]
    for prefix, group in _PREFIX_GROUP:
        if feature.startswith(prefix):
            return group
    return "operating_state"


def meaning_of(feature: str) -> str:
    if feature in _EXACT_MEANING:
        return _EXACT_MEANING[feature]
    for prefix, text in _PREFIX_MEANING:
        if feature.startswith(prefix):
            return text
    return "Window statistic of a compressor signal."


def build_evidence(
    row: pd.Series, profile: FeatureProfile, feature_names: tuple[str, ...], max_items: int = 5
) -> list[EvidenceItem]:
    """At most one evidence item per physical group, most informative first."""
    values = row[list(feature_names)].astype("float64")
    z = profile.z(values.to_frame().T).iloc[0]
    best: dict[str, tuple[float, EvidenceItem]] = {}
    for name in feature_names:
        value = float(values[name])
        if not np.isfinite(value):
            continue
        median = float(profile.median[name])
        constant = float(profile.scale[name]) <= _CONSTANT_SCALE
        if constant:
            if value == median:
                continue
            rank, z_value, novel = float("inf"), None, True
        else:
            z_value, novel = float(z[name]), False
            rank = abs(z_value)
        group = group_of(name)
        item = EvidenceItem(
            group, GROUP_LABELS[group], name, value, median, z_value, novel, meaning_of(name)
        )
        if group not in best or rank > best[group][0]:
            best[group] = (rank, item)
    ordered = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
    return [item for _, item in ordered[:max_items]]
