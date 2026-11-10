"""Shared fixtures. Simulated telemetry is used only to unit-test code paths; all reported
evaluation results come from the real MetroPT-3 data."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from factorymind.data.metropt3 import ANALOG_SENSORS, DIGITAL_SENSORS


def simulate_apu(
    start: str,
    periods: int,
    *,
    leak_from: str | None = None,
    leak_to: str | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Simple APU compressor cycle at 10 s: pressure decays while unloaded and the compressor
    loads between 8.2 and 9.5 bar. During a leak, pressure decays ~6x faster."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=periods, freq="10s")
    leak = np.zeros(periods, dtype=bool)
    if leak_from and leak_to:
        leak = (index >= pd.Timestamp(leak_from)) & (index < pd.Timestamp(leak_to))
    tp3 = np.empty(periods)
    loading = np.zeros(periods, dtype=bool)
    p, on = 9.0, False
    for i in range(periods):
        if on:
            p += 0.08
            on = p < 9.5
        else:
            p -= 0.06 if leak[i] else 0.01
            on = p <= 8.2
        tp3[i], loading[i] = p, on
    noise = rng.normal(0, 0.005, periods)
    frame = pd.DataFrame(
        {
            "TP2": np.where(loading, tp3 + 0.3, 0.0) + noise,
            "TP3": tp3 + noise,
            "H1": np.where(loading, 0.0, tp3 - 0.02) + noise,
            "DV_pressure": np.where(loading, 0.0, 0.0) + np.abs(noise),
            "Reservoirs": tp3 - 0.01 + noise,
            "Oil_temperature": np.where(loading, 70.0, 60.0) + rng.normal(0, 0.2, periods),
            "Motor_current": np.where(loading, 7.0, 4.0) + rng.normal(0, 0.05, periods),
            "COMP": np.where(loading, 0.0, 1.0),
            "DV_eletric": np.where(loading, 1.0, 0.0),
            "Towers": (np.arange(periods) // 300 % 2).astype(float),
            "MPG": np.where(loading, 0.0, 1.0),
            "LPS": (tp3 < 7.0).astype(float),
            "Pressure_switch": np.ones(periods),
            "Oil_level": np.ones(periods),
            "Caudal_impulses": np.where(loading, 1.0, 0.0),
        },
        index=pd.DatetimeIndex(index, name="timestamp"),
    )
    return frame[[*ANALOG_SENSORS, *DIGITAL_SENSORS]]


@pytest.fixture
def apu_normal() -> pd.DataFrame:
    return simulate_apu("2020-01-01", 6 * 60 * 24 * 2)  # two days


@pytest.fixture
def clear_service_cache() -> Iterator[None]:
    from factorymind.api import anomaly

    anomaly._cached_service.cache_clear()
    yield
    anomaly._cached_service.cache_clear()
