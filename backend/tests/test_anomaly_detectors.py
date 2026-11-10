import numpy as np
import pandas as pd
import pytest

from factorymind.anomaly.detectors import FeatureProfile, build_detector
from factorymind.anomaly.features import FEATURE_NAMES, WindowSpec, window_features
from tests.conftest import simulate_apu


@pytest.fixture(scope="module")
def windows() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = window_features(simulate_apu("2020-01-01", 6 * 60 * 24 * 3, seed=1), WindowSpec())
    leak = window_features(
        simulate_apu(
            "2020-01-05",
            6 * 60 * 6,
            leak_from="2020-01-05 02:00",
            leak_to="2020-01-05 06:00",
            seed=2,
        ),
        WindowSpec(),
    )
    return train[list(FEATURE_NAMES)], leak


@pytest.mark.parametrize("name", ["robust_z", "isolation_forest", "pca_spe"])
def test_learned_detectors_score_leak_above_normal(
    name: str, windows: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, leak = windows
    detector = build_detector(name, {}, seed=7).fit(train)  # type: ignore[arg-type]
    scores = detector.score(leak[list(FEATURE_NAMES)])
    assert scores.shape == (len(leak),)
    assert np.isfinite(scores).all()
    normal_part = scores[leak.index < "2020-01-05 02:00"]
    leak_part = scores[leak.index >= "2020-01-05 03:00"]
    assert leak_part.mean() > normal_part.mean()


def test_isolation_forest_is_reproducible(windows: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    train, leak = windows
    a = build_detector("isolation_forest", {"n_estimators": 50}, seed=3).fit(train)
    b = build_detector("isolation_forest", {"n_estimators": 50}, seed=3).fit(train)
    np.testing.assert_array_equal(
        a.score(leak[list(FEATURE_NAMES)]), b.score(leak[list(FEATURE_NAMES)])
    )


def test_lps_rule_uses_plant_switch() -> None:
    frame = pd.DataFrame({"LPS_mean": [0.0, 0.5]})
    detector = build_detector("lps_rule", {}, seed=0).fit(frame)
    assert detector.score(frame).tolist() == [0.0, 0.5]


def test_profile_handles_bimodal_features_without_saturating() -> None:
    # 80 % of windows unloaded (~0 bar), 20 % loaded (~10 bar): MAD alone would be ~0.
    values = np.r_[np.full(800, 0.01), np.full(200, 10.0)] + np.random.default_rng(0).normal(
        0, 0.001, 1000
    )
    profile = FeatureProfile().fit(pd.DataFrame({"TP2_mean": values}))
    z_loaded = profile.z(pd.DataFrame({"TP2_mean": [10.0]})).iloc[0, 0]
    assert abs(z_loaded) < 10


def test_unfitted_detectors_raise() -> None:
    frame = pd.DataFrame({name: [0.0] for name in FEATURE_NAMES})
    for name in ("isolation_forest", "pca_spe"):
        with pytest.raises(RuntimeError):
            build_detector(name, {}, seed=0).score(frame)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        build_detector("nope", {}, seed=0)  # type: ignore[arg-type]
