import numpy as np
import pytest

from factorymind.anomaly.calibration import ExceedanceCalibrator


def test_p_value_is_smoothed_survival_function() -> None:
    cal = ExceedanceCalibrator.fit(np.arange(100, dtype=float))  # 0..99
    assert cal.n == 100 and cal.min_p == pytest.approx(1 / 101)
    p = cal.p_value(np.array([-1.0, 0.0, 50.0, 99.0, 1000.0]))
    assert p[0] == pytest.approx(1.0)  # below everything: all 100 references are at least as high
    assert p[1] == pytest.approx(101 / 101)
    assert p[2] == pytest.approx(51 / 101)
    assert p[3] == pytest.approx(2 / 101)
    assert p[4] == pytest.approx(cal.min_p)  # beyond the reference range cannot express severity


def test_p_value_is_monotone_non_increasing() -> None:
    rng = np.random.default_rng(0)
    cal = ExceedanceCalibrator.fit(rng.normal(size=500))
    grid = np.linspace(-4, 4, 200)
    assert (np.diff(cal.p_value(grid)) <= 1e-12).all()


def test_calibration_is_valid_on_exchangeable_data() -> None:
    rng = np.random.default_rng(1)
    cal = ExceedanceCalibrator.fit(rng.normal(size=2000))
    fresh = rng.normal(size=20000)
    assert np.mean(cal.p_value(fresh) <= 0.01) == pytest.approx(0.01, abs=0.004)


def test_threshold_for_matches_quantile() -> None:
    cal = ExceedanceCalibrator.fit(np.arange(1000, dtype=float))
    thr = cal.threshold_for(0.01)
    assert np.mean(np.arange(1000) > thr) <= 0.011


@pytest.mark.parametrize("bad", [np.array([]), np.array([1.0, np.nan]), np.array([np.inf])])
def test_fit_rejects_empty_or_non_finite(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ExceedanceCalibrator.fit(bad)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1])
def test_threshold_for_validates_p(p: float) -> None:
    with pytest.raises(ValueError):
        ExceedanceCalibrator.fit(np.arange(10.0)).threshold_for(p)
