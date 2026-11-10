import numpy as np
import pandas as pd

from factorymind.anomaly.detectors import FeatureProfile
from factorymind.anomaly.evidence import GROUP_LABELS, build_evidence, group_of, meaning_of
from factorymind.anomaly.features import FEATURE_NAMES, WindowSpec, window_features
from tests.conftest import simulate_apu


def profile_and_leak_row() -> tuple[FeatureProfile, pd.Series]:
    normal = window_features(simulate_apu("2020-01-01", 6 * 60 * 24 * 2, seed=1), WindowSpec())
    profile = FeatureProfile().fit(normal[list(FEATURE_NAMES)])
    leak = window_features(
        simulate_apu(
            "2020-01-05", 6 * 60 * 5, leak_from="2020-01-05 01:00", leak_to="2020-01-05 05:00"
        ),
        WindowSpec(),
    )
    return profile, leak.loc["2020-01-05 03:00"]


def test_every_feature_has_a_group_and_a_meaning() -> None:
    for name in FEATURE_NAMES:
        assert group_of(name) in GROUP_LABELS
        assert meaning_of(name)


def test_at_most_one_item_per_group_most_deviating_first() -> None:
    profile, row = profile_and_leak_row()
    items = build_evidence(row, profile, tuple(FEATURE_NAMES))
    groups = [i.group for i in items]
    assert len(groups) == len(set(groups)) <= 5
    zs = [abs(i.robust_z) for i in items if i.robust_z is not None]
    assert zs == sorted(zs, reverse=True)
    assert all(i.meaning and i.group_label for i in items)
    # a sustained leak is visible as an operating-state deviation
    assert "operating_state" in groups


def test_features_that_never_varied_are_reported_as_novel_not_with_absurd_z() -> None:
    profile, row = profile_and_leak_row()
    assert profile.scale["Oil_level_mean"] <= 1.0001e-6  # constant in the simulated training data
    row = row.copy()
    row["Oil_level_mean"] = 0.0  # the oil-level switch activates
    items = {i.feature: i for i in build_evidence(row, profile, tuple(FEATURE_NAMES), max_items=9)}
    novel = items["Oil_level_mean"]
    assert novel.novel and novel.robust_z is None
    assert novel.group == "protection_switches"


def test_unchanged_constant_features_and_unobservable_features_are_not_evidence() -> None:
    profile, row = profile_and_leak_row()
    row = row.copy()
    row["unloaded_tp3_slope"] = np.nan  # cannot be observed in this window
    items = build_evidence(row, profile, tuple(FEATURE_NAMES), max_items=20)
    features = {i.feature for i in items}
    assert "unloaded_tp3_slope" not in features
    assert "Oil_level_mean" not in features  # constant and unchanged: nothing to report
