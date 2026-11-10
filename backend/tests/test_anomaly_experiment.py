import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from factorymind.anomaly import experiment
from factorymind.anomaly.evaluation import AlarmPolicy, evaluate
from factorymind.anomaly.experiment import Candidate, SelectionConfig, choose_default
from factorymind.anomaly.service import ModelUnavailableError, load_bundle
from factorymind.data.metropt3 import FailureReport
from tests.conftest import simulate_apu

REPORTS = (
    FailureReport("V1", datetime(2020, 1, 6, 6), datetime(2020, 1, 6, 10), "Air leak"),
    FailureReport("T1", datetime(2020, 1, 9, 6), datetime(2020, 1, 9, 10), "Air leak"),
)
QUANTILES = {"mode": "quantile", "grid": [0.99, 0.999]}


def config(**overrides: object) -> experiment.ExperimentConfig:
    base: dict[str, object] = {
        "experiment": "unit",
        "periods": {
            "train": ["2020-01-01", "2020-01-04"],
            "calibration": ["2020-01-04", "2020-01-05"],
            "evaluation": ["2020-01-05", "2020-01-12"],
        },
        "persistence_grid": [3, 6],
        "detectors": {
            "lps_rule": {"threshold": QUANTILES},
            "duty_rule": {"threshold": {"mode": "absolute", "grid": [0.3, 0.4, 0.6]}},
            "isolation_forest": {"variants": [{"n_estimators": 50}], "threshold": QUANTILES},
            "pca_spe": {
                "variants": [{"variance": 0.9}, {"variance": 0.95}],
                "threshold": QUANTILES,
            },
        },
        "selection": {
            "preference": ["lps_rule", "duty_rule", "pca_spe", "isolation_forest"],
        },
        "faults": {
            "n_per_type": 2,
            "duration_hours": 3.0,
            "seed": 3,
            "min_gap_from_reports_hours": 6,
        },
    }
    base.update(overrides)
    return experiment.ExperimentConfig.model_validate(base)


def telemetry() -> pd.DataFrame:
    parts = [
        simulate_apu(
            "2020-01-01",
            6 * 60 * 24 * 3,
            off_from="2020-01-02 00:00",
            off_to="2020-01-02 08:00",
            seed=1,
        ),
        simulate_apu("2020-01-04", 6 * 60 * 24, seed=2),
        simulate_apu("2020-01-05", 6 * 60 * 24, seed=3),
        simulate_apu(
            "2020-01-06",
            6 * 60 * 24,
            leak_from="2020-01-06 06:00",
            leak_to="2020-01-06 10:00",
            seed=4,
        ),
        simulate_apu("2020-01-07", 6 * 60 * 24 * 2, seed=5),
        simulate_apu(
            "2020-01-09",
            6 * 60 * 24,
            leak_from="2020-01-09 06:00",
            leak_to="2020-01-09 10:00",
            off_from="2020-01-09 20:00",
            off_to="2020-01-09 23:00",
            seed=6,
        ),
        simulate_apu("2020-01-10", 6 * 60 * 24 * 2, seed=7),
    ]
    return pd.concat(parts)


@pytest.fixture(scope="module")
def output() -> experiment.RunOutput:
    return experiment.run(config(), telemetry(), REPORTS)


def test_protocol_reports_loeo_per_event_for_every_detector(output: experiment.RunOutput) -> None:
    detectors = output.results["detectors"]
    assert set(detectors) == {"lps_rule", "duty_rule", "isolation_forest", "pca_spe"}
    for d in detectors.values():
        folds = d["loeo"]["folds"]
        assert [f["held_out"] for f in folds] == ["V1", "T1"]
        assert d["loeo"]["events_total"] == 2
    duty = detectors["duty_rule"]["loeo"]
    assert duty["events_detected"] == 2  # a sustained leak doubles the duty cycle
    assert detectors["lps_rule"]["loeo"]["events_detected"] == 0  # pressure never drops below 7 bar


def test_run_is_reproducible(output: experiment.RunOutput) -> None:
    again = experiment.run(config(), telemetry(), REPORTS)
    assert json.dumps(again.results["detectors"], sort_keys=True, default=str) == json.dumps(
        output.results["detectors"], sort_keys=True, default=str
    )
    assert again.results["selected_detector"] == output.results["selected_detector"]


def test_stopped_machine_is_explained_not_alarmed(output: experiment.RunOutput) -> None:
    ev = output.scored["evaluation"]
    stopped = ev[ev["stopped"]]
    assert len(stopped) >= 15  # the three-hour stop on 2020-01-09
    assert output.results["periods"]["evaluation"]["stopped_windows"] == len(stopped)
    # training excluded the eight-hour stop; calibration/training never contain stopped windows
    assert not output.scored["train"]["stopped"].any()
    assert output.results["excluded_from_fit"]["train"] >= 40


def test_fault_injection_is_reported_as_simulated_with_controls(
    output: experiment.RunOutput,
) -> None:
    faults = output.results["detectors"]["pca_spe"]["fault_injection"]
    assert faults["simulated"] is True and faults["cases"] == 10
    assert set(faults["per_type"]) == {"stuck", "offset", "flatline", "drift", "noise"}
    assert faults["control_alarms"].endswith("/10")
    assert faults["gross_rate"] == pytest.approx(faults["detected"] / 10, abs=1e-3)
    assert faults["control_alarm_rate"] == pytest.approx(faults["control_detected"] / 10, abs=1e-3)
    assert faults["clean_control_cases"] == 10 - faults["control_detected"]
    assert faults["incremental_lift"] == pytest.approx(
        (faults["detected"] - faults["control_detected"]) / 10, abs=1e-3
    )
    if faults["clean_control_cases"] > 0:
        assert faults["clean_control_sensitivity"] == pytest.approx(
            faults["clean_control_detected"] / faults["clean_control_cases"], abs=1e-3
        )
    assert output.results["detectors"]["lps_rule"]["fault_injection"] is None


def test_loeo_reports_operating_false_alarms_alongside_all_valid_days(
    output: experiment.RunOutput,
) -> None:
    for d in output.results["detectors"].values():
        loeo = d["loeo"]
        assert loeo["mean_operating_false_alarms_per_day"] >= loeo["mean_false_alarms_per_day"]
        assert all("operating_false_alarms_per_day" in f for f in loeo["folds"])


def test_loeo_selection_does_not_leak_held_out_false_alarm_masking() -> None:
    """Moving the held-out report onto a spurious alarm must not change the fold's chosen policy."""
    base = telemetry().copy()
    # V1 starts with moderate duty (0.5) for 1 h then high duty (0.85) for 3 h so threshold 0.3
    # detects V1 earlier (+30 min) than threshold 0.7 (+90 min).
    idx = base.index
    v1_early = (idx >= "2020-01-06 06:00") & (idx < "2020-01-06 07:00")
    v1_late = (idx >= "2020-01-06 07:00") & (idx < "2020-01-06 10:00")
    base.loc[v1_early, "Motor_current"] = [
        7.0 if i % 2 == 0 else 4.0 for i in range(int(v1_early.sum()))
    ]
    base.loc[v1_late, "Motor_current"] = [
        7.0 if i % 10 < 8 else 4.0 for i in range(int(v1_late.sum()))
    ]
    # Spurious 2 h burst at duty=0.5 on Jan 7 (alarms at threshold 0.3, silent at 0.7).
    burst = (idx >= "2020-01-07 12:00") & (idx < "2020-01-07 14:00")
    base.loc[burst, "Motor_current"] = [7.0 if i % 2 == 0 else 4.0 for i in range(int(burst.sum()))]

    cfg = config(
        detectors={
            "duty_rule": {"threshold": {"mode": "absolute", "grid": [0.3, 0.7]}},
        },
        selection={"preference": ["duty_rule"], "rule_detectors": ["duty_rule"]},
        persistence_grid=[3],
    )
    v1 = FailureReport("V1", datetime(2020, 1, 6, 6), datetime(2020, 1, 6, 10), "Air leak")
    # Held-out T1 placed on top of the Jan 7 burst vs placed on quiet Jan 11:
    reports_overlap = (
        v1,
        FailureReport("T1", datetime(2020, 1, 7, 12), datetime(2020, 1, 7, 14), "Air leak"),
    )
    reports_separate = (
        v1,
        FailureReport("T1", datetime(2020, 1, 11, 12), datetime(2020, 1, 11, 14), "Air leak"),
    )

    fold_overlap = experiment.run(cfg, base, reports_overlap).results["detectors"]["duty_rule"][
        "loeo"
    ]["folds"][1]
    fold_separate = experiment.run(cfg, base, reports_separate).results["detectors"]["duty_rule"][
        "loeo"
    ]["folds"][1]
    assert fold_overlap["held_out"] == "T1"
    assert fold_overlap["threshold"] == "0.7"
    assert fold_separate["threshold"] == "0.7"


def test_exceedance_and_reference_checks_are_present(output: experiment.RunOutput) -> None:
    for d in output.results["detectors"].values():
        assert d["exceedance_check"]["windows"] > 0
        assert 0 < d["final_policy"]["exceedance_at_threshold"] <= 1
        assert "share_at_or_below_alpha" in d["reference_holdout_check"]
    assert "p_pca_spe" in output.scored["evaluation"]


def test_artifacts_are_saved_with_card_and_current_pointer(
    output: experiment.RunOutput, tmp_path: Path
) -> None:
    experiment.save(output, tmp_path, "unit-model", extra={"sensitivity": {"x": 1}})
    root = tmp_path / "models" / "unit-model"
    current = (root / "CURRENT").read_text().strip()
    assert current.endswith(str(output.results["selected_detector"]))
    card = json.loads((root / current / "model_card.json").read_text())
    assert card["schema_version"] == 2 and card["dataset"]["license"] == "CC BY 4.0"
    assert len(card["artifact_sha256"]) == 64
    assert card["evaluation"]["loeo"]["events_total"] == 2
    assert "share_at_or_below_alpha" in card["evaluation"]["reference_holdout_check"]
    assert "by_month" in card["evaluation"]["exceedance_check"]
    run_dir = tmp_path / "experiments" / "unit" / str(output.results["version"])
    saved = json.loads((run_dir / "results.json").read_text())
    assert saved["sensitivity"] == {"x": 1}
    assert (run_dir / "scores_evaluation.parquet").exists()
    bundle, _ = load_bundle(tmp_path / "models", "unit-model")
    assert bundle.version == current and bundle.calibrator.n > 0


def test_load_bundle_rejects_missing_tampered_and_unsupported_models(
    output: experiment.RunOutput, tmp_path: Path
) -> None:
    with pytest.raises(ModelUnavailableError, match="No current model"):
        load_bundle(tmp_path / "none", "absent")
    experiment.save(output, tmp_path, "m")
    root = tmp_path / "models" / "m"
    current = (root / "CURRENT").read_text().strip()
    card_path = root / current / "model_card.json"
    original = card_path.read_text()
    card = json.loads(original)

    card["artifact_sha256"] = "0" * 64
    card_path.write_text(json.dumps(card))
    with pytest.raises(ModelUnavailableError, match="integrity"):
        load_bundle(tmp_path / "models", "m")

    # an M1-format card (no schema_version 2) must not be served
    card["artifact_sha256"] = json.loads(original)["artifact_sha256"]
    card["schema_version"] = 1
    card_path.write_text(json.dumps(card))
    with pytest.raises(ModelUnavailableError, match="unsupported"):
        load_bundle(tmp_path / "models", "m")
    card_path.write_text(original)


def test_errors_do_not_disclose_server_paths(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailableError) as info:
        load_bundle(tmp_path / "secret-dir", "absent")
    assert "secret-dir" not in str(info.value)


def test_training_period_with_reports_is_rejected() -> None:
    bad = (FailureReport("X", datetime(2020, 1, 2, 12), datetime(2020, 1, 2, 13), "Air leak"),)
    with pytest.raises(ValueError, match="must not contain reported failures"):
        experiment.run(config(), telemetry(), bad)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "periods": {
                "train": ["2020-01-03", "2020-01-01"],
                "calibration": ["2020-01-04", "2020-01-05"],
                "evaluation": ["2020-01-05", "2020-01-12"],
            }
        },
        {
            "periods": {
                "train": ["2020-01-01", "2020-01-04"],
                "evaluation": ["2020-01-05", "2020-01-12"],
            }
        },
        {"persistence_grid": [0]},
        {
            "detectors": {"pca_spe": {"variants": [], "threshold": QUANTILES}},
            "selection": {"preference": ["pca_spe"]},
        },
        {"selection": {"preference": ["duty_rule"]}},
    ],
)
def test_config_validation(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        config(**overrides)


def test_threshold_grid_validation() -> None:
    with pytest.raises(ValidationError):
        experiment.ThresholdConfig(mode="quantile", grid=[1.5])
    with pytest.raises(ValidationError):
        experiment.ThresholdConfig(mode="absolute", grid=[])


def test_repository_config_is_valid() -> None:
    from factorymind.anomaly.cli import DEFAULT_CONFIG

    cfg = experiment.ExperimentConfig.load(DEFAULT_CONFIG)
    assert cfg.experiment == "metropt3-anomaly" and len(cfg.fingerprint()) == 10
    assert set(cfg.selection.preference) == set(cfg.detectors)


# ---------------------------------------------------------------- selection rules


def _candidate(
    detected: list[bool], fa: float, delays: list[float | None], label: str = "x"
) -> Candidate:
    from factorymind.anomaly.evaluation import EvaluationResult, EventResult

    events = [EventResult(f"E{i + 1}", d, delays[i] if d else None) for i, d in enumerate(detected)]
    result = EvaluationResult(
        valid_windows=1,
        valid_days=1.0,
        operating_windows=1,
        operating_days=1.0,
        positive_windows=1,
        average_precision=None,
        precision=None,
        recall=None,
        f1=None,
        confusion={},
        events=events,
        events_detected=sum(detected),
        events_total=len(detected),
        alarm_episodes=0,
        false_alarm_episodes=0,
        false_alarms_per_day=fa,
        operating_false_alarms_per_day=fa,
    )
    return Candidate("d", 0, {}, label, 1.0, 1, result)


def test_select_policy_orders_by_detected_then_false_alarms_then_delay() -> None:
    a = _candidate([True, True, False], 0.0, [10, 10, None], "a")
    b = _candidate([True, True, True], 0.5, [10, 10, 10], "b")
    c = _candidate([True, True, True], 0.1, [90, 90, 90], "c")
    d = _candidate([True, True, True], 0.1, [20, 20, 20], "d")
    assert experiment.select_policy([a, b, c, d]).threshold_label == "d"
    assert experiment.select_policy([a, b]).threshold_label == "b"


def test_select_policy_can_ignore_the_held_out_event() -> None:
    # x detects only E1; y detects E2 and E3
    x = _candidate([True, False, False], 0.0, [10, None, None], "x")
    y = _candidate([False, True, True], 0.0, [None, 10, 10], "y")
    assert experiment.select_policy([x, y]).threshold_label == "y"  # 2 events beat 1
    # holding E2 out leaves one event each, a full tie, so the earlier candidate wins
    assert experiment.select_policy([x, y], exclude_event="E2").threshold_label == "x"
    # holding E1 out leaves x with nothing
    assert experiment.select_policy([x, y], exclude_event="E1").threshold_label == "y"


def _summary(detected: int, fa: float, delay: float | None = 50.0) -> dict[str, Any]:
    return {
        "events_detected": detected,
        "mean_false_alarms_per_day": fa,
        "median_abs_delay_minutes": delay,
    }


SEL = SelectionConfig(preference=["lps_rule", "duty_rule", "pca_spe", "isolation_forest"])


def _fault_dict(
    gross: float, ctrl: float, clean_sens: float | None, lift: float | None
) -> dict[str, Any]:
    return {
        "gross_rate": gross,
        "control_alarm_rate": ctrl,
        "clean_control_sensitivity": clean_sens,
        "incremental_lift": lift,
    }


def test_default_is_best_rule_when_no_learned_detector_qualifies() -> None:
    summary = {
        "lps_rule": _summary(1, 0.0),
        "duty_rule": _summary(4, 0.03),
        "pca_spe": _summary(4, 0.06),
        "isolation_forest": _summary(3, 0.04),
    }
    faults = {
        "pca_spe": _fault_dict(0.73, 0.25, 0.64, 0.48),
        "isolation_forest": _fault_dict(0.23, 0.18, 0.06, 0.05),
    }
    name, why = choose_default(summary, faults, SEL)
    assert name == "duty_rule" and "no learned detector qualified" in why


def test_learned_detector_qualifies_by_matching_the_rule() -> None:
    summary = {
        "lps_rule": _summary(1, 0.0),
        "duty_rule": _summary(4, 0.05),
        "pca_spe": _summary(4, 0.03),
        "isolation_forest": _summary(2, 0.0),
    }
    name, why = choose_default(summary, {}, SEL)
    assert name == "pca_spe" and "matches duty_rule" in why


def test_learned_detector_qualifies_through_fault_detection_without_losing_leaks() -> None:
    summary = {
        "lps_rule": _summary(1, 0.0),
        "duty_rule": _summary(4, 0.03),
        "pca_spe": _summary(4, 0.08),
        "isolation_forest": _summary(3, 0.01),
    }
    faults = {
        "pca_spe": _fault_dict(0.95, 0.05, 0.947, 0.90),
        "isolation_forest": _fault_dict(0.99, 0.0, 0.99, 0.99),
    }
    name, why = choose_default(summary, faults, SEL)
    # isolation_forest sees faults but detects fewer leaks than the rule: not eligible
    assert name == "pca_spe" and "detects injected faults" in why


def test_fault_detection_with_too_many_false_alarms_does_not_qualify() -> None:
    summary = {
        "lps_rule": _summary(1, 0.0),
        "duty_rule": _summary(4, 0.03),
        "pca_spe": _summary(4, 0.4),
        "isolation_forest": _summary(0, 0.0),
    }
    name, _ = choose_default(summary, {"pca_spe": _fault_dict(1.0, 0.0, 1.0, 1.0)}, SEL)
    assert name == "duty_rule"


def test_high_control_alarm_detector_cannot_qualify_via_gross_fault_rate() -> None:
    summary = {
        "lps_rule": _summary(1, 0.0),
        "duty_rule": _summary(4, 0.03),
        "pca_spe": _summary(4, 0.06),
        "isolation_forest": _summary(4, 0.06),
    }
    faults = {
        # Alarms on 100% of both faulted and control intervals => zero clean controls, zero lift
        "pca_spe": _fault_dict(1.0, 1.0, None, 0.0),
        # Alarms on 95% of control and 95% of faulted => gross_rate 0.95, clean_sens 0.0, lift 0.0
        "isolation_forest": _fault_dict(0.95, 0.95, 0.0, 0.0),
    }
    name, _ = choose_default(summary, faults, SEL)
    assert name == "duty_rule"


def test_sensitivity_rerun_disables_the_extreme_duty_exclusion() -> None:
    result = experiment.run_sensitivity(config(), telemetry())
    assert set(result) == {"no_extreme_duty_exclusion"}
    assert "duty_rule" in result["no_extreme_duty_exclusion"]["loeo"]


def test_evaluate_with_suppression_matches_alarm_policy_semantics() -> None:
    starts = pd.date_range("2020-01-01", periods=12, freq="10min")
    scores = pd.Series([1.0] * 12).to_numpy()
    suppress = pd.Series([False] * 5 + [True] + [False] * 6).to_numpy()
    res = evaluate(
        starts,
        scores,
        pd.Timedelta("10min"),
        AlarmPolicy(0.5, 6),
        (),
        early_horizon=pd.Timedelta(0),
        post_tolerance=pd.Timedelta(0),
        suppress=suppress,
    )
    assert res.alarm_episodes == 1  # the explained window splits 5 + 6 windows; only 6 qualifies
