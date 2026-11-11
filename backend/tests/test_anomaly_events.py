"""Tests for WP2 canonical event separation, warning tracking, and experiment reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from factorymind.anomaly import cli, experiment
from factorymind.anomaly.events import classify_window_event
from factorymind.anomaly.regimes import RegimeAssessment
from factorymind.anomaly.service import (
    AnomalyService,
    ModelUnavailableError,
    load_bundle,
    load_warning_bundle,
)
from factorymind.data import metropt3
from tests.conftest import simulate_apu


def _assessment(
    reason: str,
    *,
    regime: str = "loaded",
    quality_flags: tuple[str, ...] = (),
) -> RegimeAssessment:
    ts = pd.Timestamp("2020-01-10 00:00")
    return RegimeAssessment(
        window_start=ts,
        window_end=ts + pd.Timedelta("10min"),
        regime=regime,  # type: ignore[arg-type]
        confidence="insufficient_evidence" if regime == "insufficient_evidence" else "confirmed",
        reason=reason,
        summary="synthetic",
        quality_flags=quality_flags,
        missing_signals=(),
        off_fraction=0.0,
        unloaded_fraction=0.0,
        load_duty=0.2,
        load_starts=1.0,
        inrush_samples=0,
        valve_contradiction_fraction=0.0,
        motor_valve_mismatch_fraction=0.0,
        minutes_since_resume=60.0,
        stale_telemetry=reason == "stale_telemetry",
    )


def _wp2_config() -> experiment.ExperimentConfig:
    return experiment.ExperimentConfig(
        experiment="wp2-unit",
        seed=42,
        periods={
            "train": ("2020-01-01", "2020-01-03"),
            "calibration": ("2020-01-03", "2020-01-05"),
            "evaluation": ("2020-01-05", "2020-01-10"),
        },
        persistence_grid=[1, 2],
        detectors={
            "duty_rule": experiment.DetectorConfig(
                variants=[{}],
                threshold=experiment.ThresholdConfig(mode="absolute", grid=[0.5, 0.7]),
            ),
            "pca_spe": experiment.DetectorConfig(
                variants=[{}],
                threshold=experiment.ThresholdConfig(mode="quantile", grid=[0.99]),
            ),
        },
        selection=experiment.SelectionConfig(preference=["duty_rule", "pca_spe"]),
        faults=experiment.FaultConfig(
            n_per_type=2, duration_hours=2.0, min_gap_from_reports_hours=12.0
        ),
    )


def _wp2_telemetry_and_reports() -> tuple[pd.DataFrame, tuple[metropt3.FailureReport, ...]]:
    # 9 days at 10 s: Jan 1-3 train, Jan 3-5 cal, Jan 5-10 eval
    frame = simulate_apu(
        "2020-01-01",
        6 * 60 * 24 * 9,
        leak_from="2020-01-06 06:00",
        leak_to="2020-01-06 09:00",
        off_from="2020-01-05 12:00",
        off_to="2020-01-05 14:00",
        seed=11,
    ).copy()

    # Inject a frozen high-current block on Jan 7 (>24 h after F1):
    # WP0 would see load_duty=1.0 and fire a false air-leak alarm; WP2 must abstain & warn.
    stale_mask = (frame.index >= pd.Timestamp("2020-01-07 12:00")) & (
        frame.index < pd.Timestamp("2020-01-07 13:00")
    )
    frame.loc[stale_mask, "TP3"] = 8.8
    frame.loc[stale_mask, "TP2"] = 8.9
    frame.loc[stale_mask, "Motor_current"] = 7.0
    frame.loc[stale_mask, "COMP"] = 0.0
    frame.loc[stale_mask, "DV_eletric"] = 1.0

    # Inject a decoupled Oil_temperature spike on Jan 8 with normal compressor cycling:
    # duty_rule stays normal; pca_spe must emit sensor_consistency_warning, not leak alarm.
    decouple_mask = (frame.index >= pd.Timestamp("2020-01-08 10:00")) & (
        frame.index < pd.Timestamp("2020-01-08 11:00")
    )
    frame.loc[decouple_mask, "Oil_temperature"] = 110.0

    reports = (
        metropt3.FailureReport(
            "F1",
            "2020-01-06 06:00",
            "2020-01-06 09:00",
            "Air leak",
        ),
        metropt3.FailureReport(
            "F2",
            "2020-01-09 06:00",
            "2020-01-09 08:00",
            "Air leak",
        ),
    )
    frame_f2 = (frame.index >= pd.Timestamp("2020-01-09 06:00")) & (
        frame.index < pd.Timestamp("2020-01-09 08:00")
    )
    frame.loc[frame_f2, "Motor_current"] = 7.1
    frame.loc[frame_f2, "COMP"] = 0.0
    frame.loc[frame_f2, "DV_eletric"] = 1.0
    return frame, reports


def test_event_taxonomy_separates_all_four_classes() -> None:
    gapped = classify_window_event(
        regime=_assessment("incomplete_or_gapped", regime="insufficient_evidence"),
        row_valid=False,
        row_stopped=False,
        primary_anomalous=True,
        warning_detector_active=True,
    )
    assert gapped.status == "insufficient_data"
    assert gapped.event_class == "insufficient_evidence"
    assert not gapped.warning_active

    stale = classify_window_event(
        regime=_assessment(
            "stale_telemetry", regime="insufficient_evidence", quality_flags=("stale_telemetry",)
        ),
        row_valid=True,
        row_stopped=False,
        primary_anomalous=True,
        warning_detector_active=False,
    )
    assert stale.status == "insufficient_data"
    assert stale.event_class == "sensor_consistency_warning"
    assert stale.warning_active

    inconsistent = classify_window_event(
        regime=_assessment(
            "signal_inconsistency",
            regime="insufficient_evidence",
            quality_flags=("valve_contradiction",),
        ),
        row_valid=True,
        row_stopped=False,
        primary_anomalous=True,
        warning_detector_active=True,
    )
    assert inconsistent.status == "insufficient_data"
    assert inconsistent.event_class == "sensor_consistency_warning"
    assert "valve_contradiction" in inconsistent.warning_reasons
    assert "pca_spe_exceedance" in inconsistent.warning_reasons

    stopped = classify_window_event(
        regime=_assessment("confirmed_stopped", regime="stopped"),
        row_valid=True,
        row_stopped=True,
        primary_anomalous=False,
        warning_detector_active=False,
    )
    assert stopped.status == "stopped" and stopped.event_class == "informational"

    leak = classify_window_event(
        regime=_assessment("confirmed_loaded", regime="loaded"),
        row_valid=True,
        row_stopped=False,
        primary_anomalous=True,
        warning_detector_active=False,
    )
    assert leak.status == "anomalous" and leak.event_class == "air_leak_alarm"

    warning_only = classify_window_event(
        regime=_assessment("confirmed_loaded", regime="loaded"),
        row_valid=True,
        row_stopped=False,
        primary_anomalous=False,
        warning_detector_active=True,
    )
    assert warning_only.status == "normal"
    assert warning_only.event_class == "sensor_consistency_warning"
    assert warning_only.warning_active


def test_service_replay_separates_warnings_and_verifies_warning_bundle(tmp_path: Path) -> None:
    cfg = _wp2_config()
    telemetry, reports = _wp2_telemetry_and_reports()
    out = experiment.run(cfg, telemetry, reports)
    experiment.save(out, tmp_path / "artifacts", "wp2-model")

    processed = tmp_path / "processed"
    (processed / metropt3.DATASET_ID).mkdir(parents=True)
    telemetry.to_parquet(processed / metropt3.DATASET_ID / "telemetry.parquet")

    models_dir = tmp_path / "artifacts" / "models"
    primary, card = load_bundle(models_dir, "wp2-model")
    warning_bundle = load_warning_bundle(models_dir, "wp2-model", primary)
    assert warning_bundle is not None
    assert warning_bundle.detector_name == "pca_spe"

    service = AnomalyService(primary, card, processed, reports, warning_bundle=warning_bundle)

    # Check frozen Jan 7 block via score_block: must abstain leak alarm and emit consistency warning
    stale_block = telemetry[
        (telemetry.index >= pd.Timestamp("2020-01-07 12:00"))
        & (telemetry.index < pd.Timestamp("2020-01-07 12:10"))
    ]
    stale_score = service.score_block(stale_block)
    assert stale_score.status == "insufficient_data"
    assert stale_score.score is None
    assert stale_score.event_class == "sensor_consistency_warning"
    assert stale_score.warning_active
    assert stale_score.regime.regime == "insufficient_evidence"
    assert stale_score.regime.reason == "stale_telemetry"

    # Full replay over Jan 6 leak + Jan 7 frozen telemetry + Jan 8 decoupled sensor spike
    details = service.replay_details(
        pd.Timestamp("2020-01-06 00:00"),
        pd.Timestamp("2020-01-08 18:00"),
    )
    assert any(
        ep.event_class == "air_leak_alarm"
        and ep.episode.start >= pd.Timestamp("2020-01-06 06:00")
        and ep.episode.start < pd.Timestamp("2020-01-06 09:00")
        and ep.hypothesis
        and ep.uncertainty_note
        for ep in details.episodes
    )
    # Neither Jan 7 frozen telemetry nor Jan 8 Oil_temperature spike becomes an air_leak_alarm
    assert not any(
        pd.Timestamp("2020-01-07 12:00") <= ep.episode.start < pd.Timestamp("2020-01-08 18:00")
        for ep in details.episodes
    )
    # Both appear cleanly in WarningEpisodes
    assert any(
        w.event_class == "sensor_consistency_warning"
        and "stale_telemetry" in w.sources
        and w.start <= pd.Timestamp("2020-01-07 12:00") < w.end
        for w in details.warnings
    )
    assert any(
        w.event_class == "sensor_consistency_warning"
        and "pca_spe" in w.sources
        and w.start <= pd.Timestamp("2020-01-08 10:10") < w.end
        for w in details.warnings
    )

    # Backward-compatible 3-tuple replay signature preserved
    windows_legacy, episodes_legacy, reports_legacy = service.replay(
        pd.Timestamp("2020-01-06 00:00"),
        pd.Timestamp("2020-01-08 18:00"),
    )
    assert windows_legacy == details.windows
    assert episodes_legacy == details.episodes
    assert reports_legacy == details.reports

    # Warning bundle tampering or malformed metadata must raise ModelUnavailableError (503)
    warn_card_path = models_dir / "wp2-model" / warning_bundle.version / "model_card.json"
    bad_card = json.loads(warn_card_path.read_text(encoding="utf-8"))
    bad_card["artifact_sha256"] = "0" * 64
    warn_card_path.write_text(json.dumps(bad_card), encoding="utf-8")
    with pytest.raises(ModelUnavailableError):
        load_warning_bundle(models_dir, "wp2-model", primary)

    warn_card_path.write_text("{corrupt-json", encoding="utf-8")
    with pytest.raises(ModelUnavailableError):
        load_warning_bundle(models_dir, "wp2-model", primary)


def test_experiment_wp2_reports_regime_gating_and_fault_separation() -> None:
    cfg = _wp2_config()
    telemetry, reports = _wp2_telemetry_and_reports()
    out = experiment.run(cfg, telemetry, reports)
    wp2 = out.results["wp2"]
    card = out.bundles[out.results["selected_detector"]].card

    assert card["evaluation"]["wp2"] == wp2
    assert wp2["primary_detector"] == "duty_rule"
    assert wp2["warning_detector"] == "pca_spe"

    for split in ("train", "calibration", "evaluation"):
        assert wp2["regime_distribution"][split]["valid_windows"] > 0

    sep = wp2["event_separation"]
    assert sep["integrity_suppressed_leak_episodes"] >= 1
    assert sep["gated_false_alarm_episodes_wp2"] < sep["baseline_false_alarm_episodes_wp0"]
    assert sep["sensor_consistency_warning_episodes"] >= 2
    assert set(wp2["per_regime_metrics"]) == {
        "loaded",
        "unloaded",
        "starting",
        "stopped",
        "insufficient_evidence",
    }

    abl = wp2["regime_gating_ablations"]
    assert {"wp0_stopped_only", "wp2_integrity_gated", "starting_suppressed_ablation"} <= set(abl)
    assert (
        abl["wp2_integrity_gated"]["events_detected"] == abl["wp0_stopped_only"]["events_detected"]
    )
    assert (
        abl["wp2_integrity_gated"]["mean_false_alarms_per_day"]
        < abl["wp0_stopped_only"]["mean_false_alarms_per_day"]
    )

    sim = wp2["simulated_fault_separation"]
    assert sim["simulated"] is True
    assert sim["real_operational_data"] is False
    assert {"duty_rule", "pca_spe"} <= set(sim["detectors"])

    printed = cli._summary(out.results)
    assert "WP2 evaluation regimes:" in printed
    assert "WP2 event separation:" in printed
    assert "wp2_integrity_gated" in printed
    assert "starting_suppressed_ablation" in printed
