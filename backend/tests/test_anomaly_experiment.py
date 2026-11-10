import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from factorymind.anomaly import experiment
from factorymind.anomaly.service import ModelUnavailableError, load_bundle
from factorymind.data.metropt3 import FailureReport
from tests.conftest import simulate_apu

REPORTS = (
    FailureReport("V1", datetime(2020, 1, 4, 6), datetime(2020, 1, 4, 10), "Air leak"),
    FailureReport("T1", datetime(2020, 1, 5, 6), datetime(2020, 1, 5, 10), "Air leak"),
)


def config(**overrides: object) -> experiment.ExperimentConfig:
    base: dict[str, object] = {
        "experiment": "unit",
        "splits": {
            "train": ["2020-01-01", "2020-01-03"],
            "calibration": ["2020-01-03", "2020-01-04"],
            "validation": ["2020-01-04", "2020-01-05"],
            "test": ["2020-01-05", "2020-01-06"],
        },
        "threshold_quantiles": [0.99, 0.999],
        "persistence_grid": [1, 3],
        "detectors": {
            "lps_rule": [{}],
            "isolation_forest": [{"n_estimators": 50}],
            "pca_spe": [{"variance": 0.9}, {"variance": 0.95}],
        },
    }
    base.update(overrides)
    return experiment.ExperimentConfig.model_validate(base)


def telemetry() -> pd.DataFrame:
    parts = [
        simulate_apu("2020-01-01", 6 * 60 * 24 * 3, seed=1),
        simulate_apu(
            "2020-01-04",
            6 * 60 * 24,
            leak_from="2020-01-04 06:00",
            leak_to="2020-01-04 10:00",
            seed=2,
        ),
        simulate_apu(
            "2020-01-05",
            6 * 60 * 24,
            leak_from="2020-01-05 06:00",
            leak_to="2020-01-05 10:00",
            seed=3,
        ),
    ]
    return pd.concat(parts)


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], Path]:
    results, bundles, scored = experiment.run(config(), telemetry(), REPORTS)
    artifacts = tmp_path_factory.mktemp("artifacts")
    experiment.save(results, bundles, scored, artifacts, "unit-model")
    return results, artifacts


def test_protocol_detects_simulated_leaks_and_selects_on_validation(
    trained: tuple[dict[str, object], Path],
) -> None:
    results, _ = trained
    detectors = results["detectors"]
    assert isinstance(detectors, dict)
    assert set(detectors) == {"lps_rule", "isolation_forest", "pca_spe"}
    selected = detectors[results["selected_detector"]]
    assert selected["validation"]["events_detected"] == 1
    assert selected["test"]["events_total"] == 1
    assert results["splits"]["validation"]["reports"] == ["V1"]
    assert results["splits"]["test"]["reports"] == ["T1"]


def test_artifacts_are_saved_with_card_and_current_pointer(
    trained: tuple[dict[str, object], Path],
) -> None:
    results, artifacts = trained
    root = artifacts / "models" / "unit-model"
    current = (root / "CURRENT").read_text().strip()
    assert current.endswith(str(results["selected_detector"]))
    card = json.loads((root / current / "model_card.json").read_text())
    assert card["dataset"]["license"] == "CC BY 4.0"
    assert len(card["artifact_sha256"]) == 64
    run_dir = artifacts / "experiments" / "unit" / str(results["version"])
    assert (run_dir / "results.json").exists()
    assert (run_dir / "scores_test.parquet").exists()
    bundle, _ = load_bundle(artifacts / "models", "unit-model")
    assert bundle.version == current


def test_load_bundle_rejects_missing_and_tampered_models(
    trained: tuple[dict[str, object], Path], tmp_path: Path
) -> None:
    with pytest.raises(ModelUnavailableError, match="No current model"):
        load_bundle(tmp_path, "absent")
    _, artifacts = trained
    root = artifacts / "models" / "unit-model"
    current = (root / "CURRENT").read_text().strip()
    card_path = root / current / "model_card.json"
    original = card_path.read_text()
    card = json.loads(original)
    card["artifact_sha256"] = "0" * 64
    card_path.write_text(json.dumps(card))
    try:
        with pytest.raises(ModelUnavailableError, match="integrity"):
            load_bundle(artifacts / "models", "unit-model")
    finally:
        card_path.write_text(original)


def test_training_period_with_reports_is_rejected() -> None:
    bad = (FailureReport("X", datetime(2020, 1, 1, 6), datetime(2020, 1, 1, 7), "Air leak"),)
    with pytest.raises(ValueError, match="must not contain reported failures"):
        experiment.run(config(), telemetry(), bad)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "splits": {
                "train": ["2020-01-02", "2020-01-01"],
                "calibration": ["2020-01-03", "2020-01-04"],
                "validation": ["2020-01-04", "2020-01-05"],
                "test": ["2020-01-05", "2020-01-06"],
            }
        },
        {"threshold_quantiles": [1.5]},
        {"detectors": {"pca_spe": []}},
    ],
)
def test_config_validation(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        config(**overrides)


def test_repository_config_is_valid() -> None:
    from factorymind.anomaly.cli import DEFAULT_CONFIG

    cfg = experiment.ExperimentConfig.load(DEFAULT_CONFIG)
    assert cfg.experiment == "metropt3-anomaly"
    assert len(cfg.fingerprint()) == 10
