"""Tests for WP-C2 (M3-WP1): Multi-Model ML Engine, Uncertainty/OOD/Abstention & AI4I 2020."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from factorymind.contracts.ml_engine import ABSTENTION_MESSAGE
from factorymind.data import ai4i2020
from factorymind.ml.ai4i_benchmark import save_ai4i_artifacts, train_and_evaluate_ai4i
from factorymind.ml.orchestrator import (
    BenchmarkUnavailableError,
    MultiModelOrchestrator,
    load_verified_ai4i_bundle,
)
from factorymind.ml.uncertainty import MahalanobisOodDetector, expected_calibration_error
from tests.auth_helpers import AuthKit, make_client


def _synthetic_ai4i_frame(n: int = 360, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air_t = rng.normal(300.0, 2.0, n)
    proc_t = air_t + 10.0 + rng.normal(0.0, 0.5, n)
    rpm = rng.normal(1520.0, 140.0, n).clip(1150.0, 2800.0)
    torque = rng.normal(40.0, 9.0, n).clip(5.0, 75.0)
    wear = rng.uniform(0.0, 240.0, n)

    hdf = ((proc_t - air_t < 9.1) & (rpm < 1450.0)).astype("int64")
    osf = ((torque * wear) > 11000.0).astype("int64")
    pwf = ((rpm * torque) > 95000.0).astype("int64")
    twf = (wear > 225.0).astype("int64")
    failure = ((hdf | osf | pwf | twf) > 0).astype("int64")

    return pd.DataFrame(
        {
            "UDI": np.arange(1, n + 1),
            "Product ID": [f"{t}{i:05d}" for i, t in enumerate(types, start=1)],
            "Type": types,
            "Air temperature [K]": air_t,
            "Process temperature [K]": proc_t,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
            "Machine failure": failure,
            "TWF": twf,
            "HDF": hdf,
            "PWF": pwf,
            "OSF": osf,
            "RNF": np.zeros(n, dtype="int64"),
        }
    )


def test_ai4i_feature_contract_excludes_leakage_columns() -> None:
    assert set(ai4i2020.FEATURE_COLUMNS).isdisjoint(set(ai4i2020.EXCLUDED_COLUMNS))
    assert "UDI" in ai4i2020.EXCLUDED_COLUMNS
    assert "Machine failure" in ai4i2020.EXCLUDED_COLUMNS
    assert "TWF" in ai4i2020.EXCLUDED_COLUMNS
    assert "RNF" in ai4i2020.EXCLUDED_COLUMNS

    df = _synthetic_ai4i_frame(120)
    norm = ai4i2020.normalize(df)
    report = ai4i2020.assess_quality(norm)
    ai4i2020.validate_contract(report, expected_rows=120)
    assert report.null_values == 0
    assert report.invalid_types == 0


def test_mahalanobis_ood_and_ece_behave_monotonically() -> None:
    rng = np.random.default_rng(1)
    train = rng.normal(0.0, 1.0, size=(200, 5))
    det = MahalanobisOodDetector.fit(train, quantile=0.99)
    in_flag, in_dist = det.is_ood(np.zeros((1, 5)))
    assert in_flag is False
    assert in_dist is not None and in_dist < det.threshold

    out_flag, out_dist = det.is_ood(np.full((1, 5), 10.0))
    assert out_flag is True
    assert out_dist is not None and out_dist > det.threshold

    nan_flag, nan_dist = det.is_ood(np.array([[np.nan, 0.0, 0.0, 0.0, 0.0]]))
    assert nan_flag is True
    assert nan_dist is None

    ece = expected_calibration_error(np.array([0, 0, 1, 1]), np.array([0.05, 0.1, 0.9, 0.95]))
    assert 0.0 <= ece <= 0.1


def test_orchestrator_abstention_gates_and_bundle_integrity(tmp_path: Path) -> None:
    frame = _synthetic_ai4i_frame(360)
    bundle, report = train_and_evaluate_ai4i(frame, random_state=42)
    assert [m.model_id for m in report.models] == ["ai4i-logreg", "ai4i-rf", "ai4i-hgb"]
    assert report.leakage_free_features == list(ai4i2020.FEATURE_COLUMNS)

    bench_dir = save_ai4i_artifacts(bundle, report, tmp_path / "artifacts")
    loaded = load_verified_ai4i_bundle(tmp_path / "artifacts")
    orch = MultiModelOrchestrator(loaded)

    # Nominal in-distribution point succeeds with confident prediction
    nom = orch.predict(
        asset_id="ai4i-milling-bench",
        product_type="M",
        air_temperature_k=300.0,
        process_temperature_k=310.5,
        rotational_speed_rpm=1550.0,
        torque_nm=35.0,
        tool_wear_min=50.0,
    )
    assert nom.abstained is False
    assert nom.prediction_label == "nominal"
    assert nom.provenance_origin == "model_prediction"
    assert nom.advisory_only is True

    # Real synthetic-distribution failure row (inside Mahalanobis support) predicts failure_risk
    fail_rows = frame[frame["Machine failure"] == 1]
    chosen_fail = None
    for _, f_row in fail_rows.iterrows():
        candidate_res = orch.predict(
            asset_id="ai4i-milling-bench",
            product_type=str(f_row["Type"]),
            air_temperature_k=float(f_row["Air temperature [K]"]),
            process_temperature_k=float(f_row["Process temperature [K]"]),
            rotational_speed_rpm=float(f_row["Rotational speed [rpm]"]),
            torque_nm=float(f_row["Torque [Nm]"]),
            tool_wear_min=float(f_row["Tool wear [min]"]),
        )
        if not candidate_res.abstained and candidate_res.prediction_label == "failure_risk":
            chosen_fail = candidate_res
            break
    assert chosen_fail is not None
    assert chosen_fail.abstained is False
    assert chosen_fail.prediction_label == "failure_risk"
    assert len(chosen_fail.predicted_failure_modes) >= 1

    # 1) Cross-asset applicability mismatch -> abstain with exact safety wording
    cross = orch.predict(
        asset_id="metropt3-apu-1",
        product_type="M",
        air_temperature_k=300.0,
        process_temperature_k=310.0,
        rotational_speed_rpm=1500.0,
        torque_nm=40.0,
        tool_wear_min=50.0,
    )
    assert cross.abstained is True
    assert cross.abstention_message == ABSTENTION_MESSAGE
    assert "asset_mismatch" in cross.abstention_reasons

    # 2) Degraded data quality -> abstain
    degraded = orch.predict(
        asset_id="ai4i-milling-bench",
        product_type="M",
        air_temperature_k=300.0,
        process_temperature_k=310.0,
        rotational_speed_rpm=1500.0,
        torque_nm=40.0,
        tool_wear_min=50.0,
        quality_status="degraded",
    )
    assert degraded.abstained is True
    assert degraded.abstention_message == ABSTENTION_MESSAGE
    assert degraded.prediction_label is None

    # 3) Corrupt / NaN / out-of-range input -> abstain
    corrupt = orch.predict(
        asset_id="ai4i-milling-bench",
        product_type="M",
        air_temperature_k=float("nan"),
        process_temperature_k=310.0,
        rotational_speed_rpm=1500.0,
        torque_nm=40.0,
        tool_wear_min=50.0,
    )
    assert corrupt.abstained is True
    assert corrupt.abstention_message == ABSTENTION_MESSAGE

    # 4) In-range Mahalanobis OOD thermal inversion -> abstain with is_ood=True
    ood = orch.predict(
        asset_id="ai4i-milling-bench",
        product_type="M",
        air_temperature_k=335.0,
        process_temperature_k=280.0,
        rotational_speed_rpm=1550.0,
        torque_nm=40.0,
        tool_wear_min=60.0,
    )
    assert ood.abstained is True
    assert ood.uncertainty.is_ood is True
    assert ood.uncertainty.uncertainty_band == "abstained"
    assert ood.abstention_message == ABSTENTION_MESSAGE

    # 5) Ambiguous near-threshold prediction -> abstains with low_margin_ambiguous & band="high"
    margin_hit = False
    for rpm in np.linspace(1160.0, 2800.0, 25):
        for torque in np.linspace(30.0, 75.0, 15):
            for wear in (50.0, 180.0, 220.0):
                probe = orch.predict(
                    asset_id="ai4i-milling-bench",
                    product_type="M",
                    air_temperature_k=300.0,
                    process_temperature_k=310.0,
                    rotational_speed_rpm=float(rpm),
                    torque_nm=float(torque),
                    tool_wear_min=float(wear),
                )
                if "low_margin_ambiguous" in probe.abstention_reasons:
                    assert probe.abstained is True
                    assert probe.abstention_message == ABSTENTION_MESSAGE
                    assert probe.uncertainty.uncertainty_band == "high"
                    assert probe.prediction_label is None
                    assert probe.calibrated_failure_probability is not None
                    margin_hit = True
                    break
            if margin_hit:
                break
        if margin_hit:
            break
    if not margin_hit:
        # Directly verify low_margin_ambiguous behavior when candidate sits at threshold
        class _BoundaryCandidate:
            model_id = "boundary"
            model_family = "stub"
            threshold = 0.5

            def predict_proba(self, transformed_x: np.ndarray) -> np.ndarray:
                return np.array([0.49], dtype="float64")

        stub_bundle = loaded.__class__(
            model_version=loaded.model_version,
            asset_id=loaded.asset_id,
            dataset_id=loaded.dataset_id,
            feature_columns=loaded.feature_columns,
            numeric_features=loaded.numeric_features,
            preprocessor=loaded.preprocessor,
            numeric_scaler=loaded.numeric_scaler,
            ood_detector=loaded.ood_detector,
            selected_candidate=_BoundaryCandidate(),  # type: ignore[arg-type]
            candidates=loaded.candidates,
            mode_heads=loaded.mode_heads,
        )
        b_probe = MultiModelOrchestrator(stub_bundle).predict(
            asset_id="ai4i-milling-bench",
            product_type="M",
            air_temperature_k=300.0,
            process_temperature_k=310.5,
            rotational_speed_rpm=1550.0,
            torque_nm=35.0,
            tool_wear_min=50.0,
        )
        assert b_probe.abstained is True
        assert b_probe.abstention_reasons == ["low_margin_ambiguous"]
        assert b_probe.abstention_message == ABSTENTION_MESSAGE
        assert b_probe.uncertainty.uncertainty_band == "high"

    # 6) Tampered joblib artifact fails SHA-256 attestation before loading
    (bench_dir / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(BenchmarkUnavailableError, match="SHA-256"):
        load_verified_ai4i_bundle(tmp_path / "artifacts")


def test_ml_endpoints_rbac_and_serving(kit: AuthKit) -> None:
    client = make_client(kit)
    viewer = kit.headers(roles=["viewer"])

    # Benchmark endpoints fail cleanly with 503 when not prepared
    assert client.get("/api/v1/ml/benchmarks/ai4i2020", headers=viewer).status_code == 503

    # Persist real evaluated benchmark into kit's artifacts_dir
    frame = _synthetic_ai4i_frame(360)
    bundle, report = train_and_evaluate_ai4i(frame, random_state=42)
    save_ai4i_artifacts(bundle, report, kit.root / "artifacts")

    cat = client.get("/api/v1/ml/catalog", headers=viewer)
    assert cat.status_code == 200
    assert cat.json()["ai4i_benchmark_available"] is True
    assert "separate" in cat.json()["separation_disclaimer"].lower()

    bench = client.get("/api/v1/ml/benchmarks/ai4i2020", headers=viewer)
    assert bench.status_code == 200
    assert bench.json()["dataset_id"] == "ai4i2020"
    assert len(bench.json()["models"]) == 3

    pred = client.post(
        "/api/v1/ml/predict",
        json={
            "asset_id": "ai4i-milling-bench",
            "product_type": "M",
            "air_temperature_k": 300.0,
            "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1520.0,
            "torque_nm": 40.0,
            "tool_wear_min": 40.0,
            "quality_status": "degraded",
        },
        headers=viewer,
    )
    assert pred.status_code == 200
    assert pred.json()["abstained"] is True
    assert pred.json()["abstention_message"] == ABSTENTION_MESSAGE

    # Cross-tenant and auditor roles fail with 403
    assert (
        client.get(
            "/api/v1/ml/catalog", headers=kit.headers(tenant="beta", roles=["viewer"])
        ).status_code
        == 403
    )
    assert (
        client.get("/api/v1/ml/catalog", headers=kit.headers(roles=["auditor"])).status_code == 403
    )
