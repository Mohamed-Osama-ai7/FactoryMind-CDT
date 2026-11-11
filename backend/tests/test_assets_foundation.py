"""Tests for WP-C1 (M3-WP0): Asset Digital Passport and non-destructive Data Foundation."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from factorymind.assets.passport import get_asset_passport, read_active_model_version
from factorymind.data.foundation import inspect_metropt3_dataset, verify_dataset_integrity
from factorymind.data.metropt3 import (
    ANALOG_SENSORS,
    ARCHIVE_SHA256,
    DIGITAL_SENSORS,
    SENSORS,
    VALID_RANGES,
    assess_quality,
    clean,
)
from tests.auth_helpers import ASSET, AuthKit, make_client
from tests.conftest import simulate_apu


def _write_processed_metropt3(root: Path, *, rows: int = 36, drop_col: str | None = None) -> Path:
    data_dir = root / "data"
    raw_zip = data_dir / "raw" / "metropt3" / "metropt3.zip"
    raw_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw_zip, "w") as zf:
        zf.writestr("placeholder.txt", "raw telemetry backup preserved verbatim")

    out_dir = data_dir / "processed" / "metropt3"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = clean(simulate_apu("2020-01-01T00:00:00", periods=rows))
    quality = assess_quality(frame)
    parquet_frame = frame.drop(columns=[drop_col]) if drop_col else frame
    parquet_frame.to_parquet(out_dir / "telemetry.parquet", engine="pyarrow")
    manifest = {
        "dataset_id": "metropt3",
        "archive_sha256": ARCHIVE_SHA256,
        "raw_quality": quality.to_dict(),
        "rows_after_cleaning": len(frame),
        "failure_reports": [{"id": "F1"}, {"id": "F2"}, {"id": "F3"}, {"id": "F4"}],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return data_dir


def test_metropt3_passport_matches_canonical_sensor_contract() -> None:
    passport = get_asset_passport("metropt3-apu-1")
    assert passport is not None
    assert passport.safety_boundary == "advisory-read-only"
    assert [s.sensor_id for s in passport.sensors] == list(SENSORS)

    by_id = {s.sensor_id: s for s in passport.sensors}
    for name, unit in ANALOG_SENSORS.items():
        assert by_id[name].kind == "analog"
        assert by_id[name].unit == unit
        assert by_id[name].valid_range == VALID_RANGES[name]
    for name in DIGITAL_SENSORS:
        assert by_id[name].kind == "digital"
        assert by_id[name].valid_range == (0.0, 1.0)

    assert "Downstream air-demand flowmeter" in " ".join(passport.unobserved_quantities)
    by_model = {m.model_id: m for m in passport.model_applicability}
    assert by_model["duty_rule"].status == "active"
    assert by_model["cross_asset_transfer"].status == "not_calibrated"


def test_data_foundation_detects_missing_corrupt_artifacts_non_destructively(
    tmp_path: Path,
) -> None:
    # Empty directory -> catalog_only without failing
    empty_rec = inspect_metropt3_dataset(tmp_path / "raw", tmp_path / "processed")
    assert empty_rec.artifact_status == "catalog_only"
    assert empty_rec.raw_preserved is False
    assert empty_rec.manifest_present is False
    assert empty_rec.telemetry_present is False

    # Partial / column-missing + raw hash mismatch detected honestly without mutating raw file
    data_dir = _write_processed_metropt3(tmp_path, drop_col="TP3")
    raw_zip = data_dir / "raw" / "metropt3" / "metropt3.zip"
    before_bytes = raw_zip.read_bytes()

    verified_rec = verify_dataset_integrity(data_dir / "raw", data_dir / "processed")
    assert raw_zip.read_bytes() == before_bytes  # raw source preserved untouched
    assert verified_rec.artifact_status == "partial"
    checks = {c.check_id: c for c in verified_rec.quality_checks}
    assert checks["raw_source_preserved"].status == "warn"
    assert "mismatch" in checks["raw_source_preserved"].summary
    assert checks["parquet_schema_and_rows"].status == "warn"
    assert "TP3" in checks["parquet_schema_and_rows"].summary

    # Valid complete schema + metadata verifies parquet & manifest (non-raw-hash mode)
    data_ok = _write_processed_metropt3(tmp_path / "ok")
    ok_rec = inspect_metropt3_dataset(data_ok / "raw", data_ok / "processed", verify_raw_hash=False)
    assert ok_rec.artifact_status == "verified"
    assert ok_rec.quality is not None
    assert ok_rec.quality.rows_cleaned == 36
    assert ok_rec.quality.gap_policy == "preserve_gaps_uninterpolated"


def test_malformed_manifest_and_corrupt_parquet_degrade_gracefully(tmp_path: Path) -> None:
    out_dir = tmp_path / "processed" / "metropt3"
    out_dir.mkdir(parents=True)
    (out_dir / "manifest.json").write_text("{broken json", encoding="utf-8")
    rec = inspect_metropt3_dataset(tmp_path / "raw", tmp_path / "processed")
    assert rec.artifact_status == "partial"
    checks = {c.check_id: c for c in rec.quality_checks}
    assert checks["manifest_provenance"].status == "warn"

    # Negative counts in manifest.json must NOT crash DatasetQualitySummary with 500
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "metropt3",
                "archive_sha256": ARCHIVE_SHA256,
                "rows_after_cleaning": 10,
                "raw_quality": {
                    "rows": -5,
                    "start": "2020-01-01 00:00:00",
                    "end": "2020-01-01 00:05:00",
                    "duplicate_timestamps": -1,
                    "null_values": 0,
                    "gaps_over_60s": 0,
                    "gap_hours": -1.0,
                },
            }
        ),
        encoding="utf-8",
    )
    neg_rec = inspect_metropt3_dataset(tmp_path / "raw", tmp_path / "processed")
    assert neg_rec.quality is None
    assert {c.check_id: c for c in neg_rec.quality_checks}["manifest_provenance"].status == "warn"

    # 0-row Parquet flags "0 rows" and never claims telemetry_available=True
    data_zero = _write_processed_metropt3(tmp_path / "zero", rows=36)
    empty_frame = clean(simulate_apu("2020-01-01T00:00:00", periods=5)).iloc[0:0]
    empty_frame.to_parquet(data_zero / "processed" / "metropt3" / "telemetry.parquet")
    zero_rec = inspect_metropt3_dataset(data_zero / "raw", data_zero / "processed")
    assert zero_rec.artifact_status == "partial"
    zero_pq = {c.check_id: c for c in zero_rec.quality_checks}["parquet_schema_and_rows"]
    assert zero_pq.status == "warn"
    assert "0 rows" in zero_pq.summary


def test_read_active_model_version_blocks_traversal_and_bad_bytes(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_root = models_dir / "metropt3-anomaly"
    ver_dir = model_root / "20261111-v1"
    ver_dir.mkdir(parents=True)
    (ver_dir / "model_card.json").write_text(
        json.dumps({"model_version": "20261111-v1"}), encoding="utf-8"
    )
    (model_root / "CURRENT").write_text("../escape", encoding="utf-8")
    assert read_active_model_version(models_dir, "metropt3-anomaly") is None

    # Embedded NUL byte or invalid UTF-8 bytes fail closed to None rather than raising 500
    (model_root / "CURRENT").write_text("v1\x00bad", encoding="utf-8")
    assert read_active_model_version(models_dir, "metropt3-anomaly") is None
    (model_root / "CURRENT").write_bytes(b"\xff\xfe\xfd")
    assert read_active_model_version(models_dir, "metropt3-anomaly") is None

    (model_root / "CURRENT").write_text("20261111-v1", encoding="utf-8")
    assert read_active_model_version(models_dir, "metropt3-anomaly") == "20261111-v1"


def test_assets_and_datasets_api_scoping_and_passport_response(kit: AuthKit) -> None:
    _write_processed_metropt3(kit.root)
    ver_dir = kit.root / "artifacts" / "models" / "metropt3-anomaly" / "20261111-abc"
    ver_dir.mkdir(parents=True)
    (ver_dir / "model_card.json").write_text(
        json.dumps({"model_version": "20261111-abc"}), encoding="utf-8"
    )
    (ver_dir.parent / "CURRENT").write_text("20261111-abc", encoding="utf-8")

    client = make_client(kit, asset_ids=[ASSET, "pump-7"])
    headers = kit.headers(roles=["viewer"], assets=[ASSET])

    listed = client.get("/api/v1/assets", headers=headers)
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert len(items) == 1
    assert items[0]["asset_id"] == ASSET
    assert items[0]["sensor_count"] == 15
    assert items[0]["telemetry_available"] is True
    assert items[0]["runtime_model_version"] == "20261111-abc"

    detail = client.get(f"/api/v1/assets/{ASSET}", headers=headers)
    assert detail.status_code == 200
    body = detail.json()
    assert body["passport"]["asset_id"] == ASSET
    assert len(body["passport"]["sensors"]) == 15
    assert "42 window features" in body["passport"]["model_applicability"][1]["scope_note"]
    assert body["datasets"][0]["dataset_id"] == "metropt3"
    assert body["datasets"][0]["artifact_status"] == "verified"

    datasets = client.get(f"/api/v1/datasets?asset_id={ASSET}", headers=headers)
    assert datasets.status_code == 200
    assert datasets.json()["items"][0]["archive_sha256"] == ARCHIVE_SHA256

    # Configured-but-ungranted or unknown asset fails with 404 (not probing existence)
    assert client.get("/api/v1/assets/pump-7", headers=headers).status_code == 404
    assert client.get("/api/v1/assets/pump-99", headers=headers).status_code == 404
    assert client.get("/api/v1/datasets?asset_id=pump-7", headers=headers).status_code == 404
    assert client.get(f"/api/v1/datasets?asset_id={'a' * 80}", headers=headers).status_code == 404

    # Caller lacking data_source_asset_id grant fails with 403
    assert (
        client.get("/api/v1/assets", headers=kit.headers(roles=["viewer"], assets=[])).status_code
        == 403
    )

    # Cross-tenant and auditor roles fail with 403
    assert (
        client.get(
            "/api/v1/assets", headers=kit.headers(tenant="beta", roles=["viewer"])
        ).status_code
        == 403
    )
    assert client.get("/api/v1/assets", headers=kit.headers(roles=["auditor"])).status_code == 403


def test_corrupt_parquet_reports_telemetry_available_false_in_summary(kit: AuthKit) -> None:
    _write_processed_metropt3(kit.root, rows=36)
    empty_frame = clean(simulate_apu("2020-01-01T00:00:00", periods=5)).iloc[0:0]
    empty_frame.to_parquet(kit.root / "data" / "processed" / "metropt3" / "telemetry.parquet")

    client = make_client(kit)
    resp = client.get("/api/v1/assets", headers=kit.headers(roles=["viewer"], assets=[ASSET]))
    assert resp.status_code == 200
    assert resp.json()["items"][0]["telemetry_available"] is False
