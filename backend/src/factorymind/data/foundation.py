"""Non-destructive Data Foundation: dataset provenance, lineage, and structural quality checks.

Inspects raw archives, manifest provenance, and normalized Parquet files strictly in read-only
mode without mutating source archives or interpolating telemetry gaps.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from factorymind.contracts.assets import (
    DataLineageStep,
    DatasetQualitySummary,
    QualityCheckItem,
    SourceDatasetRecord,
)
from factorymind.data.metropt3 import (
    ARCHIVE_SHA256,
    DATASET_ID,
    EXPECTED_ROWS,
    FAILURE_REPORTS,
    LICENSE,
    SENSORS,
    SOURCE_PAGE,
    SOURCE_URL,
    sha256_file,
)

_METROPT3_LINEAGE: tuple[DataLineageStep, ...] = (
    DataLineageStep(
        step=1,
        operation="raw_acquisition",
        description=(
            "Preserve the verbatim UCI Machine Learning Repository archive "
            "(MetroPT3(AirCompressor).csv inside metropt3.zip) under raw/metropt3/ with "
            "SHA-256 verification before downstream reads."
        ),
        preserves_raw_source=True,
    ),
    DataLineageStep(
        step=2,
        operation="contract_normalization",
        description=(
            "Parse naive local timestamps (%Y-%m-%d %H:%M:%S), cast 7 analog and 8 digital "
            "channels to float64, and drop the redundant source index column."
        ),
        preserves_raw_source=True,
    ),
    DataLineageStep(
        step=3,
        operation="quality_validation_and_cleaning",
        description=(
            "Sort chronologically, deduplicate first timestamps, null out physically impossible "
            "analog readings or non-binary digital states, and preserve telemetry gaps "
            "uninterpolated (preserve_gaps_uninterpolated)."
        ),
        preserves_raw_source=True,
    ),
    DataLineageStep(
        step=4,
        operation="parquet_materialization",
        description=(
            "Persist cleaned telemetry as telemetry.parquet with machine-readable manifest.json "
            "containing raw_quality statistics and operator failure report boundaries."
        ),
        preserves_raw_source=True,
    ),
)


def _as_int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _as_float(value: Any, default: float | None = None) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        fval = float(value)
        return fval if math.isfinite(fval) else default
    return default


def _as_count_map(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    return {
        str(k): v
        for k, v in value.items()
        if isinstance(k, str) and isinstance(v, int) and not isinstance(v, bool)
    }


def _parse_manifest_quality(manifest: dict[str, Any]) -> DatasetQualitySummary | None:
    raw = manifest.get("raw_quality")
    if not isinstance(raw, dict):
        return None
    rows_cleaned = _as_int(manifest.get("rows_after_cleaning"), -1)
    if rows_cleaned < 0:
        return None
    start = raw.get("start")
    end = raw.get("end")
    if not isinstance(start, str) or not isinstance(end, str):
        return None
    return DatasetQualitySummary(
        rows_raw=_as_int(raw.get("rows")),
        rows_cleaned=rows_cleaned,
        start=start,
        end=end,
        duplicate_timestamps=_as_int(raw.get("duplicate_timestamps")),
        null_values=_as_int(raw.get("null_values")),
        out_of_range=_as_count_map(raw.get("out_of_range")),
        non_binary_digital=_as_count_map(raw.get("non_binary_digital")),
        median_period_s=_as_float(raw.get("median_period_s")),
        gaps_over_60s=_as_int(raw.get("gaps_over_60s")),
        gap_hours=_as_float(raw.get("gap_hours"), 0.0) or 0.0,
        gap_policy="preserve_gaps_uninterpolated",
    )


def inspect_metropt3_dataset(
    raw_dir: Path,
    processed_dir: Path,
    *,
    verify_raw_hash: bool = False,
) -> SourceDatasetRecord:
    """Inspect MetroPT-3 raw archive, manifest provenance, and Parquet contract cleanly."""
    raw_zip = raw_dir / DATASET_ID / "metropt3.zip"
    out_dir = processed_dir / DATASET_ID
    manifest_path = out_dir / "manifest.json"
    parquet_path = out_dir / "telemetry.parquet"

    raw_preserved = raw_zip.is_file()
    manifest_present = manifest_path.is_file()
    telemetry_present = parquet_path.is_file()
    checks: list[QualityCheckItem] = []

    if not raw_preserved:
        checks.append(
            QualityCheckItem(
                check_id="raw_source_preserved",
                status="unavailable",
                summary=f"Raw archive not found at {raw_zip}; processed-only view.",
            )
        )
    elif verify_raw_hash:
        actual = sha256_file(raw_zip)
        if actual == ARCHIVE_SHA256:
            checks.append(
                QualityCheckItem(
                    check_id="raw_source_preserved",
                    status="pass",
                    summary=f"Raw zip SHA-256 verified against {ARCHIVE_SHA256[:12]}...",
                )
            )
        else:
            checks.append(
                QualityCheckItem(
                    check_id="raw_source_preserved",
                    status="warn",
                    summary=f"Raw zip SHA-256 mismatch (got {actual[:12]}...).",
                )
            )
    else:
        checks.append(
            QualityCheckItem(
                check_id="raw_source_preserved",
                status="pass",
                summary="Raw archive present on disk and left untouched.",
            )
        )

    manifest_data: dict[str, Any] | None = None
    quality: DatasetQualitySummary | None = None
    if not manifest_present:
        checks.append(
            QualityCheckItem(
                check_id="manifest_provenance",
                status="unavailable",
                summary="Processed manifest.json not generated yet.",
            )
        )
    else:
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            loaded = None
        if not isinstance(loaded, dict):
            checks.append(
                QualityCheckItem(
                    check_id="manifest_provenance",
                    status="warn",
                    summary="manifest.json could not be parsed as a valid JSON object.",
                )
            )
        else:
            manifest_data = loaded
            quality = _parse_manifest_quality(manifest_data)
            sha_ok = manifest_data.get("archive_sha256") == ARCHIVE_SHA256
            id_ok = manifest_data.get("dataset_id") == DATASET_ID
            if sha_ok and id_ok and quality is not None:
                checks.append(
                    QualityCheckItem(
                        check_id="manifest_provenance",
                        status="pass",
                        summary=(
                            f"Manifest SHA-256 and contract verified "
                            f"({quality.rows_cleaned:,} cleaned rows)."
                        ),
                    )
                )
            else:
                checks.append(
                    QualityCheckItem(
                        check_id="manifest_provenance",
                        status="warn",
                        summary="Manifest metadata differs from documented MetroPT-3 contract.",
                    )
                )

    parquet_ok = False
    if not telemetry_present:
        checks.append(
            QualityCheckItem(
                check_id="parquet_schema_and_rows",
                status="unavailable",
                summary="Cleaned telemetry.parquet not generated yet.",
            )
        )
    else:
        try:
            meta = pq.read_metadata(parquet_path)
            schema = pq.read_schema(parquet_path)
            col_names = set(schema.names)
            missing_cols = [s for s in SENSORS if s not in col_names]
            rows_match = quality is None or meta.num_rows == quality.rows_cleaned
            if not missing_cols and rows_match and meta.num_rows > 0:
                parquet_ok = True
                checks.append(
                    QualityCheckItem(
                        check_id="parquet_schema_and_rows",
                        status="pass",
                        summary=(
                            f"Parquet footer verifies {meta.num_rows:,} rows and all "
                            f"{len(SENSORS)} canonical sensor columns."
                        ),
                    )
                )
            else:
                detail = (
                    f"missing columns {missing_cols}"
                    if missing_cols
                    else f"row count {meta.num_rows} differs from manifest"
                )
                checks.append(
                    QualityCheckItem(
                        check_id="parquet_schema_and_rows",
                        status="warn",
                        summary=f"Parquet contract check flagged issue: {detail}.",
                    )
                )
        except Exception:
            checks.append(
                QualityCheckItem(
                    check_id="parquet_schema_and_rows",
                    status="warn",
                    summary="Could not read Parquet metadata cleanly.",
                )
            )

    if quality is not None:
        gap_status: str = "warn" if quality.gaps_over_60s > 0 else "pass"
        checks.append(
            QualityCheckItem(
                check_id="temporal_continuity_honesty",
                status="warn" if gap_status == "warn" else "pass",
                summary=(
                    f"{quality.gaps_over_60s} gaps >60s ({quality.gap_hours} h total) "
                    f"preserved uninterpolated; median period {quality.median_period_s}s."
                ),
            )
        )
        checks.append(
            QualityCheckItem(
                check_id="range_and_contract_rows",
                status="pass" if quality.rows_raw == EXPECTED_ROWS else "warn",
                summary=(
                    f"Raw source rows={quality.rows_raw:,} (expected {EXPECTED_ROWS:,}), "
                    f"duplicates={quality.duplicate_timestamps}, nulls={quality.null_values}."
                ),
            )
        )

    all_critical_pass = (
        manifest_present
        and telemetry_present
        and parquet_ok
        and all(
            c.status == "pass"
            for c in checks
            if c.check_id in {"manifest_provenance", "parquet_schema_and_rows"}
        )
        and (
            not verify_raw_hash
            or any(c.check_id == "raw_source_preserved" and c.status == "pass" for c in checks)
        )
    )
    if all_critical_pass:
        artifact_status: str = "verified"
    elif manifest_present or telemetry_present or raw_preserved:
        artifact_status = "partial"
    else:
        artifact_status = "catalog_only"

    reports = manifest_data.get("failure_reports") if isinstance(manifest_data, dict) else None
    report_count = len(reports) if isinstance(reports, list) else len(FAILURE_REPORTS)

    return SourceDatasetRecord(
        dataset_id=DATASET_ID,
        title="MetroPT-3 Air Production Unit Compressor Telemetry",
        asset_ids=["metropt3-apu-1"],
        source_url=SOURCE_URL,
        source_page=SOURCE_PAGE,
        license=LICENSE,
        archive_sha256=ARCHIVE_SHA256,
        raw_preserved=raw_preserved,
        manifest_present=manifest_present,
        telemetry_present=telemetry_present,
        artifact_status="verified"
        if artifact_status == "verified"
        else ("partial" if artifact_status == "partial" else "catalog_only"),
        provenance_origin="observed",
        failure_report_count=report_count,
        quality=quality,
        quality_checks=checks,
        lineage=list(_METROPT3_LINEAGE),
    )


def verify_dataset_integrity(
    raw_dir: Path, processed_dir: Path, dataset_id: str = DATASET_ID
) -> SourceDatasetRecord:
    """Verify full dataset integrity including computing the raw archive SHA-256."""
    if dataset_id != DATASET_ID:
        raise ValueError(f"Unsupported dataset_id: {dataset_id!r}")
    return inspect_metropt3_dataset(raw_dir, processed_dir, verify_raw_hash=True)


def inspect_connected_datasets(
    raw_dir: Path,
    processed_dir: Path,
    dataset_ids: Sequence[str] | None = None,
    *,
    verify_raw_hash: bool = False,
) -> list[SourceDatasetRecord]:
    """Return data-foundation provenance records matching requested or registered dataset IDs."""
    records = [inspect_metropt3_dataset(raw_dir, processed_dir, verify_raw_hash=verify_raw_hash)]
    if dataset_ids is None:
        return records
    wanted = set(dataset_ids)
    return [r for r in records if r.dataset_id in wanted]
