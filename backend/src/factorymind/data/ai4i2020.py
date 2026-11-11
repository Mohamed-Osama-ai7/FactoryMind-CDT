"""AI4I 2020 Predictive Maintenance dataset adapter (controlled synthetic milling benchmark).

Source: UCI Machine Learning Repository, dataset 601 (CC BY 4.0).
Strictly separated from real MetroPT-3 operational compressor metrics: AI4I 2020 models a
synthetic milling benchmark across 10,000 samples (Type L/M/H + 5 physical variables + 5
failure indicator targets). Input features exclude UDI, Product ID, and all failure targets
(Machine failure, TWF, HDF, PWF, OSF, RNF) to guarantee zero target leakage.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from factorymind.data.metropt3 import DataValidationError, sha256_file

DATASET_ID: Final = "ai4i2020"
SOURCE_URL: Final = (
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
)
SOURCE_PAGE: Final = (
    "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"
)
LICENSE: Final = "CC BY 4.0"
ARCHIVE_SHA256: Final = "f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252"
CSV_MEMBER: Final = "ai4i2020.csv"
EXPECTED_ROWS: Final = 10_000

PRODUCT_TYPES: Final[tuple[str, ...]] = ("L", "M", "H")
NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
)
FEATURE_COLUMNS: Final[tuple[str, ...]] = ("Type", *NUMERIC_FEATURES)
TARGET_COLUMN: Final = "Machine failure"
MODE_COLUMNS: Final[tuple[str, ...]] = ("TWF", "HDF", "PWF", "OSF")
EXCLUDED_COLUMNS: Final[tuple[str, ...]] = (
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "UDI",
    "Product ID",
    *FEATURE_COLUMNS,
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
)

# Generous physical validity bounds detecting sensor/telemetry corruption before inference.
VALID_RANGES: Final[dict[str, tuple[float, float]]] = {
    "Air temperature [K]": (250.0, 350.0),
    "Process temperature [K]": (250.0, 360.0),
    "Rotational speed [rpm]": (500.0, 4000.0),
    "Torque [Nm]": (0.0, 120.0),
    "Tool wear [min]": (0.0, 400.0),
}


@dataclass(frozen=True)
class Ai4iQualityReport:
    rows: int
    null_values: int
    failure_count: int
    failure_rate: float
    invalid_types: int
    out_of_range: dict[str, int]
    mode_counts: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def download_archive(destination: Path) -> Path:
    """Download the official UCI #601 archive if missing and verify its SHA-256 checksum."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        tmp = destination.with_suffix(".part")
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response, tmp.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        tmp.replace(destination)
    actual = sha256_file(destination)
    if actual != ARCHIVE_SHA256:
        raise DataValidationError(
            f"Archive checksum mismatch for {destination}: expected {ARCHIVE_SHA256}, got {actual}"
        )
    return destination


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns and coerce numeric features and target flags cleanly."""
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise DataValidationError(f"Missing required AI4I columns: {missing}")
    out = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    out["Type"] = out["Type"].astype(str).str.strip()
    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    for col in (TARGET_COLUMN, *MODE_COLUMNS, "RNF"):
        parsed = pd.to_numeric(out[col], errors="coerce")
        if parsed.isna().any() or ((parsed != 0) & (parsed != 1)).any():
            raise DataValidationError(f"Column {col} must contain binary 0/1 values without nulls")
        out[col] = parsed.astype("int64")
    return out


def read_raw(archive: Path) -> pd.DataFrame:
    with zipfile.ZipFile(archive) as zf, zf.open(CSV_MEMBER) as raw:
        frame = pd.read_csv(io.TextIOWrapper(raw, encoding="utf-8"))
    return normalize(frame)


def assess_quality(frame: pd.DataFrame) -> Ai4iQualityReport:
    out_of_range = {
        col: int(((frame[col] < lo) | (frame[col] > hi)).sum())
        for col, (lo, hi) in VALID_RANGES.items()
    }
    invalid_types = int((~frame["Type"].isin(PRODUCT_TYPES)).sum())
    n_rows = len(frame)
    failures = int((frame[TARGET_COLUMN] == 1).sum())
    return Ai4iQualityReport(
        rows=n_rows,
        null_values=int(frame.isna().sum().sum()),
        failure_count=failures,
        failure_rate=round(failures / n_rows, 6) if n_rows else 0.0,
        invalid_types=invalid_types,
        out_of_range={k: v for k, v in out_of_range.items() if v},
        mode_counts={m: int((frame[m] == 1).sum()) for m in (*MODE_COLUMNS, "RNF")},
    )


def validate_contract(
    report: Ai4iQualityReport, *, expected_rows: int | None = EXPECTED_ROWS
) -> None:
    problems: list[str] = []
    if expected_rows is not None and report.rows != expected_rows:
        problems.append(f"row count {report.rows} != expected {expected_rows}")
    if report.null_values:
        problems.append(f"{report.null_values} null values")
    if report.invalid_types:
        problems.append(f"{report.invalid_types} invalid product types")
    if problems:
        raise DataValidationError("; ".join(problems))


def prepare(raw_dir: Path, processed_dir: Path) -> dict[str, object]:
    """Download/verify raw archive and store normalized Parquet without touching raw zip."""
    archive = download_archive(raw_dir / DATASET_ID / "ai4i2020.zip")
    frame = read_raw(archive)
    report = assess_quality(frame)
    validate_contract(report)
    out_dir = processed_dir / DATASET_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "data.parquet", engine="pyarrow")
    manifest: dict[str, object] = {
        "dataset_id": DATASET_ID,
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "license": LICENSE,
        "archive_sha256": ARCHIVE_SHA256,
        "leakage_free_features": list(FEATURE_COLUMNS),
        "excluded_columns": list(EXCLUDED_COLUMNS),
        "quality": report.to_dict(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_processed(processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / DATASET_ID / "data.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; prepare AI4I 2020 first")
    return pd.read_parquet(path, engine="pyarrow")
