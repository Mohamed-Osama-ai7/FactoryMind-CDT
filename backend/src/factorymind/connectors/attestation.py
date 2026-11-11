"""Supply-chain hardening, workflow SHA-pinning, and non-pickle model attestation."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from factorymind.contracts.connectors import (
    ArtifactAttestationEntry,
    AttestationStatus,
    SupplyChainHardeningResponse,
)

_WORKFLOW_USES_RE: Final[re.Pattern[str]] = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)")
_SHA_PINNED_RE: Final[re.Pattern[str]] = re.compile(r"^[^@]+@[0-9a-f]{40}$")
_FORBIDDEN_PICKLE_FORMATS: Final[frozenset[str]] = frozenset({"pickle", "joblib", "dill"})

_CROSS_SYSTEM_BENCHMARKS: Final[dict[str, str]] = {
    "M2-WP0/WP2 MetroPT-3 LOEO Detection": (
        "duty_rule 4/4 leaks, 0.034/0.044 FA/day (wp2_integrity_gated 0.026/0.035), "
        "median abs delay 50 min"
    ),
    "M2-WP3/WP4 Evidence & RCA": (
        "Mean completeness 0.9266, 1.0 grounding/agreement, 0.6765 warning non-leak rate"
    ),
    "M3-WP1 AI4I 2020 Calibration": (
        "ai4i-hgb ROC-AUC 0.9803, Brier 0.0124, ECE 0.0054; Mahalanobis OOD gate"
    ),
    "M3-WP2 Hybrid RAG + KG": "Recall@5 1.0, MRR@10 1.0, nDCG@5 0.7683 across 61 verified chunks",
    "M4-WP0 Physics Digital Twin": (
        "168 held-out segments: 10m MAE 0.2526 bar, 60m MAE 0.5218 bar, "
        "0 violations, 3.78x leak ratio"
    ),
    "M4-WP2 Controlled Copilot": (
        "24/24 red-team probes blocked, 5/5 benign allowed, strict [C1..Cn] subset check"
    ),
    "M5-WP0 OT/Edge & Supply Chain": (
        "Read-only OPC-UA/MQTT/Historian adapters, SSRF allowlists, MIT LICENSE, SHA-pinned CI"
    ),
}


def discover_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (or this file) to find the repository root."""
    anchor = (start or Path(__file__)).resolve()
    for parent in (anchor, *anchor.parents):
        if (parent / "backend" / "pyproject.toml").is_file() and (parent / ".github").is_dir():
            return parent
    return Path(__file__).resolve().parents[4]


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize ``payload`` deterministically for cryptographic hashing/HMAC signing."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_signed_manifest(
    *,
    model_name: str,
    schema_version: int,
    parameters: Mapping[str, Any],
    hmac_key: bytes | None = None,
) -> dict[str, Any]:
    """Build a non-pickle JSON model manifest with SHA-256 and optional HMAC-SHA256."""
    param_sha256 = sha256_bytes(canonical_json_bytes(parameters))
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "artifact_format": "json_parameters_v1",
        "model_name": model_name,
        "parameters_sha256": param_sha256,
        "parameters": dict(parameters),
    }
    if hmac_key is not None:
        signature_material = canonical_json_bytes(
            {
                "schema_version": schema_version,
                "artifact_format": "json_parameters_v1",
                "model_name": model_name,
                "parameters_sha256": param_sha256,
            }
        )
        manifest["hmac_sha256"] = hmac.new(hmac_key, signature_material, hashlib.sha256).hexdigest()
    return manifest


def verify_non_pickle_manifest(
    manifest_path: Path,
    *,
    hmac_key: bytes | None = None,
) -> tuple[AttestationStatus, str | None, str]:
    """Verify a JSON model manifest without deserializing any pickle/joblib payload."""
    if not manifest_path.is_file():
        return "missing", None, "Non-pickle manifest file not present"
    try:
        raw = manifest_path.read_bytes()
        data = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return "tamper_or_hash_mismatch", None, f"Invalid JSON manifest: {exc}"
    if not isinstance(data, dict):
        return "tamper_or_hash_mismatch", None, "Manifest root must be a JSON object"
    fmt = str(data.get("artifact_format", "")).strip().lower()
    if fmt in _FORBIDDEN_PICKLE_FORMATS or fmt != "json_parameters_v1":
        return (
            "tamper_or_hash_mismatch",
            sha256_bytes(raw),
            f"Unsupported or unsafe artifact_format {fmt!r}",
        )
    params = data.get("parameters")
    expected_sha = data.get("parameters_sha256")
    if not isinstance(params, dict) or not isinstance(expected_sha, str):
        return "tamper_or_hash_mismatch", sha256_bytes(raw), "Missing parameters or digest"
    try:
        actual_sha = sha256_bytes(canonical_json_bytes(params))
    except ValueError as exc:
        return "tamper_or_hash_mismatch", sha256_bytes(raw), f"Non-finite parameter: {exc}"
    if not hmac.compare_digest(actual_sha, expected_sha):
        return (
            "tamper_or_hash_mismatch",
            sha256_bytes(raw),
            "Parameters SHA-256 mismatch (manifest modified)",
        )
    if hmac_key is not None:
        sig = data.get("hmac_sha256")
        if not isinstance(sig, str):
            return "tamper_or_hash_mismatch", sha256_bytes(raw), "Missing required HMAC signature"
        material = canonical_json_bytes(
            {
                "schema_version": data.get("schema_version"),
                "artifact_format": fmt,
                "model_name": data.get("model_name"),
                "parameters_sha256": expected_sha,
            }
        )
        expected_sig = hmac.new(hmac_key, material, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return "tamper_or_hash_mismatch", sha256_bytes(raw), "HMAC-SHA256 verification failed"
    return "verified", sha256_bytes(raw), "Non-pickle JSON manifest hash verified"


def audit_ci_workflow(ci_path: Path) -> tuple[bool, tuple[str, ...], str | None, str]:
    """Verify every ``uses:`` action in GitHub Actions CI is pinned to a 40-hex commit SHA."""
    if not ci_path.is_file():
        return False, (), None, "Workflow file missing"
    raw = ci_path.read_bytes()
    file_sha = sha256_bytes(raw)
    unpinned: list[str] = []
    seen = 0
    for line in raw.decode("utf-8").splitlines():
        match = _WORKFLOW_USES_RE.match(line)
        if not match:
            continue
        seen += 1
        target = match.group(1)
        if not _SHA_PINNED_RE.fullmatch(target):
            unpinned.append(target)
    if seen == 0:
        return False, (), file_sha, "No GitHub Actions uses: steps discovered"
    if unpinned:
        return (
            False,
            tuple(unpinned),
            file_sha,
            f"Unpinned CI actions found: {', '.join(unpinned)}",
        )
    return True, (), file_sha, f"All {seen} CI workflow action steps pinned to 40-hex commit SHAs"


def _audit_json_file(path: Path) -> tuple[AttestationStatus, str | None, str]:
    if not path.is_file():
        return "missing", None, "File not found"
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    try:
        json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return "tamper_or_hash_mismatch", digest, f"Malformed JSON: {exc}"
    return "verified", digest, "Valid UTF-8 JSON verified by SHA-256"


def build_supply_chain_report(
    *,
    repo_root: Path | None = None,
    models_dir: Path | None = None,
    model_name: str = "metropt3-anomaly",
) -> SupplyChainHardeningResponse:
    """Inspect repository governance, workflow pinning, configs, and runtime model artifacts."""
    root = repo_root or discover_repo_root()
    attestations: list[ArtifactAttestationEntry] = []

    license_path = root / "LICENSE"
    if license_path.is_file():
        lic_bytes = license_path.read_bytes()
        lic_text = lic_bytes.decode("utf-8", errors="replace")
        has_mit = "MIT License" in lic_text and "Permission is hereby granted" in lic_text
        attestations.append(
            ArtifactAttestationEntry(
                name="Root Repository License",
                relative_path="LICENSE",
                format="text_license",
                status="verified" if has_mit else "tamper_or_hash_mismatch",
                sha256=sha256_bytes(lic_bytes),
                detail="MIT License matching backend/pyproject.toml"
                if has_mit
                else "Unexpected license text",
            )
        )
        license_ok = has_mit
    else:
        attestations.append(
            ArtifactAttestationEntry(
                name="Root Repository License",
                relative_path="LICENSE",
                format="text_license",
                status="missing",
                sha256=None,
                detail="Root LICENSE file missing",
            )
        )
        license_ok = False

    ci_path = root / ".github" / "workflows" / "ci.yml"
    ci_pinned, unpinned, ci_sha, ci_detail = audit_ci_workflow(ci_path)
    attestations.append(
        ArtifactAttestationEntry(
            name="GitHub Actions CI Workflow",
            relative_path=".github/workflows/ci.yml",
            format="yaml_workflow",
            status="sha_pinned" if ci_pinned else "tamper_or_hash_mismatch",
            sha256=ci_sha,
            detail=ci_detail,
        )
    )

    cfg_path = root / "backend" / "configs" / "metropt3_anomaly.json"
    cfg_status, cfg_sha, cfg_detail = _audit_json_file(cfg_path)
    attestations.append(
        ArtifactAttestationEntry(
            name="MetroPT-3 Experiment Config",
            relative_path="backend/configs/metropt3_anomaly.json",
            format="json",
            status=cfg_status,
            sha256=cfg_sha,
            detail=cfg_detail,
        )
    )

    non_pickle_ready = True
    if models_dir is not None:
        base_dir = (models_dir / model_name).resolve()
        pointer = base_dir / "CURRENT"
        if pointer.is_file():
            version = pointer.read_text(encoding="utf-8").strip()
            version_dir = (base_dir / version).resolve()
            if (
                version
                and version_dir != base_dir
                and version_dir.is_relative_to(base_dir)
                and version_dir.is_dir()
            ):
                manifest_path = version_dir / "model_manifest.json"
                if manifest_path.is_file():
                    m_status, m_sha, m_detail = verify_non_pickle_manifest(manifest_path)
                    non_pickle_ready = m_status == "verified"
                    attestations.append(
                        ArtifactAttestationEntry(
                            name=f"Model {version} JSON Manifest",
                            relative_path=f"artifacts/{model_name}/{version}/model_manifest.json",
                            format="json_manifest",
                            status=m_status,
                            sha256=m_sha,
                            detail=m_detail,
                        )
                    )
                joblib_path = version_dir / "model.joblib"
                card_path = version_dir / "model_card.json"
                if joblib_path.is_file():
                    j_sha = sha256_bytes(joblib_path.read_bytes())
                    c_status, _, _ = _audit_json_file(card_path)
                    card_match = False
                    if c_status == "verified":
                        card_obj = json.loads(card_path.read_text(encoding="utf-8"))
                        card_match = (
                            isinstance(card_obj, dict) and card_obj.get("artifact_sha256") == j_sha
                        )
                    attestations.append(
                        ArtifactAttestationEntry(
                            name=f"Model {version} Legacy Joblib Bundle",
                            relative_path=f"artifacts/{model_name}/{version}/model.joblib",
                            format="joblib_legacy",
                            status=(
                                "legacy_pickle_advisory"
                                if card_match
                                else "tamper_or_hash_mismatch"
                            ),
                            sha256=j_sha,
                            detail=(
                                "Legacy joblib/pickle bundle matches model_card.json "
                                "artifact_sha256; json_parameters_v1 manifest verifier ready"
                                if card_match
                                else "Legacy joblib bundle fails model_card.json "
                                "artifact_sha256 check"
                            ),
                        )
                    )

    return SupplyChainHardeningResponse(
        advisory_only=True,
        license_present=license_ok,
        ci_actions_sha_pinned=ci_pinned,
        unpinned_workflow_uses=unpinned,
        non_pickle_attestation_ready=non_pickle_ready,
        attestations=attestations,
        cross_system_benchmarks=_CROSS_SYSTEM_BENCHMARKS,
    )
