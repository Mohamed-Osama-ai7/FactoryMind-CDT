"""Supply-chain checks and model-manifest verification.

Every status returned here is the result of a check that actually ran in this call, and each check
states what it does *not* establish. Nothing in this module reports a component as "ready": a
missing, unreadable or unconfigured input is reported as such (``missing`` / ``not_checked``), never
as success.

What the checks establish:

- ``LICENSE``: the file exists and contains the MIT licence phrases. Nothing more.
- CI workflow: every ``uses:`` line in ``ci.yml`` names a 40-hex commit. It does not prove the
  commit exists upstream, is authentic, or that other workflows or lockfiles are pinned.
- Model manifest (``json_parameters_v1``): the manifest's parameters match the manifest's own
  SHA-256 (``self_consistent_unsigned``), and, only when an HMAC key is supplied and verifies,
  ``signature_verified``. A manifest does not cover the bytes of ``model.joblib``.
- Runtime joblib bundle: the SHA-256 of ``model.joblib`` equals the digest in ``model_card.json``
  stored in the same directory. That detects corruption; it does not stop a writer who can change
  both files, and the runtime loader still unpickles the bundle.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

from factorymind.config import IDENTIFIER
from factorymind.contracts.connectors import (
    ArtifactAttestationEntry,
    AttestationStatus,
    RuntimeModelAttestation,
    RuntimeModelFormat,
    SupplyChainHardeningResponse,
)

_WORKFLOW_USES_RE: Final[re.Pattern[str]] = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)")
_SHA_PINNED_RE: Final[re.Pattern[str]] = re.compile(r"^[^@]+@[0-9a-f]{40}$")
_FORBIDDEN_PICKLE_FORMATS: Final[frozenset[str]] = frozenset({"pickle", "joblib", "dill"})

# Static disclosures of what this report does NOT establish. They are statements about the
# implementation, not check results.
LIMITATIONS: Final[tuple[str, ...]] = (
    "The runtime loads model.joblib with joblib (pickle) after comparing its SHA-256 with "
    "model_card.json in the same directory: this detects corruption, not a writer who can change "
    "both files.",
    "No runtime code path reads model_manifest.json, and no shipped model has one; a manifest "
    "would attest its own JSON parameters, not the bytes of model.joblib.",
    "No manifest signature is verified because no HMAC key is configured in this deployment.",
    "The CI check only parses 'uses:' lines in ci.yml; it does not verify that pinned commits "
    "exist or are authentic, nor other workflows, lockfiles or dependency provenance.",
    "The LICENSE check confirms MIT licence phrases are present in the file.",
)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"Non-standard JSON constant {value!r} is forbidden")


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
    """Verify a JSON model manifest without deserializing any pickle/joblib payload.

    Returns ``signature_verified`` only when ``hmac_key`` is given and the HMAC matches.
    Without a key the best outcome is ``self_consistent_unsigned``: the parameters match the
    manifest's own digest, which anyone who can edit the file can also recompute.
    """
    if not manifest_path.is_file():
        return "missing", None, "Non-pickle manifest file not present"
    try:
        raw = manifest_path.read_bytes()
    except OSError:
        return "tamper_or_hash_mismatch", None, "Unreadable manifest file"
    digest = sha256_bytes(raw)
    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return "tamper_or_hash_mismatch", digest, f"Invalid JSON manifest: {exc}"
    if not isinstance(data, dict):
        return "tamper_or_hash_mismatch", digest, "Manifest root must be a JSON object"
    fmt = str(data.get("artifact_format", "")).strip().lower()
    if fmt in _FORBIDDEN_PICKLE_FORMATS or fmt != "json_parameters_v1":
        return (
            "tamper_or_hash_mismatch",
            digest,
            f"Unsupported or unsafe artifact_format {fmt!r}",
        )
    params = data.get("parameters")
    expected_sha = data.get("parameters_sha256")
    if not isinstance(params, dict) or not isinstance(expected_sha, str):
        return "tamper_or_hash_mismatch", digest, "Missing parameters or digest"
    try:
        actual_sha = sha256_bytes(canonical_json_bytes(params))
    except (ValueError, TypeError) as exc:
        return "tamper_or_hash_mismatch", digest, f"Non-finite parameter: {exc}"
    if not hmac.compare_digest(actual_sha, expected_sha):
        return (
            "tamper_or_hash_mismatch",
            digest,
            "Parameters SHA-256 mismatch (manifest modified)",
        )
    if hmac_key is None:
        return (
            "self_consistent_unsigned",
            digest,
            "Parameters match the manifest's own SHA-256; no signature was verified, so this does "
            "not establish authenticity or that it describes the deployed model.joblib",
        )
    sig = data.get("hmac_sha256")
    if not isinstance(sig, str):
        return "tamper_or_hash_mismatch", digest, "Missing required HMAC signature"
    try:
        material = canonical_json_bytes(
            {
                "schema_version": data.get("schema_version"),
                "artifact_format": fmt,
                "model_name": data.get("model_name"),
                "parameters_sha256": expected_sha,
            }
        )
    except (ValueError, TypeError) as exc:
        return "tamper_or_hash_mismatch", digest, f"Invalid manifest metadata: {exc}"
    expected_sig = hmac.new(hmac_key, material, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected_sig):
        return "tamper_or_hash_mismatch", digest, "HMAC-SHA256 verification failed"
    return (
        "signature_verified",
        digest,
        "HMAC-SHA256 over the manifest metadata verified with the supplied key (covers the "
        "manifest parameters, not model.joblib)",
    )


def audit_ci_workflow(ci_path: Path) -> tuple[bool, tuple[str, ...], str | None, str]:
    """Check that every ``uses:`` line in the CI workflow names a 40-hex commit SHA."""
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
        target = match.group(1).strip("'\"")
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
    return (
        True,
        (),
        file_sha,
        f"All {seen} 'uses:' lines in ci.yml name a 40-hex commit (pin syntax only; the commits "
        "were not checked upstream)",
    )


def _audit_json_file(path: Path) -> tuple[AttestationStatus, str | None, str]:
    """Hash and parse a JSON file; without a reference digest this is not an integrity check."""
    if not path.is_file():
        return "missing", None, "File not found"
    try:
        raw = path.read_bytes()
    except OSError:
        return "tamper_or_hash_mismatch", None, "Unreadable JSON file"
    digest = sha256_bytes(raw)
    try:
        json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return "tamper_or_hash_mismatch", digest, f"Malformed JSON: {exc}"
    return (
        "unattested",
        digest,
        "Valid UTF-8 JSON; SHA-256 computed but there is no reference digest to compare, so this "
        "is not an integrity check",
    )


def _entry(
    name: str,
    relative_path: str,
    fmt: Literal[
        "json",
        "json_manifest",
        "text_license",
        "yaml_workflow",
        "joblib_legacy",
    ],
    status: AttestationStatus,
    sha: str | None,
    detail: str,
) -> ArtifactAttestationEntry:
    return ArtifactAttestationEntry(
        name=name,
        relative_path=relative_path,
        format=fmt,
        status=status,
        sha256=sha,
        detail=detail,
    )


def _check_runtime_model(
    models_dir: Path,
    model_name: str,
    hmac_key: bytes | None,
    attestations: list[ArtifactAttestationEntry],
) -> tuple[RuntimeModelAttestation, RuntimeModelFormat]:
    base_dir = (models_dir / model_name).resolve()
    pointer = base_dir / "CURRENT"
    if not pointer.is_file():
        attestations.append(
            _entry(
                f"Model {model_name} CURRENT Pointer",
                f"artifacts/{model_name}/CURRENT",
                "json_manifest",
                "missing",
                None,
                "No CURRENT pointer: no runtime model was inspected",
            )
        )
        return "no_current_model", "unknown"
    try:
        version = pointer.read_text(encoding="utf-8").strip()
    except OSError:
        version = ""
    if not IDENTIFIER.fullmatch(version):
        attestations.append(
            _entry(
                f"Model {model_name} CURRENT Pointer",
                f"artifacts/{model_name}/CURRENT",
                "json_manifest",
                "tamper_or_hash_mismatch",
                None,
                "Invalid model version identifier in CURRENT pointer",
            )
        )
        return "invalid_pointer", "unknown"
    try:
        version_dir = (base_dir / version).resolve()
        valid_dir = (
            version_dir != base_dir
            and version_dir.is_relative_to(base_dir)
            and version_dir.is_dir()
        )
    except (OSError, ValueError):
        valid_dir = False
        version_dir = base_dir
    if not valid_dir:
        attestations.append(
            _entry(
                f"Model {version} Directory",
                f"artifacts/{model_name}/{version}",
                "json_manifest",
                "missing",
                None,
                "Model version directory missing or invalid",
            )
        )
        return "invalid_pointer", "unknown"

    rel = f"artifacts/{model_name}/{version}"
    # 1. the bundle the runtime actually loads
    integrity_failed = False
    fmt: RuntimeModelFormat = "unknown"
    joblib_path = version_dir / "model.joblib"
    if joblib_path.is_file():
        fmt = "joblib_pickle"
        j_sha = sha256_bytes(joblib_path.read_bytes())
        card_path = version_dir / "model_card.json"
        c_status, _, _ = _audit_json_file(card_path)
        card_match = False
        if c_status == "unattested":
            card_obj = json.loads(
                card_path.read_text(encoding="utf-8"), parse_constant=_reject_json_constant
            )
            card_match = isinstance(card_obj, dict) and card_obj.get("artifact_sha256") == j_sha
        integrity_failed = not card_match
        attestations.append(
            _entry(
                f"Model {version} joblib bundle",
                f"{rel}/model.joblib",
                "joblib_legacy",
                "sha_matches_adjacent_card" if card_match else "tamper_or_hash_mismatch",
                j_sha,
                "SHA-256 matches adjacent model_card.json (corruption check only; runtime "
                "still unpickles this file)"
                if card_match
                else "SHA-256 does not match model_card.json (or the card is missing/invalid)",
            )
        )
    # 2. the optional non-pickle manifest
    manifest_path = version_dir / "model_manifest.json"
    m_status, m_sha, m_detail = verify_non_pickle_manifest(manifest_path, hmac_key=hmac_key)
    attestations.append(
        _entry(
            f"Model {version} JSON manifest",
            f"{rel}/model_manifest.json",
            "json_manifest",
            m_status,
            m_sha,
            m_detail
            if m_status != "missing"
            else "No json_parameters_v1 manifest exists for this runtime bundle",
        )
    )
    if integrity_failed:
        return "integrity_failed", fmt
    if m_status == "signature_verified":
        return "manifest_signature_verified", fmt
    if m_status == "self_consistent_unsigned":
        return "manifest_unsigned_self_consistent", fmt
    if m_status == "missing":
        return "manifest_missing", fmt
    return "manifest_invalid", fmt


def build_supply_chain_report(
    *,
    repo_root: Path | None = None,
    models_dir: Path | None = None,
    model_name: str = "metropt3-anomaly",
    hmac_key: bytes | None = None,
) -> SupplyChainHardeningResponse:
    """Run the repository and runtime-artifact checks and report each result as executed."""
    root = repo_root or discover_repo_root()
    attestations: list[ArtifactAttestationEntry] = []

    license_path = root / "LICENSE"
    if license_path.is_file():
        lic_bytes = license_path.read_bytes()
        lic_text = lic_bytes.decode("utf-8", errors="replace")
        has_mit = "MIT License" in lic_text and "Permission is hereby granted" in lic_text
        attestations.append(
            _entry(
                "Root Repository License",
                "LICENSE",
                "text_license",
                "present" if has_mit else "tamper_or_hash_mismatch",
                sha256_bytes(lic_bytes),
                "File exists and contains the MIT licence phrases"
                if has_mit
                else "File exists but does not contain the expected MIT licence text",
            )
        )
        license_ok = has_mit
    else:
        attestations.append(
            _entry(
                "Root Repository License",
                "LICENSE",
                "text_license",
                "missing",
                None,
                "Root LICENSE file missing",
            )
        )
        license_ok = False

    ci_path = root / ".github" / "workflows" / "ci.yml"
    ci_pinned, unpinned, ci_sha, ci_detail = audit_ci_workflow(ci_path)
    attestations.append(
        _entry(
            "GitHub Actions CI Workflow",
            ".github/workflows/ci.yml",
            "yaml_workflow",
            "sha_pinned" if ci_pinned else "tamper_or_hash_mismatch",
            ci_sha,
            ci_detail,
        )
    )

    cfg_status, cfg_sha, cfg_detail = _audit_json_file(
        root / "backend" / "configs" / "metropt3_anomaly.json"
    )
    attestations.append(
        _entry(
            "MetroPT-3 Experiment Config",
            "backend/configs/metropt3_anomaly.json",
            "json",
            cfg_status,
            cfg_sha,
            cfg_detail,
        )
    )

    runtime: RuntimeModelAttestation = "not_checked"
    runtime_format: RuntimeModelFormat = "unknown"
    if models_dir is not None:
        runtime, runtime_format = _check_runtime_model(
            models_dir, model_name, hmac_key, attestations
        )

    return SupplyChainHardeningResponse(
        advisory_only=True,
        license_present=license_ok,
        ci_actions_sha_pinned=ci_pinned,
        unpinned_workflow_uses=unpinned,
        runtime_model_attestation=runtime,
        runtime_model_format=runtime_format,
        runtime_loader_consults_manifest=False,
        manifest_signature_key_configured=hmac_key is not None,
        attestations=attestations,
        limitations=LIMITATIONS,
    )
