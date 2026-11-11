"""Typed contracts for read-only OT/edge connectors and supply-chain attestation."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from factorymind.config import IDENTIFIER

ConnectorProtocol = Literal[
    "opcua_readonly",
    "mqtt_readonly",
    "historian_readonly",
]

# No adapter opens a network connection: a source is only ever a descriptor.
EdgeSourceState = Literal["descriptor_only_not_connected"]

# Outcome of validating one caller-submitted batch against an in-memory ring buffer. These are
# properties of the submitted batch, never of a live connection.
EdgeBufferState = Literal[
    "batch_within_capacity",
    "simulated_reconnect_backoff",
    "batch_exceeds_capacity",
]

ConnectorIoStatus = Literal["not_implemented"]

CadenceStatus = Literal["nominal", "gapped_or_jittered", "stale_or_invalid"]

AttestationStatus = Literal[
    "present",  # file exists and contains the expected text
    "sha_pinned",  # every CI 'uses:' line names a 40-hex commit (syntax only)
    "unattested",  # parsed and hashed, but no reference digest exists to compare against
    "self_consistent_unsigned",  # manifest matches its own digest; no signature verified
    "signature_verified",  # HMAC verified with the supplied key (manifest metadata only)
    "sha_matches_adjacent_card",  # joblib SHA equals the card beside it: corruption check only
    "missing",
    "tamper_or_hash_mismatch",
]

RuntimeModelAttestation = Literal[
    "not_checked",
    "no_current_model",
    "invalid_pointer",
    "manifest_missing",
    "manifest_invalid",
    "manifest_unsigned_self_consistent",
    "manifest_signature_verified",
    "integrity_failed",
]

RuntimeModelFormat = Literal["joblib_pickle", "unknown"]


def _validate_identifier(value: str, label: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must match {IDENTIFIER.pattern}")
    return value


class EdgeSourceConfig(BaseModel):
    """Configuration for a strictly read-only industrial telemetry source."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_id: str
    asset_id: str
    protocol: ConnectorProtocol
    endpoint_url: str = Field(min_length=8, max_length=256)
    read_only: Literal[True] = True
    allowed_hosts: tuple[str, ...] = Field(min_length=1, max_length=16)
    node_allowlist: tuple[str, ...] = Field(min_length=1, max_length=128)
    nominal_cadence_s: float = Field(default=10.0, gt=0.0, le=300.0)
    max_buffer_records: int = Field(default=512, ge=4, le=10000)

    @field_validator("source_id")
    @classmethod
    def _valid_source_id(cls, value: str) -> str:
        return _validate_identifier(value, "source_id")

    @field_validator("asset_id")
    @classmethod
    def _valid_asset_id(cls, value: str) -> str:
        return _validate_identifier(value, "asset_id")


class EdgeTelemetrySampleInput(BaseModel):
    """Single telemetry sample submitted for stateless edge validation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    timestamp: datetime
    source_id: str
    sensor_values: dict[str, float] = Field(min_length=1, max_length=64)

    @field_validator("source_id")
    @classmethod
    def _valid_source_id(cls, value: str) -> str:
        return _validate_identifier(value, "source_id")

    @field_validator("sensor_values", mode="before")
    @classmethod
    def _reject_booleans(cls, raw: object) -> object:
        if isinstance(raw, dict):
            for key, val in raw.items():
                if isinstance(val, bool):
                    raise ValueError(f"Sensor reading for {key!r} must be numeric, not boolean")
        return raw


class EdgeSampleValidationRequest(BaseModel):
    """Stateless edge buffer, cadence, and node-allowlist validation request."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_id: str
    asset_id: str = "metropt3-apu-1"
    samples: list[EdgeTelemetrySampleInput] = Field(min_length=2, max_length=500)
    simulate_disconnects: int = Field(default=0, ge=0, le=20)
    buffer_capacity_override: int | None = Field(default=None, ge=4, le=10000)

    @field_validator("source_id")
    @classmethod
    def _valid_source_id(cls, value: str) -> str:
        return _validate_identifier(value, "source_id")

    @field_validator("asset_id")
    @classmethod
    def _valid_asset_id(cls, value: str) -> str:
        return _validate_identifier(value, "asset_id")


class EdgeCadenceReport(BaseModel):
    """Sampling cadence and uninterpolated gap diagnostics."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    sample_count: int = Field(ge=0)
    expected_cadence_s: float = Field(gt=0.0)
    observed_median_interval_s: float | None = Field(default=None, ge=0.0)
    min_interval_s: float | None = Field(default=None, ge=0.0)
    max_interval_s: float | None = Field(default=None, ge=0.0)
    out_of_order_count: int = Field(ge=0)
    duplicate_timestamp_count: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    cadence_status: CadenceStatus
    unallowlisted_sensors_ignored: tuple[str, ...] = ()


class EdgeBufferStats(BaseModel):
    """Bounded ring-buffer and reconnect backoff state."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    capacity: int = Field(ge=4)
    retained_count: int = Field(ge=0)
    evicted_count: int = Field(ge=0)
    simulated_reconnect_attempts: int = Field(ge=0)
    backoff_schedule_s: tuple[float, ...] = ()
    state: EdgeBufferState


class EdgeSampleValidationResponse(BaseModel):
    """Result of stateless read-only edge connector validation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_id: str
    asset_id: str
    protocol: ConnectorProtocol
    read_only: Literal[True] = True
    # What this endpoint is: a stateless check of a caller-submitted batch. No connection to the
    # source is made and nothing is read from it.
    mode: Literal["preflight_validation_only"] = "preflight_validation_only"
    real_connector_io: Literal[False] = False
    endpoint_url_policy_ok: bool
    endpoint_check_scope: Literal["url_string_only_no_dns_resolution"] = (
        "url_string_only_no_dns_resolution"
    )
    cadence: EdgeCadenceReport
    buffer: EdgeBufferStats
    warnings: tuple[str, ...] = ()
    safety_note: str


class EdgeSourceStatusEntry(BaseModel):
    """Summary of one registered read-only OT/edge source."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_id: str
    asset_id: str
    protocol: ConnectorProtocol
    endpoint_url: str
    read_only: Literal[True] = True
    connection_state: EdgeSourceState = "descriptor_only_not_connected"
    connector_io: ConnectorIoStatus = "not_implemented"
    nominal_cadence_s: float = Field(gt=0.0)
    max_buffer_records: int = Field(ge=4)
    node_allowlist_count: int = Field(ge=1)
    endpoint_url_policy: Literal["passed", "rejected"]
    endpoint_check_scope: Literal["url_string_only_no_dns_resolution"] = (
        "url_string_only_no_dns_resolution"
    )


class EdgeSourcesResponse(BaseModel):
    """Permission-scoped listing of read-only OT/edge sources."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    advisory_only: Literal[True] = True
    # No OPC-UA, MQTT or historian client exists in this codebase, so there is nothing that could
    # write; this is a statement about missing code, not an enforced guarantee over a real client.
    connector_io_status: ConnectorIoStatus = "not_implemented"
    write_capabilities: Literal["none_no_connector_io_exists"] = "none_no_connector_io_exists"
    capability_note: str = (
        "These are registered source descriptors. FactoryMind does not currently connect to any "
        "OPC-UA server, MQTT broker or historian; endpoint checks inspect the URL text only."
    )
    sources: list[EdgeSourceStatusEntry]


class ArtifactAttestationEntry(BaseModel):
    """Verification status of one governance, workflow, config, or model artifact."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str
    relative_path: str
    format: Literal[
        "json",
        "json_manifest",
        "text_license",
        "yaml_workflow",
        "joblib_legacy",
    ]
    status: AttestationStatus
    sha256: str | None = None
    detail: str


class SupplyChainHardeningResponse(BaseModel):
    """Repository supply-chain, CI pinning, license, and artifact attestation audit."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    advisory_only: Literal[True] = True
    license_present: bool
    ci_actions_sha_pinned: bool
    unpinned_workflow_uses: tuple[str, ...] = ()
    runtime_model_attestation: RuntimeModelAttestation
    runtime_model_format: RuntimeModelFormat
    runtime_loader_consults_manifest: Literal[False] = False
    manifest_signature_key_configured: bool
    attestations: list[ArtifactAttestationEntry]
    limitations: tuple[str, ...]
