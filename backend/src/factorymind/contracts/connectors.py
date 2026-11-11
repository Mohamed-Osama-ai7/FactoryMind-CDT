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

EdgeConnectionState = Literal[
    "standby_offline",
    "connected_readonly",
    "reconnecting_backoff",
    "degraded_buffer_evicting",
]

CadenceStatus = Literal["nominal", "gapped_or_jittered", "stale_or_invalid"]

AttestationStatus = Literal[
    "verified",
    "sha_pinned",
    "missing",
    "tamper_or_hash_mismatch",
    "legacy_pickle_advisory",
]


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
    reconnect_attempts: int = Field(ge=0)
    backoff_schedule_s: tuple[float, ...] = ()
    state: EdgeConnectionState


class EdgeSampleValidationResponse(BaseModel):
    """Result of stateless read-only edge connector validation."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source_id: str
    asset_id: str
    protocol: ConnectorProtocol
    read_only: Literal[True] = True
    endpoint_allowlist_ok: bool
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
    connection_state: EdgeConnectionState
    nominal_cadence_s: float = Field(gt=0.0)
    max_buffer_records: int = Field(ge=4)
    node_allowlist_count: int = Field(ge=1)
    ssrf_policy_status: Literal["allowlisted_host_verified", "blocked"]
    buffer_utilization: float = Field(ge=0.0, le=1.0)
    reconnect_attempts: int = Field(ge=0)


class EdgeSourcesResponse(BaseModel):
    """Permission-scoped listing of read-only OT/edge sources."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    advisory_only: Literal[True] = True
    write_capabilities: Literal["none_disabled_by_architecture"] = "none_disabled_by_architecture"
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
    non_pickle_attestation_ready: bool
    attestations: list[ArtifactAttestationEntry]
    cross_system_benchmarks: dict[str, str]
