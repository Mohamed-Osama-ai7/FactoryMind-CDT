"""Read-only OT/edge source descriptors and a stateless batch preflight validator.

IMPORTANT: this module performs NO network I/O. It does not open OPC-UA, MQTT or historian
connections, subscribe to tags, or read from any plant system, and the project depends on no client
library for those protocols. What it does:

- checks endpoint *URL text* against scheme, credential, literal-IP and host-allowlist rules
  (hostnames are not resolved, so this says nothing about where a name points);
- validates a sample batch that a caller submits (allowlisted tags, finite numbers, cadence, gaps,
  duplicate/out-of-order timestamps) against an in-memory ring buffer;
- computes the exponential backoff list for a number of *simulated* disconnects.

The class names ``OpcUaReadOnlyAdapter`` / ``MqttReadOnlyAdapter`` / ``HistorianReadOnlyAdapter``
are kept for compatibility; they are protocol-specific descriptors, not connectors. The absence of
write methods is trivially true while no client exists; any future real connector must be designed
and tested for read-only behaviour on its own.
"""

from __future__ import annotations

import ipaddress
import math
import re
from abc import ABC
from collections import deque
from collections.abc import Sequence
from datetime import UTC, datetime
from statistics import median
from typing import Final, Literal
from urllib.parse import urlsplit

from factorymind.contracts.connectors import (
    CadenceStatus,
    ConnectorProtocol,
    EdgeBufferState,
    EdgeBufferStats,
    EdgeCadenceReport,
    EdgeSampleValidationRequest,
    EdgeSampleValidationResponse,
    EdgeSourceConfig,
    EdgeSourcesResponse,
    EdgeSourceStatusEntry,
    EdgeTelemetrySampleInput,
)
from factorymind.data.metropt3 import ANALOG_SENSORS, DIGITAL_SENSORS, NOMINAL_PERIOD_S

_ALLOWED_SCHEMES: Final[dict[ConnectorProtocol, frozenset[str]]] = {
    "opcua_readonly": frozenset({"opc.tcp"}),
    "mqtt_readonly": frozenset({"mqtts"}),
    "historian_readonly": frozenset({"https"}),
}

_BLOCKED_HOSTNAMES: Final[frozenset[str]] = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
        "instance-data",
    }
)
_BLOCKED_IPS: Final[frozenset[str]] = frozenset({"100.100.100.200"})
_NUMERIC_IPV4_LIKE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+)){0,3}$"
)


class EdgeSecurityError(ValueError):
    """Raised when an edge endpoint violates SSRF, scheme, credential, or host allowlist policy."""


class EdgeValidationError(ValueError):
    """Raised when a batch validation request references mismatched or unknown sources."""


def validate_edge_endpoint(
    endpoint_url: str,
    *,
    protocol: ConnectorProtocol,
    allowed_hosts: Sequence[str],
) -> tuple[str, int | None]:
    """Check an endpoint URL's *text* against SSRF-style and host-allowlist rules.

    Scope and limits: the hostname is never resolved, so a name on the allowlist that resolves to a
    private or metadata address is NOT detected; there is also no connection-time IP pinning. A
    real connector would additionally have to resolve and validate addresses at connect time.

    Rejects insecure schemes, embedded userinfo credentials, query strings/fragments, control
    characters, non-canonical numeric/octal/hex IPv4 literals, non-global/private/loopback/
    link-local/CGNAT IPs, cloud metadata hostnames, and hosts outside ``allowed_hosts``.
    """
    raw = endpoint_url.strip()
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in endpoint_url):
        raise EdgeSecurityError("Control characters are forbidden in edge endpoint URLs")
    if "?" in raw or "#" in raw:
        raise EdgeSecurityError(
            "Query parameters and fragments are not permitted in edge endpoints"
        )
    parsed = urlsplit(raw)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES[protocol]:
        raise EdgeSecurityError(
            f"Protocol {protocol} does not permit URI scheme {scheme!r}; "
            f"allowed: {sorted(_ALLOWED_SCHEMES[protocol])}"
        )
    if "@" in parsed.netloc or parsed.username is not None or parsed.password is not None:
        raise EdgeSecurityError("Embedded credentials in edge endpoint URL are forbidden")
    if re.search(r":(?=$|/)", parsed.netloc):
        raise EdgeSecurityError("Malformed empty port in edge endpoint URL")
    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise EdgeSecurityError("Edge endpoint must include a hostname")
    if host in _BLOCKED_HOSTNAMES or host.endswith(".localhost") or host.endswith(".local"):
        raise EdgeSecurityError(f"Blocked internal/metadata hostname {host!r}")
    if _NUMERIC_IPV4_LIKE_RE.fullmatch(host):
        try:
            ip_obj: ipaddress.IPv4Address | ipaddress.IPv6Address | None = ipaddress.IPv4Address(
                host
            )
        except ValueError as exc:
            raise EdgeSecurityError(f"Blocked non-canonical IPv4 literal {host!r}") from exc
    else:
        try:
            ip_obj = ipaddress.ip_address(host)
        except ValueError:
            ip_obj = None
    candidates = [ip_obj] if ip_obj is not None else []
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped is not None:
        candidates.append(ip_obj.ipv4_mapped)
    for addr in candidates:
        if (
            not addr.is_global
            or str(addr) in _BLOCKED_IPS
            or addr.is_private
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_multicast
            or addr.is_reserved
            or addr.is_unspecified
        ):
            raise EdgeSecurityError(
                f"Blocked non-global/public IP literal {host!r} in edge endpoint"
            )
    try:
        port = parsed.port
    except ValueError as exc:
        raise EdgeSecurityError("Malformed port in edge endpoint URL") from exc
    if port is not None and port == 0:
        raise EdgeSecurityError("Port 0 is not permitted in edge endpoints")
    allow_set = {item.strip().lower().rstrip(".") for item in allowed_hosts if item.strip()}
    if host not in allow_set:
        raise EdgeSecurityError(f"Endpoint hostname {host!r} is not in source host allowlist")
    return host, port


def _normalize_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class _BaseReadOnlyEdgeAdapter(ABC):
    """Source descriptor plus batch preflight (ring buffer + cadence diagnostics). No I/O."""

    expected_protocol: ConnectorProtocol

    def __init__(self, config: EdgeSourceConfig) -> None:
        if config.protocol != self.expected_protocol:
            raise EdgeValidationError(
                f"{type(self).__name__} requires protocol {self.expected_protocol!r}, "
                f"got {config.protocol!r}"
            )
        validate_edge_endpoint(
            config.endpoint_url,
            protocol=config.protocol,
            allowed_hosts=config.allowed_hosts,
        )
        self.config = config
        self.read_only: Literal[True] = True

    def snapshot_status(self) -> EdgeSourceStatusEntry:
        url_policy: Literal["passed", "rejected"]
        try:
            validate_edge_endpoint(
                self.config.endpoint_url,
                protocol=self.config.protocol,
                allowed_hosts=self.config.allowed_hosts,
            )
            url_policy = "passed"
        except EdgeSecurityError:
            url_policy = "rejected"
        return EdgeSourceStatusEntry(
            source_id=self.config.source_id,
            asset_id=self.config.asset_id,
            protocol=self.config.protocol,
            endpoint_url=self.config.endpoint_url,
            read_only=True,
            nominal_cadence_s=self.config.nominal_cadence_s,
            max_buffer_records=self.config.max_buffer_records,
            node_allowlist_count=len(self.config.node_allowlist),
            endpoint_url_policy=url_policy,
        )

    def ingest_batch(
        self,
        samples: Sequence[EdgeTelemetrySampleInput],
        *,
        simulate_disconnects: int = 0,
        buffer_capacity_override: int | None = None,
    ) -> EdgeSampleValidationResponse:
        """Validate and buffer a read-only sample batch without emitting any writes."""
        validate_edge_endpoint(
            self.config.endpoint_url,
            protocol=self.config.protocol,
            allowed_hosts=self.config.allowed_hosts,
        )
        if buffer_capacity_override is not None:
            if buffer_capacity_override < 4:
                raise EdgeValidationError("buffer_capacity_override must be at least 4")
            capacity = buffer_capacity_override
        else:
            capacity = self.config.max_buffer_records
        ring: deque[dict[str, object]] = deque(maxlen=capacity)
        evicted = 0
        allowed_sensors = set(self.config.node_allowlist)
        ignored_sensors: set[str] = set()
        accepted_rows = 0

        timestamps: list[datetime] = []
        for sample in samples:
            if sample.source_id != self.config.source_id:
                raise EdgeValidationError(
                    f"Sample source_id {sample.source_id!r} != {self.config.source_id!r}"
                )
            filtered: dict[str, float] = {}
            for key, raw_val in sample.sensor_values.items():
                if key not in allowed_sensors:
                    ignored_sensors.add(key)
                    continue
                if isinstance(raw_val, bool):
                    raise EdgeValidationError(f"Boolean sensor reading not allowed for {key!r}")
                val = float(raw_val)
                if not math.isfinite(val):
                    raise EdgeValidationError(f"Non-finite sensor reading for {key!r}")
                filtered[key] = val
            if not filtered:
                continue
            ts = _normalize_utc(sample.timestamp)
            timestamps.append(ts)
            accepted_rows += 1
            if len(ring) == capacity:
                evicted += 1
            ring.append({"timestamp": ts.isoformat(), "sensors": filtered})

        out_of_order = 0
        duplicates = 0
        positive_intervals: list[float] = []
        gaps = 0
        gap_threshold_s = 2.0 * self.config.nominal_cadence_s
        for idx in range(1, len(timestamps)):
            delta_s = (timestamps[idx] - timestamps[idx - 1]).total_seconds()
            if delta_s < 0.0:
                out_of_order += 1
            elif delta_s == 0.0:
                duplicates += 1
            else:
                positive_intervals.append(delta_s)
                if delta_s > gap_threshold_s:
                    gaps += 1

        median_interval = (
            round(float(median(positive_intervals)), 4) if positive_intervals else None
        )
        min_interval = round(float(min(positive_intervals)), 4) if positive_intervals else None
        max_interval = round(float(max(positive_intervals)), 4) if positive_intervals else None

        nominal = self.config.nominal_cadence_s
        cadence_status: CadenceStatus
        if (
            accepted_rows == 0
            or median_interval is None
            or median_interval < 0.2 * nominal
            or median_interval > 3.0 * nominal
        ):
            cadence_status = "stale_or_invalid"
        elif (
            gaps > 0
            or out_of_order > 0
            or duplicates > 0
            or median_interval < 0.5 * nominal
            or median_interval > 1.5 * nominal
        ):
            cadence_status = "gapped_or_jittered"
        else:
            cadence_status = "nominal"

        backoff_schedule = tuple(
            min(60.0, float(1 << attempt)) for attempt in range(simulate_disconnects)
        )
        if evicted > 0:
            state: EdgeBufferState = "batch_exceeds_capacity"
        elif simulate_disconnects > 0:
            state = "simulated_reconnect_backoff"
        else:
            state = "batch_within_capacity"

        warnings: list[str] = []
        if ignored_sensors:
            warnings.append(
                f"Ignored {len(ignored_sensors)} unallowlisted tag(s): "
                f"{', '.join(sorted(ignored_sensors))}."
            )
        if gaps > 0:
            warnings.append(
                f"Detected {gaps} sampling gap(s) exceeding {gap_threshold_s:.1f}s; "
                "preserved uninterpolated."
            )
        if out_of_order > 0 or duplicates > 0:
            warnings.append(
                f"Detected {out_of_order} out-of-order and {duplicates} duplicate timestamp(s)."
            )
        if evicted > 0:
            warnings.append(
                f"Ring buffer reached capacity ({capacity}); evicted {evicted} oldest sample(s)."
            )
        if accepted_rows == 0:
            warnings.append("No allowlisted sensors were present in submitted samples.")

        return EdgeSampleValidationResponse(
            source_id=self.config.source_id,
            asset_id=self.config.asset_id,
            protocol=self.config.protocol,
            read_only=True,
            endpoint_url_policy_ok=True,
            cadence=EdgeCadenceReport(
                sample_count=len(samples),
                expected_cadence_s=nominal,
                observed_median_interval_s=median_interval,
                min_interval_s=min_interval,
                max_interval_s=max_interval,
                out_of_order_count=out_of_order,
                duplicate_timestamp_count=duplicates,
                gap_count=gaps,
                cadence_status=cadence_status,
                unallowlisted_sensors_ignored=tuple(sorted(ignored_sensors)),
            ),
            buffer=EdgeBufferStats(
                capacity=capacity,
                retained_count=len(ring),
                evicted_count=evicted,
                simulated_reconnect_attempts=simulate_disconnects,
                backoff_schedule_s=backoff_schedule,
                state=state,
            ),
            warnings=tuple(warnings),
            safety_note=(
                "Preflight validation of the submitted batch only. No connection to the source was "
                "made and nothing was read from any plant system; there is no connector I/O in "
                "this build."
            ),
        )


class OpcUaReadOnlyAdapter(_BaseReadOnlyEdgeAdapter):
    """OPC-UA source descriptor (``opc.tcp://``). Not a connector: no OPC-UA client exists."""

    expected_protocol: ConnectorProtocol = "opcua_readonly"


class MqttReadOnlyAdapter(_BaseReadOnlyEdgeAdapter):
    """MQTT source descriptor (``mqtts://``). Not a connector: no MQTT client exists."""

    expected_protocol: ConnectorProtocol = "mqtt_readonly"


class HistorianReadOnlyAdapter(_BaseReadOnlyEdgeAdapter):
    """Historian source descriptor (``https://``). Not a connector: no historian client."""

    expected_protocol: ConnectorProtocol = "historian_readonly"


_ADAPTER_BY_PROTOCOL: Final[dict[ConnectorProtocol, type[_BaseReadOnlyEdgeAdapter]]] = {
    "opcua_readonly": OpcUaReadOnlyAdapter,
    "mqtt_readonly": MqttReadOnlyAdapter,
    "historian_readonly": HistorianReadOnlyAdapter,
}

_ALL_METROPT3_SENSORS: Final[tuple[str, ...]] = tuple(ANALOG_SENSORS) + tuple(DIGITAL_SENSORS)

DEFAULT_EDGE_SOURCES: Final[tuple[EdgeSourceConfig, ...]] = (
    EdgeSourceConfig(
        source_id="metropt3-opcua-ro",
        asset_id="metropt3-apu-1",
        protocol="opcua_readonly",
        endpoint_url="opc.tcp://opc-ro.factorymind.internal:4840",
        allowed_hosts=("opc-ro.factorymind.internal",),
        node_allowlist=tuple(ANALOG_SENSORS),
        nominal_cadence_s=float(NOMINAL_PERIOD_S),
        max_buffer_records=512,
    ),
    EdgeSourceConfig(
        source_id="metropt3-mqtt-ro",
        asset_id="metropt3-apu-1",
        protocol="mqtt_readonly",
        endpoint_url="mqtts://mqtt-ro.factorymind.internal:8883",
        allowed_hosts=("mqtt-ro.factorymind.internal",),
        node_allowlist=tuple(DIGITAL_SENSORS),
        nominal_cadence_s=float(NOMINAL_PERIOD_S),
        max_buffer_records=512,
    ),
    EdgeSourceConfig(
        source_id="metropt3-historian-ro",
        asset_id="metropt3-apu-1",
        protocol="historian_readonly",
        endpoint_url="https://historian-ro.factorymind.internal/api/v1/series",
        allowed_hosts=("historian-ro.factorymind.internal",),
        node_allowlist=_ALL_METROPT3_SENSORS,
        nominal_cadence_s=float(NOMINAL_PERIOD_S),
        max_buffer_records=1024,
    ),
)

_SOURCE_MAP: Final[dict[str, EdgeSourceConfig]] = {
    cfg.source_id: cfg for cfg in DEFAULT_EDGE_SOURCES
}


def build_adapter(config: EdgeSourceConfig) -> _BaseReadOnlyEdgeAdapter:
    """Instantiate the (I/O-free) source descriptor for a validated source config."""
    return _ADAPTER_BY_PROTOCOL[config.protocol](config)


def get_default_source_config(source_id: str) -> EdgeSourceConfig:
    """Look up a registered default read-only edge source by ``source_id``."""
    if source_id not in _SOURCE_MAP:
        raise KeyError(f"Unknown edge source: {source_id}")
    return _SOURCE_MAP[source_id]


def list_edge_sources(
    *,
    allowed_assets: frozenset[str] | None = None,
    asset_id: str | None = None,
) -> EdgeSourcesResponse:
    """Return registered source descriptors (URL-policy checked) scoped to allowed assets."""
    entries: list[EdgeSourceStatusEntry] = []
    for cfg in DEFAULT_EDGE_SOURCES:
        if allowed_assets is not None and cfg.asset_id not in allowed_assets:
            continue
        if asset_id is not None and cfg.asset_id != asset_id:
            continue
        entries.append(build_adapter(cfg).snapshot_status())
    return EdgeSourcesResponse(sources=entries)


def validate_edge_batch(request: EdgeSampleValidationRequest) -> EdgeSampleValidationResponse:
    """Validate a stateless telemetry batch against its registered read-only adapter."""
    cfg = get_default_source_config(request.source_id)
    if cfg.asset_id != request.asset_id:
        raise EdgeValidationError(
            f"Source {request.source_id!r} belongs to asset {cfg.asset_id!r}, "
            f"not {request.asset_id!r}"
        )
    adapter = build_adapter(cfg)
    return adapter.ingest_batch(
        request.samples,
        simulate_disconnects=request.simulate_disconnects,
        buffer_capacity_override=request.buffer_capacity_override,
    )
