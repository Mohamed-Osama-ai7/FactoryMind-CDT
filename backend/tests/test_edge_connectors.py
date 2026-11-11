"""Unit, security, and API tests for WP-H read-only OT/edge connectors and hardening."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from factorymind.connectors import (
    DEFAULT_EDGE_SOURCES,
    EdgeSecurityError,
    EdgeValidationError,
    HistorianReadOnlyAdapter,
    MqttReadOnlyAdapter,
    OpcUaReadOnlyAdapter,
    audit_ci_workflow,
    build_signed_manifest,
    build_supply_chain_report,
    discover_repo_root,
    validate_edge_endpoint,
    verify_non_pickle_manifest,
)
from factorymind.contracts.connectors import EdgeSourceConfig, EdgeTelemetrySampleInput
from tests.auth_helpers import ASSET, AuthKit, make_client

FORBIDDEN_ADAPTER_METHODS = (
    "write",
    "write_tag",
    "write_attribute",
    "publish",
    "send_command",
    "set_setpoint",
    "actuate",
    "call_method",
)


def _sample_series(
    *,
    count: int = 5,
    step_s: float = 10.0,
    gap_after_idx: int | None = None,
    extra_gap_s: float = 60.0,
    extra_sensor: str | None = None,
) -> list[dict[str, object]]:
    base = datetime(2020, 4, 15, 10, 0, 0, tzinfo=UTC)
    t = base
    rows: list[dict[str, object]] = []
    for idx in range(count):
        sensors: dict[str, float] = {"TP2": 8.4 + idx * 0.05, "TP3": 8.6 + idx * 0.05}
        if extra_sensor is not None:
            sensors[extra_sensor] = 1.0
        rows.append(
            {
                "timestamp": t.isoformat(),
                "source_id": "metropt3-opcua-ro",
                "sensor_values": sensors,
            }
        )
        t = t + timedelta(seconds=extra_gap_s if gap_after_idx == idx else step_s)
    return rows


# ---------------------------------------------------------------- SSRF & Read-Only Boundaries


def test_adapters_enforce_read_only_architecture() -> None:
    cfg_opc = DEFAULT_EDGE_SOURCES[0]
    cfg_mqtt = DEFAULT_EDGE_SOURCES[1]
    cfg_hist = DEFAULT_EDGE_SOURCES[2]
    adapters = [
        OpcUaReadOnlyAdapter(cfg_opc),
        MqttReadOnlyAdapter(cfg_mqtt),
        HistorianReadOnlyAdapter(cfg_hist),
    ]
    for adapter in adapters:
        assert adapter.read_only is True
        for forbidden in FORBIDDEN_ADAPTER_METHODS:
            assert not hasattr(adapter, forbidden), f"Unexpected {forbidden} on {type(adapter)}"

    with pytest.raises(ValidationError):
        EdgeSourceConfig(
            source_id="bad-src",
            asset_id=ASSET,
            protocol="opcua_readonly",
            endpoint_url="opc.tcp://opc-ro.factorymind.internal:4840",
            read_only=False,  # type: ignore[arg-type]
            allowed_hosts=("opc-ro.factorymind.internal",),
            node_allowlist=("TP2",),
        )


@pytest.mark.parametrize(
    ("protocol", "url", "allowlist"),
    [
        ("historian_readonly", "http://historian.plant.example/api", ("historian.plant.example",)),
        ("mqtt_readonly", "mqtt://broker.plant.example:1883", ("broker.plant.example",)),
        ("opcua_readonly", "https://opc.plant.example:4840", ("opc.plant.example",)),
        (
            "historian_readonly",
            "https://user:secret@historian.plant.example/api",
            ("historian.plant.example",),
        ),
        (
            "historian_readonly",
            "https://historian.plant.example/api?leak=1",
            ("historian.plant.example",),
        ),
        ("opcua_readonly", "opc.tcp://127.0.0.1:4840", ("127.0.0.1",)),
        ("opcua_readonly", "opc.tcp://127.1:4840", ("127.1",)),
        ("opcua_readonly", "opc.tcp://0177.0.0.1:4840", ("0177.0.0.1",)),
        ("opcua_readonly", "opc.tcp://0x7f000001:4840", ("0x7f000001",)),
        ("opcua_readonly", "opc.tcp://2130706433:4840", ("2130706433",)),
        ("opcua_readonly", "opc.tcp://169.254.169.254:4840", ("169.254.169.254",)),
        ("opcua_readonly", "opc.tcp://[::ffff:169.254.169.254]:4840", ("::ffff:169.254.169.254",)),
        ("opcua_readonly", "opc.tcp://100.64.0.1:4840", ("100.64.0.1",)),
        ("opcua_readonly", "opc.tcp://[::ffff:100.64.0.1]:4840", ("::ffff:100.64.0.1",)),
        ("opcua_readonly", "opc.tcp://100.100.100.200:4840", ("100.100.100.200",)),
        ("opcua_readonly", "opc.tcp://10.1.2.3:4840", ("10.1.2.3",)),
        ("opcua_readonly", "opc.tcp://localhost:4840", ("localhost",)),
        ("opcua_readonly", "opc.tcp://sub.localhost:4840", ("sub.localhost",)),
        (
            "opcua_readonly",
            "opc.tcp://opc-ro.factorymind.internal\n:4840",
            ("opc-ro.factorymind.internal",),
        ),
        (
            "opcua_readonly",
            "opc.tcp://opc-ro.factorymind.internal:0",
            ("opc-ro.factorymind.internal",),
        ),
        (
            "opcua_readonly",
            "opc.tcp://opc-ro.factorymind.internal:/",
            ("opc-ro.factorymind.internal",),
        ),
        (
            "opcua_readonly",
            "opc.tcp://opc-ro.factorymind.internal:4840?",
            ("opc-ro.factorymind.internal",),
        ),
        (
            "opcua_readonly",
            "opc.tcp://metadata.google.internal:4840",
            ("metadata.google.internal",),
        ),
        ("opcua_readonly", "opc.tcp://plc-gateway.local:4840", ("plc-gateway.local",)),
        (
            "mqtt_readonly",
            "mqtts://untrusted.attacker.example:8883",
            ("mqtt-ro.factorymind.internal",),
        ),
    ],
)
def test_validate_edge_endpoint_blocks_ssrf_and_unsafe_urls(
    protocol: str, url: str, allowlist: tuple[str, ...]
) -> None:
    with pytest.raises(EdgeSecurityError):
        validate_edge_endpoint(url, protocol=protocol, allowed_hosts=allowlist)  # type: ignore[arg-type]


# ---------------------------------------------------------------- Cadence, Gaps, Ring Buffer


def test_adapter_cadence_allowlist_and_ring_buffer_eviction() -> None:
    adapter = OpcUaReadOnlyAdapter(DEFAULT_EDGE_SOURCES[0])
    nominal_inputs = [
        EdgeTelemetrySampleInput.model_validate(item) for item in _sample_series(count=4)
    ]
    res = adapter.ingest_batch(nominal_inputs)
    assert res.cadence.cadence_status == "nominal"
    assert res.cadence.observed_median_interval_s == 10.0
    assert res.buffer.state == "connected_readonly"
    assert res.buffer.evicted_count == 0

    gapped_inputs = [
        EdgeTelemetrySampleInput.model_validate(item)
        for item in _sample_series(count=10, gap_after_idx=3, extra_sensor="SETPOINT_WRITE")
    ]
    gapped = adapter.ingest_batch(gapped_inputs, simulate_disconnects=3, buffer_capacity_override=4)
    assert gapped.cadence.cadence_status == "gapped_or_jittered"
    assert gapped.cadence.gap_count == 1
    assert gapped.cadence.unallowlisted_sensors_ignored == ("SETPOINT_WRITE",)
    assert gapped.buffer.retained_count == 4
    assert gapped.buffer.evicted_count == 6
    assert gapped.buffer.backoff_schedule_s == (1.0, 2.0, 4.0)
    assert gapped.buffer.state == "degraded_buffer_evicting"

    with pytest.raises(EdgeValidationError):
        adapter.ingest_batch(nominal_inputs, buffer_capacity_override=0)

    with pytest.raises(ValidationError):
        EdgeTelemetrySampleInput.model_validate(
            {
                "timestamp": "2020-04-15T10:00:00Z",
                "source_id": "metropt3-opcua-ro",
                "sensor_values": {"TP2": True},
            }
        )


# ---------------------------------------------------------------- Supply-Chain & Manifests


def test_ci_actions_pinned_and_root_license_present() -> None:
    repo_root = discover_repo_root()
    report = build_supply_chain_report(repo_root=repo_root)
    assert report.license_present is True
    assert report.ci_actions_sha_pinned is True
    assert report.unpinned_workflow_uses == ()
    assert report.non_pickle_attestation_ready is True
    assert any(entry.status == "sha_pinned" for entry in report.attestations)


def test_audit_ci_workflow_flags_unpinned_tag_and_handles_quotes(tmp_path: Path) -> None:
    bad_ci = tmp_path / "ci.yml"
    bad_ci.write_text("steps:\n  - uses: actions/checkout@v4\n", encoding="utf-8")
    pinned, unpinned, _, _ = audit_ci_workflow(bad_ci)
    assert pinned is False
    assert unpinned == ("actions/checkout@v4",)

    quoted_ci = tmp_path / "quoted.yml"
    quoted_ci.write_text(
        'steps:\n  - uses: "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"\n',
        encoding="utf-8",
    )
    pinned_q, unpinned_q, _, _ = audit_ci_workflow(quoted_ci)
    assert pinned_q is True
    assert unpinned_q == ()


def test_signed_non_pickle_manifest_verification_and_tamper_detection(tmp_path: Path) -> None:
    key = b"factorymind-test-hmac-key"
    manifest = build_signed_manifest(
        model_name="metropt3-twin",
        schema_version=1,
        parameters={"k_charge": 0.948, "k_drain": 0.084},
        hmac_key=key,
    )
    path = tmp_path / "model_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    status_ok, sha_ok, _ = verify_non_pickle_manifest(path, hmac_key=key)
    assert status_ok == "verified"
    assert sha_ok is not None and len(sha_ok) == 64

    # Non-RFC NaN metadata must be rejected cleanly without raising ValueError
    path.write_text(
        json.dumps(manifest).replace('"schema_version": 1', '"schema_version": NaN'),
        encoding="utf-8",
    )
    status_nan, _, _ = verify_non_pickle_manifest(path, hmac_key=key)
    assert status_nan == "tamper_or_hash_mismatch"

    # Parameter tampering must fail cleanly
    manifest["parameters"]["k_drain"] = 0.999
    path.write_text(json.dumps(manifest), encoding="utf-8")
    status_tampered, _, _ = verify_non_pickle_manifest(path, hmac_key=key)
    assert status_tampered == "tamper_or_hash_mismatch"

    # Unsafe pickle format claim must fail cleanly even if JSON parses
    manifest["artifact_format"] = "pickle"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    status_pickle, _, _ = verify_non_pickle_manifest(path, hmac_key=key)
    assert status_pickle == "tamper_or_hash_mismatch"


def test_supply_chain_report_audits_runtime_models_dir(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    vdir = models_dir / "metropt3-anomaly" / "v1"
    vdir.mkdir(parents=True)
    (models_dir / "metropt3-anomaly" / "CURRENT").write_text("v1", encoding="utf-8")
    blob = b"fake-bundle-bytes"
    (vdir / "model.joblib").write_bytes(blob)
    (vdir / "model_card.json").write_text(
        json.dumps({"artifact_sha256": hashlib.sha256(blob).hexdigest()}),
        encoding="utf-8",
    )

    # Without model_manifest.json present, non_pickle_attestation_ready is honestly False
    missing_manifest_report = build_supply_chain_report(
        models_dir=models_dir, model_name="metropt3-anomaly"
    )
    assert missing_manifest_report.non_pickle_attestation_ready is False
    by_format_missing = {item.format: item for item in missing_manifest_report.attestations}
    assert by_format_missing["json_manifest"].status == "missing"

    # Traversal / null-byte CURRENT must never crash or pass
    (models_dir / "metropt3-anomaly" / "CURRENT").write_text("v1\x00evil", encoding="utf-8")
    bad_ptr_report = build_supply_chain_report(models_dir=models_dir, model_name="metropt3-anomaly")
    assert bad_ptr_report.non_pickle_attestation_ready is False
    assert any(item.status == "tamper_or_hash_mismatch" for item in bad_ptr_report.attestations)

    # Restoring valid pointer + signed JSON manifest passes cleanly
    (models_dir / "metropt3-anomaly" / "CURRENT").write_text("v1", encoding="utf-8")
    manifest = build_signed_manifest(
        model_name="metropt3-anomaly",
        schema_version=1,
        parameters={"threshold": 0.5},
    )
    (vdir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = build_supply_chain_report(models_dir=models_dir, model_name="metropt3-anomaly")
    assert report.non_pickle_attestation_ready is True
    by_format = {item.format: item for item in report.attestations}
    assert by_format["json_manifest"].status == "verified"
    assert by_format["joblib_legacy"].status == "legacy_pickle_advisory"


# ---------------------------------------------------------------- API Routes


def test_edge_api_sources_validation_hardening_and_rbac(kit: AuthKit) -> None:
    client = make_client(kit)
    engineer = kit.headers(roles=["engineer"])
    auditor = kit.headers(roles=["auditor"])
    other_asset = kit.headers(roles=["engineer"], assets=["other-asset"])

    assert client.get("/api/v1/edge/sources", headers=auditor).status_code == 403

    sources = client.get("/api/v1/edge/sources", headers=engineer)
    assert sources.status_code == 200
    body = sources.json()
    assert body["advisory_only"] is True
    assert body["write_capabilities"] == "none_disabled_by_architecture"
    assert len(body["sources"]) == 3
    assert all(s["read_only"] is True for s in body["sources"])

    # Asset access isolation (require_data_source denies ungranted assets with 403)
    assert (
        client.get(f"/api/v1/edge/sources?asset_id={ASSET}", headers=other_asset).status_code == 403
    )
    assert (
        client.get("/api/v1/edge/sources?asset_id=unknown-asset", headers=engineer).status_code
        == 404
    )

    # Batch validation endpoint
    resp = client.post(
        "/api/v1/edge/validate-sample",
        json={
            "source_id": "metropt3-opcua-ro",
            "asset_id": ASSET,
            "samples": _sample_series(count=5),
            "simulate_disconnects": 2,
        },
        headers=engineer,
    )
    assert resp.status_code == 200
    val = resp.json()
    assert val["cadence"]["cadence_status"] == "nominal"
    assert val["buffer"]["backoff_schedule_s"] == [1.0, 2.0]
    assert val["buffer"]["state"] == "reconnecting_backoff"

    # Unknown source -> 404; ungranted asset -> 403; bad telemetry/mismatched source -> 422
    assert (
        client.post(
            "/api/v1/edge/validate-sample",
            json={
                "source_id": "unknown-src",
                "asset_id": ASSET,
                "samples": _sample_series(count=2),
            },
            headers=engineer,
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/api/v1/edge/validate-sample",
            json={
                "source_id": "metropt3-opcua-ro",
                "asset_id": ASSET,
                "samples": _sample_series(count=2),
            },
            headers=other_asset,
        ).status_code
        == 403
    )
    bad_samples = _sample_series(count=2)
    bad_samples[1]["source_id"] = "metropt3-mqtt-ro"
    assert (
        client.post(
            "/api/v1/edge/validate-sample",
            json={
                "source_id": "metropt3-opcua-ro",
                "asset_id": ASSET,
                "samples": bad_samples,
            },
            headers=engineer,
        ).status_code
        == 422
    )
    nan_payload = (
        b'{"source_id":"metropt3-opcua-ro","asset_id":"metropt3-apu-1","samples":['
        b'{"timestamp":"2020-04-15T10:00:00Z","source_id":"metropt3-opcua-ro",'
        b'"sensor_values":{"TP2":NaN}},'
        b'{"timestamp":"2020-04-15T10:00:10Z","source_id":"metropt3-opcua-ro",'
        b'"sensor_values":{"TP2":8.5}}]}'
    )
    assert (
        client.post(
            "/api/v1/edge/validate-sample",
            content=nan_payload,
            headers={**engineer, "Content-Type": "application/json"},
        ).status_code
        == 422
    )

    hardening = client.get("/api/v1/edge/hardening", headers=engineer)
    assert hardening.status_code == 200
    hbody = hardening.json()
    assert hbody["license_present"] is True
    assert hbody["ci_actions_sha_pinned"] is True
    assert hbody["non_pickle_attestation_ready"] is True
