"""Unit, security, failure-path, and API tests for WP-P0A edge preflight and attestation."""

from __future__ import annotations

import hashlib
import json
import socket
import urllib.request
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
    list_edge_sources,
    validate_edge_batch,
    validate_edge_endpoint,
    verify_non_pickle_manifest,
)
from factorymind.contracts.connectors import (
    EdgeSampleValidationRequest,
    EdgeSourceConfig,
    EdgeTelemetrySampleInput,
)
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
    only_sensor: str | None = None,
) -> list[dict[str, object]]:
    base = datetime(2020, 4, 15, 10, 0, 0, tzinfo=UTC)
    t = base
    rows: list[dict[str, object]] = []
    for idx in range(count):
        if only_sensor is not None:
            sensors: dict[str, float] = {only_sensor: 1.0}
        else:
            sensors = {"TP2": 8.4 + idx * 0.05, "TP3": 8.6 + idx * 0.05}
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


# ---------------------------------------------------------------- Descriptor & No-I/O Boundaries


def test_descriptors_expose_no_write_methods_and_no_socket_io(
    kit: AuthKit, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _forbid_network(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("Descriptor-only edge preflight must perform zero network/DNS I/O")

    with monkeypatch.context() as mp:
        mp.setattr(socket, "socket", _forbid_network)
        mp.setattr(socket, "create_connection", _forbid_network)
        mp.setattr(socket, "getaddrinfo", _forbid_network)
        mp.setattr(socket, "gethostbyname", _forbid_network)
        mp.setattr(urllib.request, "urlopen", _forbid_network)

        descriptors = [
            OpcUaReadOnlyAdapter(DEFAULT_EDGE_SOURCES[0]),
            MqttReadOnlyAdapter(DEFAULT_EDGE_SOURCES[1]),
            HistorianReadOnlyAdapter(DEFAULT_EDGE_SOURCES[2]),
        ]
        for desc in descriptors:
            assert desc.read_only is True
            snap = desc.snapshot_status()
            assert snap.connection_state == "descriptor_only_not_connected"
            assert snap.connector_io == "not_implemented"
            assert snap.endpoint_url_policy == "passed"
            assert snap.endpoint_check_scope == "url_string_only_no_dns_resolution"
            for forbidden in FORBIDDEN_ADAPTER_METHODS:
                assert not hasattr(desc, forbidden), f"Unexpected {forbidden} on {type(desc)}"

        sources = list_edge_sources(allowed_assets=frozenset({ASSET}))
        assert sources.connector_io_status == "not_implemented"
        assert sources.write_capabilities == "none_no_connector_io_exists"

        req = EdgeSampleValidationRequest.model_validate(
            {
                "source_id": "metropt3-opcua-ro",
                "asset_id": ASSET,
                "samples": _sample_series(count=3),
            }
        )
        batch_res = validate_edge_batch(req)
        assert batch_res.mode == "preflight_validation_only"
        assert batch_res.real_connector_io is False

    # HTTP routes must also run with zero outbound connects or DNS lookups
    with make_client(kit) as client:
        monkeypatch.setattr(socket.socket, "connect", _forbid_network)
        monkeypatch.setattr(socket, "create_connection", _forbid_network)
        monkeypatch.setattr(socket, "getaddrinfo", _forbid_network)
        monkeypatch.setattr(socket, "gethostbyname", _forbid_network)
        monkeypatch.setattr(urllib.request, "urlopen", _forbid_network)
        engineer = kit.headers(roles=["engineer"])
        assert client.get("/api/v1/edge/sources", headers=engineer).status_code == 200
        assert (
            client.post(
                "/api/v1/edge/validate-sample",
                json={
                    "source_id": "metropt3-opcua-ro",
                    "asset_id": ASSET,
                    "samples": _sample_series(count=3),
                },
                headers=engineer,
            ).status_code
            == 200
        )
        assert client.get("/api/v1/edge/hardening", headers=engineer).status_code == 200

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


def test_descriptor_reports_rejected_url_policy_and_protocol_mismatch() -> None:
    desc = OpcUaReadOnlyAdapter(DEFAULT_EDGE_SOURCES[0])
    # Simulate config mutation to an unallowlisted/SSRF target after construction
    desc.config = desc.config.model_copy(update={"endpoint_url": "opc.tcp://127.0.0.1:4840"})
    snap = desc.snapshot_status()
    assert snap.endpoint_url_policy == "rejected"
    with pytest.raises(EdgeSecurityError):
        desc.ingest_batch(
            [EdgeTelemetrySampleInput.model_validate(item) for item in _sample_series(count=2)]
        )

    with pytest.raises(EdgeValidationError):
        MqttReadOnlyAdapter(DEFAULT_EDGE_SOURCES[0])


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


def test_preflight_cadence_allowlist_gaps_and_failure_cases() -> None:
    adapter = OpcUaReadOnlyAdapter(DEFAULT_EDGE_SOURCES[0])
    nominal_inputs = [
        EdgeTelemetrySampleInput.model_validate(item) for item in _sample_series(count=4)
    ]
    res = adapter.ingest_batch(nominal_inputs)
    assert res.mode == "preflight_validation_only"
    assert res.real_connector_io is False
    assert res.endpoint_url_policy_ok is True
    assert res.endpoint_check_scope == "url_string_only_no_dns_resolution"
    assert res.cadence.cadence_status == "nominal"
    assert res.cadence.observed_median_interval_s == 10.0
    assert res.buffer.state == "batch_within_capacity"
    assert res.buffer.evicted_count == 0
    assert res.buffer.simulated_reconnect_attempts == 0

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
    assert gapped.buffer.simulated_reconnect_attempts == 3
    assert gapped.buffer.backoff_schedule_s == (1.0, 2.0, 4.0)
    assert gapped.buffer.state == "batch_exceeds_capacity"

    # Unallowlisted-only samples produce stale_or_invalid cleanly without network/interval math bugs
    unallowlisted_only = [
        EdgeTelemetrySampleInput.model_validate(item)
        for item in _sample_series(count=3, only_sensor="UNKNOWN_TAG")
    ]
    stale = adapter.ingest_batch(unallowlisted_only)
    assert stale.cadence.cadence_status == "stale_or_invalid"
    assert stale.cadence.observed_median_interval_s is None
    assert stale.buffer.retained_count == 0

    # Out-of-order and duplicate timestamps trigger accurate diagnostics
    raw_order = _sample_series(count=3)
    raw_order[1]["timestamp"] = raw_order[0]["timestamp"]
    raw_order[2]["timestamp"] = "2020-04-15T09:59:50+00:00"
    ordered_res = adapter.ingest_batch(
        [EdgeTelemetrySampleInput.model_validate(item) for item in raw_order]
    )
    assert ordered_res.cadence.duplicate_timestamp_count == 1
    assert ordered_res.cadence.out_of_order_count == 1
    assert ordered_res.cadence.cadence_status == "stale_or_invalid"

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
    assert report.runtime_model_attestation == "not_checked"
    assert report.runtime_model_format == "unknown"
    assert report.runtime_loader_consults_manifest is False
    assert report.manifest_signature_key_configured is False
    assert report.limitations
    by_name = {entry.name: entry for entry in report.attestations}
    assert by_name["Root Repository License"].status == "present"
    assert by_name["GitHub Actions CI Workflow"].status == "sha_pinned"
    assert by_name["MetroPT-3 Experiment Config"].status == "unattested"


def test_repo_checks_report_missing_unpinned_and_malformed_artifacts(tmp_path: Path) -> None:
    # Empty directory -> missing LICENSE, missing CI workflow, missing config JSON
    missing_report = build_supply_chain_report(repo_root=tmp_path)
    assert missing_report.license_present is False
    assert missing_report.ci_actions_sha_pinned is False
    assert {e.status for e in missing_report.attestations} == {"missing", "tamper_or_hash_mismatch"}

    # Non-MIT license and malformed JSON config must fail cleanly
    (tmp_path / "LICENSE").write_text("Proprietary license text", encoding="utf-8")
    (tmp_path / ".github" / "workflows").mkdir(parents=True)
    (tmp_path / ".github" / "workflows" / "ci.yml").write_text(
        "steps:\n  - uses: actions/checkout@v4\n", encoding="utf-8"
    )
    (tmp_path / "backend" / "configs").mkdir(parents=True)
    (tmp_path / "backend" / "configs" / "metropt3_anomaly.json").write_text(
        '{"seed": NaN}', encoding="utf-8"
    )
    bad_repo_report = build_supply_chain_report(repo_root=tmp_path)
    assert bad_repo_report.license_present is False
    assert bad_repo_report.ci_actions_sha_pinned is False
    assert bad_repo_report.unpinned_workflow_uses == ("actions/checkout@v4",)
    by_path = {e.relative_path: e for e in bad_repo_report.attestations}
    assert by_path["LICENSE"].status == "tamper_or_hash_mismatch"
    assert by_path[".github/workflows/ci.yml"].status == "tamper_or_hash_mismatch"
    assert by_path["backend/configs/metropt3_anomaly.json"].status == "tamper_or_hash_mismatch"


def test_audit_ci_workflow_flags_unpinned_tag_and_handles_quotes(tmp_path: Path) -> None:
    bad_ci = tmp_path / "ci.yml"
    bad_ci.write_text("steps:\n  - uses: actions/checkout@v4\n", encoding="utf-8")
    pinned, unpinned, _, _ = audit_ci_workflow(bad_ci)
    assert pinned is False
    assert unpinned == ("actions/checkout@v4",)

    empty_ci = tmp_path / "empty.yml"
    empty_ci.write_text("name: Empty\n", encoding="utf-8")
    pinned_e, unpinned_e, _, _ = audit_ci_workflow(empty_ci)
    assert pinned_e is False
    assert unpinned_e == ()

    quoted_ci = tmp_path / "quoted.yml"
    quoted_ci.write_text(
        'steps:\n  - uses: "actions/checkout@11d5960a326750d5838078e36cf38b85af677262"\n',
        encoding="utf-8",
    )
    pinned_q, unpinned_q, _, _ = audit_ci_workflow(quoted_ci)
    assert pinned_q is True
    assert unpinned_q == ()


def test_unsigned_and_signed_manifest_verification_distinction(tmp_path: Path) -> None:
    key = b"factorymind-test-hmac-key"
    # 1. Unsigned self-consistent manifest must NEVER count as signature_verified
    unsigned = build_signed_manifest(
        model_name="metropt3-twin",
        schema_version=1,
        parameters={"k_charge": 0.948, "k_drain": 0.084},
        hmac_key=None,
    )
    path = tmp_path / "model_manifest.json"
    path.write_text(json.dumps(unsigned), encoding="utf-8")

    st_unsigned, sha_unsigned, _ = verify_non_pickle_manifest(path, hmac_key=None)
    assert st_unsigned == "self_consistent_unsigned"
    assert sha_unsigned is not None and len(sha_unsigned) == 64

    # Asking to verify signatures when manifest lacks hmac_sha256 must fail
    st_missing_sig, _, _ = verify_non_pickle_manifest(path, hmac_key=key)
    assert st_missing_sig == "tamper_or_hash_mismatch"

    # 2. Signed manifest with key -> unsigned check still reports self_consistent_unsigned!
    signed = build_signed_manifest(
        model_name="metropt3-twin",
        schema_version=1,
        parameters={"k_charge": 0.948, "k_drain": 0.084},
        hmac_key=key,
    )
    path.write_text(json.dumps(signed), encoding="utf-8")

    st_no_key, _, _ = verify_non_pickle_manifest(path, hmac_key=None)
    assert st_no_key == "self_consistent_unsigned"

    # Valid key establishes signature_verified; wrong key rejects
    st_verified, _, _ = verify_non_pickle_manifest(path, hmac_key=key)
    assert st_verified == "signature_verified"
    st_wrong_key, _, _ = verify_non_pickle_manifest(path, hmac_key=b"wrong-key")
    assert st_wrong_key == "tamper_or_hash_mismatch"

    # Non-RFC NaN metadata must be rejected cleanly without raising ValueError
    path.write_text(
        json.dumps(signed).replace('"schema_version": 1', '"schema_version": NaN'),
        encoding="utf-8",
    )
    st_nan, _, _ = verify_non_pickle_manifest(path, hmac_key=key)
    assert st_nan == "tamper_or_hash_mismatch"

    # Parameter tampering must fail cleanly
    signed["parameters"]["k_drain"] = 0.999
    path.write_text(json.dumps(signed), encoding="utf-8")
    st_tampered, _, _ = verify_non_pickle_manifest(path, hmac_key=key)
    assert st_tampered == "tamper_or_hash_mismatch"

    # Unsafe pickle format claim must fail cleanly even if JSON parses
    signed["artifact_format"] = "pickle"
    path.write_text(json.dumps(signed), encoding="utf-8")
    st_pickle, _, _ = verify_non_pickle_manifest(path, hmac_key=key)
    assert st_pickle == "tamper_or_hash_mismatch"


def test_runtime_model_attestation_unavailable_unverified_and_failed_states(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    model_root = models_dir / "metropt3-anomaly"
    vdir = model_root / "v1"
    vdir.mkdir(parents=True)

    # 1. Missing CURRENT -> no_current_model
    no_curr = build_supply_chain_report(models_dir=models_dir, model_name="metropt3-anomaly")
    assert no_curr.runtime_model_attestation == "no_current_model"
    assert no_curr.runtime_model_format == "unknown"

    # 2. Traversal / null-byte / missing target dir -> invalid_pointer
    for bad_ptr in ("v1\x00evil", "../outside", "nonexistent-version"):
        (model_root / "CURRENT").write_text(bad_ptr, encoding="utf-8")
        rep = build_supply_chain_report(models_dir=models_dir, model_name="metropt3-anomaly")
        assert rep.runtime_model_attestation == "invalid_pointer"

    # 3. Valid joblib + card without manifest -> manifest_missing + sha_matches_adjacent_card
    (model_root / "CURRENT").write_text("v1", encoding="utf-8")
    blob = b"fake-bundle-bytes"
    (vdir / "model.joblib").write_bytes(blob)
    (vdir / "model_card.json").write_text(
        json.dumps({"artifact_sha256": hashlib.sha256(blob).hexdigest()}),
        encoding="utf-8",
    )
    missing_manifest = build_supply_chain_report(
        models_dir=models_dir, model_name="metropt3-anomaly"
    )
    assert missing_manifest.runtime_model_attestation == "manifest_missing"
    assert missing_manifest.runtime_model_format == "joblib_pickle"
    assert missing_manifest.runtime_loader_consults_manifest is False
    by_fmt = {item.format: item for item in missing_manifest.attestations}
    assert by_fmt["json_manifest"].status == "missing"
    assert by_fmt["joblib_legacy"].status == "sha_matches_adjacent_card"

    # 4. Signed manifest requires valid hmac_key; otherwise self_consistent_unsigned
    key = b"runtime-hmac-key"
    manifest = build_signed_manifest(
        model_name="metropt3-anomaly",
        schema_version=1,
        parameters={"threshold": 0.5},
        hmac_key=key,
    )
    (vdir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    unkeyed = build_supply_chain_report(models_dir=models_dir, model_name="metropt3-anomaly")
    assert unkeyed.runtime_model_attestation == "manifest_unsigned_self_consistent"
    assert unkeyed.manifest_signature_key_configured is False
    assert {i.format: i for i in unkeyed.attestations}["json_manifest"].status == (
        "self_consistent_unsigned"
    )

    keyed = build_supply_chain_report(
        models_dir=models_dir, model_name="metropt3-anomaly", hmac_key=key
    )
    assert keyed.runtime_model_attestation == "manifest_signature_verified"
    assert keyed.manifest_signature_key_configured is True
    assert {i.format: i for i in keyed.attestations}["json_manifest"].status == (
        "signature_verified"
    )

    wrong_keyed = build_supply_chain_report(
        models_dir=models_dir, model_name="metropt3-anomaly", hmac_key=b"wrong"
    )
    assert wrong_keyed.runtime_model_attestation == "manifest_invalid"

    # 5. Tampered model.joblib fails integrity even if manifest HMAC verifies!
    (vdir / "model.joblib").write_bytes(b"corrupted-bundle-bytes")
    tampered_joblib = build_supply_chain_report(
        models_dir=models_dir, model_name="metropt3-anomaly", hmac_key=key
    )
    assert tampered_joblib.runtime_model_attestation == "integrity_failed"
    assert {i.format: i for i in tampered_joblib.attestations}["joblib_legacy"].status == (
        "tamper_or_hash_mismatch"
    )


# ---------------------------------------------------------------- API Routes


def test_edge_api_sources_validation_hardening_and_rbac(kit: AuthKit) -> None:
    client = make_client(kit)
    engineer = kit.headers(roles=["engineer"])
    auditor = kit.headers(roles=["auditor"])
    other_asset = kit.headers(roles=["engineer"], assets=["other-asset"])

    assert client.get("/api/v1/edge/sources").status_code == 401
    assert client.get("/api/v1/edge/sources", headers=auditor).status_code == 403

    sources = client.get("/api/v1/edge/sources", headers=engineer)
    assert sources.status_code == 200
    body = sources.json()
    assert body["advisory_only"] is True
    assert body["connector_io_status"] == "not_implemented"
    assert body["write_capabilities"] == "none_no_connector_io_exists"
    assert len(body["sources"]) == 3
    for src in body["sources"]:
        assert src["read_only"] is True
        assert src["connection_state"] == "descriptor_only_not_connected"
        assert src["connector_io"] == "not_implemented"
        assert src["endpoint_url_policy"] == "passed"
        assert src["endpoint_check_scope"] == "url_string_only_no_dns_resolution"

    # Asset access isolation (require_data_source denies ungranted assets with 403)
    assert (
        client.get(f"/api/v1/edge/sources?asset_id={ASSET}", headers=other_asset).status_code == 403
    )
    assert (
        client.get("/api/v1/edge/sources?asset_id=unknown-asset", headers=engineer).status_code
        == 404
    )

    # Batch preflight validation endpoint
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
    assert val["mode"] == "preflight_validation_only"
    assert val["real_connector_io"] is False
    assert val["endpoint_url_policy_ok"] is True
    assert val["endpoint_check_scope"] == "url_string_only_no_dns_resolution"
    assert val["cadence"]["cadence_status"] == "nominal"
    assert val["buffer"]["backoff_schedule_s"] == [1.0, 2.0]
    assert val["buffer"]["simulated_reconnect_attempts"] == 2
    assert val["buffer"]["state"] == "simulated_reconnect_backoff"

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
    assert (
        client.post(
            "/api/v1/edge/validate-sample",
            json={
                "source_id": "metropt3-opcua-ro",
                "asset_id": ASSET,
                "samples": _sample_series(count=1),
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
    assert hbody["runtime_model_attestation"] == "no_current_model"
    assert hbody["runtime_loader_consults_manifest"] is False
    assert hbody["manifest_signature_key_configured"] is False
    assert len(hbody["limitations"]) >= 4
