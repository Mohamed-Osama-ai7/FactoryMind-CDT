# M5-WP0 (`WP-H`) & WP-P0A — Edge Source Descriptors, Preflight Validation, and Supply-Chain Checks

M5-WP0 added descriptor contracts, stateless batch preflight checks, repository workflow pinning, and the `/edge` view. **WP-P0A** remediates overclaimed security and connectivity language so every API response, test, UI view, and document states only what is actually implemented and checked at runtime.

---

## 1. Connector Scope: Descriptors & Preflight Only (`factorymind.connectors.edge`)

**OPC-UA, MQTT, and historian connector I/O is NOT implemented** (`connector_io_status="not_implemented"`). This repository installs no OPC-UA, MQTT, or historian client library, opens no network sockets, never connects to plant systems, and reads no live plant data:

- **`OpcUaReadOnlyAdapter` (`opcua_readonly`), `MqttReadOnlyAdapter` (`mqtt_readonly`), `HistorianReadOnlyAdapter` (`historian_readonly`):** Compatibility class names for protocol-specific **source descriptors** (`connection_state="descriptor_only_not_connected"`), not live connectors. The absence of write methods holds trivially while no client exists; any future real connector must independently design and test read-only enforcement.
- **URL-String Policy Check (`validate_edge_endpoint`, `endpoint_check_scope="url_string_only_no_dns_resolution"`):** Checks raw endpoint URL text against scheme rules (`opc.tcp`, `mqtts`, `https`), embedded credentials, query/fragment components, control characters, non-canonical numeric/octal/hex IPv4 literals, non-global/loopback/link-local/CGNAT IPs, metadata hostnames, `.local`/`.localhost` suffixes, and `allowed_hosts`. **No DNS resolution or connect-time IP pinning is performed.**
- **Stateless Batch Preflight (`ingest_batch`, `mode="preflight_validation_only"`, `real_connector_io=False`):** Validates caller-submitted sample batches against allowlisted sensor names, finite values, sampling cadence (`~0.1 Hz` / `10 s`), uninterpolated gaps (`> 2 * nominal_cadence_s`), out-of-order/duplicate timestamps, in-memory ring-buffer capacity, and synthetic disconnect backoff schedules (`simulated_reconnect_attempts`).

---

## 2. Supply-Chain & Model Artifact Checks (`factorymind.connectors.attestation`)

Every field returned by [`build_supply_chain_report`](backend/src/factorymind/connectors/attestation.py) reflects the concrete check executed and explicitly discloses its limits in `limitations`:

- **`LICENSE` (`present` / `missing` / `tamper_or_hash_mismatch`):** Verifies the root MIT `LICENSE` file exists and contains MIT license phrases.
- **GitHub Actions CI (`.github/workflows/ci.yml`, `sha_pinned` / `tamper_or_hash_mismatch`):** Verifies via `audit_ci_workflow()` that every `uses:` line in `ci.yml` uses a 40-hex commit pin syntax (`owner/repo@<40-hex>`). It does **not** prove upstream commits exist or are authentic, and does not check lockfiles or transitive dependencies.
- **Runtime Model Bundle (`runtime_model_format="joblib_pickle"` / `"unknown"`):** The runtime loader (`AnomalyService`) still loads `model.joblib` using `joblib` (**pickle**) after comparing its SHA-256 against `model_card.json` in the same directory (`sha_matches_adjacent_card`). That detects file corruption, not an adversary able to modify both files.
- **Non-Pickle Manifest (`json_parameters_v1`, `verify_non_pickle_manifest`):** No shipped model includes `model_manifest.json`, `runtime_loader_consults_manifest=False`, and `manifest_signature_key_configured=False` in the default API deployment. Without an HMAC key, the best status is `self_consistent_unsigned` (digest computed over manifest JSON parameters only, never verified as a signature and never covering `model.joblib` bytes). `signature_verified` requires an explicit HMAC key (`hmac_key`) that verifies against `hmac_sha256`. Missing, unconfigured, or invalid states are reported as `no_current_model`, `invalid_pointer`, `manifest_missing`, `manifest_invalid`, or `integrity_failed`.

---

## 3. Safe API, Boundaries & `/edge` UI

Route paths use `/api/v1/edge/...` and expose zero write or actuation operations:

- `GET  /api/v1/edge/sources` (`Permission.ASSET_READ`): Lists registered source descriptors scoped by asset permissions (`connector_io_status="not_implemented"`, `write_capabilities="none_no_connector_io_exists"`).
- `POST /api/v1/edge/validate-sample` (`Permission.ASSET_READ`, allowlisted stateless endpoint in `test_safety_boundary.py`): Runs caller-submitted batch preflight validation (`mode="preflight_validation_only"`, `real_connector_io=False`).
- `GET  /api/v1/edge/hardening` (`Permission.ASSET_READ`): Runs repository and runtime-artifact checks and returns explicit `runtime_model_attestation`, `runtime_loader_consults_manifest=False`, `manifest_signature_key_configured`, per-artifact `attestations`, and `limitations`.
- **Frontend (`/edge`, `EdgeConnectorsWorkspace.tsx`, `routes/edge.tsx`):** Linked in the sidebar as **`Edge Preflight`** (`asset:read`), displaying explicit non-connected descriptor state, URL-text check scope, supply-chain limitations, unavailable/unverified/failed statuses, and the stateless batch preflight form.

---

## 4. Verified Test Results Across the Repository

- **Backend:** `uv run ruff format --check .` (`112 files`), `uv run ruff check .`, `uv run mypy` (`75 source files`), `uv run pytest -q` (**604 passed**, including socket/DNS/urllib no-I/O checks, HMAC-vs-unsigned distinction checks, and unavailable/failed attestation state tests in `tests/test_edge_connectors.py`).
- **Frontend:** `npm run lint`, `npm run typecheck`, `npm test` (**17 files, 63 tests passed**), `npm run build`, and `npm run generate:api` Orval client parity.
