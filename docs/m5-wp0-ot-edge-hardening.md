# M5-WP0 (`WP-H`) — Read-Only OT/Edge Connector Foundations & Supply-Chain Hardening

M5-WP0 closes the FactoryMind CDT master roadmap by adding strictly read-only industrial telemetry connector foundations, SSRF/host-allowlist and bounded ring-buffer validation, repository supply-chain hardening (`LICENSE` + SHA-pinned GitHub Actions CI + non-pickle HMAC/SHA-256 manifest verification), an `/edge` diagnostics UI, and end-to-end verification across all milestones M0–M5.

---

## 1. Read-Only OT & Edge Architecture (`factorymind.connectors.edge`)

All adapters strictly prohibit write, command, setpoint, publish, or physical actuation interfaces at both contract (`read_only: Literal[True] = True`) and class levels:

- **`OpcUaReadOnlyAdapter` (`opcua_readonly`):** `opc.tcp://` read-only tag subscription descriptor.
- **`MqttReadOnlyAdapter` (`mqtt_readonly`):** `mqtts://` TLS read-only topic subscriber descriptor.
- **`HistorianReadOnlyAdapter` (`historian_readonly`):** `https://` read-only time-series pull descriptor.
- **SSRF & URI Hygiene (`validate_edge_endpoint`):** Rejects insecure schemes (`http://`, `mqtt://`, `file://`), embedded URL userinfo credentials, query/fragment smuggling, non-public/loopback/link-local/IPv4-mapped IPv6 literals (`127.0.0.1`, `169.254.169.254`, `10.*`, `::1`, `::ffff:169.254.169.254`), `localhost`/`.local`/cloud-metadata hostnames (even if misconfigured inside `allowed_hosts`), and un-allowlisted hosts.
- **Cadence, Gap Honesty & Bounded Ring Buffers:** Validates MetroPT-3 `~0.1 Hz` (`10 s`) sampling cadence, rejects non-allowlisted sensor keys, counts out-of-order/duplicate timestamps, preserves gaps (`> 2 * nominal_cadence_s`) uninterpolated, enforces bounded FIFO ring buffers (`collections.deque`), and computes exponential reconnect backoff (`min(60s, 2^attempt)`).

---

## 2. Supply-Chain & Model Artifact Hardening (`factorymind.connectors.attestation`)

- **Root `LICENSE`:** MIT license at repository root matching `backend/pyproject.toml` (`license = "MIT"`).
- **SHA-Pinned GitHub Actions (`.github/workflows/ci.yml`):** Every `uses:` action (`actions/checkout`, `astral-sh/setup-uv`, `actions/setup-node`) is pinned to its verified 40-hex commit SHA and regression-checked via `audit_ci_workflow()` in pytest.
- **Non-Pickle Model Attestation (`build_signed_manifest` / `verify_non_pickle_manifest`):** Canonical sorted JSON (`json_parameters_v1`) verification with SHA-256 parameter digest and constant-time HMAC-SHA256 verification without executing `pickle`/`joblib` deserialization, while runtime `build_supply_chain_report()` audits `LICENSE`, CI pinning, configs, and runtime model bundles.

---

## 3. Safe API, Safety Boundaries & `/edge` UI

Route paths intentionally use `/api/v1/edge/...` and expose zero write/actuation primitives:

- `GET  /api/v1/edge/sources` (`Permission.ASSET_READ`): Tenant/asset-scoped listing of read-only registered adapters and SSRF status.
- `POST /api/v1/edge/validate-sample` (`Permission.ASSET_READ`, allowlisted stateless validator in `test_safety_boundary.py`): Validates telemetry batches, simulated disconnects, and ring-buffer eviction.
- `GET  /api/v1/edge/hardening` (`Permission.ASSET_READ`): Returns repository license, CI SHA pinning, artifact attestations, and cross-system verified benchmarks.
- **Frontend (`/edge`, `EdgeConnectorsWorkspace.tsx`, `routes/edge.tsx`):** Accessible via the `OT & Edge Health` sidebar link (`asset:read`), displaying read-only adapter posture, supply-chain audit status, M2–M5 verified metrics, and an interactive stateless cadence/ring-buffer validator.

---

## 4. Final Cross-System Ablation & Verification Summary (M0–M5)

| Layer / Milestone | Authoritative Verified Result |
| --- | --- |
| **M2-WP0 / WP2 Anomaly Detection (Real MetroPT-3)** | `duty_rule` LOEO `4/4` leaks, `0.034/0.044` FA/day (`wp2_integrity_gated` `0.026/0.035`), `50 min` median `\|delay\|`, `1` frozen leak prevented, `34` warnings |
| **M2-WP3 / WP4 Evidence & RCA (Real MetroPT-3)** | `mean_completeness_score 0.9266`, `1.0` grounding/agreement/disclaimer on `F1..F4`, `0.6765` warning non-leak rate |
| **M3-WP1 Calibration & OOD (AI4I 2020 Separate Bench)** | `ai4i-hgb` `ROC-AUC 0.9803`, `Brier 0.0124`, `ECE 0.0054`; Ledoit-Wolf Mahalanobis OOD abstention |
| **M3-WP2 Hybrid RAG + KG** | `Recall@5 1.0`, `MRR@10 1.0`, `nDCG@5 0.7683` across `61` live chunks |
| **M4-WP0 Physics Digital Twin** | `168` held-out segments: `10m MAE 0.2526 bar`, `60m MAE 0.5218 bar`, `RMSE 0.5853`, `0` violations, `3.78x` leak ratio |
| **M4-WP1 / WP2 Optimization & Copilot** | 4-tier impact stratification + constrained optimizer; `24/24` red-team probes blocked, `5/5` benign queries allowed, strict `[C1..Cn]` subset check |
| **M5-WP0 OT/Edge & Full Suite** | Zero PLC write methods, SSRF IP/scheme/host enforcement, MIT `LICENSE`, SHA-pinned CI, **`590` backend tests & `62` frontend tests (17 files) passing** |

