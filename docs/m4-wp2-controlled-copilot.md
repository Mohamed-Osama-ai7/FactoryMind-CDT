# M4-WP2 (`WP-G`) — Controlled Industrial Copilot

M4-WP2 adds a permission-scoped, strictly read-only **Controlled Industrial Copilot** orchestrating across stored incident evidence, RCA hypotheses, partition-isolated Hybrid RAG documentation, typed Knowledge Graph traversal, the calibrated Digital Twin, and constrained maintenance optimization.

---

## 1. Read-Only Tool Orchestration & Strict Schemas (`factorymind.copilot.service`)

Every tool enforces principal permissions + tenant/asset grants before execution and prohibits physical actuation:

- `get_event_evidence` & `get_rca_hypotheses`: `Permission.INCIDENT_READ` + `IncidentRepo` tenant/asset scoping.
- `search_docs`: `Permission.ASSET_READ` + partition-isolated `HybridRagIndex`.
- `query_kg`: `Permission.ASSET_READ` + typed `IndustrialKnowledgeGraph` (causal vs correlational distinction).
- `simulate_scenario`: `Permission.ASSET_READ` (`metropt3-apu-1` APU digital twin, `simulated_projection`).
- `compare_options`: `Permission.ASSET_READ` (`repair_now` vs `schedule` vs `monitor` under safety constraints).

---

## 2. Offline Citation Grounding & Red-Team Firewall

- **Input Red-Team Firewall (`detect_safety_or_injection_risk`):** Immediately refuses PLC write / valve actuation commands and prompt-injection attempts (`safety_refusal_triggered=True`, `abstained=True`, all tools skipped).
- **Prompt Sanitization & Output Verification:** Strips suspicious instruction directives from corpus/user text before optional LLM invocation and falls back deterministically (`synthesis_mode="deterministic_grounded"`) if an external LLM output omits verified citations `[C1..Cn]` or emits unsafe actuation phrasing.

---

## 3. API & `/copilot` UI Workspace

- `GET /api/v1/copilot/tools` (`Permission.ASSET_READ`): Lists all 6 read-only tools and caller permissions.
- `POST /api/v1/copilot/query` (`Permission.ASSET_READ`, allowlisted in `test_safety_boundary.py`): Returns structured citations, tool execution trace, red-team flags, and causal-vs-correlation reminder.
- **Frontend (`/copilot`, `CopilotWorkspace.tsx`, `routes/copilot.tsx`):** Accessible via `Copilot` sidebar link (`asset:read`), displaying tool permissions, citation provenance badges, red-team safety alerts, and tool execution audit cards.

