# M4-WP2 (`WP-G`) — Controlled Industrial Copilot

M4-WP2 adds a permission-scoped, strictly read-only **Controlled Industrial Copilot** orchestrating across stored incident evidence, RCA hypotheses, partition-isolated Hybrid RAG documentation, typed Knowledge Graph traversal, the calibrated Digital Twin, and constrained maintenance optimization.

---

## 1. Read-Only Tool Orchestration & Real RCA Integration (`factorymind.copilot.service`)

Every tool enforces principal permissions + tenant/asset grants before execution and prohibits physical actuation:

- `get_event_evidence` & `get_rca_hypotheses`: `Permission.INCIDENT_READ` + `IncidentRepo` tenant/asset scoping; `get_rca_hypotheses` builds a multi-phase `EventEvidencePackage` (incorporating recorded `HumanObservation` rows) and runs the real deterministic `analyze_event_evidence` RCA engine rather than fixed detector shortcuts. If full telemetry/model artifacts are unavailable (`status="unavailable"`), Copilot abstains conservatively and passes `"insufficient_evidence"` to downstream tools. Rule-scored consistency hypotheses are explicitly reported as heuristic evidence rankings, not proven causal inference.
- `search_docs`: `Permission.ASSET_READ` + partition-isolated `HybridRagIndex` with stopword exclusion and IDF-weighted minimum relevance gating.
- `query_kg`: `Permission.ASSET_READ` + typed `IndustrialKnowledgeGraph` (depth=3 traversing physical, causal hypothesis, and correlational edges).
- `simulate_scenario`: `Permission.ASSET_READ` (`metropt3-apu-1` APU digital twin, `simulated_projection`, shaped by the supported RCA hypothesis).
- `compare_options`: `Permission.ASSET_READ` (`repair_now` vs `schedule` vs `monitor` receiving the real RCA `dominant_hypothesis`).

---

## 2. Deterministic Grounded Synthesis, Abstention & Red-Team Firewall

- **Relevance-Gated Abstention:** Out-of-scope, unsupported, or empty-retrieval queries (or incidents where the RCA engine abstains or telemetry is missing) set `abstained=True` with an explicit `abstention_reason`. KG, Digital Twin, and Optimizer tools are skipped rather than emitting unrelated filler citations for ungrounded questions.
- **Input Red-Team Firewall (`detect_safety_or_injection_risk`):** Immediately refuses PLC write / valve actuation commands and prompt-injection attempts (`safety_refusal_triggered=True`, `abstained=True`, all tools skipped).
- **Deterministic Grounded Synthesizer:** The deployed API always runs deterministic grounded synthesis (`synthesis_mode="deterministic_grounded"`, `llm_adapter=None`); no external LLM provider integration is implemented. An in-Python `LlmAdapter` callback protocol hook exists strictly with input/snippet directive redaction and citation-subset fallback (`[C1..Cn]`).

---

## 3. API & `/copilot` UI Workspace

- `GET /api/v1/copilot/tools` (`Permission.ASSET_READ`): Lists all 6 read-only tools and caller permissions.
- `POST /api/v1/copilot/query` (`Permission.ASSET_READ`, allowlisted in `test_safety_boundary.py`): Returns structured citations, tool execution trace, red-team flags, and causal-vs-correlation reminder.
- **Frontend (`/copilot`, `CopilotWorkspace.tsx`, `routes/copilot.tsx`):** Accessible via `Copilot` sidebar link (`asset:read`), displaying tool permissions, citation provenance badges, red-team safety alerts, and tool execution audit cards.

