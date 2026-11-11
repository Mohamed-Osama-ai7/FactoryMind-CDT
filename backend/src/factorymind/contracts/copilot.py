"""Canonical contracts for Controlled Industrial Copilot & read-only tool orchestration."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from factorymind.config import IDENTIFIER

CopilotToolName = Literal[
    "get_event_evidence",
    "get_rca_hypotheses",
    "search_docs",
    "query_kg",
    "simulate_scenario",
    "compare_options",
]

CopilotToolStatus = Literal["executed", "denied_permission", "unavailable", "skipped", "error"]


class CopilotToolSpec(BaseModel):
    """Metadata describing one permission-scoped read-only Copilot tool."""

    model_config = ConfigDict(extra="forbid")

    tool: CopilotToolName
    required_permission: str
    permitted_for_caller: bool
    read_only: Literal[True] = True
    description: str


class CopilotToolsResponse(BaseModel):
    """Catalog of read-only tools visible to the authenticated principal."""

    model_config = ConfigDict(extra="forbid")

    read_only_enforced: Literal[True] = True
    tools: list[CopilotToolSpec]


class CopilotToolInvocation(BaseModel):
    """Execution audit trace for one Copilot tool invocation."""

    model_config = ConfigDict(extra="forbid")

    tool: CopilotToolName
    permitted: bool
    status: CopilotToolStatus
    record_count: int = Field(ge=0)
    summary: str


class CopilotCitation(BaseModel):
    """Verified evidence/doc/KG/twin citation supporting statements in the Copilot reply."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str
    source_type: Literal[
        "incident_evidence",
        "rca",
        "knowledge_doc",
        "knowledge_graph",
        "digital_twin",
        "maintenance_optimizer",
    ]
    title: str
    provenance_origin: str
    detail: str


class CopilotQueryRequest(BaseModel):
    """Bounded read-only question to the Industrial Copilot orchestrator."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    asset_id: str = Field(
        default="metropt3-apu-1", min_length=1, max_length=64, pattern=IDENTIFIER.pattern
    )
    incident_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    leak_rate_bar_per_min: float = Field(default=0.35, ge=0.0, le=1.5)
    planning_horizon_hours: float = Field(default=8.0, ge=1.0, le=72.0)
    use_llm_adapter: bool = False

    @field_validator("question")
    @classmethod
    def _non_blank_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be blank or whitespace-only")
        return cleaned


class CopilotQueryResponse(BaseModel):
    """Citation-grounded, red-team-checked Copilot advisory briefing."""

    model_config = ConfigDict(extra="forbid")

    asset_id: str
    incident_id: str | None = None
    answer: str
    abstained: bool = False
    abstention_reason: str | None = None
    synthesis_mode: Literal["deterministic_grounded", "configured_llm_verified"]
    safety_refusal_triggered: bool = False
    red_team_flags: list[str] = Field(default_factory=list)
    tools_executed: list[CopilotToolInvocation]
    citations: list[CopilotCitation]
    causal_vs_correlation_note: str
    advisory_disclaimer: str = (
        "Strictly advisory decision support. Copilot possesses zero actuating or PLC write tools."
    )
