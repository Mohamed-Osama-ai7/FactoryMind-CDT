"""Typed industrial Knowledge Graph with explicit causal vs correlational edge semantics."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

from factorymind.contracts.knowledge import EpistemicType, KgEdge, KgNode, KgSubgraphResponse

EPISTEMIC_LEGEND: dict[str, str] = {
    "physical_mechanism": (
        "Direct hardware containment, physical balance, or engineered switch link."
    ),
    "causal_hypothesis": (
        "Candidate fault-to-symptom physical hypothesis; requires field verification "
        "against competing explanations before causal certainty."
    ),
    "correlational_indicator": (
        "Statistical association or projection residual only; correlation does NOT imply causation."
    ),
    "structural_membership": "Taxonomic grouping across operating regimes, event tiers, and docs.",
}

_BASE_NODES: tuple[KgNode, ...] = (
    KgNode(
        node_id="asset:metropt3-apu-1",
        kind="asset",
        label="MetroPT-3 APU Compressor #1",
        asset_id="metropt3-apu-1",
        summary="Railway screw compressor Air Production Unit supplying pneumatic brakes/doors.",
        attributes={"dataset": "UCI #791", "cadence": "~0.1 Hz"},
    ),
    KgNode(
        node_id="comp:compressor-stage",
        kind="component",
        label="Screw Compressor & Intake/Outlet Valves",
        asset_id="metropt3-apu-1",
        summary="Compresses air under COMP/DV_eletric valve control and Motor_current draw.",
    ),
    KgNode(
        node_id="comp:air-dryer-towers",
        kind="component",
        label="Twin-Tower Air Dryer & Separator",
        asset_id="metropt3-apu-1",
        summary="Switches drying/regeneration across Towers with H1/DV_pressure purge cycles.",
    ),
    KgNode(
        node_id="comp:pneumatic-reservoirs",
        kind="component",
        label="Pneumatic Panel & Reservoirs",
        asset_id="metropt3-apu-1",
        summary=(
            "Panel pressure TP3 and downstream Reservoirs regulated by MPG (~8.2 bar) & LPS (<7)."
        ),
    ),
    KgNode(
        node_id="comp:oil-circuit",
        kind="component",
        label="Lubrication Oil Circuit",
        asset_id="metropt3-apu-1",
        summary="Thermal and level monitoring via Oil_temperature and Oil_level switch.",
    ),
    KgNode(
        node_id="sensor:TP3",
        kind="sensor",
        label="TP3 (Pneumatic Panel Pressure, bar)",
        asset_id="metropt3-apu-1",
        summary="Primary panel pressure (~8.2-10.2 bar). Decays rapidly during unloaded air leaks.",
    ),
    KgNode(
        node_id="sensor:TP2",
        kind="sensor",
        label="TP2 (Compressor Pressure, bar)",
        asset_id="metropt3-apu-1",
        summary="Upstream compressor stage pressure (~0 bar off/unloaded, ~8.2-10.2 bar loaded).",
    ),
    KgNode(
        node_id="sensor:Reservoirs",
        kind="sensor",
        label="Reservoirs (Downstream Pressure, bar)",
        asset_id="metropt3-apu-1",
        summary="Train air reservoir pressure tracking TP3 envelope during nominal cycles.",
    ),
    KgNode(
        node_id="sensor:Motor_current",
        kind="sensor",
        label="Motor_current (Phase Current, A)",
        asset_id="metropt3-apu-1",
        summary=(
            "Separates stopped (<1 A), unloaded (1-5.5 A), loaded (>5.5 A), and starting (>8 A)."
        ),
    ),
    KgNode(
        node_id="sensor:Oil_temperature",
        kind="sensor",
        label="Oil_temperature (degC)",
        asset_id="metropt3-apu-1",
        summary="Rises monotonically with extended loaded compressor duty.",
    ),
    KgNode(
        node_id="sensor:H1",
        kind="sensor",
        label="H1 (Cyclonic Filter Pressure Drop, bar)",
        asset_id="metropt3-apu-1",
        summary="Near 0 bar when loaded; excluded from univariate sensor-freeze heuristics.",
    ),
    KgNode(
        node_id="sensor:DV_pressure",
        kind="sensor",
        label="DV_pressure (Air-Dryer Discharge Drop, bar)",
        asset_id="metropt3-apu-1",
        summary="Reads ~0 bar under healthy loaded compression; spikes on tower purge.",
    ),
    KgNode(
        node_id="sensor:LPS",
        kind="sensor",
        label="LPS (Low-Pressure Switch <7 bar)",
        asset_id="metropt3-apu-1",
        summary="Emergency binary switch triggered when pneumatic pressure drops below 7 bar.",
    ),
    KgNode(
        node_id="sensor:COMP_DV_eletric",
        kind="sensor",
        label="COMP & DV_eletric Intake/Outlet Valves",
        asset_id="metropt3-apu-1",
        summary="Complementary binary solenoid valves gated for signal consistency.",
    ),
    KgNode(
        node_id="regime:loaded",
        kind="regime",
        label="Operating Regime: loaded",
        asset_id="metropt3-apu-1",
        summary="Motor_current > 5.5 A, COMP == 0, DV_eletric == 1 for >= 90% of 10-min window.",
    ),
    KgNode(
        node_id="regime:unloaded",
        kind="regime",
        label="Operating Regime: unloaded",
        asset_id="metropt3-apu-1",
        summary=(
            "1.0 <= Motor_current <= 5.5 A, COMP == 1, DV_eletric == 0 (pressure hold decay)."
        ),
    ),
    KgNode(
        node_id="regime:insufficient_evidence",
        kind="regime",
        label="Operating Regime: insufficient_evidence",
        asset_id="metropt3-apu-1",
        summary="Stale freeze, valve contradiction, mixed_duty_cycle, or low window coverage.",
    ),
    KgNode(
        node_id="event:air_leak_alarm",
        kind="event_class",
        label="Event Class: air_leak_alarm",
        asset_id="metropt3-apu-1",
        summary="Primary calibrated duty_rule exceedance episode (LOEO 4/4 on F1..F4).",
    ),
    KgNode(
        node_id="event:non_leak_warning",
        kind="event_class",
        label="Event Class: non_leak_statistical_warning",
        asset_id="metropt3-apu-1",
        summary="Separated pca_spe residual excursion kept distinct from primary leak alarms.",
    ),
    KgNode(
        node_id="mode:air_leakage",
        kind="failure_mode",
        label="Failure Mode: Pneumatic Air Leakage",
        asset_id="metropt3-apu-1",
        summary=("Pneumatic loss raising load_duty, negative unloaded_tp3_slope, and LPS trips."),
    ),
    KgNode(
        node_id="mode:abnormal_high_demand",
        kind="failure_mode",
        label="Competing Mode: Abnormal High Demand",
        asset_id="metropt3-apu-1",
        summary=("Elevated train braking/door duty raising load_duty without an internal leak."),
    ),
    KgNode(
        node_id="mode:sensor_malfunction",
        kind="failure_mode",
        label="Failure Mode: Sensor Freeze / Channel Fault",
        asset_id="metropt3-apu-1",
        summary=("Channel flatline or isolated PCA SPE departure requiring harness inspection."),
    ),
    KgNode(
        node_id="mode:telemetry_inconsistency",
        kind="failure_mode",
        label="Failure Mode: Control / Telemetry Contradiction",
        asset_id="metropt3-apu-1",
        summary="Valve or motor/valve state conflict forcing conservative alarm abstention.",
    ),
    KgNode(
        node_id="doc:metropt3-pdf",
        kind="document",
        label="UCI MetroPT-3 Data Description PDF",
        asset_id="metropt3-apu-1",
        summary="Official 3-page dataset specification defining all 15 analog/digital channels.",
    ),
    KgNode(
        node_id="doc:m2-wp4-explainable-rca",
        kind="document",
        label="M2-WP4 Explainable RCA Architecture",
        asset_id="metropt3-apu-1",
        summary="Physical multi-hypothesis ranking, abstention, and citation grounding contract.",
    ),
    KgNode(
        node_id="asset:ai4i-milling-bench",
        kind="asset",
        label="AI4I 2020 Milling Benchmark Bench",
        asset_id="ai4i-milling-bench",
        summary="Controlled UCI #601 synthetic milling benchmark strictly separate from MetroPT-3.",
    ),
    KgNode(
        node_id="mode:ai4i-failure-risk",
        kind="failure_mode",
        label="AI4I Benchmark Failure Modes (TWF/HDF/PWF/OSF)",
        asset_id="ai4i-milling-bench",
        summary=("Calibrated multi-model ML failure prediction gated by Mahalanobis OOD distance."),
    ),
)

_BASE_EDGES: tuple[KgEdge, ...] = (
    KgEdge(
        source_id="asset:metropt3-apu-1",
        target_id="comp:compressor-stage",
        relation="contains_component",
        epistemic_type="structural_membership",
        rationale="APU assembly contains the rotary screw compressor and intake/outlet valves.",
    ),
    KgEdge(
        source_id="asset:metropt3-apu-1",
        target_id="comp:air-dryer-towers",
        relation="contains_component",
        epistemic_type="structural_membership",
        rationale="APU assembly contains twin-tower dryer and cyclonic filter separator.",
    ),
    KgEdge(
        source_id="asset:metropt3-apu-1",
        target_id="comp:pneumatic-reservoirs",
        relation="contains_component",
        epistemic_type="structural_membership",
        rationale="APU feeds panel circuit (TP3) and downstream train reservoirs.",
    ),
    KgEdge(
        source_id="asset:metropt3-apu-1",
        target_id="comp:oil-circuit",
        relation="contains_component",
        epistemic_type="structural_membership",
        rationale="Screw compressor lubrication circuit monitored for thermal & oil-level safety.",
    ),
    KgEdge(
        source_id="comp:compressor-stage",
        target_id="sensor:TP2",
        relation="monitored_by",
        epistemic_type="physical_mechanism",
        rationale="TP2 measures direct compressor stage discharge pressure.",
    ),
    KgEdge(
        source_id="comp:compressor-stage",
        target_id="sensor:Motor_current",
        relation="monitored_by",
        epistemic_type="physical_mechanism",
        rationale="Motor phase current tracks compression torque load across regimes.",
    ),
    KgEdge(
        source_id="comp:compressor-stage",
        target_id="sensor:COMP_DV_eletric",
        relation="monitored_by",
        epistemic_type="physical_mechanism",
        rationale="COMP and DV_eletric control intake/outlet loading state.",
    ),
    KgEdge(
        source_id="comp:air-dryer-towers",
        target_id="sensor:H1",
        relation="monitored_by",
        epistemic_type="physical_mechanism",
        rationale="H1 measures cyclonic filter separator pressure drop.",
    ),
    KgEdge(
        source_id="comp:air-dryer-towers",
        target_id="sensor:DV_pressure",
        relation="monitored_by",
        epistemic_type="physical_mechanism",
        rationale="DV_pressure monitors air-dryer tower discharge drop.",
    ),
    KgEdge(
        source_id="comp:pneumatic-reservoirs",
        target_id="sensor:TP3",
        relation="monitored_by",
        epistemic_type="physical_mechanism",
        rationale="TP3 monitors pneumatic control panel pressure.",
    ),
    KgEdge(
        source_id="comp:pneumatic-reservoirs",
        target_id="sensor:Reservoirs",
        relation="monitored_by",
        epistemic_type="physical_mechanism",
        rationale="Reservoirs measures downstream train storage tank pressure.",
    ),
    KgEdge(
        source_id="comp:pneumatic-reservoirs",
        target_id="sensor:LPS",
        relation="monitored_by",
        epistemic_type="physical_mechanism",
        rationale="LPS trips whenever pneumatic line pressure falls below 7 bar.",
    ),
    KgEdge(
        source_id="comp:oil-circuit",
        target_id="sensor:Oil_temperature",
        relation="monitored_by",
        epistemic_type="physical_mechanism",
        rationale="Oil_temperature rises under prolonged compression work.",
    ),
    KgEdge(
        source_id="mode:air_leakage",
        target_id="sensor:TP3",
        relation="induces_pressure_decay",
        epistemic_type="causal_hypothesis",
        rationale="Loss of air inventory causes rapid negative TP3 slope during unloaded hold.",
    ),
    KgEdge(
        source_id="mode:air_leakage",
        target_id="regime:loaded",
        relation="increases_loaded_duty",
        epistemic_type="causal_hypothesis",
        rationale="Control governor (MPG <8.2 bar) engages loaded compression more frequently.",
    ),
    KgEdge(
        source_id="mode:air_leakage",
        target_id="sensor:LPS",
        relation="triggers_low_pressure_trip",
        epistemic_type="causal_hypothesis",
        rationale="Severe air loss drops circuit below 7 bar safety switch threshold.",
    ),
    KgEdge(
        source_id="mode:air_leakage",
        target_id="event:air_leak_alarm",
        relation="detected_by_alarm_tier",
        epistemic_type="causal_hypothesis",
        rationale="Primary duty_rule exceedance episodes flag sustained high-duty windows.",
    ),
    KgEdge(
        source_id="mode:air_leakage",
        target_id="sensor:Oil_temperature",
        relation="co_occurs_with_thermal_rise",
        epistemic_type="correlational_indicator",
        rationale=(
            "Extended runtime raises oil temperature statistically; thermal rise is non-specific."
        ),
    ),
    KgEdge(
        source_id="mode:air_leakage",
        target_id="mode:abnormal_high_demand",
        relation="competes_with",
        epistemic_type="structural_membership",
        rationale=("Unobserved brake/door air demand also raises loaded duty without an APU leak."),
    ),
    KgEdge(
        source_id="mode:abnormal_high_demand",
        target_id="regime:loaded",
        relation="increases_loaded_duty",
        epistemic_type="causal_hypothesis",
        rationale="High train air consumption forces frequent compression cycles.",
    ),
    KgEdge(
        source_id="mode:sensor_malfunction",
        target_id="event:non_leak_warning",
        relation="surfaces_in_warning_tier",
        epistemic_type="correlational_indicator",
        rationale=("Isolated pca_spe departures without duty exceedance surface as warnings."),
    ),
    KgEdge(
        source_id="mode:telemetry_inconsistency",
        target_id="regime:insufficient_evidence",
        relation="abstains_into",
        epistemic_type="physical_mechanism",
        rationale=(
            "Valve contradiction (COMP vs DV_eletric) blocks scoring via WP2 integrity gate."
        ),
    ),
    KgEdge(
        source_id="sensor:Motor_current",
        target_id="regime:unloaded",
        relation="defines_regime_envelope",
        epistemic_type="physical_mechanism",
        rationale="1.0 <= Motor_current <= 5.5 A identifies idle unloaded hold windows.",
    ),
    KgEdge(
        source_id="asset:metropt3-apu-1",
        target_id="doc:metropt3-pdf",
        relation="documented_in",
        epistemic_type="structural_membership",
        rationale="Canonical UCI dataset description PDF.",
    ),
    KgEdge(
        source_id="mode:air_leakage",
        target_id="doc:m2-wp4-explainable-rca",
        relation="documented_in",
        epistemic_type="structural_membership",
        rationale="WP4 multi-hypothesis ranking and correlation boundary spec.",
    ),
    KgEdge(
        source_id="asset:ai4i-milling-bench",
        target_id="mode:ai4i-failure-risk",
        relation="exhibits_failure_mode",
        epistemic_type="causal_hypothesis",
        rationale="Controlled synthetic milling failure classification benchmark (non-MetroPT-3).",
    ),
)


class IndustrialKnowledgeGraph:
    """In-memory typed property graph supporting asset-scoped BFS traversal."""

    def __init__(
        self,
        nodes: Iterable[KgNode] = _BASE_NODES,
        edges: Iterable[KgEdge] = _BASE_EDGES,
    ) -> None:
        self._nodes: dict[str, KgNode] = {n.node_id: n for n in nodes}
        self._edges: list[KgEdge] = list(edges)

    def node(self, node_id: str) -> KgNode | None:
        return self._nodes.get(node_id)

    def edges_for_nodes(
        self,
        node_ids: Sequence[str],
        *,
        allowed_asset_ids: set[str],
    ) -> list[KgEdge]:
        wanted = set(node_ids)
        return [
            e
            for e in self._edges
            if (e.source_id in wanted or e.target_id in wanted)
            and (s := self._nodes.get(e.source_id)) is not None
            and (t := self._nodes.get(e.target_id)) is not None
            and s.asset_id in allowed_asset_ids
            and t.asset_id in allowed_asset_ids
        ]

    def traverse(
        self,
        *,
        allowed_asset_ids: set[str],
        center_node_id: str | None = None,
        depth: int = 2,
        epistemic_type: EpistemicType | None = None,
    ) -> KgSubgraphResponse:
        bounded_depth = min(max(depth, 1), 3)
        accessible_nodes = {
            nid: node for nid, node in self._nodes.items() if node.asset_id in allowed_asset_ids
        }

        if center_node_id is None:
            filtered_edges = [
                e
                for e in self._edges
                if e.source_id in accessible_nodes
                and e.target_id in accessible_nodes
                and (epistemic_type is None or e.epistemic_type == epistemic_type)
            ]
            return KgSubgraphResponse(
                center_node_id=None,
                depth=bounded_depth,
                nodes=list(accessible_nodes.values()),
                edges=filtered_edges,
                epistemic_legend=EPISTEMIC_LEGEND,
            )

        if center_node_id not in accessible_nodes:
            raise KeyError(center_node_id)

        visited: set[str] = {center_node_id}
        queue: deque[tuple[str, int]] = deque([(center_node_id, 0)])
        collected_edges: list[KgEdge] = []
        seen_edges: set[tuple[str, str, str]] = set()

        while queue:
            cur_id, cur_depth = queue.popleft()
            if cur_depth >= bounded_depth:
                continue
            for edge in self._edges:
                if epistemic_type is not None and edge.epistemic_type != epistemic_type:
                    continue
                if edge.source_id != cur_id and edge.target_id != cur_id:
                    continue
                other_id = edge.target_id if edge.source_id == cur_id else edge.source_id
                if other_id not in accessible_nodes:
                    continue
                edge_key = (edge.source_id, edge.target_id, edge.relation)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    collected_edges.append(edge)
                if other_id not in visited:
                    visited.add(other_id)
                    queue.append((other_id, cur_depth + 1))

        return KgSubgraphResponse(
            center_node_id=center_node_id,
            depth=bounded_depth,
            nodes=[accessible_nodes[nid] for nid in visited],
            edges=collected_edges,
            epistemic_legend=EPISTEMIC_LEGEND,
        )


def infer_linked_kg_nodes(text: str) -> list[str]:
    """Deterministically link corpus chunks to matching canonical Knowledge Graph entities."""
    lower = text.lower()
    mapping: tuple[tuple[str, str], ...] = (
        ("tp3", "sensor:TP3"),
        ("tp2", "sensor:TP2"),
        ("reservoir", "sensor:Reservoirs"),
        ("motor_current", "sensor:Motor_current"),
        ("oil_temperature", "sensor:Oil_temperature"),
        ("h1", "sensor:H1"),
        ("dv_pressure", "sensor:DV_pressure"),
        ("lps", "sensor:LPS"),
        ("dv_eletric", "sensor:COMP_DV_eletric"),
        ("dv_electric", "sensor:COMP_DV_eletric"),
        ("air leak", "mode:air_leakage"),
        ("air_leakage", "mode:air_leakage"),
        ("abnormal_high_demand", "mode:abnormal_high_demand"),
        ("high demand", "mode:abnormal_high_demand"),
        ("sensor_malfunction", "mode:sensor_malfunction"),
        ("telemetry_inconsistency", "mode:telemetry_inconsistency"),
        ("pca_spe", "event:non_leak_warning"),
        ("duty_rule", "event:air_leak_alarm"),
        ("unloaded", "regime:unloaded"),
        ("loaded", "regime:loaded"),
        ("ai4i", "asset:ai4i-milling-bench"),
    )
    hits: list[str] = []
    for needle, node_id in mapping:
        if needle in lower and node_id not in hits:
            hits.append(node_id)
    return hits
