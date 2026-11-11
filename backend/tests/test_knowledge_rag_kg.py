"""Tests for WP-C3 (M3-WP2): Hybrid RAG, hardened PDF ingestion, Knowledge Graph, and eval."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter

from factorymind.contracts.knowledge import KnowledgeDocumentChunk
from factorymind.knowledge.corpus import MAX_PDF_BYTES, build_verified_corpus, extract_pdf_chunks
from factorymind.knowledge.eval import evaluate_retrieval_modes
from factorymind.knowledge.kg import IndustrialKnowledgeGraph
from factorymind.knowledge.rag import HybridRagIndex
from tests.auth_helpers import ASSET, AuthKit, make_client


def _minimal_text_pdf(text: str) -> bytes:
    """Construct a valid single-page PDF containing searchable ASCII text without external libs."""
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET\n".encode("latin-1")
    objs = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        (
            b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
        ),
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        (
            f"5 0 obj << /Length {len(stream)} >> stream\n".encode("ascii")
            + stream
            + b"endstream endobj\n"
        ),
    ]
    buf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for obj in objs:
        offsets.append(len(buf))
        buf.extend(obj)
    xref_pos = len(buf)
    buf.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets:
        buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    buf.extend(
        f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return bytes(buf)


def test_extract_pdf_chunks_enforces_size_bounds_and_handles_corruption(tmp_path: Path) -> None:
    # Corrupt bytes degrade cleanly to []
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a real pdf at all")
    assert extract_pdf_chunks(corrupt) == []

    # Blank / empty page skipped safely
    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with blank.open("wb") as handle:
        writer.write(handle)
    assert extract_pdf_chunks(blank) == []

    # Oversized file (> MAX_PDF_BYTES) skipped cleanly without parsing
    huge = tmp_path / "huge.pdf"
    with huge.open("wb") as handle:
        handle.seek(MAX_PDF_BYTES + 10)
        handle.write(b"\x00")
    assert extract_pdf_chunks(huge) == []

    # Real PDF with searchable text extracts cleanly with license tag
    valid = tmp_path / "Data Description_Metro.pdf"
    valid.write_bytes(
        _minimal_text_pdf(
            "MetroPT3 TP3 pneumatic panel pressure decays rapidly during unloaded air leakage."
        )
    )
    chunks = extract_pdf_chunks(valid)
    assert len(chunks) == 1
    assert "MetroPT3 TP3 pneumatic panel" in chunks[0].text
    assert chunks[0].license == "CC BY 4.0 (UCI #791)"


def test_hybrid_rag_enforces_tenant_and_asset_isolation(tmp_path: Path) -> None:
    chunks = [
        KnowledgeDocumentChunk(
            chunk_id="acme:apu-leak",
            doc_id="doc-1",
            title="APU Air Leak Maintenance Note",
            section="Unloaded TP3 Decay",
            asset_id=ASSET,
            tenant_scope="acme",
            source_path="note1.md",
            license="MIT",
            provenance_origin="observed",
            text="Unloaded TP3 pressure drop and elevated loaded duty indicate pneumatic leak.",
        ),
        KnowledgeDocumentChunk(
            chunk_id="beta:secret-doc",
            doc_id="doc-secret",
            title="Beta Confidential Maintenance Log",
            section="Secret",
            asset_id=ASSET,
            tenant_scope="beta",
            source_path="secret.md",
            license="Internal",
            provenance_origin="observed",
            text="Confidential beta leak turbine keyword TP3 diagnostic tokenxyz999.",
        ),
        KnowledgeDocumentChunk(
            chunk_id="acme:other-asset",
            doc_id="doc-other",
            title="Pump 99 Cavitation Manual",
            section="Pump",
            asset_id="pump-99",
            tenant_scope="acme",
            source_path="pump.md",
            license="Internal",
            provenance_origin="observed",
            text="Pump 99 TP3 secret pumpnote123.",
        ),
    ]
    index = HybridRagIndex(chunks)

    res = index.search(
        "TP3 leak tokenxyz999 pumpnote123",
        tenant_id="acme",
        allowed_asset_ids={ASSET},
        top_k=5,
        mode="hybrid_rrf",
    )
    assert res.total_indexed_chunks == 3
    assert res.accessible_chunks == 1
    assert [h.chunk_id for h in res.items] == ["acme:apu-leak"]
    assert "Confidential beta leak" not in str(res.model_dump())
    assert "Cavitation Manual" not in str(res.model_dump())


def test_knowledge_graph_separates_causal_vs_correlation_and_blocks_ungranted_nodes() -> None:
    kg = IndustrialKnowledgeGraph()

    corr = kg.traverse(
        allowed_asset_ids={ASSET},
        center_node_id="mode:air_leakage",
        depth=2,
        epistemic_type="correlational_indicator",
    )
    assert len(corr.edges) >= 1
    assert all(e.epistemic_type == "correlational_indicator" for e in corr.edges)
    assert any(e.target_id == "sensor:Oil_temperature" for e in corr.edges)

    # AI4I node belongs to ai4i-milling-bench, inaccessible when only metropt3-apu-1 is granted
    try:
        kg.traverse(allowed_asset_ids={ASSET}, center_node_id="asset:ai4i-milling-bench")
        raise AssertionError("Expected KeyError for ungranted KG asset node")
    except KeyError:
        pass


def test_knowledge_eval_and_api_endpoints(kit: AuthKit, tmp_path: Path) -> None:
    pdf_dir = kit.root / "data" / "raw" / "metropt3"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "Data Description_Metro.pdf").write_bytes(
        _minimal_text_pdf(
            "MPG starts compressor under load below 8.2 bar and LPS triggers below 7 bar."
        )
    )

    chunks = build_verified_corpus(kit.root / "data" / "raw")
    index = HybridRagIndex(chunks)
    ev_report = evaluate_retrieval_modes(index, tenant_id="acme", allowed_asset_ids={ASSET})
    assert [m.mode for m in ev_report.modes] == ["bm25_only", "dense_svd_only", "hybrid_rrf"]
    for m in ev_report.modes:
        assert 0.0 <= m.recall_at_5 <= 1.0
        assert 0.0 <= m.mrr_at_10 <= 1.0
        assert 0.0 <= m.ndcg_at_5 <= 1.0

    client = make_client(kit)
    viewer = kit.headers(roles=["viewer"], assets=[ASSET])

    search_resp = client.get(
        "/api/v1/knowledge/search",
        params={"q": "unloaded TP3 air leakage", "top_k": 5, "mode": "hybrid_rrf"},
        headers=viewer,
    )
    assert search_resp.status_code == 200
    s_body = search_resp.json()
    assert len(s_body["items"]) >= 1
    assert s_body["items"][0]["rrf_score"] > 0.0
    assert s_body["items"][0]["license"]

    kg_resp = client.get(
        "/api/v1/knowledge/kg",
        params={"node_id": "mode:air_leakage", "depth": 1},
        headers=viewer,
    )
    assert kg_resp.status_code == 200
    assert any(n["node_id"] == "mode:air_leakage" for n in kg_resp.json()["nodes"])

    # Ungranted or unknown KG node / asset filter fails cleanly with 404
    assert (
        client.get(
            "/api/v1/knowledge/kg",
            params={"node_id": "asset:ai4i-milling-bench"},
            headers=viewer,
        ).status_code
        == 404
    )
    assert (
        client.get(
            "/api/v1/knowledge/search",
            params={"q": "leak", "asset_id": "pump-99"},
            headers=viewer,
        ).status_code
        == 404
    )

    eval_resp = client.get("/api/v1/knowledge/eval", headers=viewer)
    assert eval_resp.status_code == 200
    assert len(eval_resp.json()["modes"]) == 3

    # Auditor and cross-tenant denied with 403
    assert (
        client.get("/api/v1/knowledge/eval", headers=kit.headers(roles=["auditor"])).status_code
        == 403
    )
    assert (
        client.get(
            "/api/v1/knowledge/search",
            params={"q": "leak"},
            headers=kit.headers(tenant="beta", roles=["viewer"]),
        ).status_code
        == 403
    )
