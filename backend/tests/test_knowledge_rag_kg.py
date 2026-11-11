"""Tests for WP-C3 (M3-WP2): Hybrid RAG, hardened PDF ingestion, Knowledge Graph, and eval."""

from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfWriter

from factorymind.contracts.knowledge import KnowledgeDocumentChunk
from factorymind.knowledge.corpus import (
    MAX_EXTRACTED_CHARS,
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    build_verified_corpus,
    extract_pdf_chunks,
)
from factorymind.knowledge.eval import evaluate_abstention_rate, evaluate_retrieval_modes
from factorymind.knowledge.kg import IndustrialKnowledgeGraph, infer_linked_kg_nodes
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


def test_extract_pdf_chunks_enforces_size_page_and_char_bounds(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.pdf"
    corrupt.write_bytes(b"not a real pdf at all")
    assert extract_pdf_chunks(corrupt) == []

    blank = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with blank.open("wb") as handle:
        writer.write(handle)
    assert extract_pdf_chunks(blank) == []

    over_pages = tmp_path / "too_many_pages.pdf"
    p_writer = PdfWriter()
    for _ in range(MAX_PDF_PAGES + 1):
        p_writer.add_blank_page(width=200, height=200)
    with over_pages.open("wb") as handle:
        p_writer.write(handle)
    assert extract_pdf_chunks(over_pages) == []

    huge = tmp_path / "huge.pdf"
    with huge.open("wb") as handle:
        handle.seek(MAX_PDF_BYTES + 10)
        handle.write(b"\x00")
    assert extract_pdf_chunks(huge) == []

    # Verify MAX_EXTRACTED_CHARS total cap across multi-page PDFs
    multi = PdfWriter()
    long_line = "TP3 pneumatic compressor " * 1500
    for _ in range(8):
        page_reader_pdf = _minimal_text_pdf(long_line)
        temp_p = tmp_path / "one.pdf"
        temp_p.write_bytes(page_reader_pdf)
        multi.append(str(temp_p))
    capped_pdf = tmp_path / "capped.pdf"
    with capped_pdf.open("wb") as handle:
        multi.write(handle)
    capped_chunks = extract_pdf_chunks(capped_pdf)
    assert sum(len(c.text) for c in capped_chunks) <= MAX_EXTRACTED_CHARS

    # Valid PDF extracts cleanly with license tag; missing docs_root degrades cleanly
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

    fallback = build_verified_corpus(tmp_path / "raw", docs_root=tmp_path / "missing_docs")
    assert len(fallback) >= 1


def test_hybrid_rag_enforces_zero_cross_tenant_oracle_and_zero_score_filtering() -> None:
    acme_chunks = [
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
            chunk_id="acme:apu-oil",
            doc_id="doc-2",
            title="APU Oil Thermal Note",
            section="Oil Temperature",
            asset_id=ASSET,
            tenant_scope="acme",
            source_path="note2.md",
            license="MIT",
            provenance_origin="observed",
            text="Oil temperature rises smoothly during continuous runtime cycles mentioning ai4i.",
        ),
    ]
    beta_secret = KnowledgeDocumentChunk(
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
    )
    other_asset = KnowledgeDocumentChunk(
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
    )

    isolated_idx = HybridRagIndex(acme_chunks)
    shared_idx = HybridRagIndex([*acme_chunks, beta_secret, other_asset])

    # Cross-tenant secret token MUST NOT match via BM25, dense SVD co-occurrence, or RRF
    for mode in ("bm25_only", "dense_svd_only", "hybrid_rrf"):
        leak_probe = shared_idx.search(
            "tokenxyz999",
            tenant_id="acme",
            allowed_asset_ids={ASSET},
            top_k=5,
            mode=mode,  # type: ignore[arg-type]
        )
        assert leak_probe.items == []
        assert leak_probe.related_kg_edges == []
        assert leak_probe.total_indexed_chunks == 2
        assert leak_probe.accessible_chunks == 2

    # Nonexistent token returns [] in hybrid_rrf (no zero-score filler hits)
    nonexistent = shared_idx.search(
        "nonexistenttoken000",
        tenant_id="acme",
        allowed_asset_ids={ASSET},
        top_k=5,
        mode="hybrid_rrf",
    )
    assert nonexistent.items == []

    # Shared index scores on acme-only query are invariant to beta/pump-99 corpus additions
    q_shared = shared_idx.search("TP3 leak", tenant_id="acme", allowed_asset_ids={ASSET})
    q_iso = isolated_idx.search("TP3 leak", tenant_id="acme", allowed_asset_ids={ASSET})
    assert [h.bm25_score for h in q_shared.items] == [h.bm25_score for h in q_iso.items]
    assert [h.semantic_score for h in q_shared.items] == [h.semantic_score for h in q_iso.items]

    # Mention of "ai4i" inside an acme chunk MUST NOT expose ungranted ai4i KG links
    oil_hit = shared_idx.search(
        "Oil temperature runtime", tenant_id="acme", allowed_asset_ids={ASSET}, top_k=2
    )
    assert oil_hit.items
    assert "asset:ai4i-milling-bench" not in oil_hit.items[0].linked_node_ids


def test_knowledge_graph_word_boundaries_and_epistemic_isolation() -> None:
    kg = IndustrialKnowledgeGraph()

    # Substring false positives eliminated
    assert "sensor:LPS" not in infer_linked_kg_nodes("This helps diagnose compressor wear.")
    assert "regime:loaded" not in infer_linked_kg_nodes("Unloaded pressure decay.")
    assert infer_linked_kg_nodes("LPS dropped below 7 bar.") == ["sensor:LPS"]
    assert infer_linked_kg_nodes("ai4i milling", allowed_asset_ids={ASSET}, kg=kg) == []
    assert infer_linked_kg_nodes(
        "ai4i milling", allowed_asset_ids={"ai4i-milling-bench"}, kg=kg
    ) == ["asset:ai4i-milling-bench"]

    corr = kg.traverse(
        allowed_asset_ids={ASSET},
        center_node_id="mode:air_leakage",
        depth=2,
        epistemic_type="correlational_indicator",
    )
    assert len(corr.edges) >= 1
    assert all(e.epistemic_type == "correlational_indicator" for e in corr.edges)
    assert any(e.target_id == "sensor:Oil_temperature" for e in corr.edges)

    uncentered_corr = kg.traverse(
        allowed_asset_ids={ASSET},
        center_node_id=None,
        epistemic_type="correlational_indicator",
    )
    connected = {e.source_id for e in uncentered_corr.edges} | {
        e.target_id for e in uncentered_corr.edges
    }
    assert {n.node_id for n in uncentered_corr.nodes} == connected

    with pytest.raises(KeyError):
        kg.traverse(allowed_asset_ids={ASSET}, center_node_id="asset:ai4i-milling-bench")


def test_knowledge_eval_subset_and_api_endpoints(kit: AuthKit) -> None:
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

    # Asset without ground-truth benchmark queries returns empty modes cleanly (no fake 0.0s)
    unmapped = evaluate_retrieval_modes(index, tenant_id="acme", allowed_asset_ids={"pump-99"})
    assert unmapped.corpus_chunks == 0
    assert unmapped.modes == []

    client = make_client(kit)
    viewer = kit.headers(roles=["viewer"], assets=[ASSET])

    # Whitespace-only query rejected cleanly with 422
    assert (
        client.get("/api/v1/knowledge/search", params={"q": "   "}, headers=viewer).status_code
        == 422
    )

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


def test_hybrid_rag_relevance_gate_rejects_out_of_scope_and_preserves_in_scope(
    kit: AuthKit,
) -> None:
    pdf_dir = kit.root / "data" / "raw" / "metropt3"
    pdf_dir.mkdir(parents=True)
    (pdf_dir / "Data Description_Metro.pdf").write_bytes(
        _minimal_text_pdf(
            "MPG starts compressor under load below 8.2 bar and LPS triggers below 7 bar."
        )
    )
    chunks = build_verified_corpus(kit.root / "data" / "raw")
    index = HybridRagIndex(chunks)
    empty_index = HybridRagIndex([])

    # Empty corpus never invents hits or KG links
    assert (
        empty_index.search(
            "unloaded TP3 air leakage", tenant_id="acme", allowed_asset_ids={ASSET}
        ).items
        == []
    )

    out_of_scope_queries = (
        "What is the capital of France?",
        "What's the capital of France?",
        "Chocolate cake recipe",
        "How to bake sourdough bread with yeast and flour?",
        "Who won the FIFA World Cup in 2022?",
        "What is the stock price of Apple today?",
        "Write a Python binary search algorithm.",
        "What is the turbine blade temperature limit?",
        "capital of France compressor",
        "Capital of compressor",
        "What is 1?",
        "What is 3?",
        "What is 3 + 7?",
        "What is the price of oil and air?",
        "Recipe for oil pressure",
        "Give me 1 chocolate",
        "Description of banana",
        "What is this?",
    )
    for query in out_of_scope_queries:
        for mode in ("bm25_only", "dense_svd_only", "hybrid_rrf"):
            res = index.search(
                query,
                tenant_id="acme",
                allowed_asset_ids={ASSET},
                top_k=5,
                mode=mode,  # type: ignore[arg-type]
            )
            assert res.items == [], f"Unexpected hit for {query!r} in mode={mode}: {res.items}"
            assert res.related_kg_edges == []

    # Small (2-doc) slice also blocks partial OOV common-word pairs & numeric-only queries
    two_doc_index = HybridRagIndex(chunks[:2])
    for small_oos in (
        "Capital of compressor",
        "What is 1?",
        "What is the capital of France for this compressor?",
        "Chocolate cake recipe",
    ):
        assert (
            two_doc_index.search(small_oos, tenant_id="acme", allowed_asset_ids={ASSET}).items == []
        )

    # In-scope technical & multi-term queries pass across all modes with zero bm25==0 tail hits
    positive_checks = (
        ("unloaded TP3 air leakage", ("air",)),
        ("Explain COMP valve contradiction semantics.", ("valve",)),
        ("What is the oil temperature limit?", ("oil",)),
        ("What's the compressor's oil temperature limit?", ("oil",)),
        ("pca_spe non-leak statistical warning tier separation", ("pca_spe",)),
        ("compressor duty limit", ("duty", "limit")),
        ("tp3 pressure limits", ("pressure",)),
        ("Explain pca_spe warning limits", ("pca_spe",)),
        ("What is H1?", ("h1",)),
        ("Explain cross_asset_transfer", ("cross_asset_transfer",)),
        ("Summarize unloaded TP3 air leakage risk and compare repair options.", ("air",)),
    )
    for query, expected_terms in positive_checks:
        for mode in ("bm25_only", "dense_svd_only", "hybrid_rrf"):
            res = index.search(
                query,
                tenant_id="acme",
                allowed_asset_ids={ASSET},
                top_k=3,
                mode=mode,  # type: ignore[arg-type]
            )
            assert len(res.items) >= 1, f"False negative for {query!r} in mode={mode}"
            assert all(h.bm25_score > 0.0 for h in res.items), (
                f"Zero-BM25 tail hit for {query!r} in mode={mode}: {res.items}"
            )
            combined = " ".join(f"{h.title} {h.section} {h.snippet}".lower() for h in res.items)
            assert any(term in combined for term in expected_terms), f"{query!r} -> {combined!r}"

    # Evaluation & OOS abstention metrics remain at 1.0 across all supported modes
    ev_report = evaluate_retrieval_modes(index, tenant_id="acme", allowed_asset_ids={ASSET})
    for m in ev_report.modes:
        assert m.recall_at_5 == 1.0
        assert m.mrr_at_10 == 1.0
    abstention = evaluate_abstention_rate(index, tenant_id="acme", allowed_asset_ids={ASSET})
    assert all(rate == 1.0 for rate in abstention.values())

    # HTTP /knowledge/search rejects out-of-scope queries cleanly without unrelated hits
    client = make_client(kit)
    client.app.state.knowledge_index = index  # type: ignore[union-attr]
    viewer = kit.headers(roles=["viewer"], assets=[ASSET])
    for query in (
        "What is the capital of France?",
        "Chocolate cake recipe",
        "What is the price of oil and air?",
        "What is 3 + 7?",
    ):
        resp = client.get(
            "/api/v1/knowledge/search",
            params={"q": query, "top_k": 5, "mode": "hybrid_rrf"},
            headers=viewer,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["related_kg_edges"] == []
        assert "abstained" in body["safety_note"].lower()
