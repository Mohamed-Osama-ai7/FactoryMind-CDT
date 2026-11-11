"""Hardened verified technical corpus loader (PDF + Asset Passports + repository docs)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Final

from pypdf import PdfReader

from factorymind.assets.passport import get_asset_passport
from factorymind.contracts.knowledge import KnowledgeDocumentChunk

log = logging.getLogger("factorymind.knowledge")

MAX_PDF_BYTES: Final = 10_000_000
MAX_PDF_PAGES: Final = 50
MAX_EXTRACTED_CHARS: Final = 250_000
_MIN_CHUNK_CHARS: Final = 30

_REPO_DOC_FILES: Final[tuple[tuple[str, str, str], ...]] = (
    ("docs/m1-anomaly-detection.md", "M1 Anomaly Detection Design", "metropt3-apu-1"),
    (
        "docs/m2-wp0-detection-hardening.md",
        "M2-WP0 Nested LOEO & Detection Hardening",
        "metropt3-apu-1",
    ),
    (
        "docs/m2-wp2-operating-regime-warning-separation.md",
        "M2-WP2 Operating Regimes & Warning Separation",
        "metropt3-apu-1",
    ),
    (
        "docs/m2-wp3-event-evidence-foundation.md",
        "M2-WP3 Event Evidence Packages",
        "metropt3-apu-1",
    ),
    ("docs/m2-wp4-explainable-rca.md", "M2-WP4 Explainable Physical RCA", "metropt3-apu-1"),
    ("docs/m3-wp0-asset-digital-passport.md", "M3-WP0 Asset Digital Passport", "metropt3-apu-1"),
    ("docs/m3-wp1-multimodel-ml.md", "M3-WP1 Multi-Model ML & AI4I 2020", "ai4i-milling-bench"),
)


def find_repo_docs_dir(start: Path | None = None) -> Path | None:
    """Locate repository ``docs/`` folder safely when running from a source checkout."""
    cur = (start or Path(__file__)).resolve()
    for candidate in (cur, *cur.parents):
        docs_dir = candidate / "docs"
        if docs_dir.is_dir() and (docs_dir / "m2-wp4-explainable-rca.md").is_file():
            return docs_dir
    return None


def extract_pdf_chunks(
    pdf_path: Path,
    *,
    doc_id: str = "metropt3-pdf",
    title: str = "UCI MetroPT-3 Official Data Description PDF",
    asset_id: str = "metropt3-apu-1",
    license_name: str = "CC BY 4.0 (UCI #791)",
) -> list[KnowledgeDocumentChunk]:
    """Safely extract text chunks from a bounded local PDF with size, page, and char caps."""
    if not pdf_path.is_file():
        return []
    try:
        size = pdf_path.stat().st_size
    except OSError:
        return []
    if size <= 0 or size > MAX_PDF_BYTES:
        log.warning("Skipping PDF %s: size %d outside safe bounds", pdf_path, size)
        return []

    try:
        reader = PdfReader(str(pdf_path))
        if len(reader.pages) > MAX_PDF_PAGES:
            log.warning("Skipping PDF %s: page count exceeds %d", pdf_path, MAX_PDF_PAGES)
            return []
        chunks: list[KnowledgeDocumentChunk] = []
        total_chars = 0
        for page_num, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            normalized = " ".join(raw_text.split())
            if len(normalized) < _MIN_CHUNK_CHARS:
                continue
            remaining = MAX_EXTRACTED_CHARS - total_chars
            if remaining <= 0:
                break
            excerpt = normalized[:remaining]
            total_chars += len(excerpt)
            chunks.append(
                KnowledgeDocumentChunk(
                    chunk_id=f"{doc_id}:page-{page_num}",
                    doc_id=doc_id,
                    title=title,
                    section=f"Page {page_num}",
                    asset_id=asset_id,
                    source_path=pdf_path.name,
                    license=license_name,
                    provenance_origin="observed",
                    text=excerpt,
                )
            )
        return chunks
    except Exception:
        log.warning("Failed to parse PDF %s cleanly", pdf_path, exc_info=True)
        return []


def _passport_chunks() -> list[KnowledgeDocumentChunk]:
    passport = get_asset_passport("metropt3-apu-1")
    if passport is None:
        return []
    chunks: list[KnowledgeDocumentChunk] = []

    sensor_lines = [
        f"{s.sensor_id} ({s.display_name}, {s.kind}, {s.unit}): {s.physical_role} "
        f"Envelope: {s.nominal_envelope}."
        + (f" Note: {s.unobservability_note}" if s.unobservability_note else "")
        for s in passport.sensors
    ]
    chunks.append(
        KnowledgeDocumentChunk(
            chunk_id="passport:metropt3-sensors",
            doc_id="passport-metropt3-apu-1",
            title="MetroPT-3 APU Digital Passport",
            section="Sensor Specifications & Envelopes",
            asset_id=passport.asset_id,
            source_path="factorymind/assets/passport.py",
            license="MIT (Repository Contract)",
            provenance_origin="derived",
            text=" ".join(sensor_lines),
        )
    )

    regime_text = (
        " ".join(f"Regime {k}: {v}." for k, v in passport.operating_envelopes.items())
        + " Unobserved quantities: "
        + "; ".join(passport.unobserved_quantities)
    )
    chunks.append(
        KnowledgeDocumentChunk(
            chunk_id="passport:metropt3-regimes",
            doc_id="passport-metropt3-apu-1",
            title="MetroPT-3 APU Digital Passport",
            section="Operating Regimes & Unobserved Boundaries",
            asset_id=passport.asset_id,
            source_path="factorymind/assets/passport.py",
            license="MIT (Repository Contract)",
            provenance_origin="derived",
            text=regime_text,
        )
    )

    for fm in passport.failure_modes:
        text = (
            f"{fm.display_name} ({fm.mode_id}): {fm.description} "
            f"Primary signatures: {'; '.join(fm.primary_signatures)}. "
            f"Competing explanations: {', '.join(fm.competing_explanations)}. "
            f"Recommended advisory checks: {'; '.join(fm.recommended_checks)}."
        )
        chunks.append(
            KnowledgeDocumentChunk(
                chunk_id=f"passport:mode-{fm.mode_id}",
                doc_id="passport-metropt3-apu-1",
                title="MetroPT-3 APU Failure Mode Catalog",
                section=fm.display_name,
                asset_id=passport.asset_id,
                source_path="factorymind/assets/passport.py",
                license="MIT (Repository Contract)",
                provenance_origin="derived",
                text=text,
            )
        )

    for model in passport.model_applicability:
        chunks.append(
            KnowledgeDocumentChunk(
                chunk_id=f"passport:model-{model.model_id}",
                doc_id="passport-metropt3-apu-1",
                title="MetroPT-3 Model Applicability Matrix",
                section=f"Model {model.model_id} ({model.status})",
                asset_id=passport.asset_id,
                source_path="factorymind/assets/passport.py",
                license="MIT (Repository Contract)",
                provenance_origin="derived",
                text=f"{model.scope_note} Limitations: {'; '.join(model.limitations)}",
            )
        )
    return chunks


def _markdown_chunks(
    path: Path,
    *,
    doc_id: str,
    title: str,
    rel_path: str,
    asset_id: str,
) -> list[KnowledgeDocumentChunk]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    parts = re.split(r"(?m)^##\s+", raw)
    chunks: list[KnowledgeDocumentChunk] = []
    for idx, part in enumerate(parts):
        cleaned = " ".join(part.split())
        if len(cleaned) < _MIN_CHUNK_CHARS:
            continue
        heading_line = part.splitlines()[0].strip() if part.splitlines() else f"Section {idx}"
        section = heading_line.lstrip("# ").strip()[:80] or f"Section {idx}"
        chunks.append(
            KnowledgeDocumentChunk(
                chunk_id=f"{doc_id}:sec-{idx}",
                doc_id=doc_id,
                title=title,
                section=section,
                asset_id=asset_id,
                source_path=rel_path,
                license="MIT (Repository Documentation)",
                provenance_origin="derived",
                text=cleaned[:1800],
            )
        )
    return chunks


def build_verified_corpus(
    raw_dir: Path,
    *,
    docs_root: Path | None = None,
) -> list[KnowledgeDocumentChunk]:
    """Assemble verified technical chunks from asset contracts, MetroPT-3 PDF, and repo docs."""
    chunks: list[KnowledgeDocumentChunk] = []
    chunks.extend(_passport_chunks())

    pdf_path = raw_dir / "metropt3" / "Data Description_Metro.pdf"
    chunks.extend(extract_pdf_chunks(pdf_path))

    resolved_docs = docs_root if docs_root is not None else find_repo_docs_dir()
    if resolved_docs is not None:
        repo_root = resolved_docs.parent
        for rel_path, title, asset_id in _REPO_DOC_FILES:
            candidate = (repo_root / rel_path).resolve()
            if not candidate.is_relative_to(repo_root.resolve()) or not candidate.is_file():
                continue
            doc_id = Path(rel_path).stem
            chunks.extend(
                _markdown_chunks(
                    candidate,
                    doc_id=doc_id,
                    title=title,
                    rel_path=rel_path,
                    asset_id=asset_id,
                )
            )
    return chunks
