"""Hybrid RAG, hardened PDF ingestion, typed Knowledge Graph, and retrieval evaluation."""

from factorymind.knowledge.corpus import build_verified_corpus, extract_pdf_chunks
from factorymind.knowledge.eval import evaluate_retrieval_modes
from factorymind.knowledge.kg import EPISTEMIC_LEGEND, IndustrialKnowledgeGraph
from factorymind.knowledge.rag import HybridRagIndex, SearchMode

__all__ = [
    "EPISTEMIC_LEGEND",
    "HybridRagIndex",
    "IndustrialKnowledgeGraph",
    "SearchMode",
    "build_verified_corpus",
    "evaluate_retrieval_modes",
    "extract_pdf_chunks",
]
