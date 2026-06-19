"""
RAG Engine package — Universal Document Q&A Assistant.
Agentic RAG architecture with Groq LLM.
"""

from rag_engine.pdf_extractor import extract_text_from_pdf
from rag_engine.chunker import chunk_documents
from rag_engine.embedder import get_embedder, embed_texts
from rag_engine.vector_store import (
    store_chunks, 
    search_chunks, 
    collection_exists, 
    get_collection_stats
)
from rag_engine.groq_client import ask_groq
from rag_engine.agent import run_rag_agent

__all__ = [
    "extract_text_from_pdf",
    "chunk_documents",
    "get_embedder",
    "embed_texts",
    "store_chunks",
    "search_chunks",
    "collection_exists",
    "get_collection_stats",
    "ask_groq",
    "run_rag_agent",
]
