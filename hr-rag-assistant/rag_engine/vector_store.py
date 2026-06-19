"""
Vector Store Module — pure in-memory store using numpy.
Replaces ChromaDB to avoid the chromadb → opentelemetry → protobuf
dependency conflict on Streamlit Cloud (and other ephemeral environments).
"""

import numpy as np
import streamlit as st
from rag_engine.embedder import embed_texts

_STORE_KEY = "_vector_store"


def _get_store() -> dict:
    """Return (or initialise) the in-memory store from session state."""
    if _STORE_KEY not in st.session_state:
        st.session_state[_STORE_KEY] = {
            "embeddings": [],   # list[np.ndarray]
            "documents": [],    # list[str]
            "metadatas": [],    # list[dict]
        }
    return st.session_state[_STORE_KEY]


def collection_exists() -> bool:
    """Return True if any documents have been indexed this session."""
    return len(_get_store()["documents"]) > 0


def store_chunks(chunks: list[dict], embedder):
    """
    Embed and store document chunks in the session-level vector store.
    Replaces any previously indexed data.

    Args:
        chunks: list of dicts with keys 'chunk_text', 'source', 'page'.
        embedder: SentenceTransformer model instance.
    """
    store = _get_store()
    store["embeddings"] = []
    store["documents"] = []
    store["metadatas"] = []

    if not chunks:
        return

    texts     = [c["chunk_text"] for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    embeddings = embed_texts(texts, embedder)

    store["embeddings"] = [np.array(e, dtype=np.float32) for e in embeddings]
    store["documents"]  = texts
    store["metadatas"]  = metadatas


def search_chunks(query: str, embedder, n_results: int = 5) -> list[dict]:
    """
    Return the top-N most similar chunks to the query using cosine similarity.

    Args:
        query: The search string.
        embedder: SentenceTransformer model instance.
        n_results: Maximum number of results to return.

    Returns:
        List of dicts: {text, source, page, score}
    """
    store = _get_store()
    if not store["documents"]:
        return []

    query_vec = np.array(embed_texts([query], embedder)[0], dtype=np.float32)

    matrix = np.array(store["embeddings"], dtype=np.float32)          # (N, D)
    q_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    d_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
    similarities = d_norms @ q_norm                                    # (N,)

    n = min(n_results, len(similarities))
    top_indices = np.argsort(similarities)[::-1][:n]

    return [
        {
            "text":   store["documents"][i],
            "source": store["metadatas"][i].get("source", "Unknown"),
            "page":   store["metadatas"][i].get("page", 0),
            "score":  float(similarities[i]),
        }
        for i in top_indices
    ]


def get_collection_stats() -> dict:
    """Return stats about the current in-memory index."""
    store = _get_store()
    if not store["documents"]:
        return {"document_count": 0, "chunk_count": 0, "sources": []}

    sources = sorted({m.get("source", "") for m in store["metadatas"]})
    return {
        "document_count": len(sources),
        "chunk_count":    len(store["documents"]),
        "sources":        sources,
    }
