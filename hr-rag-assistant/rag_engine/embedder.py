"""
Embedder Module for the HR Policy & Payroll Assistant RAG system.
Manages the SentenceTransformer model for generating vector embeddings locally.
"""

import streamlit as st
from sentence_transformers import SentenceTransformer

def get_embedder():
    """
    Loads and returns the SentenceTransformer model 'all-MiniLM-L6-v2'.

    Returns:
        SentenceTransformer: Loaded model.
    """
    try:
        # Load the model locally. It will download once and cache.
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model
    except Exception as e:
        error_msg = f"Failed to load embedding model: {str(e)}"
        st.error(error_msg)
        raise RuntimeError(error_msg) from e

def embed_texts(texts: list[str], embedder) -> list[list[float]]:
    """
    Converts list of texts into vector embeddings.

    Args:
        texts (list[str]): List of raw text strings to embed.
        embedder (SentenceTransformer): The loaded SentenceTransformer model.

    Returns:
        list[list[float]]: List of float vector embeddings.
    """
    try:
        if not texts:
            return []
        # Encode texts to embeddings list
        embeddings = embedder.encode(texts, show_progress_bar=True)
        return embeddings.tolist()
    except Exception as e:
        error_msg = f"Failed to generate text embeddings: {str(e)}"
        st.error(error_msg)
        raise RuntimeError(error_msg) from e
