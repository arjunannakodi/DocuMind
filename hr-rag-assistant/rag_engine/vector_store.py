"""
Vector Store Module for the HR Policy & Payroll Assistant RAG system.
Manages connection to ChromaDB, document insertion, and similarity search.
"""

import os
import streamlit as st
import chromadb
from rag_engine.embedder import embed_texts

COLLECTION_NAME = "hr_policies"

def get_chroma_client():
    """
    Returns an in-memory ChromaDB client stored in Streamlit session state.
    This avoids any disk writes, making it compatible with ephemeral environments
    like Streamlit Community Cloud.
    """
    if "_chroma_client" not in st.session_state:
        st.session_state["_chroma_client"] = chromadb.EphemeralClient()
    return st.session_state["_chroma_client"]

def collection_exists() -> bool:
    """
    Checks if the 'hr_policies' collection exists and has documents.

    Returns:
        bool: True if collection exists and has at least one document, False otherwise.
    """
    try:
        client = get_chroma_client()
        # list_collections() returns collections
        collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in collections:
            collection = client.get_collection(name=COLLECTION_NAME)
            return collection.count() > 0
        return False
    except Exception as e:
        # Don't fail the app initialization, just return False and log
        print(f"Checking collection existence failed: {str(e)}")
        return False

def store_chunks(chunks: list[dict], embedder):
    """
    Stores document chunks with their embeddings and metadata in ChromaDB.
    Recreates the collection to ensure it only contains the newly indexed documents.

    Args:
        chunks (list[dict]): List of chunk dictionaries containing:
            "chunk_text", "source", "page".
        embedder: The SentenceTransformer embedding model.
    """
    try:
        client = get_chroma_client()
        
        # Recreate the collection to clear out old documents and avoid duplicates
        try:
            client.delete_collection(name=COLLECTION_NAME)
        except Exception:
            # Collection may not exist, ignore
            pass
            
        # Create a new collection with cosine similarity configuration
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        if not chunks:
            return
            
        # Extract fields for ChromaDB
        texts = [chunk["chunk_text"] for chunk in chunks]
        metadatas = [{"source": chunk["source"], "page": chunk["page"]} for chunk in chunks]
        ids = [f"id_{i}" for i in range(len(chunks))]
        
        # Embed the texts using embedder
        embeddings = embed_texts(texts, embedder)
        
        # ChromaDB limits batch sizes in some configurations, but for our scale,
        # we can add them in batches or all at once. Let's add them.
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        
    except Exception as e:
        error_msg = f"Failed to store chunks in ChromaDB: {str(e)}"
        st.error(error_msg)
        raise RuntimeError(error_msg) from e

def search_chunks(query: str, embedder, n_results=5) -> list[dict]:
    """
    Searches ChromaDB for the top N most similar chunks to the query.

    Args:
        query (str): The search query.
        embedder: The SentenceTransformer model.
        n_results (int): Number of top results to retrieve.

    Returns:
        list[dict]: A list of results, each having:
            {"text": "...", "source": "...", "page": ..., "score": float}
    """
    try:
        client = get_chroma_client()
        
        # Check if collection exists first
        collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME not in collections:
            return []
            
        collection = client.get_collection(name=COLLECTION_NAME)
        
        # Embed query text
        query_embedding = embed_texts([query], embedder)[0]
        
        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Format the query results
        formatted_results = []
        if results and "documents" in results and results["documents"]:
            # ChromaDB returns nested list for queries
            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            for doc, meta, dist in zip(documents, metadatas, distances):
                # ChromaDB cosine distance is in [0, 2] (0=identical, 2=opposite).
                # Convert to similarity in [0, 1]: similarity = 1 - (dist / 2)
                similarity = 1.0 - (float(dist) / 2.0)
                
                formatted_results.append({
                    "text": doc,
                    "source": meta.get("source", "Unknown"),
                    "page": meta.get("page", 0),
                    "score": similarity
                })
                
        return formatted_results
        
    except Exception as e:
        error_msg = f"Failed to search vector store: {str(e)}"
        st.error(error_msg)
        raise RuntimeError(error_msg) from e

def get_collection_stats() -> dict:
    """
    Retrieves statistics about the stored collection in ChromaDB.

    Returns:
        dict: A dictionary containing:
            - "document_count" (int): Number of unique PDF documents.
            - "chunk_count" (int): Number of chunks stored in vector database.
            - "sources" (list[str]): List of unique filenames stored.
    """
    try:
        client = get_chroma_client()
        collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in collections:
            collection = client.get_collection(name=COLLECTION_NAME)
            count = collection.count()
            if count == 0:
                return {"document_count": 0, "chunk_count": 0, "sources": []}
            
            # Retrieve all metadata items to parse out unique filenames
            data = collection.get(include=["metadatas"])
            metadatas = data.get("metadatas", []) if data else []
            
            sources = set()
            for meta in metadatas:
                if meta and "source" in meta:
                    sources.add(meta["source"])
                    
            return {
                "document_count": len(sources),
                "chunk_count": count,
                "sources": sorted(list(sources))
            }
        return {"document_count": 0, "chunk_count": 0, "sources": []}
    except Exception as e:
        print(f"Failed to query collection stats: {str(e)}")
        return {"document_count": 0, "chunk_count": 0, "sources": []}

