"""
Agentic RAG Engine for the Universal Document Q&A Assistant.

This agent autonomously decides:
- Whether one search is enough
- When to rewrite the query and search again
- How to combine multi-iteration results
- When to give up and return a safe fallback

This is what separates this project from basic RAG chatbots.
"""

import streamlit as st
from rag_engine.vector_store import search_chunks
from rag_engine.groq_client import ask_groq, rewrite_query_with_groq

# Thresholds
STRONG_MATCH_THRESHOLD = 0.45   # Score above this = confident answer
WEAK_MATCH_THRESHOLD = 0.15     # Score below this = no relevant content
MAX_ITERATIONS = 3              # Maximum search attempts by agent

def run_rag_agent(
    question: str,
    embedder,
    show_agent_steps: bool = True
) -> dict:
    """
    Agentic RAG pipeline that autonomously searches, evaluates,
    rewrites queries, and combines results across multiple iterations.

    Args:
        question: The user's natural language question
        embedder: SentenceTransformer embedding model
        show_agent_steps: Whether to show agent thinking in Streamlit UI

    Returns:
        dict with keys:
            - answer (str): Final generated answer
            - sources (list): All retrieved chunks used
            - iterations (int): Number of search rounds performed
            - sub_queries (list): All queries attempted by agent
            - status (str): "success", "low_confidence", or "no_match"
    """
    all_chunks = []
    sub_queries = [question]
    current_query = question
    iterations = 0
    
    for iteration in range(MAX_ITERATIONS):
        iterations += 1
        
        if show_agent_steps:
            st.caption(
                f"🤖 Agent Iteration {iteration + 1}: "
                f"Searching for → \"{current_query}\""
            )
        
        # Search ChromaDB with current query
        retrieved = search_chunks(current_query, embedder, n_results=5)
        
        if not retrieved:
            break
        
        # Add new unique chunks to combined pool
        existing_texts = {c["text"] for c in all_chunks}
        new_chunks = [
            c for c in retrieved 
            if c["text"] not in existing_texts
        ]
        all_chunks.extend(new_chunks)
        
        # Check best score in this iteration
        best_score = max(c["score"] for c in retrieved)
        
        if show_agent_steps:
            st.caption(
                f"   📊 Best similarity score: {best_score:.3f} | "
                f"Chunks found: {len(new_chunks)} new"
            )
        
        # AGENT DECISION LOGIC:
        
        # Case 1: Strong match found → stop searching, generate answer
        if best_score >= STRONG_MATCH_THRESHOLD:
            if show_agent_steps:
                st.caption(
                    f"   ✅ Strong match found! "
                    f"Generating answer from "
                    f"{len(all_chunks)} total chunks."
                )
            break
        
        # Case 2: Weak match → agent rewrites query and tries again
        if best_score < STRONG_MATCH_THRESHOLD and iteration < MAX_ITERATIONS - 1:
            if show_agent_steps:
                st.caption(
                    f"   🔄 Weak match (score: {best_score:.3f}). "
                    f"Agent is rewriting the search query..."
                )
            
            # Agent rewrites the query using Groq
            rewritten = rewrite_query_with_groq(question)
            
            # Avoid repeating the same query
            if rewritten not in sub_queries:
                sub_queries.append(rewritten)
                current_query = rewritten
                if show_agent_steps:
                    st.caption(f"   ✏️ New query: \"{rewritten}\"")
            else:
                # Query already tried, stop iterating
                break
    
    # Final decision after all iterations
    if not all_chunks:
        return {
            "answer": (
                "⚠️ No relevant content found in your uploaded documents. "
                "Please upload the correct PDF and try again."
            ),
            "sources": [],
            "iterations": iterations,
            "sub_queries": sub_queries,
            "status": "no_match"
        }
    
    # Check if best overall score is too low
    best_overall = max(c["score"] for c in all_chunks)
    
    if best_overall < WEAK_MATCH_THRESHOLD:
        return {
            "answer": (
                "⚠️ No strong match found in your uploaded documents. "
                "Try rephrasing your question using keywords from the PDF, "
                "or upload a more relevant document."
            ),
            "sources": all_chunks,
            "iterations": iterations,
            "sub_queries": sub_queries,
            "status": "low_confidence"
        }
    
    # Sort all chunks by score, take top 5
    top_chunks = sorted(
        all_chunks, 
        key=lambda x: x["score"], 
        reverse=True
    )[:5]
    
    # Generate final answer using Groq
    answer = ask_groq(question, top_chunks)
    
    return {
        "answer": answer,
        "sources": top_chunks,
        "iterations": iterations,
        "sub_queries": sub_queries,
        "status": "success"
    }
