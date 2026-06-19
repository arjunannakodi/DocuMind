"""
Groq LLM Client for the Universal Document Q&A Assistant.
Primary LLM provider — fast, free, no rate limit issues.
"""

import os
import time
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

def get_groq_client():
    """Returns authenticated Groq client."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to your .env file. "
            "Get a free key at https://console.groq.com"
        )
    return Groq(api_key=api_key)

def ask_groq(question: str, context_chunks: list[dict]) -> str:
    """
    Sends question + retrieved context to Groq Mixtral model.
    
    Args:
        question: User's question
        context_chunks: Retrieved chunks with text, source, page, score
        
    Returns:
        str: Generated answer grounded in document context
    """
    context_parts = []
    for chunk in context_chunks:
        header = f"[Source: {chunk['source']} | Page: {chunk['page']}]"
        context_parts.append(f"{header}\n{chunk['text']}")
    
    context_string = "\n\n".join(context_parts)
    
    system_prompt = """You are an intelligent document assistant.
Answer the user's question ONLY using the context provided.
Do NOT use your own training knowledge.
Always cite the source document and page number.
If answer is not in context, say: 
'This information is not found in your uploaded documents.'
Be concise, accurate, and always mention: (Source: filename | Page: X)"""

    user_prompt = f"""CONTEXT FROM UPLOADED DOCUMENTS:
{context_string}

USER QUESTION:
{question}

ANSWER:"""

    # Retry logic for any rate limit
    for attempt in range(3):
        try:
            client = get_groq_client()
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                wait_time = 10 * (attempt + 1)
                st.warning(f"⏳ Rate limit hit. Retrying in {wait_time}s... "
                          f"(Attempt {attempt + 1}/3)")
                time.sleep(wait_time)
            else:
                raise RuntimeError(f"Groq API error: {str(e)}") from e
    
    raise RuntimeError(
        "Rate limit reached after 3 retries. "
        "Please wait 1 minute and try again."
    )

def rewrite_query_with_groq(original_question: str) -> str:
    """
    Uses Groq to rewrite the user question into a better search query.
    This is the CORE of Agentic RAG — the agent rewrites bad queries.
    
    Args:
        original_question: The user's original question
        
    Returns:
        str: A better, more specific search query
    """
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query optimizer. "
                        "Rewrite the given question into a better, "
                        "more specific search query for finding "
                        "relevant text in documents. "
                        "Return ONLY the rewritten query. "
                        "No explanation. One line only."
                    )
                },
                {
                    "role": "user",
                    "content": f"Original question: {original_question}\n"
                               f"Rewritten search query:"
                }
            ],
            temperature=0.1,
            max_tokens=100
        )
        rewritten = response.choices[0].message.content.strip()
        return rewritten if rewritten else original_question
        
    except Exception:
        # If rewrite fails, fall back to original question
        return original_question
