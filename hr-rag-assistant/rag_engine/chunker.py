"""
Chunker Module for the HR Policy & Payroll Assistant RAG system.
Responsible for splitting page text into smaller, overlapping chunks.
"""

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(pages: list[dict]) -> list[dict]:
    """
    Splits text from pages into overlapping chunks using RecursiveCharacterTextSplitter.

    Args:
        pages (list[dict]): List of page dictionaries, each having keys:
            "page" (int), "text" (str), and "source" (str).

    Returns:
        list[dict]: A list of chunk dictionaries, where each contains:
            - "chunk_text" (str): The chunked text segment.
            - "source" (str): The filename of the source PDF.
            - "page" (int): The page number the chunk originated from.
    """
    chunks_data = []
    
    try:
        # Initialize the LangChain text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=60,
            length_function=len
        )
        
        for page_data in pages:
            text = page_data["text"]
            source = page_data["source"]
            page_num = page_data["page"]
            
            # Split the text of this page
            split_texts = splitter.split_text(text)
            
            for split_text in split_texts:
                if split_text.strip():
                    chunks_data.append({
                        "chunk_text": split_text.strip(),
                        "source": source,
                        "page": page_num
                    })
                    
    except Exception as e:
        error_msg = f"Error during document chunking: {str(e)}"
        st.error(error_msg)
        raise RuntimeError(error_msg) from e
        
    return chunks_data
