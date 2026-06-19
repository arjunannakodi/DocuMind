"""
PDF Extractor Module for the HR Policy & Payroll Assistant RAG system.
Responsible for extracting text page-by-page from uploaded PDF documents.
"""

import os
import fitz  # PyMuPDF
import streamlit as st

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts raw text from every page of the given PDF file.

    Args:
        pdf_path (str): Absolute or relative path to the PDF file.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary contains:
            - "page" (int): The page number (1-based index).
            - "text" (str): The text content extracted from the page.
            - "source" (str): The filename of the source PDF.
    """
    pages_data = []
    filename = os.path.basename(pdf_path)
    
    try:
        # Open the PDF file
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text().strip()
            
            # Skip blank pages
            if not text:
                continue
            
            pages_data.append({
                "page": page_num + 1,  # 1-based page numbers for user readability
                "text": text,
                "source": filename
            })
            
        doc.close()
        
    except Exception as e:
        error_msg = f"Error extracting text from PDF '{filename}': {str(e)}"
        st.error(error_msg)
        raise RuntimeError(error_msg) from e
        
    return pages_data
