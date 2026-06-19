# 📄 DocuMind — Agentic Document Intelligence
### Ask anything about your documents. Get cited answers.

> Powered by Groq LLaMA 3.1 · ChromaDB · Agentic RAG

---

## 🧠 What is DocuMind?

DocuMind is a universal document Q&A assistant built 
on Agentic RAG architecture. Upload any PDF and ask 
questions in natural language — the AI agent searches 
your documents, reflects on results, rewrites queries 
if needed, and delivers precise cited answers.

It never guesses. It only answers from your documents.

---

## 🏗️ Architecture

### Indexing Pipeline
[PDF Upload] → [PyMuPDF Extractor] → [Text Chunker]
→ [SentenceTransformer Embedder] → [ChromaDB]

### Agentic Query Pipeline
[Question] → [Embed] → [ChromaDB Search]
→ [Score Check]
  → Strong match: Generate answer
  → Weak match: Groq rewrites query → Search again
→ [Combine Results] → [Groq LLaMA generates answer]
→ [Answer + Source + Page Citations]

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| PDF Reading | PyMuPDF |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | SentenceTransformer all-MiniLM-L6-v2 (local) |
| Vector DB | ChromaDB (local persistent) |
| LLM | Groq LLaMA 3.1 8B Instant (free) |
| Agent Layer | Custom Agentic RAG loop |

---

## ⚡ Quick Start

### 1. Install
```
pip install -r requirements.txt
```

### 2. Add API Key
```
cp .env.example .env
# Add GROQ_API_KEY — free at https://console.groq.com
```

### 3. Run
```
streamlit run app.py
```

---

## 📁 Project Structure

```
documind/
├── app.py                  # Streamlit UI
├── requirements.txt
├── .env
├── .gitignore
├── rag_engine/
│   ├── __init__.py
│   ├── agent.py            # Agentic RAG loop
│   ├── groq_client.py      # Groq LLM + query rewriter
│   ├── chunker.py
│   ├── embedder.py
│   ├── pdf_extractor.py
│   └── vector_store.py
└── data/
    └── sample_hr_docs/
```

---

## 🛡️ Hallucination Prevention

- Similarity threshold gate (score < 0.15 → blocked)
- Agent retry with rewritten query on weak matches
- Prompt constraint: LLM must cite source or say not found
- Source + page number shown for every answer

---

## 🤖 Agentic RAG vs Normal RAG

```
Normal RAG:  Search once → Answer
Agentic RAG: Search → Evaluate → Rewrite if weak
             → Search again → Combine → Better answer
```

The agent autonomously decides how many search 
iterations are needed before generating a final answer.

---

## 📄 Supported Documents

Any text-based PDF: HR policies, payroll manuals,
research papers, legal contracts, product manuals,
financial reports, resumes, study notes.

---

Built as a Generative AI portfolio project.
Demonstrates Agentic RAG, hallucination prevention,
and production-ready modular Python architecture.
