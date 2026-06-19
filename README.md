# 📄 DocuMind — Agentic Document Intelligence

<div align="center">

**Ask anything about your documents. Get precise, cited answers.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-F55036?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-00C4CC?style=for-the-badge)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

## 🧠 What is DocuMind?

**DocuMind** is a universal document Q&A assistant built on **Agentic RAG** (Retrieval-Augmented Generation) architecture.

Upload any PDF — HR policies, research papers, legal contracts, payroll manuals, resumes — and ask questions in plain English. The AI agent autonomously searches your documents, evaluates the results, rewrites queries when needed, and delivers **precise, cited answers** grounded entirely in your uploaded files.

> **It never guesses. It only answers from your documents.**

---

## 🎬 Demo

```
User  → "How many sick leaves do I get per year?"

Agent → Iteration 1: Searching → "How many sick leaves do I get per year?"
      → Best similarity score: 0.821 | Chunks found: 5 new
      → ✅ Strong match found! Generating answer from 5 total chunks.

DocuMind → "Sick leaves are capped at 12 days per year and require
            a doctor certificate for more than 3 consecutive days.
            (Source: leave_policy.pdf | Page: 1)"
```

---

## ✨ Key Features

- 📂 **Universal PDF Support** — HR policies, research papers, contracts, manuals, resumes
- 🤖 **Agentic Search Loop** — autonomously retries with rewritten queries on weak matches
- 🛡️ **Hallucination Guard** — blocked from answering when no relevant content found
- 📎 **Source Citations** — every answer shows document name + page number
- ⚡ **Sub-second Responses** — powered by Groq's ultra-fast inference
- 🔍 **Show Agent Thinking** — toggle to see the agent's reasoning steps live
- 🏠 **Fully Local Embeddings** — no data sent for embedding (SentenceTransformer runs locally)

---

## 🏗️ Architecture

### Phase 1 — Indexing Pipeline (runs once per upload)

```
[PDF Upload]
     │
     ▼
[PyMuPDF Extractor]        → Extracts text page by page
     │
     ▼
[LangChain Text Splitter]  → 300-char chunks, 60-char overlap
     │
     ▼
[SentenceTransformer]      → Converts chunks to 384-dim vectors (local)
     │
     ▼
[ChromaDB Vector Store]    → Persists vectors + metadata to disk
```

### Phase 2 — Agentic Query Pipeline (runs every question)

```
[User Question]
     │
     ▼
[SentenceTransformer]        → Embeds question to vector
     │
     ▼
[ChromaDB Similarity Search] → Top 5 most relevant chunks
     │
     ▼
[Score Evaluation]
     ├── Score ≥ 0.45  →  Strong match → Generate answer ✅
     ├── Score < 0.45  →  Weak match → Groq rewrites query
     │                              → Search again (up to 3x)
     └── Score < 0.15  →  No match → Safe fallback message 🛡️
     │
     ▼
[Combine Multi-Iteration Results]
     │
     ▼
[Groq LLaMA 3.1 — Answer Generation]  → Reads ONLY retrieved context
     │
     ▼
[Answer + Source Citations → Streamlit UI]
```

---

## 🛡️ Hallucination Prevention — 3 Layers

| Layer | Mechanism | Effect |
|---|---|---|
| **Gate 1** | Similarity score < 0.15 → LLM call blocked | No guessing when docs are irrelevant |
| **Gate 2** | Weak match → Agent rewrites query + searches again | Better retrieval before giving up |
| **Gate 3** | Prompt constraint: "Answer ONLY from context" | LLM instructed to say "not found" |

---

## 🤖 Agentic RAG vs Normal RAG

| | Normal RAG | DocuMind (Agentic RAG) |
|---|---|---|
| Search attempts | 1 | Up to 3 (autonomous) |
| Query rewriting | ❌ | ✅ Auto via Groq |
| Result combining | ❌ | ✅ Multi-iteration pool |
| Hallucination guard | Basic threshold | 3-layer defense |
| Transparency | Hidden | Agent steps visible |

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **UI** | Streamlit | Full chat interface in Python |
| **PDF Reading** | PyMuPDF | Fastest, most reliable PDF parser |
| **Text Splitting** | LangChain RecursiveCharacterTextSplitter | Industry standard chunking |
| **Embeddings** | SentenceTransformer `all-MiniLM-L6-v2` | Free, local, no API needed |
| **Vector DB** | ChromaDB | Local persistent vector store |
| **LLM** | Groq LLaMA 3.1 8B Instant | Free, sub-second inference |
| **Agent Layer** | Custom Python | Autonomous search + query rewriting |

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+
- Free Groq API key → [console.groq.com](https://console.groq.com)

### Step 1 — Clone the repo
```bash
git clone https://github.com/yourusername/documind.git
cd documind
```

### Step 2 — Create virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

> ⏳ First run downloads the SentenceTransformer model (~90MB). One-time only.

### Step 4 — Configure API key
```bash
cp .env.example .env
```

Open `.env` and add your Groq key:
```
GROQ_API_KEY=your_groq_key_here
```

### Step 5 — Run the app
```bash
streamlit run app.py
```

Open → `http://localhost:8501`

---

## 🖥️ How to Use

```
1. Upload any PDF in the sidebar
        ↓
2. Click "Index Documents"
        ↓
3. Ask any question in the chat
        ↓
4. Watch the agent search, reflect, and answer
        ↓
5. Expand "Sources" to see which page the answer came from
```

---

## 📁 Project Structure

```
documind/
│
├── app.py                      # 🖥️  Streamlit chat UI
│
├── rag_engine/                 # 🧠  Core RAG pipeline
│   ├── __init__.py             #     Package exports
│   ├── agent.py                #     Agentic RAG loop (the brain)
│   ├── groq_client.py          #     Groq LLM + query rewriter
│   ├── chunker.py              #     Text splitting
│   ├── embedder.py             #     Vector embeddings (local)
│   ├── pdf_extractor.py        #     PDF text extraction
│   └── vector_store.py         #     ChromaDB operations
│
├── data/
│   └── sample_hr_docs/         # 📂  PDF storage (gitignored)
│
├── chroma_db/                  # 🗃️   Vector store (auto-created)
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 💬 Sample Questions to Try

After uploading any HR policy PDF:
```
✅ "How many annual leaves do I get?"
✅ "What is the gratuity calculation formula?"
✅ "What documents are needed for reimbursement?"
✅ "What is the notice period for resignation?"
✅ "Are sick leaves carried forward to next year?"
```

After uploading a research paper:
```
✅ "What is the main conclusion of this paper?"
✅ "What methodology was used in this study?"
✅ "What datasets were used for evaluation?"
```

---

## 🎤 Interview Explanation (90 Seconds)

> *"I built DocuMind — a universal document Q&A system on Agentic RAG architecture.*
>
> *First, I index uploaded PDFs into ChromaDB using a local SentenceTransformer embedding model — no data leaves the machine during indexing.*
>
> *When a user asks a question, the system doesn't just search once. An autonomous agent evaluates the similarity score of retrieved chunks. If the score is weak, the agent uses Groq LLaMA to rewrite the query into a better search term and tries again — up to 3 iterations — combining results across all searches.*
>
> *The hardest engineering challenge was hallucination prevention. I built a 3-layer guard: a similarity gate that blocks the LLM entirely when context is irrelevant, an agent retry loop for weak matches, and a strict prompt constraint that forces the LLM to cite sources or say 'not found'.*
>
> *The result is a system that answers in under a second, always cites the source document and page number, and refuses to guess."*

---

## 🚀 Future Improvements

| Feature | Description |
|---|---|
| Multi-language support | Answer in Tamil, Hindi, etc. |
| Chat memory | Remember previous questions in session |
| Hybrid search | BM25 + vector search combined |
| Re-ranking layer | Cross-encoder for better accuracy |
| Auto re-indexing | Detect PDF changes automatically |
| Authentication | User login before accessing |

---

## 📄 License

MIT License — Free to use, modify, and distribute.

---

<div align="center">

**Built as a Generative AI portfolio project**

Demonstrates · Agentic RAG Architecture · Hallucination Prevention · Production Python Engineering

⭐ Star this repo if it helped you in your interview preparation!

</div>
