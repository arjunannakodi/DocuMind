"""
Universal Document Q&A Assistant — Agentic RAG
Powered by Groq Mixtral + ChromaDB + SentenceTransformers
"""

import os
import streamlit as st
from rag_engine import (
    extract_text_from_pdf,
    chunk_documents,
    get_embedder,
    store_chunks,
    collection_exists,
    get_collection_stats,
    run_rag_agent,
)

# Page config
st.set_page_config(
    page_title="DocuMind",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0d1117;
        color: #c9d1d9;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    .main-title {
        font-weight: 700;
        font-size: 2.5rem;
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .subtitle { font-size: 1.1rem; color: #8b949e; margin-bottom: 2rem; }
    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
    }
    .source-tag {
        display: inline-block; padding: 3px 10px;
        background: rgba(59,130,246,0.12);
        border: 1px solid rgba(59,130,246,0.3);
        border-radius: 6px; color: #60a5fa;
        font-size: 0.8rem; font-weight: 600;
        margin-right: 6px; margin-top: 6px;
    }
    .page-tag {
        display: inline-block; padding: 3px 10px;
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 6px; color: #34d399;
        font-size: 0.8rem; font-weight: 600;
        margin-right: 6px; margin-top: 6px;
    }
    .score-tag {
        display: inline-block; padding: 3px 10px;
        background: rgba(245,158,11,0.12);
        border: 1px solid rgba(245,158,11,0.3);
        border-radius: 6px; color: #fbbf24;
        font-size: 0.8rem; font-weight: 600;
        margin-top: 6px;
    }
    .agent-tag {
        display: inline-block; padding: 3px 10px;
        background: rgba(139,92,246,0.12);
        border: 1px solid rgba(139,92,246,0.3);
        border-radius: 6px; color: #a78bfa;
        font-size: 0.8rem; font-weight: 600;
        margin-right: 6px; margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='font-size:1.6rem;margin-bottom:0.5rem;'>"
        "📄 DocuMind</h2>", 
        unsafe_allow_html=True
    )
    st.markdown(
        "<div style='border-bottom:1px solid #30363d;"
        "margin-bottom:1.5rem;'></div>", 
        unsafe_allow_html=True
    )

    db_ready = collection_exists()
    stats = (
        get_collection_stats() if db_ready 
        else {"document_count": 0, "chunk_count": 0, "sources": []}
    )

    uploaded_files = st.file_uploader(
        "Upload Any PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload any PDF — policies, resumes, manuals, reports, research papers."
    )

    # Show agent steps toggle
    show_steps = st.toggle(
        "🤖 Show Agent Thinking Steps", 
        value=True,
        help="Shows what the AI agent is doing while searching"
    )

    if st.button("📚 Index Documents", use_container_width=True):
        if not uploaded_files:
            st.error("⚠️ Please upload at least one PDF file first.")
        else:
            with st.spinner("📚 Indexing documents, please wait..."):
                try:
                    data_dir = "./data/sample_hr_docs"
                    os.makedirs(data_dir, exist_ok=True)

                    for existing_file in os.listdir(data_dir):
                        fp = os.path.join(data_dir, existing_file)
                        if os.path.isfile(fp) and not existing_file.endswith(".gitkeep"):
                            os.remove(fp)

                    pdf_paths = []
                    for uf in uploaded_files:
                        target = os.path.join(data_dir, uf.name)
                        with open(target, "wb") as f:
                            f.write(uf.getbuffer())
                        pdf_paths.append(target)

                    all_pages = []
                    for path in pdf_paths:
                        all_pages.extend(extract_text_from_pdf(path))

                    if not all_pages:
                        st.warning("⚠️ No extractable text found in uploaded PDFs.")
                    else:
                        chunks = chunk_documents(all_pages)
                        embedder = get_embedder()
                        store_chunks(chunks, embedder)
                        st.success(
                            f"✅ Indexed {len(pdf_paths)} document(s) "
                            f"({len(chunks)} chunks) successfully!"
                        )
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Indexing failed: {str(e)}")

    # API Key check
    groq_key = bool(os.getenv("GROQ_API_KEY"))
    if not groq_key:
        ui_key = st.text_input(
            "Enter Groq API Key", 
            type="password",
            help="Get free key at console.groq.com"
        )
        if ui_key:
            os.environ["GROQ_API_KEY"] = ui_key
            st.success("✅ Groq API Key set for this session!")
            st.rerun()
        else:
            st.warning(
                "🔑 No Groq API key found. "
                "[Get a free key here](https://console.groq.com)"
            )

    # DB Status
    st.markdown("### 📊 Database Status")
    if db_ready:
        st.markdown(
            f"<div class='glass-card' style='padding:12px;"
            f"border-color:rgba(16,185,129,0.4);'>"
            f"<span style='color:#10b981;font-weight:600;'>● System Ready</span><br/>"
            f"<span style='font-size:0.9rem;'>Documents: {stats['document_count']}</span><br/>"
            f"<span style='font-size:0.9rem;'>Total Chunks: {stats['chunk_count']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        if stats["sources"]:
            with st.expander("📄 Indexed Files", expanded=False):
                for src in stats["sources"]:
                    st.caption(f"✓ {src}")
    else:
        st.markdown(
            "<div class='glass-card' style='padding:12px;"
            "border-color:rgba(239,68,68,0.4);'>"
            "<span style='color:#ef4444;font-weight:600;'>● No Documents Indexed</span><br/>"
            "<span style='font-size:0.85rem;color:#8b949e;'>"
            "Upload PDFs and click Index Documents.</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    st.caption("⚡ Powered by Agentic RAG + Groq Mixtral")

# ── Main Area ─────────────────────────────────────
st.markdown(
    "<div class='main-title'>Ask Questions About Your Documents</div>",
    unsafe_allow_html=True
)
st.markdown(
    "<div class='subtitle'>Upload any PDF — reports, manuals, resumes, "
    "policies, research papers — and ask anything.</div>",
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📎 Sources + Agent Info"):
                # Show agent metadata
                if "iterations" in message:
                    st.markdown(
                        f"<span class='agent-tag'>"
                        f"🔄 {message['iterations']} search iteration(s)</span>",
                        unsafe_allow_html=True
                    )
                if "sub_queries" in message and len(message["sub_queries"]) > 1:
                    for q in message["sub_queries"]:
                        st.markdown(
                            f"<span class='agent-tag'>🔍 \"{q}\"</span>",
                            unsafe_allow_html=True
                        )
                st.markdown("---")
                for src in message["sources"]:
                    st.markdown(
                        f"<span class='source-tag'>📄 {src['source']}</span>"
                        f"<span class='page-tag'>Page {src['page']}</span>"
                        f"<span class='score-tag'>"
                        f"Similarity: {src['score']:.2f}</span>",
                        unsafe_allow_html=True
                    )
                    st.markdown(f"> *{src['text'][:200]}...*")
                    st.markdown("<div style='margin-bottom:10px;'></div>",
                               unsafe_allow_html=True)

# Chat input & agent pipeline
if user_question := st.chat_input(
    "Ask anything about your uploaded documents..."
):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    if not db_ready:
        with st.chat_message("assistant"):
            st.warning(
                "⚠️ No documents indexed yet. "
                "Upload PDFs and click Index Documents first."
            )
    else:
        with st.chat_message("assistant"):
            with st.spinner("🤖 Agent is searching your documents..."):
                try:
                    result = run_rag_agent(
                        question=user_question,
                        embedder=get_embedder(),
                        show_agent_steps=show_steps
                    )

                    st.markdown(result["answer"])

                    if result["sources"]:
                        with st.expander("📎 Sources + Agent Info"):
                            st.markdown(
                                f"<span class='agent-tag'>"
                                f"🔄 {result['iterations']} "
                                f"search iteration(s)</span>",
                                unsafe_allow_html=True
                            )
                            if len(result["sub_queries"]) > 1:
                                for q in result["sub_queries"]:
                                    st.markdown(
                                        f"<span class='agent-tag'>"
                                        f"🔍 \"{q}\"</span>",
                                        unsafe_allow_html=True
                                    )
                            st.markdown("---")
                            for src in result["sources"]:
                                st.markdown(
                                    f"<span class='source-tag'>"
                                    f"📄 {src['source']}</span>"
                                    f"<span class='page-tag'>"
                                    f"Page {src['page']}</span>"
                                    f"<span class='score-tag'>"
                                    f"Similarity: {src['score']:.2f}</span>",
                                    unsafe_allow_html=True
                                )
                                st.markdown(
                                    f"> *{src['text'][:200]}...*"
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                        "iterations": result["iterations"],
                        "sub_queries": result["sub_queries"]
                    })

                except Exception as e:
                    st.error(f"❌ Agent error: {str(e)}")
