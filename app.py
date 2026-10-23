import os
import streamlit as st
from pathlib import Path
from typing import Optional

# Import local modules
from config import (
    PDF_PATH,
    CHROMA_PERSIST_DIRECTORY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    TOP_K,
    NOT_FOUND_RESPONSE,
    get_api_key,
    is_api_key_configured,
)
from src.document_loader import get_pdf_metadata, load_and_chunk_pdf
from src.vector_store import (
    is_vectorstore_indexed,
    get_index_stats,
    get_vectorstore,
    get_embedding_model,
    index_documents,
)
from src.rag_engine import (
    HistoryRAGService,
    MissingAPIKeyError,
    InvalidAPIKeyError,
    APIQuotaError,
    IndexNotReadyError,
    RAGException,
)
from src.utils import get_system_status

# --- Page Configuration ---
st.set_page_config(
    page_title="Orient BlackSwan History Bot",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- History-Themed Custom Styling ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    :root {
        --obs-burgundy: #6B1724;
        --obs-deep-burgundy: #480D17;
        --obs-gold: #D4AF37;
        --obs-gold-light: #F7E7A9;
        --obs-dark: #0D1117;
        --obs-surface: #161B22;
        --obs-border: rgba(212, 175, 55, 0.25);
        --obs-text: #E6EDF3;
        --obs-muted: #8B949E;
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Header */
    .obs-header {
        background: linear-gradient(135deg, #2D0B12 0%, #161B22 50%, #0D1117 100%);
        border: 1px solid var(--obs-border);
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    
    .obs-header::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #D4AF37, #F7E7A9, #D4AF37);
    }

    .obs-title {
        font-family: 'Cinzel', serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #F7E7A9;
        margin: 0;
        letter-spacing: 0.5px;
    }

    .obs-subtitle {
        color: #C9D1D9;
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 400;
    }

    .obs-badge {
        display: inline-block;
        background: rgba(212, 175, 55, 0.15);
        border: 1px solid rgba(212, 175, 55, 0.4);
        color: #F7E7A9;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    /* Sidebar cards */
    .sidebar-card {
        background: #161B22;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }

    .sidebar-card-title {
        font-family: 'Cinzel', serif;
        font-size: 0.9rem;
        color: #D4AF37;
        font-weight: 700;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Citation Cards */
    .citation-card {
        background: #1A202C;
        border-left: 3px solid #D4AF37;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 0.88rem;
    }

    .citation-badge {
        font-weight: 700;
        color: #F7E7A9;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }

    .citation-text {
        color: #CBD5E0;
        font-style: italic;
        line-height: 1.4;
    }

    /* Not Found Callout */
    .not-found-banner {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 8px;
        padding: 12px 16px;
        color: #FCA5A5;
        font-size: 0.9rem;
        margin-top: 8px;
    }

    /* Example questions buttons */
    .stButton>button {
        border-radius: 8px;
        transition: all 0.2s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Helper: Initialize Cached Vector Store & RAG Service ---
@st.cache_resource(show_spinner=False)
def load_rag_service(api_key: str, persist_dir: str):
    """
    Initializes and caches the persistent vector store and HistoryRAGService.
    Does NOT re-index on every query.
    """
    embeddings = get_embedding_model(api_key=api_key)
    vector_store = get_vectorstore(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    return HistoryRAGService(
        vector_store=vector_store,
        api_key=api_key,
        model_name=GEMINI_MODEL,
        top_k=TOP_K,
    )


def handle_indexing(pdf_path: str, persist_dir: str, api_key: str):
    """
    Triggers indexing pipeline with UI progress bar.
    """
    progress_bar = st.progress(0, text="Reading and parsing PDF pages...")
    status_text = st.empty()

    try:
        # Step 1: Chunking
        status_text.text("Extracting text and chunking with page metadata...")
        chunks = load_and_chunk_pdf(pdf_path)
        progress_bar.progress(35, text=f"Extracted {len(chunks)} chunks. Connecting to Gemini Embeddings...")

        # Step 2: Embeddings (verify compatibility with gemini-embedding-001)
        try:
            embeddings = get_embedding_model(api_key=api_key)
            embeddings.embed_query("ping")
        except Exception as emb_test_err:
            err_msg = str(emb_test_err).lower()
            if "404" in err_msg or "not found" in err_msg or "not supported" in err_msg:
                # Try alternate candidates
                candidates = ["models/gemini-embedding-001", "gemini-embedding-001", "models/embedding-001", "embedding-001"]
                matched = None
                for c in candidates:
                    try:
                        test_emb = get_embedding_model(api_key=api_key, model_name=c)
                        test_emb.embed_query("ping")
                        matched = test_emb
                        break
                    except Exception:
                        continue
                if matched:
                    embeddings = matched
                else:
                    raise emb_test_err
            else:
                raise emb_test_err

        # Step 3: Batch indexing into ChromaDB
        status_text.text("Embedding chunks and saving to persistent ChromaDB...")

        # Safely remove partial or unindexed previous directory
        if Path(persist_dir).exists() and not is_vectorstore_indexed(persist_dir):
            import shutil
            try:
                shutil.rmtree(persist_dir)
            except Exception:
                pass

        def on_progress(done, total):
            pct = 35 + int((done / total) * 60)
            progress_bar.progress(pct, text=f"Indexing chunks: {done}/{total} ({int(done/total*100)}%)")

        index_documents(
            documents=chunks,
            persist_directory=persist_dir,
            embedding_function=embeddings,
            batch_size=75,
            progress_callback=on_progress,
        )

        progress_bar.progress(100, text="Indexing complete!")
        status_text.empty()
        # Clear cached service so new index loads immediately
        load_rag_service.clear()
        st.success(f"✓ Successfully indexed {len(chunks)} chunks into ChromaDB!")
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Indexing failed: {str(e)}")


# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# --- Sidebar ---
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 16px;">
            <div style="font-size: 2.2rem; margin-bottom: 4px;">🏛️</div>
            <div style="font-family: 'Cinzel', serif; font-size: 1.15rem; font-weight: 700; color: #F7E7A9;">ORIENT BLACKSWAN</div>
            <div style="font-size: 0.75rem; letter-spacing: 1.5px; color: #8B949E; text-transform: uppercase;">History Archives RAG</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. API Key Section
    env_key = get_api_key()
    has_env_key = is_api_key_configured()

    st.markdown('<div class="sidebar-card-title">🔑 API Authentication</div>', unsafe_allow_html=True)
    if has_env_key:
        active_api_key = env_key
        st.caption("🟢 Using Gemini API Key from environment (.env)")
    else:
        user_key_input = st.text_input(
            "Enter Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Acquire a free API key from Google AI Studio (https://aistudio.google.com/)",
        )
        active_api_key = user_key_input.strip() if user_key_input else ""
        if not active_api_key:
            st.warning("⚠️ Please provide a Gemini API Key to proceed.")

    st.divider()

    # 2. Document & Index Diagnostics
    status = get_system_status()
    pdf_info = status["pdf"]
    idx_info = status["vector_store"]

    st.markdown('<div class="sidebar-card-title">📄 Document Repository</div>', unsafe_allow_html=True)
    if pdf_info["exists"]:
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div style="font-weight: 600; color: #E6EDF3;">{pdf_info['filename']}</div>
                <div style="color: #8B949E; font-size: 0.8rem; margin-top: 4px;">
                    Pages: <b>{pdf_info['pages']}</b> • Size: <b>{pdf_info['size_mb']} MB</b>
                </div>
                <div style="margin-top: 6px; font-size: 0.75rem; color: #3FB950;">
                    ✓ Source file verified
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error(f"❌ Missing PDF: `{PDF_PATH}`")

    # Vector Index State
    st.markdown('<div class="sidebar-card-title">💾 ChromaDB Persistent Index</div>', unsafe_allow_html=True)
    if idx_info["indexed"]:
        st.markdown(
            f"""
            <div class="sidebar-card">
                <div style="color: #3FB950; font-weight: 600;">🟢 Index Active & Ready</div>
                <div style="color: #8B949E; font-size: 0.8rem; margin-top: 4px;">
                    Total Chunks: <b>{idx_info['count']}</b><br>
                    Storage: <code>{CHROMA_PERSIST_DIRECTORY}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🔄 Re-index Document", use_container_width=True, help="Force rebuild of ChromaDB vector store"):
            if not active_api_key:
                st.error("Please configure an API key before indexing.")
            else:
                handle_indexing(PDF_PATH, CHROMA_PERSIST_DIRECTORY, active_api_key)
    else:
        st.warning("Index not found or empty.")
        if st.button("⚡ Index Document Now", use_container_width=True, type="primary"):
            if not active_api_key:
                st.error("Please enter a valid Gemini API Key first.")
            elif not pdf_info["exists"]:
                st.error(f"PDF not found at {PDF_PATH}.")
            else:
                handle_indexing(PDF_PATH, CHROMA_PERSIST_DIRECTORY, active_api_key)

    st.divider()

    # 3. Example Assignment Questions
    st.markdown('<div class="sidebar-card-title">💡 Example Questions</div>', unsafe_allow_html=True)
    st.caption("Click any question to ask the bot:")

    example_questions = [
        "When was Orient BlackSwan incorporated and what is the consistent emphasis of its publishing programme?",
        "What is the title and author of the work on Women's Health in Modern Sri Lanka?",
        "What themes does 'The Mughal Empire and its Decline' explore?",
        "What publications are featured under Dalit Studies or Subaltern Studies?",
        "Who designed the iPhone in 2007? (Negative test: absent evidence)",
    ]

    for q in example_questions:
        # Truncate text for button label
        btn_label = q if len(q) <= 55 else q[:52] + "..."
        if st.button(btn_label, key=f"ex_{hash(q)}", use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    # 4. Controls & System Info
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.caption(
        f"**Model:** `{GEMINI_MODEL}`  \n"
        f"**Embeddings:** `{EMBEDDING_MODEL}`  \n"
        f"**Top-K Retrieval:** `{TOP_K}` passages  \n"
        f"**Strict Evidence Mode:** Enforced"
    )


# --- Main Screen ---

# Header Section
st.markdown(
    """
    <div class="obs-header">
        <div class="obs-badge">Academic Assignment • RAG System</div>
        <h1 class="obs-title">Orient BlackSwan History Bot</h1>
        <div class="obs-subtitle">
            Authoritative, evidence-grounded research assistant for the <i>History 2024–25</i> catalogue.
            Answers are drawn strictly from retrieved PDF evidence with verified page citations.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🎓" if msg["role"] == "user" else "📖"):
        st.markdown(msg["content"])
        
        # If assistant message has source citations, render them
        if msg.get("sources"):
            with st.expander(f"📚 Verified Sources & Citations ({len(msg['sources'])} retrieved)", expanded=False):
                for s in msg["sources"]:
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-badge">Page {s['page']} • {s['source']}</div>
                            <div class="citation-text">"{s['excerpt']}"</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        elif msg["role"] == "assistant" and msg.get("is_not_found"):
            st.markdown(
                """
                <div class="not-found-banner">
                    ⚠️ <b>Strict Evidence Guardrail:</b> No supporting evidence was found in <i>History_2024-25.pdf</i>. 
                    In accordance with academic integrity rules, this system returns exactly <code>Not found</code> rather than using general outside knowledge.
                </div>
                """,
                unsafe_allow_html=True,
            )

# Check for pending question from example buttons
user_input = st.chat_input("Ask a question about Orient BlackSwan History 2024-25...")
if st.session_state.pending_question:
    user_input = st.session_state.pending_question
    st.session_state.pending_question = None

# Process User Input
if user_input:
    # Append and render user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(user_input)

    # Validate readiness
    if not active_api_key:
        with st.chat_message("assistant", avatar="📖"):
            st.error("⚠️ **API Key Required:** Please configure `GEMINI_API_KEY` in your `.env` file or enter it in the sidebar.")
    elif not is_vectorstore_indexed(CHROMA_PERSIST_DIRECTORY):
        with st.chat_message("assistant", avatar="📖"):
            st.warning("⚠️ **Index Not Built:** Please click **'⚡ Index Document Now'** in the sidebar to build the ChromaDB index before asking questions.")
    else:
        with st.chat_message("assistant", avatar="📖"):
            with st.spinner("Searching document index and verifying citations..."):
                try:
                    rag_service = load_rag_service(
                        api_key=active_api_key,
                        persist_dir=CHROMA_PERSIST_DIRECTORY,
                    )
                    response = rag_service.query(user_input)
                    answer = response["answer"]
                    sources = response["sources"]
                    is_not_found = not response["is_found"]

                    # Display Answer
                    st.markdown(answer)

                    # Display citations or Not found alert
                    if sources:
                        with st.expander(f"📚 Verified Sources & Citations ({len(sources)} retrieved)", expanded=True):
                            for s in sources:
                                st.markdown(
                                    f"""
                                    <div class="citation-card">
                                        <div class="citation-badge">Page {s['page']} • {s['source']}</div>
                                        <div class="citation-text">"{s['excerpt']}"</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                    elif is_not_found:
                        st.markdown(
                            """
                            <div class="not-found-banner">
                                ⚠️ <b>Strict Evidence Guardrail:</b> No supporting evidence was found in <i>History_2024-25.pdf</i>. 
                                In accordance with academic integrity rules, this system returns exactly <code>Not found</code> rather than using general outside knowledge.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Store in chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "is_not_found": is_not_found,
                    })

                except MissingAPIKeyError:
                    st.error("🔑 **API Key Error:** Gemini API key is missing. Please provide it in the sidebar.")
                except InvalidAPIKeyError:
                    st.error("❌ **Invalid API Key:** The provided Gemini API Key was rejected. Please verify it at Google AI Studio.")
                except APIQuotaError:
                    st.error("⏳ **Quota Exceeded:** Google Gemini API rate limit reached. Please wait a moment before trying again.")
                except Exception as e:
                    st.error(f"⚠️ **Error:** {str(e)}")
