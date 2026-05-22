from __future__ import annotations

import os
from html import escape
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from utils.chat_handler import ask_pdf_question, build_qa_chain
from utils.embeddings import (
    DEFAULT_MODE,
    build_chunk_ids,
    chunk_documents,
    get_vectorstore,
    load_processed_registry,
    normalize_mode,
    update_processed_registry,
)
from utils.pdf_loader import (
    ensure_project_directories,
    load_documents_from_saved_files,  
    save_uploaded_files,
)
from utils.retriever import get_retriever, vectorstore_has_documents


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
DATABASE_DIR = BASE_DIR / "database"

st.set_page_config(
    page_title="AskMyPDF AI",
    page_icon="📄",
    layout="wide",
)

MODE_LABELS = {
    "local": "Local (Free, no API billing)",
    "openai": "OpenAI API",
}


def initialize_session_state() -> None:
    """Create the session keys we need across reruns."""
    defaults = {
        "chat_history": [],
        "qa_chain": None,
        "vectorstore_ready": False,
        "process_report": None,
        "startup_error": None,
        "initialized": False,
        "ai_mode": DEFAULT_MODE,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def has_openai_api_key() -> bool:
    """Check whether the app has the required OpenAI API key."""
    return bool(os.getenv("OPENAI_API_KEY"))


def explain_openai_error(exc: Exception) -> tuple[str, str]:
    """Convert raw OpenAI errors into clearer UI messages."""
    error_text = str(exc)
    lower_text = error_text.lower()

    if "insufficient_quota" in lower_text or "exceeded your current quota" in lower_text:
        return (
            "Your OpenAI API key is being accepted, but that OpenAI project has no available API quota or billing left.",
            "Add credits or billing in your OpenAI API account, or replace the key with another API key that has available quota.",
        )

    if "rate limit" in lower_text:
        return (
            "OpenAI rate limits were reached for this API key.",
            "Wait a little and try again, or use an API project with higher limits.",
        )

    if "incorrect api key" in lower_text or "invalid api key" in lower_text or "authentication" in lower_text:
        return (
            "The OpenAI API key appears to be invalid or no longer active.",
            "Double-check the key in your `.env` file and create a new one in the OpenAI dashboard if needed.",
        )

    return (
        f"An error occurred: {error_text}",
        "Please review your OpenAI key, billing, and network connection, then try again.",
    )


def explain_runtime_error(mode: str, exc: Exception) -> tuple[str, str]:
    """Explain errors based on the current answer engine."""
    if normalize_mode(mode) == "openai":
        return explain_openai_error(exc)

    return (
        f"Local mode ran into an error: {exc}",
        "Try processing the PDFs again. If you want GPT-generated answers later, you can switch to OpenAI mode.",
    )


def apply_app_styles() -> None:
    """Small style improvements for a cleaner dark Streamlit UI."""
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
            }
            .app-card {
                border: 1px solid #2a3240;
                border-radius: 18px;
                padding: 1.1rem;
                background: #171923;
                box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
            }
            .answer-card {
                border: 1px solid #2a3240;
                border-radius: 14px;
                padding: 0.95rem 1rem;
                background: #151823;
                color: #f8fafc;
                margin-bottom: 0.85rem;
                line-height: 1.7;
            }
            .source-card {
                border-left: 4px solid #0ea5e9;
                border: 1px solid #243447;
                background: #101826;
                padding: 0.9rem;
                border-radius: 12px;
                margin-bottom: 0.7rem;
                color: #e5eefb;
            }
            .status-chip {
                display: inline-block;
                padding: 0.3rem 0.6rem;
                border-radius: 999px;
                background: #e0f2fe;
                color: #075985;
                font-size: 0.85rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
            }
            .source-card strong {
                color: #7dd3fc;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def refresh_qa_chain(mode: str | None = None) -> None:
    """Reconnect the retriever and QA chain from the persisted Chroma database."""
    current_mode = normalize_mode(mode or st.session_state.ai_mode)

    if current_mode == "openai" and not has_openai_api_key():
        st.session_state.qa_chain = None
        st.session_state.vectorstore_ready = False
        st.session_state.startup_error = None
        return

    try:
        vectorstore = get_vectorstore(DATABASE_DIR, current_mode)
        if vectorstore_has_documents(vectorstore):
            retriever = get_retriever(DATABASE_DIR, current_mode)
            st.session_state.qa_chain = build_qa_chain(retriever, current_mode)
            st.session_state.vectorstore_ready = True
            st.session_state.startup_error = None
        else:
            st.session_state.qa_chain = None
            st.session_state.vectorstore_ready = False
            st.session_state.startup_error = None
    except Exception as exc:
        st.session_state.qa_chain = None
        st.session_state.vectorstore_ready = False
        st.session_state.startup_error = str(exc)


def initialize_app() -> None:
    """Prepare folders and bootstrap any existing vector database."""
    ensure_project_directories(UPLOADS_DIR, DATABASE_DIR)
    initialize_session_state()

    if not st.session_state.initialized:
        refresh_qa_chain()
        st.session_state.initialized = True


def show_process_report() -> None:
    """Show the latest PDF processing summary."""
    report = st.session_state.process_report
    if not report:
        return

    if report["added_files"]:
        st.success(
            f"Processed {len(report['added_files'])} new PDF file(s) and created {report['chunk_count']} text chunk(s)."
        )

    if report["already_indexed"]:
        st.info("Already indexed and skipped: " + ", ".join(report["already_indexed"]))

    if report["duplicate_uploads"]:
        st.info("Duplicate uploads in the current batch were skipped: " + ", ".join(report["duplicate_uploads"]))

    if report["failed_files"]:
        failed_messages = [f"{item['name']} ({item['error']})" for item in report["failed_files"]]
        st.warning("Some files could not be processed: " + "; ".join(failed_messages))


def process_uploaded_pdfs(uploaded_files: list, mode: str) -> None:
    """Save, split, embed, and store PDF data in ChromaDB."""
    normalized_mode = normalize_mode(mode)
    registry = load_processed_registry(DATABASE_DIR, normalized_mode)
    saved_files = save_uploaded_files(uploaded_files, UPLOADS_DIR)

    seen_hashes: set[str] = set()
    new_files: list[dict] = []
    duplicate_uploads: list[str] = []
    already_indexed: list[str] = []

    for file_info in saved_files:
        file_hash = file_info["file_hash"]
        original_name = file_info["original_name"]

        if file_hash in seen_hashes:
            duplicate_uploads.append(original_name)
            continue

        seen_hashes.add(file_hash)

        if file_hash in registry:
            already_indexed.append(original_name)
            continue

        new_files.append(file_info)

    report = {
        "added_files": [],
        "already_indexed": already_indexed,
        "duplicate_uploads": duplicate_uploads,
        "failed_files": [],
        "chunk_count": 0,
    }

    if not new_files:
        st.session_state.process_report = report
        return

    documents, processed_files, failed_files = load_documents_from_saved_files(new_files)
    report["failed_files"] = failed_files

    if not documents:
        st.session_state.process_report = report
        return

    chunks = chunk_documents(documents)
    chunk_ids = build_chunk_ids(chunks)

    vectorstore = get_vectorstore(DATABASE_DIR, normalized_mode)
    vectorstore.add_documents(documents=chunks, ids=chunk_ids)

    update_processed_registry(DATABASE_DIR, processed_files, normalized_mode)
    refresh_qa_chain(normalized_mode)

    report["added_files"] = [file_info["original_name"] for file_info in processed_files]
    report["chunk_count"] = len(chunks)
    st.session_state.process_report = report


def render_sidebar() -> str:
    """Sidebar controls and quick project status."""
    current_mode = normalize_mode(st.session_state.ai_mode)
    registry = load_processed_registry(DATABASE_DIR, current_mode)

    with st.sidebar:
        st.header("Project Status")

        selected_mode = st.radio(
            "Answer Engine",
            options=list(MODE_LABELS.keys()),
            index=list(MODE_LABELS.keys()).index(current_mode),
            format_func=lambda mode: MODE_LABELS[mode],
            help="Local mode is free and deploys without OpenAI billing. OpenAI mode is optional.",
        )

        if selected_mode != current_mode:
            st.session_state.ai_mode = selected_mode
            st.session_state.chat_history = []
            st.session_state.process_report = None
            refresh_qa_chain(selected_mode)
            st.rerun()

        st.metric("Indexed PDFs", len(registry))
        st.metric("Chat Turns", len(st.session_state.chat_history))
        st.metric("Database Ready", "Yes" if st.session_state.vectorstore_ready else "No")
        st.metric("Current Engine", "Local" if current_mode == "local" else "OpenAI")

        st.divider()
        st.subheader("Setup")

        if current_mode == "openai":
            st.markdown(
                "1. Add your `OPENAI_API_KEY` to `.env`.\n"
                "2. Make sure your OpenAI API project has billing/credits.\n"
                "3. Upload one or more PDFs.\n"
                "4. Click **Process PDFs**.\n"
                "5. Ask questions in natural language."
            )
            if not has_openai_api_key():
                st.warning("`OPENAI_API_KEY` is missing. Add it to your `.env` file first.")
        else:
            st.markdown(
                "1. Keep **Local (Free)** mode selected.\n"
                "2. Upload one or more PDFs.\n"
                "3. Click **Process PDFs**.\n"
                "4. Ask questions in natural language.\n"
                "5. No OpenAI billing is required."
            )
            st.success("Local free mode is active. This mode does not need an API key.")

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.success("Chat history cleared for this session.")

    return current_mode


def render_source_chunks(sources: list[dict]) -> None:
    """Show the retrieved source chunks under each answer."""
    if not sources:
        st.info("No source chunks were returned for this answer.")
        return

    st.markdown("**Source Chunks**")
    for source in sources:
        safe_citation = escape(source["citation"])
        safe_content = escape(source["content"])
        st.markdown(
            f"""
            <div class="source-card">
                <strong>{safe_citation}</strong><br>
                {safe_content}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_answer_block(answer: str) -> None:
    """Render assistant answers using a dark card for readability."""
    safe_answer = escape(answer).replace("\n", "<br>")
    st.markdown(
        f'<div class="answer-card">{safe_answer}</div>',
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    """Replay the stored chat history on every rerun."""
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(item["question"])

        with st.chat_message("assistant"):
            render_answer_block(item["answer"])
            render_source_chunks(item["sources"])


def main() -> None:
    initialize_app()
    apply_app_styles()
    current_mode = render_sidebar()

    st.title("AskMyPDF AI")
    st.caption(
        "Upload PDFs, build a ChromaDB knowledge base, and ask questions with local or OpenAI-powered retrieval."
    )

    if st.session_state.startup_error:
        st.error(
            "The app could not connect to the local vector database. "
            f"Details: {st.session_state.startup_error}"
        )

    if current_mode == "openai" and not has_openai_api_key():
        st.warning("Add your OpenAI API key to `.env` before processing PDFs or asking questions.")

    upload_col, info_col = st.columns([1.4, 1], gap="large")

    with upload_col:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="status-chip">PDF Upload and Indexing</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload one or multiple PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            help="Only new files are embedded. Previously indexed PDFs are skipped automatically.",
        )

        process_button = st.button("Process PDFs", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if process_button:
            if current_mode == "openai" and not has_openai_api_key():
                st.warning("Please add `OPENAI_API_KEY` to your `.env` file before processing PDFs.")
            elif not uploaded_files:
                st.warning("Please upload at least one PDF file before processing.")
            else:
                spinner_message = (
                    "Reading PDFs, splitting text, and storing embeddings..."
                    if current_mode == "openai"
                    else "Reading PDFs, building local embeddings, and storing them in ChromaDB..."
                )
                with st.spinner(spinner_message):
                    try:
                        process_uploaded_pdfs(uploaded_files, current_mode)
                    except Exception as exc:
                        st.session_state.process_report = None
                        error_title, error_help = explain_runtime_error(current_mode, exc)
                        st.error(error_title)
                        st.info(error_help)

    with info_col:
        st.markdown('<div class="app-card">', unsafe_allow_html=True)
        st.markdown('<div class="status-chip">How It Works</div>', unsafe_allow_html=True)
        if current_mode == "openai":
            st.markdown(
                """
                This app follows an OpenAI-powered Retrieval-Augmented Generation workflow:

                - PDF text is extracted page by page.
                - Text is split into overlapping chunks.
                - OpenAI embeddings are created for each chunk.
                - Chunks are stored in ChromaDB.
                - Relevant chunks are retrieved for each question.
                - GPT-3.5-turbo generates the final answer with citations.
                """
            )
        else:
            st.markdown(
                """
                This app follows a fully local retrieval workflow:

                - PDF text is extracted page by page.
                - Text is split into overlapping chunks.
                - Lightweight local hash embeddings are created for each chunk.
                - Chunks are stored in ChromaDB.
                - Relevant chunks are retrieved for each question.
                - A local extractive answerer builds the reply from the best matching sentences.
                """
            )
        st.markdown("</div>", unsafe_allow_html=True)

    show_process_report()

    st.divider()
    st.subheader("Ask Questions About Your PDFs")

    if not st.session_state.vectorstore_ready:
        st.info("Process at least one PDF to start asking questions.")

    render_chat_history()

    user_question = st.chat_input(
        "Ask something about your uploaded PDFs...",
        disabled=not st.session_state.vectorstore_ready or (current_mode == "openai" and not has_openai_api_key()),
    )

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching the PDFs and generating an answer..."):
                try:
                    response = ask_pdf_question(
                        qa_chain=st.session_state.qa_chain,
                        question=user_question,
                        chat_history=st.session_state.chat_history,
                    )

                    render_answer_block(response["answer"])
                    render_source_chunks(response["sources"])

                    st.session_state.chat_history.append(
                        {
                            "question": user_question,
                            "answer": response["answer"],
                            "sources": response["sources"],
                        }
                    )
                except Exception as exc:
                    error_title, error_help = explain_runtime_error(current_mode, exc)
                    st.error(error_title)
                    st.info(error_help)


if __name__ == "__main__":
    main()
