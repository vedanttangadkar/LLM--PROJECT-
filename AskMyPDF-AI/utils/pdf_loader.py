from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from pypdf import PdfReader


def ensure_project_directories(upload_dir: Path, database_dir: Path) -> None:
    """Create folders used by the Streamlit app."""
    upload_dir.mkdir(parents=True, exist_ok=True)
    database_dir.mkdir(parents=True, exist_ok=True)


def _sanitize_filename(filename: str) -> str:
    """Keep filenames simple and safe for local storage."""
    clean_name = Path(filename).name.strip()
    clean_name = clean_name.replace(" ", "_")
    return clean_name or "uploaded_file.pdf"


def calculate_file_hash(file_bytes: bytes) -> str:
    """Use SHA256 so we can detect duplicate uploads reliably."""
    return hashlib.sha256(file_bytes).hexdigest()


def save_uploaded_files(uploaded_files: Iterable, upload_dir: Path) -> list[dict]:
    """
    Save Streamlit uploaded files to disk.

    We store files locally so they can be reused across app reruns.
    """
    saved_files: list[dict] = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        if not file_bytes:
            continue

        file_hash = calculate_file_hash(file_bytes)
        safe_name = _sanitize_filename(uploaded_file.name)
        saved_name = f"{file_hash[:12]}_{safe_name}"
        saved_path = upload_dir / saved_name

        if not saved_path.exists():
            saved_path.write_bytes(file_bytes)

        saved_files.append(
            {
                "original_name": uploaded_file.name,
                "saved_name": saved_name,
                "saved_path": str(saved_path),
                "file_hash": file_hash,
                "size_bytes": len(file_bytes),
            }
        )

    return saved_files


def extract_documents_from_pdf(file_info: dict) -> list[Document]:
    """Read one PDF and convert each non-empty page into a LangChain document."""
    file_path = Path(file_info["saved_path"])
    reader = PdfReader(str(file_path))

    if not reader.pages:
        raise ValueError("The PDF does not contain any readable pages.")

    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = " ".join(raw_text.split())

        if not cleaned_text:
            continue

        documents.append(
            Document(
                page_content=cleaned_text,
                metadata={
                    "source": file_info["original_name"],
                    "file_id": file_info["file_hash"],
                    "page": page_number,
                    "saved_path": str(file_path),
                },
            )
        )

    if not documents:
        raise ValueError("No extractable text was found in this PDF.")

    return documents


def load_documents_from_saved_files(saved_files: list[dict]) -> tuple[list[Document], list[dict], list[dict]]:
    """
    Load documents from many PDFs.

    Returns:
    - all extracted documents
    - successfully processed file records
    - failures with error messages
    """
    all_documents: list[Document] = []
    processed_files: list[dict] = []
    failed_files: list[dict] = []

    for file_info in saved_files:
        try:
            documents = extract_documents_from_pdf(file_info)
            all_documents.extend(documents)

            processed_files.append(
                {
                    **file_info,
                    "page_count": len(documents),
                }
            )
        except Exception as exc:
            failed_files.append(
                {
                    "name": file_info["original_name"],
                    "error": str(exc),
                }
            )

    return all_documents, processed_files, failed_files
