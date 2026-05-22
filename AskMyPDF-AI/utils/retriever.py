from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma

from utils.embeddings import DEFAULT_MODE, get_vectorstore


def vectorstore_has_documents(vectorstore: Chroma) -> bool:
    """Check whether the Chroma collection already contains indexed chunks."""
    try:
        result = vectorstore.get(limit=1)
        return len(result.get("ids", [])) > 0
    except Exception:
        return False


def get_retriever(
    persist_directory: str | Path,
    mode: str = DEFAULT_MODE,
    k: int = 4,
):
    """Create the retriever used by the QA chains."""
    vectorstore = get_vectorstore(persist_directory, mode)
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
