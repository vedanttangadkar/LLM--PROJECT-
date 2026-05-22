from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


MODE_CONFIG = {
    "local": {
        "collection_name": "askmypdf_ai_local",
        "manifest_file": "processed_files_local.json",
    },
    "openai": {
        "collection_name": "askmypdf_ai_openai",
        "manifest_file": "processed_files_openai.json",
    },
}
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MODE = "local"


class LocalHashEmbeddings(Embeddings):
    """Lightweight local embeddings using a hashing trick."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions
        self.token_pattern = re.compile(r"[a-zA-Z0-9]{2,}")

    def _tokenize(self, text: str) -> list[str]:
        return self.token_pattern.findall(text.lower())

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = self._tokenize(text)

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            first_index = int.from_bytes(digest[:4], "big") % self.dimensions
            second_index = int.from_bytes(digest[4:8], "big") % self.dimensions
            first_sign = 1.0 if digest[8] % 2 == 0 else -1.0
            second_sign = 1.0 if digest[9] % 2 == 0 else -1.0
            weight = 1.0 + (min(len(token), 12) / 12.0)

            vector[first_index] += first_sign * weight
            vector[second_index] += second_sign * weight * 0.5

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)


def normalize_mode(mode: str | None) -> str:
    """Return a valid backend mode."""
    if mode in MODE_CONFIG:
        return mode
    return DEFAULT_MODE


def get_embeddings_model(mode: str = DEFAULT_MODE) -> Embeddings:
    """Create the embedding model used by ChromaDB."""
    normalized_mode = normalize_mode(mode)
    if normalized_mode == "openai":
        return OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return LocalHashEmbeddings()


def get_collection_name(mode: str = DEFAULT_MODE) -> str:
    """Get the Chroma collection name for the chosen mode."""
    return MODE_CONFIG[normalize_mode(mode)]["collection_name"]


def get_manifest_file(mode: str = DEFAULT_MODE) -> str:
    """Get the manifest filename for the chosen mode."""
    return MODE_CONFIG[normalize_mode(mode)]["manifest_file"]


def get_mode_label(mode: str = DEFAULT_MODE) -> str:
    """Human-readable label for the selected mode."""
    if normalize_mode(mode) == "openai":
        return "OpenAI API"
    return "Local Free Mode"


def get_text_splitter(
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> RecursiveCharacterTextSplitter:
    """Use overlapping chunks so retrieval keeps enough context."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split PDF pages into smaller chunks."""
    splitter = get_text_splitter()
    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks, start=1):
        chunk.metadata["global_chunk_number"] = index

    return chunks


def build_chunk_ids(chunks: list[Document]) -> list[str]:
    """Build stable chunk IDs for Chroma."""
    counters: dict[tuple[str, int], int] = {}
    chunk_ids: list[str] = []

    for chunk in chunks:
        file_id = chunk.metadata.get("file_id", "unknown-file")
        page = int(chunk.metadata.get("page", 0))
        counter_key = (file_id, page)
        counters[counter_key] = counters.get(counter_key, 0) + 1
        chunk_number = counters[counter_key]

        chunk.metadata["chunk_number"] = chunk_number
        chunk_ids.append(f"{file_id}-page-{page}-chunk-{chunk_number}")

    return chunk_ids


def get_vectorstore(persist_directory: str | Path, mode: str = DEFAULT_MODE) -> Chroma:
    """Open the persisted Chroma collection."""
    normalized_mode = normalize_mode(mode)
    return Chroma(
        collection_name=get_collection_name(normalized_mode),
        persist_directory=str(persist_directory),
        embedding_function=get_embeddings_model(normalized_mode),
    )


def _manifest_path(database_dir: str | Path, mode: str = DEFAULT_MODE) -> Path:
    return Path(database_dir) / get_manifest_file(mode)


def load_processed_registry(database_dir: str | Path, mode: str = DEFAULT_MODE) -> dict:
    """Read the registry of already processed PDFs."""
    manifest_path = _manifest_path(database_dir, mode)

    if not manifest_path.exists():
        return {}

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_processed_registry(database_dir: str | Path, registry: dict, mode: str = DEFAULT_MODE) -> None:
    """Persist the registry of processed PDFs."""
    manifest_path = _manifest_path(database_dir, mode)
    manifest_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def update_processed_registry(
    database_dir: str | Path,
    processed_files: list[dict],
    mode: str = DEFAULT_MODE,
) -> None:
    """Add newly indexed files to the local manifest."""
    registry = load_processed_registry(database_dir, mode)
    timestamp = datetime.utcnow().isoformat() + "Z"

    for file_info in processed_files:
        registry[file_info["file_hash"]] = {
            "original_name": file_info["original_name"],
            "saved_name": file_info["saved_name"],
            "saved_path": file_info["saved_path"],
            "size_bytes": file_info["size_bytes"],
            "page_count": file_info["page_count"],
            "processed_at": timestamp,
        }

    save_processed_registry(database_dir, registry, mode)
