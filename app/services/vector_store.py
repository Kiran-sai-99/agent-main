"""
ChromaDB vector store service for document embeddings.
"""
import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from app.core.config import get_settings
from app.services.llm_factory import get_embeddings

logger = logging.getLogger(__name__)

# Module-level store to reuse across requests (lazy init)
_vector_store: VectorStore | None = None
_embeddings: Embeddings | None = None


def _get_embeddings() -> Embeddings:
    """Return (and cache) the embeddings instance used by the vector store."""
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings


def get_vector_store() -> VectorStore:
    """Return the Chroma vector store (singleton, persistent on disk)."""
    global _vector_store
    if _vector_store is None:
        settings = get_settings()
        client = chromadb.PersistentClient(
            path=settings.CHROMA_PERSIST_DIR,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        _vector_store = Chroma(
            client=client,
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=_get_embeddings(),
        )
        logger.info("Chroma vector store initialized at %s", settings.CHROMA_PERSIST_DIR)
    return _vector_store


def add_documents(docs: list[Document]) -> list[str]:
    """Add documents to the vector store. Returns list of IDs."""
    store = get_vector_store()
    ids = store.add_documents(docs)
    logger.info("Added %d documents to vector store", len(ids))
    return ids


def similarity_search(
    query: str,
    k: int | None = None,
    filter: dict[str, Any] | None = None,
) -> list[tuple[Document, float]]:
    """
    Search for similar documents. Returns list of (Document, score).
    Default k from config (TOP_K_RETRIEVAL).
    """
    settings = get_settings()
    store = get_vector_store()
    top_k = k or settings.TOP_K_RETRIEVAL
    # Chroma's similarity_search_with_score returns (Doc, score)
    results = store.similarity_search_with_score(query, k=top_k, filter=filter)
    return results


def clear_collection() -> None:
    """Delete all data in the Chroma collection (for /clear)."""
    global _vector_store
    settings = get_settings()
    client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        client.delete_collection(settings.CHROMA_COLLECTION_NAME)
    except Exception as e:
        logger.warning("Delete collection: %s", e)
    # Recreate empty collection on next get_vector_store
    _vector_store = None
    logger.info("Vector store cleared.")


def list_document_metadata() -> list[dict[str, Any]]:
    """
    List unique documents and chunk counts from Chroma metadata.
    Requires reading the collection; returns list of {document_name, chunk_count, ...}.
    """
    settings = get_settings()
    client = chromadb.PersistentClient(
        path=settings.CHROMA_PERSIST_DIR,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    try:
        coll = client.get_or_create_collection(
            settings.CHROMA_COLLECTION_NAME,
            metadata={"description": "Document chunks"},
        )
    except Exception:
        return []
    data = coll.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    # Aggregate by document_name
    by_doc: dict[str, dict[str, Any]] = {}
    for m in metadatas:
        if not m:
            continue
        name = m.get("document_name") or "unknown"
        if name not in by_doc:
            by_doc[name] = {
                "document_name": name,
                "chunk_count": 0,
                "page_count": None,
                "upload_timestamp": m.get("upload_timestamp"),
            }
        by_doc[name]["chunk_count"] += 1
    return list(by_doc.values())
