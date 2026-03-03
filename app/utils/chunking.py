"""
Semantic chunking utilities for document text.
Splits by paragraphs/sentences when possible, with size and overlap constraints.
"""
import re
from dataclasses import dataclass
from datetime import datetime

from app.core.config import get_settings


@dataclass
class TextChunk:
    """A single chunk with content and metadata."""

    content: str
    document_name: str
    page_number: int | None
    upload_timestamp: datetime
    chunk_index: int


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences (simple heuristic)."""
    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs (double newline or more)."""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def semantic_chunk(
    text: str,
    document_name: str,
    page_number: int | None = None,
    upload_timestamp: datetime | None = None,
) -> list[TextChunk]:
    """
    Chunk text semantically: prefer paragraph boundaries, then sentence boundaries,
    with chunk_size and chunk_overlap from config.
    """
    settings = get_settings()
    chunk_size = settings.CHUNK_SIZE
    chunk_overlap = settings.CHUNK_OVERLAP
    ts = upload_timestamp or datetime.utcnow()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    if not text:
        return []

    # First try paragraph-level chunks
    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        paragraphs = _split_into_sentences(text)
    if not paragraphs:
        paragraphs = [text] if text else []

    chunks: list[TextChunk] = []
    current: list[str] = []
    current_len = 0
    chunk_index = 0

    for i, block in enumerate(paragraphs):
        block_len = len(block) + (1 if current else 0)
        if current_len + block_len <= chunk_size and current:
            current.append(block)
            current_len += block_len
        else:
            if current:
                content = " ".join(current)
                chunks.append(
                    TextChunk(
                        content=content,
                        document_name=document_name,
                        page_number=page_number,
                        upload_timestamp=ts,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1
                # Overlap: keep last few segments
                overlap_len = 0
                overlap_blocks: list[str] = []
                for j in range(len(current) - 1, -1, -1):
                    if overlap_len + len(current[j]) + 1 <= chunk_overlap:
                        overlap_blocks.insert(0, current[j])
                        overlap_len += len(current[j]) + 1
                    else:
                        break
                current = overlap_blocks
                current_len = overlap_len
            current = [block]
            current_len = len(block)

    if current:
        content = " ".join(current)
        chunks.append(
            TextChunk(
                content=content,
                document_name=document_name,
                page_number=page_number,
                upload_timestamp=ts,
                chunk_index=chunk_index,
            )
        )

    return chunks
