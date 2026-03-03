"""
Pydantic request/response models for API.
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Upload ---
class UploadResponse(BaseModel):
    """Response after successful document upload."""

    document_name: str
    chunks_created: int
    message: str = "Document ingested successfully."


# --- Query ---
class QueryRequest(BaseModel):
    """Request body for /query endpoint."""

    question: str = Field(..., min_length=1, description="User question")


class SourceItem(BaseModel):
    """A single source citation."""

    document: str
    page: int | None = None
    chunk: str
    score: float | None = None


class ReasoningStep(BaseModel):
    """One step in the agent reasoning trace."""

    step: int
    thought: str | None = None
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None
    conclusion: str | None = None


class QueryResponse(BaseModel):
    """Response from /query with answer, sources, and reasoning."""

    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    retrieval_used: bool = False
    confidence: float | None = None


# --- Chat ---
class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""

    session_id: str = Field(..., min_length=1, description="Conversation session ID")
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    """Response from /chat (same structure as QueryResponse)."""

    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    reasoning_trace: list[ReasoningStep] = Field(default_factory=list)
    retrieval_used: bool = False
    confidence: float | None = None


# --- Documents ---
class DocumentMeta(BaseModel):
    """Metadata for one uploaded document."""

    document_name: str
    page_count: int | None = None
    chunk_count: int
    upload_timestamp: datetime | None = None


class DocumentsResponse(BaseModel):
    """Response for GET /documents."""

    documents: list[DocumentMeta]
    total: int


# --- Clear ---
class ClearResponse(BaseModel):
    """Response after clearing vector store."""

    message: str = "Vector store and sessions cleared."

