"""
FastAPI route handlers for upload, query, chat, documents, clear.
Adds basic observability (request IDs, timings, structured logs).
"""
import asyncio
import logging
from datetime import datetime
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ClearResponse,
    DocumentMeta,
    DocumentsResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.services.agent import run_react_agent
from app.services.ingestion import ingest_file
from app.services.memory import (
    append_ai_message,
    append_user_message,
    clear_all_sessions,
    get_session_messages,
)
from langchain_core.messages import HumanMessage

from app.core.config import get_settings
from app.services.llm_factory import get_llm
from app.services.vector_store import clear_collection, list_document_metadata

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["rag"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}
AZURE_AUTH_ERROR_MSG = (
    "Azure authentication failed. Check endpoint, key, and deployment name."
)


def _is_azure_auth_error(exc: Exception) -> bool:
    """True if the exception indicates Azure auth failure (e.g. 401)."""
    msg = str(exc).lower()
    return (
        "401" in str(exc)
        or "authentication" in msg
        or "access denied" in msg
        or "invalid subscription key" in msg
    )


def _test_azure_connectivity() -> tuple[str, str]:
    """Call Azure chat with a simple message. Returns (response_text, deployment_name). Sync for asyncio.to_thread."""
    settings = get_settings()
    deployment = (settings.AZURE_OPENAI_DEPLOYMENT_NAME or "").strip()
    llm = get_llm()
    response = llm.invoke([HumanMessage(content="Respond with OK")])
    text = response.content if hasattr(response, "content") else str(response)
    return (text.strip(), deployment)


@router.get("/test-azure")
async def test_azure():
    """
    Validate Azure OpenAI connectivity. Calls the chat deployment with a simple prompt.
    Returns 500 with a clear message if authentication fails.
    """
    settings = get_settings()
    if settings.LLM_PROVIDER != "azure":
        raise HTTPException(
            status_code=400,
            detail="Azure is not the configured provider. Set LLM_PROVIDER=azure and required Azure env vars.",
        )
    try:
        response_text, deployment = await asyncio.to_thread(_test_azure_connectivity)
    except Exception as e:
        if _is_azure_auth_error(e):
            logger.warning("Azure connectivity test failed (auth): %s", e)
            raise HTTPException(status_code=500, detail=AZURE_AUTH_ERROR_MSG)
        logger.exception("Azure connectivity test failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "status": "success",
        "response": response_text,
        "deployment": deployment,
        "provider": "azure",
    }


MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


def _allowed_file(filename: str) -> bool:
    """Return True if file has an allowed extension."""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept PDF, DOCX, or TXT. Extract text, semantic chunk, add metadata,
    generate embeddings, store in ChromaDB.
    """
    request_id = uuid4().hex
    if not file.filename or not _allowed_file(file.filename):
        logger.warning(
            "Rejected upload (request_id=%s, filename=%s, reason=invalid_extension)",
            request_id,
            file.filename,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        logger.warning(
            "Rejected upload (request_id=%s, filename=%s, reason=file_too_large,size=%d)",
            request_id,
            file.filename,
            len(content),
        )
        raise HTTPException(status_code=400, detail="File too large (max 20MB).")
    if len(content) == 0:
        logger.warning(
            "Rejected upload (request_id=%s, filename=%s, reason=empty_file)",
            request_id,
            file.filename,
        )
        raise HTTPException(status_code=400, detail="Empty file.")
    try:
        chunks_created = await ingest_file(content, file.filename or "document")
    except ValueError as e:
        logger.warning(
            "Upload failed (request_id=%s, filename=%s, error=%s)", request_id, file.filename, e
        )
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.exception(
            "Upload failed (request_id=%s, filename=%s, error=%s)", request_id, file.filename, e
        )
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        "Upload succeeded (request_id=%s, filename=%s, chunks_created=%d)",
        request_id,
        file.filename,
        chunks_created,
    )
    return UploadResponse(
        document_name=file.filename or "document",
        chunks_created=chunks_created,
    )


@router.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """
    Ask a question. Agent classifies query and decides document_search vs direct_llm vs hybrid.
    Returns answer, sources, reasoning_trace, retrieval_used, and confidence.
    """
    request_id = uuid4().hex
    start = perf_counter()
    try:
        answer, sources, reasoning_trace, retrieval_used, confidence = await asyncio.to_thread(
            run_react_agent,
            req.question,
            None,
            request_id,
        )
    except Exception as e:
        logger.exception("Query failed (request_id=%s, question=%s)", request_id, req.question)
        detail = AZURE_AUTH_ERROR_MSG if get_settings().LLM_PROVIDER == "azure" and _is_azure_auth_error(e) else str(e)
        raise HTTPException(status_code=500, detail=detail)
    duration_ms = (perf_counter() - start) * 1000.0
    logger.info(
        "Query completed (request_id=%s, duration_ms=%.1f, retrieval_used=%s, confidence=%.2f)",
        request_id,
        duration_ms,
        retrieval_used,
        confidence or -1.0,
    )
    return QueryResponse(
        answer=answer,
        sources=sources,
        reasoning_trace=reasoning_trace,
        retrieval_used=retrieval_used,
        confidence=confidence,
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """
    Multi-turn conversation with session_id. Maintains conversation memory.
    """
    request_id = uuid4().hex
    append_user_message(req.session_id, req.message)
    history = get_session_messages(req.session_id)
    start = perf_counter()
    try:
        answer, sources, reasoning_trace, retrieval_used, confidence = await asyncio.to_thread(
            run_react_agent,
            req.message,
            history[:-1],  # exclude current so agent sees context only
            request_id,
        )
    except Exception as e:
        logger.exception(
            "Chat failed (request_id=%s, session_id=%s, error=%s)",
            request_id,
            req.session_id,
            e,
        )
        append_ai_message(req.session_id, f"Error: {e}")
        detail = AZURE_AUTH_ERROR_MSG if get_settings().LLM_PROVIDER == "azure" and _is_azure_auth_error(e) else str(e)
        raise HTTPException(status_code=500, detail=detail)
    duration_ms = (perf_counter() - start) * 1000.0
    logger.info(
        "Chat completed (request_id=%s, session_id=%s, duration_ms=%.1f, retrieval_used=%s, confidence=%.2f)",
        request_id,
        req.session_id,
        duration_ms,
        retrieval_used,
        confidence or -1.0,
    )
    append_ai_message(req.session_id, answer)
    return ChatResponse(
        answer=answer,
        sources=sources,
        reasoning_trace=reasoning_trace,
        retrieval_used=retrieval_used,
        confidence=confidence,
    )


@router.get("/documents", response_model=DocumentsResponse)
async def documents() -> DocumentsResponse:
    """List all uploaded documents with metadata (name, chunk count, etc.)."""
    meta_list = list_document_metadata()
    docs = []
    for m in meta_list:
        ts = m.get("upload_timestamp")
        if ts is not None and isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except ValueError:
                ts = None
        elif ts is not None and not isinstance(ts, datetime):
            ts = None
        docs.append(
            DocumentMeta(
                document_name=m["document_name"],
                page_count=m.get("page_count"),
                chunk_count=m["chunk_count"],
                upload_timestamp=ts,
            )
        )
    return DocumentsResponse(documents=docs, total=len(docs))


@router.delete("/clear", response_model=ClearResponse)
async def clear() -> ClearResponse:
    """Clear vector DB and all session memories."""
    clear_collection()
    clear_all_sessions()
    return ClearResponse()
