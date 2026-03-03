"""
GenAI Agentic RAG API - FastAPI application entrypoint.
"""
import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.services.llm_factory import validate_provider

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GenAI Agentic RAG API",
    description="Upload documents, ask questions; agent decides retrieval vs direct LLM.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    """Health check for Docker/load balancers."""
    return {"status": "ok", "service": "agentic-rag"}


@app.on_event("startup")
async def startup():
    """Validate provider config and log active LLM provider."""
    get_settings()
    validate_provider()
