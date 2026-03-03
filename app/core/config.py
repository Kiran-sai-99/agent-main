"""
Application configuration loaded from environment variables.
"""
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation."""

    # LLM Provider: azure (default) | openai | ollama
    LLM_PROVIDER: Literal["azure", "openai", "ollama"] = "azure"

    # Azure OpenAI (default). All required when LLM_PROVIDER=azure; no hardcoded deployment defaults.
    AZURE_OPENAI_API_KEY: str | None = None
    AZURE_OPENAI_ENDPOINT: str | None = None
    AZURE_OPENAI_API_VERSION: str | None = None
    AZURE_OPENAI_DEPLOYMENT_NAME: str | None = None
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str | None = None

    # OpenAI (optional)
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Ollama (optional)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_data"
    CHROMA_COLLECTION_NAME: str = "documents"

    # RAG
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_RETRIEVAL: int = 3

    # Agent
    AGENT_MAX_ITERATIONS: int = 10
    AGENT_TEMPERATURE: float = 0.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance for dependency injection."""
    return Settings()
