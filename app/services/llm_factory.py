"""
LLM and Embeddings factory based on LLM_PROVIDER environment variable.
Supports: azure (default), openai, ollama.
Only this module branches on provider; agent and ingestion remain provider-agnostic.
"""
import logging
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _validate_azure(settings: Any) -> None:
    """Raise if required Azure env vars are missing. No fallbacks; all read from environment."""
    missing: list[str] = []
    if not (settings.AZURE_OPENAI_API_KEY or "").strip():
        missing.append("AZURE_OPENAI_API_KEY")
    if not (settings.AZURE_OPENAI_ENDPOINT or "").strip():
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not (settings.AZURE_OPENAI_API_VERSION or "").strip():
        missing.append("AZURE_OPENAI_API_VERSION")
    if not (settings.AZURE_OPENAI_DEPLOYMENT_NAME or "").strip():
        missing.append("AZURE_OPENAI_DEPLOYMENT_NAME")
    if not (settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "").strip():
        missing.append("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if missing:
        raise ValueError(
            f"When LLM_PROVIDER=azure the following must be set in .env: {', '.join(missing)}. "
            "Use deployment names from Azure Portal, not model names."
        )


def _validate_openai(settings: Any) -> None:
    """Raise if required OpenAI env vars are missing."""
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY must be set when LLM_PROVIDER=openai. Set it in .env or export it."
        )


def _validate_ollama(settings: Any) -> None:
    """Ollama has no required secrets; base URL is optional."""
    pass


def validate_provider() -> str:
    """
    Validate that required environment variables for the chosen provider are set.
    Returns a short message describing the active provider for logging.
    Raises ValueError with a clear message if validation fails.
    """
    settings = get_settings()
    provider = settings.LLM_PROVIDER

    if provider == "azure":
        _validate_azure(settings)
        logger.info("Using Azure OpenAI provider")
        logger.info("Chat deployment: %s", settings.AZURE_OPENAI_DEPLOYMENT_NAME)
        logger.info("Embedding deployment: %s", settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
        logger.info("Endpoint: %s", (settings.AZURE_OPENAI_ENDPOINT or "").rstrip("/"))
        return f"Azure OpenAI (chat={settings.AZURE_OPENAI_DEPLOYMENT_NAME})"
    if provider == "openai":
        _validate_openai(settings)
        msg = f"Using OpenAI provider with model: {settings.OPENAI_MODEL}"
        logger.info(msg)
        return msg
    if provider == "ollama":
        _validate_ollama(settings)
        msg = (
            f"Using Ollama provider with model: {settings.OLLAMA_MODEL} "
            f"(base_url={settings.OLLAMA_BASE_URL})"
        )
        logger.info(msg)
        return msg
    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider}. Must be one of: azure, openai, ollama."
    )


def get_llm(**kwargs: Any) -> BaseChatModel:
    """
    Return the configured chat LLM based on LLM_PROVIDER.
    Default: Azure OpenAI (deployment-based). temperature=0 from config.
    """
    settings = get_settings()
    provider = settings.LLM_PROVIDER

    if provider == "azure":
        _validate_azure(settings)
        endpoint = (settings.AZURE_OPENAI_ENDPOINT or "").rstrip("/")
        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=(settings.AZURE_OPENAI_API_KEY or "").strip(),
            api_version=(settings.AZURE_OPENAI_API_VERSION or "").strip(),
            azure_deployment=(settings.AZURE_OPENAI_DEPLOYMENT_NAME or "").strip(),
            temperature=settings.AGENT_TEMPERATURE,
            **kwargs,
        )
    if provider == "openai":
        _validate_openai(settings)
        return ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.AGENT_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
            **kwargs,
        )
    if provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        _validate_ollama(settings)
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=settings.AGENT_TEMPERATURE,
            **kwargs,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def get_embeddings(**kwargs: Any) -> Embeddings:
    """
    Return the configured embeddings model based on LLM_PROVIDER.
    azure -> AzureOpenAIEmbeddings, openai -> OpenAIEmbeddings, ollama -> OllamaEmbeddings.
    """
    settings = get_settings()
    provider = settings.LLM_PROVIDER

    if provider == "azure":
        _validate_azure(settings)
        endpoint = (settings.AZURE_OPENAI_ENDPOINT or "").rstrip("/")
        return AzureOpenAIEmbeddings(
            azure_endpoint=endpoint,
            api_key=(settings.AZURE_OPENAI_API_KEY or "").strip(),
            api_version=(settings.AZURE_OPENAI_API_VERSION or "").strip(),
            azure_deployment=(settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT or "").strip(),
            **kwargs,
        )
    if provider == "openai":
        _validate_openai(settings)
        return OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
            **kwargs,
        )
    if provider == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings

        _validate_ollama(settings)
        return OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBEDDING_MODEL,
            **kwargs,
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")
