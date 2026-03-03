"""
Tool-calling agent (OpenAI / Azure): document_search, direct_llm, calculator.
Adds query classification and hybrid handling.
Unified response: (answer, sources, reasoning_trace, retrieval_used, confidence).
"""
import logging
import re
from time import perf_counter
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool

from app.core.config import get_settings
from app.models.schemas import ReasoningStep, SourceItem
from app.services.llm_factory import get_llm
from app.services.vector_store import similarity_search

logger = logging.getLogger(__name__)

QueryType = Literal["GENERAL", "DOCUMENT", "HYBRID"]


# --- Tool implementations ---


def _document_search_impl(query: str, request_id: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Query vector store; return (observation_string, sources_for_response)."""
    try:
        results = similarity_search(query, k=get_settings().TOP_K_RETRIEVAL)
    except Exception as exc:  # defensive: handle vector store issues gracefully
        logger.exception("document_search error (request_id=%s): %s", request_id, exc)
        return "Error while searching documents.", []

    sources: list[dict[str, Any]] = []
    parts: list[str] = []
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata or {}
        score_val = float(score) if score is not None else None
        item = {
            "document": meta.get("document_name", "unknown"),
            "page": meta.get("page_number"),
            "chunk": doc.page_content[:500],
            "score": score_val,
        }
        sources.append(item)
        parts.append(
            f"[{i}] (doc: {item['document']}, page: {item['page']}) {doc.page_content[:400]}..."
        )
    if not sources:
        logger.info("document_search returned no results (request_id=%s)", request_id)
        return "No relevant documents found.", []

    logger.info(
        "document_search used %d results (request_id=%s, scores=%s)",
        len(sources),
        request_id,
        [s["score"] for s in sources],
    )
    observation = "\n\n".join(parts)
    return observation, sources


def _direct_llm_impl(llm: BaseChatModel, question: str, request_id: str | None = None) -> str:
    """Answer from LLM internal knowledge only."""
    try:
        msg = HumanMessage(content=question)
        resp = llm.invoke([msg])
        content = resp.content if hasattr(resp, "content") else str(resp)
        logger.info("direct_llm completed (request_id=%s)", request_id)
        return content
    except Exception as exc:
        logger.exception("direct_llm error (request_id=%s): %s", request_id, exc)
        return "I encountered an error while answering your question."


def _calculator_impl(expression: str, request_id: str | None = None) -> str:
    """Safely evaluate a single math expression."""
    expression = expression.strip()
    if not re.match(r"^[\d\s+\-*/.()]+$", expression):
        return "Error: Only numbers and + - * / ( ) allowed."
    try:
        result = str(eval(expression))
        logger.info("calculator used (request_id=%s, expression=%s)", request_id, expression)
        return result
    except Exception as exc:
        logger.exception("calculator error (request_id=%s): %s", request_id, exc)
        return f"Error: {exc}"


# --- Query classification ---

CLASSIFIER_SYSTEM = """You are a routing classifier for a RAG system.
Classify the user question into exactly one of:
- GENERAL: can be answered from model knowledge alone.
- DOCUMENT: should be answered from uploaded/internal documents only.
- HYBRID: should use both documents and model knowledge.

Answer with a single token: GENERAL, DOCUMENT, or HYBRID."""


def _classify_query(llm: BaseChatModel, question: str, request_id: str | None = None) -> QueryType:
    """Classify query as GENERAL, DOCUMENT, or HYBRID using the LLM."""
    try:
        messages = [
            SystemMessage(content=CLASSIFIER_SYSTEM),
            HumanMessage(content=question),
        ]
        resp = llm.invoke(messages)
        raw = (resp.content if hasattr(resp, "content") else str(resp)).strip().upper()
        if "DOCUMENT" in raw and "HYBRID" in raw:
            label: QueryType = "HYBRID"
        elif "HYBRID" in raw:
            label = "HYBRID"
        elif "DOCUMENT" in raw:
            label = "DOCUMENT"
        elif "GENERAL" in raw:
            label = "GENERAL"
        else:
            label = "GENERAL"
        logger.info("Query classified (request_id=%s): %s -> %s", request_id, question, label)
        return label
    except Exception as exc:
        logger.exception("Query classification error (request_id=%s): %s", request_id, exc)
        return "GENERAL"


# --- Tool-calling agent ---

TOOL_CALLING_SYSTEM = """You are a helpful assistant with access to tools.

Use document_search when the user asks about uploaded documents, internal reports, or files they have provided.
Use direct_llm for general knowledge or when no document retrieval is needed.
Use calculator for math or arithmetic.

Respond to the user clearly. If you used a tool, summarize the result in your final answer."""


def _build_tools_for_calling(
    llm: BaseChatModel,
    sources_collector: list[dict[str, Any]],
    retrieval_used_flag: list[bool],
    request_id: str | None,
) -> list[StructuredTool]:
    """Build LangChain tools that collect sources and set retrieval_used when document_search is called."""

    def document_search(query: str) -> str:
        obs, sources = _document_search_impl(query, request_id=request_id)
        sources_collector.extend(sources)
        if sources:
            retrieval_used_flag[0] = True
        return obs

    def direct_llm(question: str) -> str:
        return _direct_llm_impl(llm, question, request_id=request_id)

    def calculator(expression: str) -> str:
        return _calculator_impl(expression, request_id=request_id)

    doc_tool = StructuredTool.from_function(
        func=document_search,
        name="document_search",
        description="Search uploaded documents for relevant passages. Use when the user asks about uploaded files, internal reports, or content they have provided. Input: search query string.",
    )
    direct_tool = StructuredTool.from_function(
        func=direct_llm,
        name="direct_llm",
        description="Answer from general knowledge without searching documents. Use for facts, definitions, or when no document retrieval is needed. Input: the user's question or a rephrased question.",
    )
    calc_tool = StructuredTool.from_function(
        func=calculator,
        name="calculator",
        description="Evaluate a single mathematical expression. Use for arithmetic or math. Input: expression with numbers and + - * / ( ) e.g. '2 + 3' or '100 * 0.15'.",
    )
    return [doc_tool, direct_tool, calc_tool]


def _intermediate_steps_to_reasoning_trace(
    intermediate_steps: list[tuple[Any, str]],
) -> list[ReasoningStep]:
    """Convert AgentExecutor intermediate_steps (AgentAction, observation) to ReasoningStep list."""
    from langchain_core.agents import AgentAction

    trace: list[ReasoningStep] = []
    for step_num, (action, observation) in enumerate(intermediate_steps, 1):
        if isinstance(action, AgentAction):
            trace.append(
                ReasoningStep(
                    step=step_num,
                    action=action.tool,
                    action_input=action.tool_input if isinstance(action.tool_input, str) else str(action.tool_input),
                    observation=(observation[:500] if observation else None),
                )
            )
        else:
            trace.append(ReasoningStep(step=step_num, observation=observation[:500] if observation else None))
    return trace


def _estimate_confidence(
    retrieval_used: bool,
    sources: list[SourceItem],
    query_type: QueryType,
) -> float:
    """Simple heuristic confidence score in [0, 1]."""
    if not sources and retrieval_used:
        # Retrieval attempted but no sources; lower confidence.
        base = 0.4
    elif retrieval_used and sources:
        base = 0.8
    else:
        base = 0.7

    if query_type == "HYBRID":
        base += 0.05
    elif query_type == "DOCUMENT" and not retrieval_used:
        base -= 0.1

    return max(0.0, min(1.0, base))


def _run_tool_calling_agent(
    question: str,
    chat_history: list[Any] | None,
    llm: BaseChatModel,
    request_id: str | None,
    query_type: QueryType,
) -> tuple[str, list[SourceItem], list[ReasoningStep], bool, float]:
    """
    Native tool-calling agent (create_tool_calling_agent + AgentExecutor).
    Returns (answer, sources, reasoning_trace, retrieval_used, confidence).
    """
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

    logger.info("Using tool-calling agent (request_id=%s, query_type=%s)", request_id, query_type)
    settings = get_settings()
    sources_collector: list[dict[str, Any]] = []
    retrieval_used_flag: list[bool] = [False]

    tools = _build_tools_for_calling(llm, sources_collector, retrieval_used_flag, request_id=request_id)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", TOOL_CALLING_SYSTEM),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=settings.AGENT_MAX_ITERATIONS,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

    inputs: dict[str, Any] = {"input": question}
    if chat_history:
        inputs["chat_history"] = chat_history

    result = agent_executor.invoke(inputs)
    output = result.get("output", "")
    intermediate_steps = result.get("intermediate_steps", [])

    reasoning_trace = _intermediate_steps_to_reasoning_trace(intermediate_steps)
    if output and reasoning_trace:
        reasoning_trace.append(ReasoningStep(step=len(reasoning_trace) + 1, conclusion=output[:500]))
    elif output and not reasoning_trace:
        reasoning_trace = [ReasoningStep(step=1, conclusion=output[:500])]

    answer = output.strip() if output else "No response generated."
    source_items = [SourceItem(**s) for s in sources_collector]
    retrieval_used = retrieval_used_flag[0]
    confidence = _estimate_confidence(retrieval_used, source_items, query_type=query_type)
    logger.info(
        "Agent completed (request_id=%s, retrieval_used=%s, confidence=%.2f)",
        request_id,
        retrieval_used,
        confidence,
    )
    return answer, source_items, reasoning_trace, retrieval_used, confidence


def _run_hybrid_agent(
    question: str,
    chat_history: list[Any] | None,
    llm: BaseChatModel,
    request_id: str | None,
) -> tuple[str, list[SourceItem], list[ReasoningStep], bool, float]:
    """
    HYBRID path: retrieve top-k chunks, then let the LLM answer using both document context and internal knowledge.
    Returns (answer, sources, reasoning_trace, retrieval_used, confidence).
    """
    logger.info("Using hybrid path (request_id=%s)", request_id)
    observation, raw_sources = _document_search_impl(question, request_id=request_id)
    if not raw_sources:
        # Fallback: no retrieval available; use normal tool-calling agent.
        return _run_tool_calling_agent(
            question=question,
            chat_history=chat_history,
            llm=llm,
            request_id=request_id,
            query_type="HYBRID",
        )

    # Build combined prompt: context + question.
    context_snippets = "\n\n".join([s["chunk"] for s in raw_sources][:3])
    user_content = (
        "You have access to both retrieved document excerpts and your general knowledge.\n"
        "Use the documents as primary evidence, and fill gaps with model knowledge when needed.\n\n"
        f"User question:\n{question}\n\n"
        f"Retrieved context:\n{context_snippets}\n"
    )

    try:
        resp = llm.invoke([HumanMessage(content=user_content)])
        output = resp.content if hasattr(resp, "content") else str(resp)
    except Exception as exc:
        logger.exception("Hybrid LLM error (request_id=%s): %s", request_id, exc)
        output = "I encountered an error while combining document context and model knowledge."

    answer = output.strip() if output else "No response generated."
    sources = [SourceItem(**s) for s in raw_sources]

    reasoning_trace: list[ReasoningStep] = [
        ReasoningStep(
            step=1,
            thought="Query classified as HYBRID; combining retrieval with model knowledge.",
            action="classify",
            action_input=question,
        ),
        ReasoningStep(
            step=2,
            action="document_search",
            action_input=question,
            observation=observation[:500] if observation else None,
        ),
        ReasoningStep(
            step=3,
            action="direct_llm",
            action_input="question + retrieved context",
            observation=answer[:500],
            conclusion=answer[:500],
        ),
    ]

    retrieval_used = True
    confidence = _estimate_confidence(retrieval_used, sources, query_type="HYBRID")
    logger.info(
        "Hybrid path completed (request_id=%s, retrieval_used=%s, confidence=%.2f)",
        request_id,
        retrieval_used,
        confidence,
    )
    return answer, sources, reasoning_trace, retrieval_used, confidence


def run_react_agent(
    question: str,
    chat_history: list[Any] | None = None,
    request_id: str | None = None,
) -> tuple[str, list[SourceItem], list[ReasoningStep], bool, float]:
    """
    Public entrypoint: classify query and run appropriate agent path.
    Returns (answer, sources, reasoning_trace, retrieval_used, confidence).
    """
    start = perf_counter()
    settings = get_settings()
    llm = get_llm()

    query_type = _classify_query(llm, question, request_id=request_id)

    if query_type == "HYBRID":
        answer, sources, reasoning_trace, retrieval_used, confidence = _run_hybrid_agent(
            question=question,
            chat_history=chat_history,
            llm=llm,
            request_id=request_id,
        )
    else:
        answer, sources, reasoning_trace, retrieval_used, confidence = _run_tool_calling_agent(
            question=question,
            chat_history=chat_history,
            llm=llm,
            request_id=request_id,
            query_type=query_type,
        )

    duration_ms = (perf_counter() - start) * 1000.0
    logger.info(
        "run_react_agent completed (request_id=%s, duration_ms=%.1f, query_type=%s, retrieval_used=%s)",
        request_id,
        duration_ms,
        query_type,
        retrieval_used,
    )
    return answer, sources, reasoning_trace, retrieval_used, confidence
