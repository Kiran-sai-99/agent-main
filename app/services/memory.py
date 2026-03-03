"""
In-memory conversation history per session for /chat.
Production systems would use Redis or a database.
"""
import logging
from collections import defaultdict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

logger = logging.getLogger(__name__)

# session_id -> list of messages (HumanMessage, AIMessage)
_session_messages: dict[str, list[BaseMessage]] = defaultdict(list)
# Max messages to keep per session (trim oldest)
_MAX_MESSAGES_PER_SESSION = 50


def get_session_messages(session_id: str) -> list[BaseMessage]:
    """Return conversation history for session (copy)."""
    return list(_session_messages.get(session_id, []))


def append_user_message(session_id: str, content: str) -> None:
    """Add a user message to the session."""
    _session_messages[session_id].append(HumanMessage(content=content))
    _trim_session(session_id)


def append_ai_message(session_id: str, content: str) -> None:
    """Add an AI message to the session."""
    _session_messages[session_id].append(AIMessage(content=content))
    _trim_session(session_id)


def _trim_session(session_id: str) -> None:
    """Keep only the last N messages."""
    if session_id not in _session_messages:
        return
    msgs = _session_messages[session_id]
    if len(msgs) > _MAX_MESSAGES_PER_SESSION:
        _session_messages[session_id] = msgs[-_MAX_MESSAGES_PER_SESSION:]


def clear_all_sessions() -> None:
    """Clear all session histories (e.g. on /clear)."""
    _session_messages.clear()
    logger.info("All session memories cleared.")
