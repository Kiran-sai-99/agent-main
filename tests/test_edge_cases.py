"""
Edge cases: empty vector store, missing/wrong config (mocked), tool failure (mocked).
"""
import io
import pytest


def test_query_empty_vector_store(client):
    """Query when no documents uploaded: no crash; retrieval_used false or empty sources."""
    client.delete("/api/clear")
    response = client.post(
        "/api/query",
        json={"question": "What documents do we have about revenue?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    # Either no retrieval or empty sources
    assert data.get("retrieval_used") is False or len(data.get("sources", [])) == 0


def test_query_empty_question_rejected(client):
    """POST /api/query with empty question -> 422 or 400."""
    response = client.post("/api/query", json={"question": ""})
    assert response.status_code in (400, 422)


def test_chat_empty_message_rejected(client):
    """POST /api/chat with empty message -> 422 or 400."""
    response = client.post("/api/chat", json={"session_id": "s1", "message": ""})
    assert response.status_code in (400, 422)


def test_chat_missing_session_id_rejected(client):
    """POST /api/chat without session_id -> 422."""
    response = client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 422


def test_documents_empty_after_clear(client):
    """GET /api/documents after clear returns empty list."""
    client.delete("/api/clear")
    response = client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert data.get("total", 0) == 0
    assert data.get("documents", []) == []


def test_azure_auth_error_message_mock(client, monkeypatch):
    """Mock Azure auth failure; API returns 500 with auth-related detail."""
    from app.core.config import get_settings

    if get_settings().LLM_PROVIDER != "azure":
        pytest.skip("Only when Azure is provider")

    def _mock_connectivity(*args, **kwargs):
        raise Exception("Error code: 401 - Invalid subscription key")

    monkeypatch.setattr("app.api.routes._test_azure_connectivity", _mock_connectivity)
    response = client.get("/api/test-azure")
    assert response.status_code == 500
    assert "detail" in response.json()
