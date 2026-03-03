"""
General knowledge path.
Query that does not require documents; assert retrieval_used==false, sources empty.
"""
import pytest


def test_general_knowledge_no_retrieval(client):
    """Query 'What is the capital of France?' -> retrieval_used false, sources empty."""
    # Clear so no docs influence
    client.delete("/api/clear")
    response = client.post(
        "/api/query",
        json={"question": "What is the capital of France?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("retrieval_used") is False
    assert data.get("sources") == [] or len(data.get("sources", [])) == 0
    assert "answer" in data
    assert "Paris" in data["answer"] or "paris" in data["answer"].lower()
    assert "reasoning_trace" in data
