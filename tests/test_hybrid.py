"""
Hybrid path: document + general knowledge.
Upload doc with Project Phoenix budget; ask budget + inflation; assert retrieval_used and reasoning.
"""
import io
import pytest

from tests.conftest import HYBRID_PROJECT_PHRASE, HYBRID_TEST_CONTENT


def test_hybrid_retrieval_and_reasoning(client, sample_hybrid_txt_file):
    """Upload doc with Project Phoenix budget; ask budget and inflation; retrieval_used true, reasoning present."""
    filename, content = sample_hybrid_txt_file
    client.delete("/api/clear")
    up = client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(content), "text/plain")},
    )
    assert up.status_code == 200
    response = client.post(
        "/api/query",
        json={
            "question": "What was Project Phoenix budget and what is inflation in simple terms?"
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("retrieval_used") is True
    assert len(data.get("sources", [])) > 0
    assert "reasoning_trace" in data
    # Should mention document (Project Phoenix / 5 million) and possibly general (inflation)
    answer = data.get("answer", "").lower()
    assert "5" in answer or "five" in answer or "million" in answer or "phoenix" in answer or "budget" in answer
    # Trace should show document_search and possibly direct_llm
    actions = [s.get("action") for s in data["reasoning_trace"] if s.get("action")]
    assert "document_search" in actions
