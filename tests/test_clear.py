"""
Clear endpoint: DELETE /api/clear wipes vector store and sessions; query after returns no doc sources.
"""
import io
import pytest

from tests.conftest import RETRIEVAL_UNIQUE_PHRASE


def test_clear_then_query_no_sources(client, sample_txt_file):
    """Upload document, DELETE /api/clear, query document-based question -> retrieval_used false or no sources."""
    filename, content = sample_txt_file
    client.delete("/api/clear")
    up = client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(content), "text/plain")},
    )
    assert up.status_code == 200
    # Clear
    clear_resp = client.delete("/api/clear")
    assert clear_resp.status_code == 200
    data_clear = clear_resp.json()
    assert "message" in data_clear
    assert "clear" in data_clear["message"].lower() or "clear" in str(data_clear).lower()
    # Query same topic; should not find doc (store is empty)
    query_resp = client.post(
        "/api/query",
        json={"question": f"What is the revenue target or {RETRIEVAL_UNIQUE_PHRASE}?"},
    )
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    # After clear, either no retrieval or empty sources
    assert query_data.get("retrieval_used") is False or len(query_data.get("sources", [])) == 0


def test_clear_returns_200(client):
    """DELETE /api/clear returns 200."""
    response = client.delete("/api/clear")
    assert response.status_code == 200
