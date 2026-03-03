"""
Retrieval correctness.
Upload document with unique phrase, query that phrase, assert retrieval_used and sources.
"""
import io
import pytest

from tests.conftest import RETRIEVAL_UNIQUE_PHRASE


def test_retrieval_used_and_sources_populated(client, sample_txt_file):
    """Upload doc with unique phrase, query it; retrieval_used==true, sources not empty, correct document name."""
    filename, content = sample_txt_file
    # Ensure clean state
    client.delete("/api/clear")
    # Upload
    up = client.post(
        "/api/upload",
        files={"file": (filename, io.BytesIO(content), "text/plain")},
    )
    assert up.status_code == 200
    # Query using the unique phrase so agent uses document_search
    response = client.post(
        "/api/query",
        json={"question": f"What is the key phrase or revenue target? Look for {RETRIEVAL_UNIQUE_PHRASE} or 10 million."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data.get("retrieval_used") is True, "Expected retrieval_used true when querying uploaded content"
    assert isinstance(data.get("sources"), list)
    assert len(data["sources"]) > 0, "Expected at least one source"
    # Document name should match uploaded file
    doc_names = [s.get("document") for s in data["sources"] if s.get("document")]
    assert filename in doc_names or any(filename in str(d) for d in doc_names), "Expected document name in sources"
    assert "reasoning_trace" in data
    assert any(
        step.get("action") == "document_search" for step in data["reasoning_trace"] if step
    ), "Expected document_search in reasoning trace"
