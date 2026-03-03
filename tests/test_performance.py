"""
Performance sanity: measure response time and log duration.
"""
import time
import pytest


def test_health_response_time(client):
    """GET /health responds within 2 seconds."""
    start = time.perf_counter()
    response = client.get("/health")
    elapsed = time.perf_counter() - start
    assert response.status_code == 200
    assert elapsed < 2.0, f"Health took {elapsed:.2f}s"


def test_query_response_time_sanity(client):
    """POST /api/query (simple) completes within 30 seconds (LLM call)."""
    client.delete("/api/clear")
    start = time.perf_counter()
    response = client.post(
        "/api/query",
        json={"question": "What is 2 + 2? Reply with one number."},
    )
    elapsed = time.perf_counter() - start
    assert response.status_code == 200
    assert elapsed < 30.0, f"Query took {elapsed:.2f}s"
    # Log for visibility
    print(f"\n[perf] query duration: {elapsed:.2f}s")
