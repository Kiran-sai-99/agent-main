"""
Health and startup validation.
GET /health returns 200 and expected body.
"""
import pytest


def test_health_returns_200(client):
    """GET /health returns 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_body(client):
    """GET /health returns status and service name."""
    response = client.get("/health")
    data = response.json()
    assert data.get("status") == "ok"
    assert data.get("service") == "agentic-rag"


def test_health_content_type(client):
    """GET /health returns JSON."""
    response = client.get("/health")
    assert "application/json" in response.headers.get("content-type", "")
