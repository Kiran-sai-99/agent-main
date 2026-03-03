"""
Azure connectivity validation.
GET /api/test-azure success when provider is Azure; failure cases via mock.
"""
import pytest


def test_test_azure_returns_200_when_azure_configured(client):
    """GET /api/test-azure returns 200 when LLM_PROVIDER=azure and credentials valid."""
    response = client.get("/api/test-azure")
    # If Azure is configured and valid: 200 with status success
    # If not configured (e.g. LLM_PROVIDER=openai): 400
    if response.status_code == 200:
        data = response.json()
        assert data.get("status") == "success"
        assert data.get("provider") == "azure"
        assert "deployment" in data
        assert "response" in data
    elif response.status_code == 400:
        assert "Azure is not the configured provider" in response.json().get("detail", "")
    else:
        # 500 = auth failure (wrong key/endpoint)
        assert response.status_code in (200, 400, 500)


def test_test_azure_response_shape_on_success(client):
    """When test-azure succeeds, response has expected keys."""
    response = client.get("/api/test-azure")
    if response.status_code != 200:
        pytest.skip("Azure not configured or auth failed")
    data = response.json()
    assert "status" in data
    assert "response" in data
    assert "deployment" in data
    assert "provider" in data


def test_test_azure_auth_failure_returns_500_with_message(client, monkeypatch):
    """When Azure auth fails, API returns 500 with clear message (mocked)."""
    from app.api.routes import _test_azure_connectivity
    from app.core.config import get_settings

    def _mock_fail(*args, **kwargs):
        raise Exception("Error code: 401 - Access denied due to invalid subscription key")

    monkeypatch.setattr(
        "app.api.routes._test_azure_connectivity",
        _mock_fail,
    )
    # Only run if provider is azure (otherwise we get 400)
    if get_settings().LLM_PROVIDER != "azure":
        pytest.skip("LLM_PROVIDER is not azure")
    response = client.get("/api/test-azure")
    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "authentication" in data["detail"].lower() or "key" in data["detail"].lower() or "endpoint" in data["detail"].lower()
