from types import SimpleNamespace

import pytest

from admin.services.openrouter import request_service_order_readiness


def test_openrouter_requires_provider_evidence(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "test-model")

    def fake_post(url, **kwargs):
        assert url == "https://openrouter.ai/api/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer test-key"
        return SimpleNamespace(
            status_code=200,
            json=lambda: {
                "id": "generation-123",
                "model": "provider/test-model",
                "choices": [{"message": {"content": "Authorize the technician; retain dispatch evidence; record customer approval."}}],
            },
        )

    monkeypatch.setattr("admin.services.openrouter.requests.post", fake_post)
    evidence = request_service_order_readiness("Emergency repair dispatch and customer sign-off")
    assert evidence["providerReceipt"]["requestId"] == "generation-123"
    assert "dispatch evidence" in evidence["result"]


def test_openrouter_rejects_noncanonical_endpoint(monkeypatch):
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://example.invalid/api/v1")
    with pytest.raises(RuntimeError, match="canonical OpenRouter API endpoint"):
        request_service_order_readiness("Emergency repair dispatch and customer sign-off")
