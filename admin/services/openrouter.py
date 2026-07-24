from __future__ import annotations

from datetime import datetime, timezone
import os

import requests

CANONICAL_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def request_service_order_readiness(order_summary: str) -> dict:
    base_url = os.getenv("OPENROUTER_BASE_URL", CANONICAL_OPENROUTER_BASE_URL).rstrip("/")
    if base_url != CANONICAL_OPENROUTER_BASE_URL:
        raise RuntimeError("OPENROUTER_BASE_URL must use the canonical OpenRouter API endpoint")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model = os.getenv("OPENROUTER_MODEL", "").strip()
    if not api_key or not model:
        raise RuntimeError("OpenRouter credentials and model must be configured")

    response = requests.post(
        f"{CANONICAL_OPENROUTER_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("FRONTEND_URL", "http://127.0.0.1"),
            "X-Title": "Generic Service Order Readiness",
        },
        json={
            "model": model,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a field-service operations reviewer. Give concise, concrete controls and never invent operational evidence.",
                },
                {
                    "role": "user",
                    "content": (
                        "Review this service-order workflow: " + order_summary
                        + ". Return exactly three short controls covering technician authorization, dispatch evidence, and customer approval."
                    ),
                },
            ],
        },
        timeout=45,
    )
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(f"OpenRouter request failed with status {response.status_code}")
    payload = response.json()
    request_id = str(payload.get("id") or "").strip()
    provider_model = str(payload.get("model") or "").strip()
    choices = payload.get("choices") or []
    content = str((choices[0].get("message") or {}).get("content") or "").strip() if choices else ""
    if not request_id or not provider_model or len(content) < 40:
        raise RuntimeError("OpenRouter response did not include substantive provider evidence")
    return {
        "result": content,
        "providerReceipt": {
            "provider": "openrouter",
            "requestId": request_id,
            "model": provider_model,
            "completedAt": datetime.now(timezone.utc).isoformat(),
        },
    }
