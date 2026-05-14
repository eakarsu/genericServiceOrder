"""Order copilot for support agents — handle returns/refunds with policies."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter(prefix="/api/order-copilot", tags=["order-copilot"])

DEFAULT_POLICY = (
    "Returns: full refund within 14 days if unopened; 50% restocking fee 15-30 days; no returns after 30 days. "
    "Shipping: refundable only if defective. Loyalty points refunded proportionally."
)


class TriageReq(BaseModel):
    order: Dict[str, Any]
    customer_message: str
    policy: Optional[str] = None


class ActionPlanReq(BaseModel):
    order: Dict[str, Any]
    decision: str  # "approve_refund", "partial_refund", "deny", "replace"
    customer_message: str
    policy: Optional[str] = None


async def _ai(messages: List[Dict[str, str]], max_tokens: int = 600) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail="OPENROUTER_API_KEY not configured")
    model = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.3},
        )
        j = r.json()
        if "error" in j:
            raise HTTPException(status_code=502, detail=j["error"])
        return j["choices"][0]["message"]["content"]


@router.post("/triage")
async def triage(req: TriageReq):
    sys = (
        "You are a support copilot. Read the order and customer message, then return JSON "
        '{"intent":"refund|return|complaint|question","severity":"low|med|high","suggested_action":string,"empathy_score":number}. '
        f"Policy: {req.policy or DEFAULT_POLICY}"
    )
    usr = f"Order: {req.order}\nCustomer: {req.customer_message}"
    out = await _ai([{"role": "system", "content": sys}, {"role": "user", "content": usr}])
    return {"raw": out}


@router.post("/action-plan")
async def action_plan(req: ActionPlanReq):
    sys = (
        "You are a support agent assistant. Given an order and decision, produce: "
        "1) Suggested response to customer (markdown), 2) Internal action steps (list), 3) Policy citations. "
        f"Policy: {req.policy or DEFAULT_POLICY}"
    )
    usr = f"Decision: {req.decision}\nOrder: {req.order}\nCustomer message: {req.customer_message}"
    out = await _ai([{"role": "system", "content": sys}, {"role": "user", "content": usr}], max_tokens=900)
    return {"plan": out}
