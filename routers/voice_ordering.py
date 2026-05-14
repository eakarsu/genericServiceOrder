"""Voice ordering agent — wires ElevenLabs + OpenRouter to a real intake flow.

TODO: configure credentials — ELEVEN_LABS_API_KEY for TTS, OPENROUTER_API_KEY for chat.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/voice-ordering", tags=["voice-ordering"])

# In-memory session store; replace with Redis for production.
_sessions: Dict[str, Dict[str, Any]] = {}


class StartReq(BaseModel):
    sector: Optional[str] = None
    channel: str = "phone"


class UtteranceReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class FinalizeReq(BaseModel):
    cart: Optional[List[Dict[str, Any]]] = None


async def _call_openrouter(messages: List[Dict[str, str]]) -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise HTTPException(status_code=503, detail={"error": "AI not configured"})
    model = os.environ.get("OPENROUTER_MODEL", "anthropic/claude-haiku-4.5")
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"model": model, "messages": messages, "max_tokens": 400, "temperature": 0.5},
        )
        j = r.json()
        if "error" in j:
            raise HTTPException(status_code=502, detail=j["error"])
        return j["choices"][0]["message"]["content"]


@router.post("/start")
async def start(req: StartReq):
    sid = f"voice_{int(time.time()*1000)}"
    _sessions[sid] = {"id": sid, "sector": req.sector, "channel": req.channel, "transcript": [], "cart": []}
    return {"sessionId": sid}


@router.post("/{session_id}/utterance")
async def utterance(session_id: str, req: UtteranceReq):
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    s["transcript"].append({"role": "user", "text": req.text})
    system = (
        f"You are a friendly voice ordering agent for the {s['sector'] or 'general'} sector. "
        "Take orders, confirm items, suggest add-ons. Reply in <=2 short sentences."
    )
    msgs = [{"role": "system", "content": system}]
    for t in s["transcript"][-6:]:
        msgs.append({"role": t["role"], "content": t["text"]})
    reply = await _call_openrouter(msgs)
    s["transcript"].append({"role": "assistant", "text": reply})
    return {"reply": reply}


@router.post("/{session_id}/finalize")
async def finalize(session_id: str, req: FinalizeReq):
    s = _sessions.pop(session_id, None)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    cart = req.cart or s.get("cart", [])
    total = sum((c.get("price", 0) * c.get("qty", 1)) for c in cart)
    return {"ok": True, "cart": cart, "total": total, "transcriptLen": len(s["transcript"])}


@router.get("/{session_id}")
async def get_session(session_id: str):
    s = _sessions.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return s
