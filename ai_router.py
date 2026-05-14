"""
AI router for GenericOrderingService.

Wraps the existing AI logic from ``updated_universal_service_bot.py`` and
exposes it as proper HTTP POST endpoints under ``/api/ai/*``.

Design notes:
- Endpoints DO NOT rewrite the bot logic. They wrap the existing
  ``UniversalServiceBot`` methods (chatAway, chatAway2, detect_with_keywords,
  detect_sector_with_ai, getIngredients, process_conversation,
  get_available_sectors, get_sector_info, is_sector_available).
- The bot is initialized lazily on first use; if initialization fails (e.g.,
  ChromaDB not seeded) the endpoints still respond, returning the failure
  message instead of crashing the server.
- If ``OPENROUTER_API_KEY`` is not set we return HTTP 503 with
  ``{"error": "AI not configured"}`` per the canonical pattern documented in
  ``_AUDIT/APPLY_SUMMARY.md``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/ai", tags=["AI"])


# ---------------------------------------------------------------------------
# Lazy bot singleton
# ---------------------------------------------------------------------------

_bot_lock = threading.Lock()
_bot_instance: Any = None
_bot_init_error: Optional[str] = None


def _require_api_key() -> None:
    """Canonical 503 if OPENROUTER_API_KEY isn't set."""
    if not os.getenv("OPENROUTER_API_KEY"):
        raise HTTPException(status_code=503, detail={"error": "AI not configured"})


def _get_bot() -> Any:
    """
    Lazily import and construct the UniversalServiceBot.
    Import is deferred so the FastAPI app can boot even if heavy deps
    (chromadb, langchain) are missing or the chroma DB hasn't been
    populated yet.
    """
    global _bot_instance, _bot_init_error

    if _bot_instance is not None:
        return _bot_instance

    with _bot_lock:
        if _bot_instance is not None:
            return _bot_instance
        if _bot_init_error is not None:
            raise HTTPException(status_code=503, detail={"error": _bot_init_error})

        try:
            # Local import to avoid loading heavy modules at FastAPI boot.
            from updated_universal_service_bot import UniversalServiceBot

            _bot_instance = UniversalServiceBot(
                sectors_directory="sectors",
                db_path="universal_chroma_database",
            )
            return _bot_instance
        except Exception as exc:  # noqa: BLE001
            _bot_init_error = f"UniversalServiceBot init failed: {exc}"
            raise HTTPException(status_code=503, detail={"error": _bot_init_error})


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    user_input: str = Field(..., description="User message to the bot")
    detected_sector: Optional[str] = Field(
        None, description="Force a specific sector; if omitted, sector is auto-detected"
    )


class ChatResponse(BaseModel):
    response: str
    detected_sector: Optional[str] = None


class DetectSectorRequest(BaseModel):
    user_input: str
    conversation_context: str = ""


class DetectSectorResponse(BaseModel):
    sector: str


class IngredientsRequest(BaseModel):
    sector: str
    user_input: str
    sector_prompt: str = ""


class IngredientsResponse(BaseModel):
    enhanced_prompt: str


class ConversationRequest(BaseModel):
    messages: List[str] = Field(..., description="Ordered list of user turns")


class ConversationResponse(BaseModel):
    responses: List[str]


class SectorInfoResponse(BaseModel):
    sector: str
    has_prompt: bool = False
    has_menu: bool = False
    has_rules: bool = False
    processor_loaded: bool = False
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Wraps UniversalServiceBot.chatAway."""
    _require_api_key()
    bot = _get_bot()
    response = bot.chatAway(req.user_input, req.detected_sector)
    sector = req.detected_sector
    if sector is None:
        # chatAway updates internal state; best-effort surface the sector it picked
        sector = getattr(bot, "state", {}).get("detected_sector")
    return ChatResponse(response=response, detected_sector=sector)


@router.post("/chat-stateless", response_model=ChatResponse)
def chat_stateless(req: ChatRequest) -> ChatResponse:
    """Wraps UniversalServiceBot.chatAway2 (no internal history)."""
    _require_api_key()
    bot = _get_bot()
    response = bot.chatAway2(req.user_input, req.detected_sector)
    return ChatResponse(response=response, detected_sector=req.detected_sector)


@router.post("/detect-sector", response_model=DetectSectorResponse)
def detect_sector(req: DetectSectorRequest) -> DetectSectorResponse:
    """Wraps UniversalServiceBot.detect_sector_with_ai."""
    _require_api_key()
    bot = _get_bot()
    sector = bot.detect_sector_with_ai(req.user_input, req.conversation_context)
    return DetectSectorResponse(sector=sector)


@router.post("/detect-sector-keywords", response_model=DetectSectorResponse)
def detect_sector_keywords(req: DetectSectorRequest) -> DetectSectorResponse:
    """Wraps UniversalServiceBot.detect_with_keywords (no LLM call -> no key check)."""
    bot = _get_bot()
    sector = bot.detect_with_keywords(req.user_input, req.conversation_context)
    return DetectSectorResponse(sector=sector)


@router.post("/ingredients", response_model=IngredientsResponse)
def ingredients(req: IngredientsRequest) -> IngredientsResponse:
    """Wraps UniversalServiceBot.getIngredients (menu/intent enrichment of a prompt)."""
    bot = _get_bot()
    prompt = req.sector_prompt
    if not prompt:
        try:
            prompt = bot.load_sector_prompt(req.sector)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"load_sector_prompt failed: {exc}")
    enhanced = bot.getIngredients(req.sector, req.user_input, prompt)
    return IngredientsResponse(enhanced_prompt=enhanced)


@router.post("/process-conversation", response_model=ConversationResponse)
def process_conversation(req: ConversationRequest) -> ConversationResponse:
    """Wraps UniversalServiceBot.process_conversation (multi-turn batch)."""
    _require_api_key()
    bot = _get_bot()
    responses = bot.process_conversation(req.messages)
    return ConversationResponse(responses=responses)


@router.get("/sectors", response_model=List[str])
def list_sectors() -> List[str]:
    """Wraps UniversalServiceBot.get_available_sectors."""
    bot = _get_bot()
    return list(bot.get_available_sectors())


@router.get("/sectors/{sector}", response_model=SectorInfoResponse)
def sector_info(sector: str) -> SectorInfoResponse:
    """Wraps UniversalServiceBot.get_sector_info."""
    bot = _get_bot()
    info: Dict[str, Any] = bot.get_sector_info(sector)
    if "error" in info:
        return SectorInfoResponse(sector=sector, error=info["error"])
    return SectorInfoResponse(**info)


@router.get("/status")
def status() -> Dict[str, Any]:
    """Operational status: whether key is set and whether bot has initialized."""
    return {
        "openrouter_configured": bool(os.getenv("OPENROUTER_API_KEY")),
        "bot_initialized": _bot_instance is not None,
        "bot_init_error": _bot_init_error,
    }
